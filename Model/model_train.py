import copy
import itertools
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import KFold, train_test_split

from Model.model_config import (
    BATCH_SIZE,
    DEVICE,
    EARLY_STOPPING_PATIENCE,
    EPOCHS,
    LABEL_THRESHOLD,
    MIN_DELTA,
    OPTIMIZERS,
    PARAM_GRID,
    SEED,
)
from Model.model_definition import GNNClassifier
from Model.model_utils import (
    calc_pos_class_weight,
    calculate_evaluation_metrics,
    get_prediction_probs,
    make_data_loaders,
    set_seed,
)
from Model.graph_dataset import TabularDataset


def _move_batch_to_device(inputs):
    if isinstance(inputs, dict):
        return {
            "node_x": {k: v.to(DEVICE) for k, v in inputs["node_x"].items()},
            "edge_index": {k: v.to(DEVICE) for k, v in inputs["edge_index"].items()},
            "node_batch": {k: v.to(DEVICE) for k, v in inputs["node_batch"].items()},
            "num_graphs": inputs["num_graphs"],
            "pr_number": inputs["pr_number"].to(DEVICE),
        }
    return inputs.to(DEVICE)


def _infer_graph_schema(X_df: pd.DataFrame, y_df: pd.DataFrame) -> tuple[Dict[str, int], Tuple[Tuple[str, str, str], ...]]:
    dataset = TabularDataset(X_df.head(1), y_df.head(1))
    if not dataset.graph_mode:
        raise ValueError("Expected graph mode with 'pr_number' column for GNN training.")

    sample, _ = dataset[0]
    node_x = sample["node_x"]
    edge_index = sample["edge_index"]

    in_dim_by_type = {nt: int(x.shape[1]) for nt, x in node_x.items()}
    rel_types = tuple(sorted(edge_index.keys()))

    if not in_dim_by_type:
        raise ValueError("Could not infer node feature dimensions from graph sample.")

    return in_dim_by_type, rel_types


def _build_gnn_model(in_dim_by_type, rel_types, hidden_dims, dropout, out_dim):
    if isinstance(hidden_dims, (tuple, list)):
        hidden_dim = int(hidden_dims[0])
        num_layers = max(1, len(hidden_dims))
    else:
        hidden_dim = int(hidden_dims)
        num_layers = 2

    if isinstance(dropout, (tuple, list)):
        dropout_value = float(dropout[0])
    else:
        dropout_value = float(dropout)

    return GNNClassifier(
        in_dim_by_type=in_dim_by_type,
        rel_types=rel_types,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout_value,
        out_dim=out_dim,
    ).to(DEVICE)


def train_final_GNN(
    X: pd.DataFrame,
    y: pd.DataFrame,
    hidden_dims: List[int],
    dropout: float,
    lr: float,
    weight_decay: float,
    batch_size: int,
    optimizer_choice: str,
):
    """Kept function name for backward compatibility; now trains GNN."""
    print(
        f"\nFinal GNN Training with HParams: "
        f"LR={lr}, WeightDecay={weight_decay}, Dropout={dropout}, HiddenDims={hidden_dims}"
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=SEED, shuffle=True
    )

    in_dim_by_type, rel_types = _infer_graph_schema(X_train, y_train)
    output_dim = y_train.shape[1]

    model = _build_gnn_model(
        in_dim_by_type=in_dim_by_type,
        rel_types=rel_types,
        hidden_dims=hidden_dims,
        dropout=dropout,
        out_dim=output_dim,
    )

    optimizer = OPTIMIZERS[optimizer_choice](model.parameters(), lr=lr, weight_decay=weight_decay)
    pos_weight = calc_pos_class_weight(y_train)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_loader, _ = make_data_loaders(X=X_train, y=y_train, batch_size=batch_size, shuffle=True)
    val_loader, _ = make_data_loaders(X=X_val, y=y_val, batch_size=batch_size, shuffle=False)

    n_no_improvement_loop = 0
    best_f1_micro = -np.inf

    for epoch in range(EPOCHS):
        model.train()

        for inputs, labels in train_loader:
            inputs = _move_batch_to_device(inputs)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        validation_probs, validation_labels = get_prediction_probs(model, val_loader)
        eval_metrics = calculate_evaluation_metrics(
            probs=validation_probs,
            y_true=validation_labels,
            threshold=LABEL_THRESHOLD,
        )

        val_accuracy, val_precision, val_recall, val_micro_f1 = eval_metrics.values()
        print(
            f"Epoch {epoch+1}/{EPOCHS}, Val F1-micro: {val_micro_f1:.4f}  "
            f"Val Recall: {val_recall:.4f}  Val Precision: {val_precision:.4f}  Val Acc: {val_accuracy:.4f}"
        )

        if val_micro_f1 > best_f1_micro + MIN_DELTA:
            best_f1_micro = val_micro_f1
            n_no_improvement_loop = 0
        else:
            n_no_improvement_loop += 1

        if n_no_improvement_loop > EARLY_STOPPING_PATIENCE:
            print(f"Early Stopping at Epoch: {epoch + 1}")
            break

    return model, None


def train_gnn_with_cv(X: pd.DataFrame, y: pd.DataFrame, optimizer_choice: str, k: int = 5, **kwargs):
    """Kept function name for backward compatibility; now does GNN CV."""
    kf = KFold(n_splits=k, shuffle=True, random_state=SEED)

    fold_f1_scores = []
    best_overall_f1 = -np.inf
    best_model_state = None
    best_hparams = None

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\n{'='*50}")
        print(f"Fold {fold+1}/{k}")
        print(f"{'='*50}")

        X_train = X.iloc[train_idx].reset_index(drop=True)
        y_train = y.iloc[train_idx].reset_index(drop=True)
        X_val = X.iloc[val_idx].reset_index(drop=True)
        y_val = y.iloc[val_idx].reset_index(drop=True)

        model_state, f1_score, hparams = tune_hyper_param(
            X_train, y_train, X_val, y_val, optimizer_choice, k_i=fold + 1, **kwargs
        )

        fold_f1_scores.append(f1_score)

        if f1_score > best_overall_f1:
            best_overall_f1 = f1_score
            best_model_state = model_state
            best_hparams = hparams

    average_f1 = np.mean(fold_f1_scores)
    std_f1 = np.std(fold_f1_scores)

    print(f"\n{'='*50}")
    print("Cross Validation Results:")
    print(f"{'='*50}")
    print(f"Average F1-micro: {average_f1:.4f} ± {std_f1:.4f}")
    print(f"Best Fold F1-micro: {best_overall_f1:.4f}")
    print(f"Best Hyperparameters: {best_hparams}")
    print(f"Fold F1 scores: {fold_f1_scores}")

    return best_model_state, average_f1, best_hparams, None


def tune_hyper_param(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_val: pd.DataFrame,
    optimizer_choice: str,
    k_i: int,
    batch_size: int = BATCH_SIZE,
):
    set_seed(seed=SEED)

    best_f1_micro = -np.inf
    best_hparams = None
    best_model_state = None

    in_dim_by_type, rel_types = _infer_graph_schema(X_train, y_train)
    output_dim = y_train.shape[1]

    for hparams in itertools.product(
        PARAM_GRID["lr"],
        PARAM_GRID["weight_decay"],
        PARAM_GRID["hidden_dims"],
        PARAM_GRID["dropout"],
    ):
        lr, weight_decay, hidden_dims, dropout = hparams

        print(
            f"\nTraining GNN with HParams(k={k_i}): "
            f"LR={lr}, WeightDecay={weight_decay}, Dropout={dropout}, HiddenDims={hidden_dims}"
        )

        model = _build_gnn_model(
            in_dim_by_type=in_dim_by_type,
            rel_types=rel_types,
            hidden_dims=hidden_dims,
            dropout=dropout,
            out_dim=output_dim,
        )

        optimizer = OPTIMIZERS[optimizer_choice](model.parameters(), lr=lr, weight_decay=weight_decay)
        pos_weight = calc_pos_class_weight(y_train)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        train_loader, _ = make_data_loaders(X=X_train, y=y_train, batch_size=batch_size, shuffle=True)
        val_loader, _ = make_data_loaders(X=X_val, y=y_val, batch_size=batch_size, shuffle=False)

        n_no_improvement_loop = 0

        for epoch in range(EPOCHS):
            model.train()

            for inputs, labels in train_loader:
                inputs = _move_batch_to_device(inputs)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            validation_probs, validation_labels = get_prediction_probs(model, val_loader)
            eval_metrics = calculate_evaluation_metrics(
                probs=validation_probs,
                y_true=validation_labels,
                threshold=LABEL_THRESHOLD,
            )

            val_accuracy, val_precision, val_recall, val_micro_f1 = eval_metrics.values()
            print(
                f"Epoch {epoch+1}/{EPOCHS}, Val F1-micro: {val_micro_f1:.4f}  "
                f"Val Recall: {val_recall:.4f}  Val Precision: {val_precision:.4f}  Val Acc: {val_accuracy:.4f}"
            )

            if val_micro_f1 > best_f1_micro + MIN_DELTA:
                best_f1_micro = val_micro_f1
                best_hparams = {
                    "lr": lr,
                    "weight_decay": weight_decay,
                    "hidden_dims": hidden_dims,
                    "dropout": dropout,
                }
                best_model_state = copy.deepcopy(model.state_dict())
                n_no_improvement_loop = 0
            else:
                n_no_improvement_loop += 1

            if n_no_improvement_loop > EARLY_STOPPING_PATIENCE:
                print(f"Early Stopping at Epoch: {epoch + 1}")
                break

    print("\nHyperparameter tuning complete!")
    print(f"Best Hyperparameters: {best_hparams}")
    print(f"Best Validation F1-micro: {best_f1_micro:.4f}")

    return best_model_state, best_f1_micro, best_hparams
