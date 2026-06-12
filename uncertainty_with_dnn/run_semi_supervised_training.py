from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Add repo root to path so `Model` package imports work when run as a script.
sys.path.insert(0, str(Path(__file__).parent.parent))

from Model.data_config import LABEL_COLS
from Model.graph_dataset import TabularDataset
from Model.model_config import BATCH_SIZE, DEVICE, LABEL_THRESHOLD, SEED
from Model.model_definition import GNNClassifier
from Model.model_utils import (
    calc_pos_class_weight,
    calculate_evaluation_metrics,
    get_prediction_probs,
    make_data_loaders,
    set_seed,
)
from uncertainty_with_dnn.prepare_data import (
    load_labeled_test_data,
    load_labeled_train_data,
    resolve_dir,
)


def _parse_hidden_dims(raw: str) -> tuple[int, ...]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise ValueError("hidden_dims cannot be empty.")
    return tuple(int(x) for x in parts)


def _parse_dropout(raw: str) -> tuple[float, ...]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise ValueError("dropout cannot be empty.")
    return tuple(float(x) for x in parts)


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


def _infer_graph_schema(X_df: pd.DataFrame, y_df: pd.DataFrame) -> tuple[dict[str, int], tuple[tuple[str, str, str], ...]]:
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


def _build_model(
    in_dim_by_type: dict[str, int],
    rel_types: tuple[tuple[str, str, str], ...],
    hidden_dims: tuple[int, ...],
    dropout: tuple[float, ...],
    out_dim: int,
) -> GNNClassifier:
    hidden_dim = int(hidden_dims[0])
    num_layers = max(1, len(hidden_dims))
    dropout_value = float(dropout[0])

    return GNNClassifier(
        in_dim_by_type=in_dim_by_type,
        rel_types=rel_types,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout_value,
        out_dim=out_dim,
    ).to(DEVICE)


def _split_features_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    y = df[LABEL_COLS].astype(np.float32)
    X = df.drop(columns=LABEL_COLS)
    return X, y


def _soften_binary_targets(targets: torch.Tensor, epsilon: float) -> torch.Tensor:
    # 1 -> (1 - epsilon), 0 -> epsilon
    return targets * (1.0 - 2.0 * epsilon) + epsilon


def train_semi_supervised(
    loop_number: int,
    data_root: Path,
    model_monitor_dir: Path,
    hidden_dims: tuple[int, ...],
    dropout: tuple[float, ...],
    lr: float,
    weight_decay: float,
    batch_size: int,
    epochs: int,
    soft_epsilon: float,
    soft_loss_weight: float,
) -> dict:
    if not (0.0 <= soft_epsilon < 0.5):
        raise ValueError("soft_epsilon must be in [0.0, 0.5).")

    loop0_dir = resolve_dir(data_root, 0)
    target_loop_dir = resolve_dir(data_root, loop_number)

    hard_df = load_labeled_train_data(loop0_dir)
    loop_df = load_labeled_train_data(target_loop_dir)
    test_df = load_labeled_test_data(loop0_dir)

    hard_prs = set(hard_df["pr_number"].astype(int).tolist())
    soft_df = loop_df[~loop_df["pr_number"].astype(int).isin(hard_prs)].reset_index(drop=True)

    X_hard, y_hard = _split_features_labels(hard_df)
    X_soft, y_soft = _split_features_labels(soft_df)
    X_test, y_test = _split_features_labels(test_df)

    if len(X_hard) == 0:
        raise ValueError("No hard-labeled samples found in loop_0_data/labeled_train_data.csv.")

    in_dim_by_type, rel_types = _infer_graph_schema(X_hard, y_hard)
    model = _build_model(
        in_dim_by_type=in_dim_by_type,
        rel_types=rel_types,
        hidden_dims=hidden_dims,
        dropout=dropout,
        out_dim=y_hard.shape[1],
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    hard_pos_weight = calc_pos_class_weight(y_hard)
    hard_criterion = nn.BCEWithLogitsLoss(pos_weight=hard_pos_weight)
    soft_criterion = nn.BCEWithLogitsLoss()

    hard_loader, hard_n = make_data_loaders(X_hard, y_hard, batch_size=batch_size, shuffle=True)
    soft_loader, soft_n = make_data_loaders(X_soft, y_soft, batch_size=batch_size, shuffle=True)
    test_loader, _ = make_data_loaders(X_test, y_test, batch_size=batch_size, shuffle=False)

    print(f"[SemiSupervised] Loop: {loop_number}")
    print(f"[SemiSupervised] Hard samples (manual, loop_0): {hard_n}")
    print(f"[SemiSupervised] Soft samples (LLM, loop_{loop_number} minus loop_0): {soft_n}")
    print(
        f"[SemiSupervised] hidden_dims={hidden_dims}, dropout={dropout}, lr={lr}, "
        f"weight_decay={weight_decay}, epochs={epochs}, soft_epsilon={soft_epsilon}, "
        f"soft_loss_weight={soft_loss_weight}"
    )

    for epoch in range(max(1, int(epochs))):
        model.train()

        hard_loss_sum = 0.0
        hard_steps = 0
        for xb, yb in hard_loader:
            xb = _move_batch_to_device(xb)
            yb = yb.to(DEVICE)

            optimizer.zero_grad()
            logits = model(xb)
            loss_hard = hard_criterion(logits, yb)
            loss_hard.backward()
            optimizer.step()

            hard_loss_sum += float(loss_hard.item())
            hard_steps += 1

        soft_loss_sum = 0.0
        soft_steps = 0
        for xb, yb in soft_loader:
            xb = _move_batch_to_device(xb)
            yb = yb.to(DEVICE)
            yb_soft = _soften_binary_targets(yb, soft_epsilon)

            optimizer.zero_grad()
            logits = model(xb)
            loss_soft = soft_criterion(logits, yb_soft) * soft_loss_weight
            loss_soft.backward()
            optimizer.step()

            soft_loss_sum += float(loss_soft.item())
            soft_steps += 1

        hard_avg = hard_loss_sum / max(1, hard_steps)
        soft_avg = soft_loss_sum / max(1, soft_steps)
        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"hard_loss={hard_avg:.5f} ({hard_steps} steps) | "
            f"soft_loss={soft_avg:.5f} ({soft_steps} steps)"
        )

    test_probs, test_labels = get_prediction_probs(model, test_loader)
    test_metrics = calculate_evaluation_metrics(
        probs=test_probs,
        y_true=test_labels,
        threshold=LABEL_THRESHOLD,
    )

    print("\n[SemiSupervised] Evaluation on loop_0 labeled_test_data.csv")
    print(f"accuracy : {test_metrics['accuracy']:.4f}")
    print(f"precision: {test_metrics['precision']:.4f}")
    print(f"recall   : {test_metrics['recall']:.4f}")
    print(f"f1       : {test_metrics['f1']:.4f}")

    model_store_path = model_monitor_dir / "model_store" / f"semi_supervised_model_loop_{loop_number}.pt"
    model_store_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_store_path)

    metrics_path = model_monitor_dir / "eval_metrics" / "semi_supervised_model_evaluation.csv"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "loop_number": loop_number,
        "hard_samples": hard_n,
        "soft_samples": soft_n,
        "accuracy": test_metrics["accuracy"],
        "precision": test_metrics["precision"],
        "recall": test_metrics["recall"],
        "f1": test_metrics["f1"],
        "soft_epsilon": soft_epsilon,
        "soft_loss_weight": soft_loss_weight,
        "lr": lr,
        "weight_decay": weight_decay,
        "epochs": epochs,
        "hidden_dims": "|".join(str(x) for x in hidden_dims),
        "dropout": "|".join(str(x) for x in dropout),
        "model_path": str(model_store_path),
    }
    pd.DataFrame([row]).to_csv(
        metrics_path,
        mode="a",
        header=not metrics_path.exists(),
        index=False,
    )
    print(f"[SemiSupervised] Saved model: {model_store_path}")
    print(f"[SemiSupervised] Appended metrics: {metrics_path}")

    return test_metrics


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train semi-supervised GNN: hard labels from loop_0, soft labels from selected loop."
    )
    parser.add_argument("--loop-number", type=int, required=True, help="Target loop number (e.g., 19).")
    parser.add_argument("--data-root", type=Path, default=Path("SamplingLoopData"))
    parser.add_argument("--model-monitor-dir", type=Path, default=Path("ModelMonitoring"))
    parser.add_argument("--hidden-dims", type=str, default="256,128,64")
    parser.add_argument("--dropout", type=str, default="0.1,0.1,0.2")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument(
        "--soft-epsilon",
        type=float,
        default=0.1,
        help="Label smoothing factor for soft labels: 1->(1-e), 0->e.",
    )
    parser.add_argument(
        "--soft-loss-weight",
        type=float,
        default=0.7,
        help="Multiplier for soft-label loss term.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    set_seed(SEED)
    hidden_dims = _parse_hidden_dims(args.hidden_dims)
    dropout = _parse_dropout(args.dropout)

    train_semi_supervised(
        loop_number=args.loop_number,
        data_root=args.data_root,
        model_monitor_dir=args.model_monitor_dir,
        hidden_dims=hidden_dims,
        dropout=dropout,
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        epochs=args.epochs,
        soft_epsilon=args.soft_epsilon,
        soft_loss_weight=args.soft_loss_weight,
    )


if __name__ == "__main__":
    main()
