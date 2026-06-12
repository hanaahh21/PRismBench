from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

import Model.model_train as mt
from Model.data_config import LABEL_COLS
from Model.model_config import BATCH_SIZE, DEVICE, LABEL_THRESHOLD, MIN_DELTA, SEED
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
    load_unlabeled_data,
    resolve_dir,
)


def _move_graph_batch_to_device(graph_batch: dict) -> dict:
    return {
        "node_x": {k: v.to(DEVICE) for k, v in graph_batch["node_x"].items()},
        "edge_index": {k: v.to(DEVICE) for k, v in graph_batch["edge_index"].items()},
        "node_batch": {k: v.to(DEVICE) for k, v in graph_batch["node_batch"].items()},
        "num_graphs": graph_batch["num_graphs"],
        "pr_number": graph_batch["pr_number"].to(DEVICE),
    }


def _forward_graph_repr(model: nn.Module, batch_graph: dict) -> torch.Tensor:
    original_head = model.head
    model.head = nn.Identity()
    try:
        graph_repr = model(batch_graph)
    finally:
        model.head = original_head
    return graph_repr


def _augment_batch_graph(batch_graph: dict, feat_drop_prob: float, edge_drop_prob: float) -> dict:
    node_x_aug = {}
    for ntype, x in batch_graph["node_x"].items():
        if feat_drop_prob <= 0:
            node_x_aug[ntype] = x
            continue
        keep_mask = (torch.rand_like(x) > feat_drop_prob).float()
        node_x_aug[ntype] = x * keep_mask

    edge_index_aug = {}
    for rel, ei in batch_graph["edge_index"].items():
        if edge_drop_prob <= 0 or ei.numel() == 0:
            edge_index_aug[rel] = ei
            continue
        keep = torch.rand(ei.shape[1], device=ei.device) > edge_drop_prob
        if keep.any():
            edge_index_aug[rel] = ei[:, keep]
        else:
            edge_index_aug[rel] = ei[:, :0]

    return {
        "node_x": node_x_aug,
        "edge_index": edge_index_aug,
        "node_batch": batch_graph["node_batch"],
        "num_graphs": batch_graph["num_graphs"],
        "pr_number": batch_graph["pr_number"],
    }


def _info_nce_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    z1 = F.normalize(z1, p=2, dim=1)
    z2 = F.normalize(z2, p=2, dim=1)
    reps = torch.cat([z1, z2], dim=0)
    sim = torch.mm(reps, reps.t()) / temperature

    n = z1.size(0)
    labels = torch.arange(n, device=z1.device)
    labels = torch.cat([labels + n, labels], dim=0)

    mask = torch.eye(2 * n, device=z1.device, dtype=torch.bool)
    sim = sim.masked_fill(mask, -1e9)
    return F.cross_entropy(sim, labels)


def _tune_threshold(probs: np.ndarray, y_true: np.ndarray) -> Tuple[float, Dict]:
    best_t = LABEL_THRESHOLD
    best_metrics = calculate_evaluation_metrics(probs, y_true, best_t)
    best_f1 = best_metrics["f1"]
    for t in np.arange(0.1, 0.901, 0.05):
        metrics = calculate_evaluation_metrics(probs, y_true, float(t))
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_t = float(t)
            best_metrics = metrics
    return best_t, best_metrics


def _train_supervised(
    model: nn.Module,
    X: pd.DataFrame,
    y: pd.DataFrame,
    lr: float,
    weight_decay: float,
    epochs: int,
) -> Tuple[nn.Module, float]:
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=SEED, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    pos_weight = calc_pos_class_weight(y_train)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    train_loader, _ = make_data_loaders(X_train, y_train, BATCH_SIZE, shuffle=True)
    val_loader, _ = make_data_loaders(X_val, y_val, BATCH_SIZE, shuffle=False)

    best_f1 = -np.inf
    best_threshold = LABEL_THRESHOLD
    best_state = None
    no_improve = 0

    for epoch in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb = _move_graph_batch_to_device(xb) if isinstance(xb, dict) else xb.to(DEVICE)
            yb = yb.to(DEVICE)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

        val_probs, val_labels = get_prediction_probs(model, val_loader)
        tuned_t, val_metrics = _tune_threshold(val_probs, val_labels)
        val_f1 = val_metrics["f1"]
        if val_f1 > best_f1 + MIN_DELTA:
            best_f1 = val_f1
            best_threshold = tuned_t
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1

        if no_improve > mt.EARLY_STOPPING_PATIENCE:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_threshold


def _pretrain_ssl(
    model: nn.Module,
    X_pretrain: pd.DataFrame,
    epochs: int,
    lr: float,
    feat_drop_prob: float,
    edge_drop_prob: float,
) -> None:
    dummy_y = pd.DataFrame(np.zeros((len(X_pretrain), len(LABEL_COLS)), dtype=np.float32), columns=LABEL_COLS)
    loader, _ = make_data_loaders(X_pretrain, dummy_y, BATCH_SIZE, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(epochs):
        model.train()
        for xb, _ in loader:
            if not isinstance(xb, dict):
                continue
            xb = _move_graph_batch_to_device(xb)
            v1 = _augment_batch_graph(xb, feat_drop_prob=feat_drop_prob, edge_drop_prob=edge_drop_prob)
            v2 = _augment_batch_graph(xb, feat_drop_prob=feat_drop_prob, edge_drop_prob=edge_drop_prob)
            z1 = _forward_graph_repr(model, v1)
            z2 = _forward_graph_repr(model, v2)
            loss = _info_nce_loss(z1, z2, temperature=0.2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


def _filter_rows_with_graphs(df: pd.DataFrame, graph_root: Path) -> pd.DataFrame:
    if "pr_number" not in df.columns:
        return df
    keep = df["pr_number"].astype(int).apply(lambda pr: (graph_root / f"PR_{pr}.pkl").exists())
    return df.loc[keep].copy().reset_index(drop=True)


def run_experiment(
    loop_number: int,
    data_root: Path,
    out_csv: Path,
    pretrain_epochs: int,
    finetune_epochs: int,
) -> None:
    set_seed(SEED)
    mt.EPOCHS = finetune_epochs

    loop_dir = resolve_dir(data_root, loop_number - 1)
    labeled_train = load_labeled_train_data(loop_dir)
    unlabeled = load_unlabeled_data(loop_dir)
    labeled_test = load_labeled_test_data(resolve_dir(data_root, 0))

    y = labeled_train[LABEL_COLS]
    X = labeled_train.drop(columns=LABEL_COLS)
    y_test = labeled_test[LABEL_COLS]
    X_test = labeled_test.drop(columns=LABEL_COLS)

    graph_root = Path("graphs/apache_kafka")
    X = _filter_rows_with_graphs(X, graph_root)
    y = y.loc[X.index].reset_index(drop=True)
    X_test = _filter_rows_with_graphs(X_test, graph_root)
    y_test = y_test.loc[X_test.index].reset_index(drop=True)

    _, _, best_hparams, _ = mt.train_gnn_with_cv(X, y, "adam", k=3)

    in_dim_by_type, rel_types = mt._infer_graph_schema(X, y)
    out_dim = y.shape[1]

    baseline = mt._build_gnn_model(
        in_dim_by_type=in_dim_by_type,
        rel_types=rel_types,
        hidden_dims=best_hparams["hidden_dims"],
        dropout=best_hparams["dropout"],
        out_dim=out_dim,
    )
    baseline, baseline_t = _train_supervised(
        baseline,
        X=X,
        y=y,
        lr=best_hparams["lr"],
        weight_decay=best_hparams["weight_decay"],
        epochs=finetune_epochs,
    )

    ssl_model = mt._build_gnn_model(
        in_dim_by_type=in_dim_by_type,
        rel_types=rel_types,
        hidden_dims=best_hparams["hidden_dims"],
        dropout=best_hparams["dropout"],
        out_dim=out_dim,
    )
    pretrain_pool = pd.concat([X, unlabeled], ignore_index=True, sort=False)
    pretrain_pool = pretrain_pool.drop_duplicates(subset=["pr_number"], keep="first")
    pretrain_pool = _filter_rows_with_graphs(pretrain_pool, graph_root)
    _pretrain_ssl(
        ssl_model,
        X_pretrain=pretrain_pool,
        epochs=pretrain_epochs,
        lr=best_hparams["lr"],
        feat_drop_prob=0.1,
        edge_drop_prob=0.1,
    )
    ssl_model, ssl_t = _train_supervised(
        ssl_model,
        X=X,
        y=y,
        lr=best_hparams["lr"],
        weight_decay=best_hparams["weight_decay"],
        epochs=finetune_epochs,
    )

    test_loader, _ = make_data_loaders(X_test, y_test, BATCH_SIZE, shuffle=False)
    b_probs, b_labels = get_prediction_probs(baseline, test_loader)
    s_probs, s_labels = get_prediction_probs(ssl_model, test_loader)
    b_metrics = calculate_evaluation_metrics(b_probs, b_labels, baseline_t)
    s_metrics = calculate_evaluation_metrics(s_probs, s_labels, ssl_t)

    rows = [
        {
            "loop_number": loop_number,
            "model": "baseline_supervised",
            "threshold": baseline_t,
            "accuracy": b_metrics["accuracy"],
            "precision": b_metrics["precision"],
            "recall": b_metrics["recall"],
            "f1": b_metrics["f1"],
        },
        {
            "loop_number": loop_number,
            "model": "ssl_pretrain_then_finetune",
            "threshold": ssl_t,
            "accuracy": s_metrics["accuracy"],
            "precision": s_metrics["precision"],
            "recall": s_metrics["recall"],
            "f1": s_metrics["f1"],
        },
    ]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    if out_csv.exists():
        old = pd.read_csv(out_csv)
        old = old[old["loop_number"] != loop_number]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(out_csv, index=False)

    print("\n=== SSL Pretraining Ablation ===")
    print(df[df["loop_number"] == loop_number].to_string(index=False))
    print(
        f"\nDelta (ssl - baseline) F1: "
        f"{(s_metrics['f1'] - b_metrics['f1']):.6f}"
    )
    print(f"Saved: {out_csv}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run baseline vs SSL-pretraining ablation on one active-learning loop.")
    parser.add_argument("--loop-number", type=int, default=19)
    parser.add_argument("--data-root", type=Path, default=Path("SamplingLoopData"))
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("ModelMonitoring/eval_metrics/ssl_pretrain_ablation.csv"),
    )
    parser.add_argument("--pretrain-epochs", type=int, default=8)
    parser.add_argument("--finetune-epochs", type=int, default=40)
    args = parser.parse_args()

    run_experiment(
        loop_number=args.loop_number,
        data_root=args.data_root,
        out_csv=args.out_csv,
        pretrain_epochs=args.pretrain_epochs,
        finetune_epochs=args.finetune_epochs,
    )


if __name__ == "__main__":
    main()

