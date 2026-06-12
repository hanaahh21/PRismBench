from __future__ import annotations

import os
import sys
from pathlib import Path

# Add parent directory to path to import Model package
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import torch

from .prepare_data import (
    resolve_dir,
    load_labeled_train_data,
    load_unlabeled_data,
    load_labeled_test_data,
)
from .uncertainty_metric_calc import calculate_prediction_entropy
from .k_center_greedy import k_center_greedy_from_uncertain
from Model.model_train import (
    train_final_GNN,
    train_gnn_with_cv,
    calculate_evaluation_metrics,
    retrain_final_GNN_from_checkpoint,
    train_gnn_with_cv_warm_start,
)
from Model.data_config import LABEL_COLS
from Model.model_config import BATCH_SIZE, GNN_AUTO_LABEL_F1_THRESHOLD, LABEL_THRESHOLD
from Model.model_utils import get_prediction_probs, make_data_loaders

WARM_START_MAX_EPOCHS = 20


def run_uncertainty_selection(
    ml_features_csv: str,
    loop_number: int = 1,
    data_root: Path = Path("SamplingLoopData"),
    output_dir: Path = Path("UncertainPoints"),
    model_monitor_dir: Path = Path("ModelMonitoring"),
    n_top_uncertain: int = 100,
    k_diverse: int = 25,
    metric: str = "euclidean",
    verbose: bool = False,
) -> tuple[pd.DataFrame, bool]:
    """
    Run uncertainty sampling with entropy + k-center greedy diversity.

    Returns:
        (selected_pr_df, gnn_auto_labeled_all)
    """
########################## Load unlabeled train data ################################################################
    loop_unlabeled_dir = resolve_dir(data_root, loop_number - 1)
    unlabeled = load_unlabeled_data(loop_unlabeled_dir)

    # filtering only the prs with szz issues
    szz_issue_path = Path(__file__).parent.parent / ml_features_csv
    szz_origin_check = pd.read_csv(szz_issue_path)
    prs_with_szz = szz_origin_check.loc[~szz_origin_check["szz_origin_issues"].isna(), "pr_number"]

    unlabeled = unlabeled[unlabeled["pr_number"].isin(prs_with_szz)]

########################## Load labeled train data ###################################################################
    labeled_train = load_labeled_train_data(loop_unlabeled_dir)

########################## Load labeled test data ####################################################################
    labeled_test_dir = resolve_dir(data_root, 0)
    labeled_test = load_labeled_test_data(labeled_test_dir)

#####################################################################################################################

    if "pr_number" not in unlabeled.columns:
        raise ValueError("Unlabeled CSV must contain 'pr_number' column (needed for output mapping).")

    X_unlabeled = unlabeled.copy()

########################## Train GNN with K-fold CV to pick best hparams ############################################
    y = labeled_train[LABEL_COLS]
    X = labeled_train.drop(columns=LABEL_COLS)

    # keep pr_number for graph lookup
    _, average_f1, best_hparams, _ = train_gnn_with_cv(X, y, "adam")

########################## Train final GNN with best hparams #########################################################

    final_model, _ = train_final_GNN(
        X=X,
        y=y,
        hidden_dims=best_hparams["hidden_dims"],
        dropout=best_hparams["dropout"],
        lr=best_hparams["lr"],
        weight_decay=best_hparams["weight_decay"],
        batch_size=BATCH_SIZE,
        optimizer_choice="adam",
    )

########################## Store the final trained model in each loop ################################################

    model_store_path = model_monitor_dir / "model_store" / f"final_model_{loop_number}.pt"
    model_store_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(final_model.state_dict(), model_store_path)

########################## Test final GNN on Golden Seed test set ####################################################
    y_test = labeled_test[LABEL_COLS]
    X_test = labeled_test.drop(columns=LABEL_COLS)

    test_loader, _ = make_data_loaders(X_test, y_test, BATCH_SIZE, shuffle=False)
    test_prob, test_label = get_prediction_probs(final_model, test_loader)

    test_eval_metric = calculate_evaluation_metrics(test_prob, test_label, LABEL_THRESHOLD)
    current_f1 = float(test_eval_metric["f1"])
    threshold_reached = current_f1 > GNN_AUTO_LABEL_F1_THRESHOLD

    print(
        f"[Loop {loop_number}] GNN F1: {current_f1:.4f} | "
        f"Threshold: {GNN_AUTO_LABEL_F1_THRESHOLD:.4f} | "
        f"Status: {'REACHED' if threshold_reached else 'NOT REACHED'}"
    )

########################## Visualize evaluation results of testing of final GNN ######################################

    eval_metric_csv = model_monitor_dir / "eval_metrics" / "final_model_evaluation.csv"
    eval_metric_csv.parent.mkdir(parents=True, exist_ok=True)
    eval_metric_row = {
        "loop_number": loop_number,
        "accuracy": test_eval_metric["accuracy"],
        "precision": test_eval_metric["precision"],
        "recall": test_eval_metric["recall"],
        "f1": test_eval_metric["f1"],
    }
    pd.DataFrame([eval_metric_row]).to_csv(
        eval_metric_csv,
        mode="a",
        header=not os.path.exists(eval_metric_csv),
        index=False,
    )

########################## Optional warm-start retraining evaluation (separate metrics file) #########################

    previous_model_path = model_monitor_dir / "model_store" / f"final_model_{loop_number - 1}.pt"
    if previous_model_path.exists():
        _, warm_cv_average_f1, warm_best_hparams, _ = train_gnn_with_cv_warm_start(
            X=X,
            y=y,
            optimizer_choice="adam",
            checkpoint_state_dict=torch.load(previous_model_path, map_location="cpu"),
            hidden_dims=best_hparams["hidden_dims"],
            dropout=best_hparams["dropout"],
        )

        warm_start_model, _ = retrain_final_GNN_from_checkpoint(
            X=X,
            y=y,
            hidden_dims=warm_best_hparams["hidden_dims"],
            dropout=warm_best_hparams["dropout"],
            lr=warm_best_hparams["lr"],
            weight_decay=warm_best_hparams["weight_decay"],
            batch_size=BATCH_SIZE,
            optimizer_choice="adam",
            checkpoint_state_dict=torch.load(previous_model_path, map_location="cpu"),
            max_epochs=WARM_START_MAX_EPOCHS,
        )

        retrained_model_store_path = model_monitor_dir / "model_store" / f"retrained_model_{loop_number}.pt"
        retrained_model_store_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(warm_start_model.state_dict(), retrained_model_store_path)

        warm_test_prob, warm_test_label = get_prediction_probs(warm_start_model, test_loader)
        warm_test_eval_metric = calculate_evaluation_metrics(warm_test_prob, warm_test_label, LABEL_THRESHOLD)

        warm_eval_metric_csv = model_monitor_dir / "eval_metrics" / "final_model_evaluation_retrained.csv"
        warm_eval_metric_csv.parent.mkdir(parents=True, exist_ok=True)
        warm_eval_metric_row = {
            "loop_number": loop_number,
            "warm_cv_f1": warm_cv_average_f1,
            "baseline_f1": test_eval_metric["f1"],
            "retrained_f1": warm_test_eval_metric["f1"],
            "accuracy": warm_test_eval_metric["accuracy"],
            "precision": warm_test_eval_metric["precision"],
            "recall": warm_test_eval_metric["recall"],
            "f1": warm_test_eval_metric["f1"],
        }
        pd.DataFrame([warm_eval_metric_row]).to_csv(
            warm_eval_metric_csv,
            mode="a",
            header=not os.path.exists(warm_eval_metric_csv),
            index=False,
        )
        print(
            f"[Loop {loop_number}] Warm-start retrain F1: {warm_test_eval_metric['f1']:.4f} "
            f"(baseline {test_eval_metric['f1']:.4f}) | "
            f"saved: {retrained_model_store_path}"
        )
    else:
        print(
            f"[Loop {loop_number}] Skipping warm-start retrain evaluation: "
            f"no previous model found at {previous_model_path}"
        )

########################## Switch to full GNN auto-labeling once model quality is high enough ########################

    if threshold_reached:
        print(
            f"GNN F1 {current_f1:.4f} reached auto-label threshold "
            f"{GNN_AUTO_LABEL_F1_THRESHOLD:.4f}. Labeling all remaining unlabeled PRs with GNN."
        )

        unlabeled_targets = pd.DataFrame(
            data=np.zeros((len(X_unlabeled), len(LABEL_COLS)), dtype=np.float32),
            columns=LABEL_COLS,
        )
        unlabeled_loader, _ = make_data_loaders(X_unlabeled, unlabeled_targets, BATCH_SIZE, shuffle=False)
        unlabeled_probs, _ = get_prediction_probs(final_model, unlabeled_loader)
        unlabeled_preds = (unlabeled_probs > LABEL_THRESHOLD).astype(int)

        auto_labeled_df = unlabeled.copy()
        for label_index, label_name in enumerate(LABEL_COLS):
            auto_labeled_df[label_name] = unlabeled_preds[:, label_index]

        current_loop_dir = data_root / f"loop_{loop_number}_data"
        current_loop_dir.mkdir(parents=True, exist_ok=True)

        auto_labeled_path = current_loop_dir / "labeled_train_data.csv"
        auto_labeled_df.to_csv(auto_labeled_path, index=False)

        current_loop_unlabeled_path = current_loop_dir / "unlabeled_data.csv"
        unlabeled.iloc[0:0].to_csv(current_loop_unlabeled_path, index=False)

        selected_pr_df = pd.DataFrame(
            {"pr_number": unlabeled["pr_number"].values, "uncertainty_score": np.zeros(len(unlabeled))}
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        out_scores_path = output_dir / f"loop_{loop_number}_selected.csv"
        selected_pr_df.to_csv(out_scores_path, index=False)

        if verbose:
            print(f"Wrote: {auto_labeled_path}")
            print(f"Wrote: {current_loop_unlabeled_path}")
            print(f"Wrote: {out_scores_path}")
            print(f"Average CV F1: {average_f1:.4f}")

        return selected_pr_df, True

########################## Calculate prediction entropy of unlabeled set using final GNN #############################

    entropy_values = calculate_prediction_entropy(final_model, X_unlabeled, LABEL_COLS)

    assert len(unlabeled) == len(entropy_values), "Unlabeled Data Frame and entropy value list has different lengths."

    pr_entropy_df = pd.DataFrame(
        {"pr_number": unlabeled["pr_number"].values, "uncertainty_score": entropy_values}
    ).sort_values("uncertainty_score", ascending=False)

    # Pool then diversify
    pool_df = pr_entropy_df.head(n_top_uncertain)
    pool_idxs = pool_df.index.to_numpy()

########################## Apply k-center greedy algorithm to pick diverse points #####################################

    selected_idxs = k_center_greedy_from_uncertain(
        X_unlabeled=X_unlabeled.values,
        uncertain_idxs=pool_idxs,
        metric=metric,
        k=k_diverse,
    )

    selected_pr_df = pr_entropy_df.loc[selected_idxs].sort_values("uncertainty_score", ascending=False)

    # Output
    output_dir.mkdir(parents=True, exist_ok=True)
    out_scores_path = output_dir / f"loop_{loop_number}_selected.csv"
    selected_pr_df.to_csv(out_scores_path, index=False)

########################## Prepare next loop unlabeled data ###########################################################

    next_loop_unlabeled_df = unlabeled[~unlabeled["pr_number"].isin(selected_pr_df["pr_number"])]

    current_loop_dir = data_root / f"loop_{loop_number}_data"
    current_loop_dir.mkdir(parents=True, exist_ok=True)
    current_loop_unlabeled_path = current_loop_dir / "unlabeled_data.csv"
    next_loop_unlabeled_df.to_csv(current_loop_unlabeled_path, index=False)

    if verbose:
        print(f"Wrote: {out_scores_path}")
        print(f"Wrote: {current_loop_unlabeled_path}")
        print(f"Average CV F1: {average_f1:.4f}")

    print(f"Lenth of unlabled df: {len(unlabeled)}")
    print(f"Lenth of next unlabeled df: {len(next_loop_unlabeled_df)}")

    return selected_pr_df, False
