from __future__ import annotations

import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

# Add parent directory to path to import Model package
sys.path.insert(0, str(Path(__file__).parent.parent))

from Model.model_definition import GNNClassifier
from Model.model_utils import get_prediction_probs, make_data_loaders
from Model.model_config import BATCH_SIZE


def calculate_prediction_entropy(
    model: GNNClassifier,
    unlabeled_X: pd.DataFrame,
    label_columns: Sequence[str],
) -> np.ndarray:
    """
    Prediction entropy based on mean predicted probability from the final model of each Active learning loop.
    Returns array shape (n_samples,).
    """
    full_entropy = np.full(len(unlabeled_X), -1.0, dtype=np.float32)

    if "pr_number" in unlabeled_X.columns:
        graph_root = Path("graphs/apache_kafka")
        missing_mask = ~unlabeled_X["pr_number"].astype(int).apply(
            lambda pr: (graph_root / f"PR_{pr}.pkl").exists()
        )
        if missing_mask.any():
            missing_count = int(missing_mask.sum())
            print(
                f"WARNING: {missing_count} unlabeled PRs have no graph file in {graph_root}. "
                "They will be assigned uncertainty_score=-1 and skipped from selection."
            )
        valid_positions = np.flatnonzero((~missing_mask).to_numpy())
        valid_df = unlabeled_X.loc[~missing_mask].copy().reset_index(drop=True)
    else:
        valid_positions = np.arange(len(unlabeled_X))
        valid_df = unlabeled_X.copy().reset_index(drop=True)

    if valid_df.empty:
        return full_entropy

    # DataLoader expects label columns thus preparing dummy column
    dummy_y = pd.DataFrame(np.zeros((len(valid_df), len(label_columns)), dtype=np.float32))

    # Disable shuffle to preserve row order for PR-to-entropy mapping.
    loader, _ = make_data_loaders(valid_df, dummy_y, BATCH_SIZE, shuffle=False)

    # probs shape = (n_samples, n_labels)
    probs, _ = get_prediction_probs(model, loader)

    # avoid 0/1 probability values
    eps = 1e-7
    probs = np.clip(probs, eps, 1.0 - eps)

    # entropy calculation
    entropy = -(
        probs * np.log(probs) +
        (1.0 - probs) * np.log(1.0 - probs)
    )

    entropy = np.nan_to_num(entropy, nan=0.0)
    mean_entropy = entropy.mean(axis=1).astype(np.float32)
    full_entropy[valid_positions] = mean_entropy

    return full_entropy
