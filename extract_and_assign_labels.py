from pathlib import Path

import pandas as pd

from llm_oracle_labeling.config import GLOBAL_ACCEPTED_FILE, LAYERS_BASE_DIR


ACCEPTED_TO_TRAINING_LABEL_MAP = {
    "bug": "bug",
    "security": "security",
    "performance": "performance",
    "maintainability": "code_quality_or_maintenability",
}


def _require_columns(df: pd.DataFrame, required_columns: list[str], source_name: str) -> None:
    missing_columns = [column for column in required_columns if column not in df.columns]
    if missing_columns:
        raise ValueError(f"{source_name} is missing required columns: {missing_columns}")


def extract_and_assign_labels(loop_number: int) -> None:
    """
    Convert accepted consensus labels into training rows for the next loop.

    Workflow:
    1. Read accepted PR-level consensus labels from Layers/accepted.csv
    2. Read the previous loop's unlabeled feature rows
    3. Join by pr_number to recover the full ML feature rows
    4. Convert Yes/No label decisions into 1/0 training targets
    5. Save as SamplingLoopData/loop_{loop_number}_data/labeled_train_data.csv
    """

    sampling_dir = Path("SamplingLoopData")
    accepted_labels_path = Path(LAYERS_BASE_DIR) / GLOBAL_ACCEPTED_FILE
    previous_loop_dir = sampling_dir / f"loop_{loop_number - 1}_data"
    previous_unlabeled_path = previous_loop_dir / "unlabeled_data.csv"
    next_loop_dir = sampling_dir / f"loop_{loop_number}_data"
    next_loop_dir.mkdir(parents=True, exist_ok=True)
    label_save_path = next_loop_dir / "labeled_train_data.csv"

    if not accepted_labels_path.exists():
        raise FileNotFoundError(
            f"Accepted consensus file not found: {accepted_labels_path}. "
            "Run the LLM convergence layers first."
        )

    if not previous_unlabeled_path.exists():
        raise FileNotFoundError(
            f"Previous loop unlabeled data not found: {previous_unlabeled_path}"
        )

    accepted_df = pd.read_csv(accepted_labels_path)
    unlabeled_df = pd.read_csv(previous_unlabeled_path)

    _require_columns(
        accepted_df,
        ["pr_number", *ACCEPTED_TO_TRAINING_LABEL_MAP.keys()],
        str(accepted_labels_path),
    )
    _require_columns(unlabeled_df, ["pr_number"], str(previous_unlabeled_path))

    accepted_labels_df = accepted_df[["pr_number", *ACCEPTED_TO_TRAINING_LABEL_MAP.keys()]].copy()

    labeled_train_df = unlabeled_df.merge(
        accepted_labels_df,
        on="pr_number",
        how="inner",
    )

    if len(labeled_train_df) != len(accepted_labels_df):
        raise ValueError(
            "Mismatch between accepted PRs and recovered feature rows. "
            f"Recovered {len(labeled_train_df)} rows for {len(accepted_labels_df)} accepted PRs."
        )

    for accepted_column, training_column in ACCEPTED_TO_TRAINING_LABEL_MAP.items():
        labeled_train_df[training_column] = (
            labeled_train_df[accepted_column]
            .astype(str)
            .str.strip()
            .str.lower()
            .map({"yes": 1, "no": 0})
        )

        if labeled_train_df[training_column].isna().any():
            invalid_values = sorted(labeled_train_df[accepted_column].dropna().astype(str).unique().tolist())
            raise ValueError(
                f"Unexpected label values found in column '{accepted_column}': {invalid_values}"
            )

    labeled_train_df = labeled_train_df.drop(columns=list(ACCEPTED_TO_TRAINING_LABEL_MAP.keys()))
    labeled_train_df.to_csv(label_save_path, index=False)

    print(f"Accepted consensus records: {len(accepted_df)}")
    print(f"Recovered labeled training rows: {len(labeled_train_df)}")
    print(f"Wrote labeled training data to: {label_save_path}")


if __name__ == "__main__":
    extract_and_assign_labels(1)
