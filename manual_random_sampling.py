from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import pandas as pd


LABEL_COLUMNS = ["bug", "security", "performance", "maintainability"]


def _parse_loop_number(loop_dir_name: str) -> int:
    try:
        return int(loop_dir_name.split("_")[-1])
    except ValueError:
        return 10**9


def _discover_loop_files(layers_dir: Path) -> List[Tuple[int, Path]]:
    loop_files: List[Tuple[int, Path]] = []
    for loop_dir in layers_dir.glob("loop_*"):
        if not loop_dir.is_dir():
            continue
        loop_number = _parse_loop_number(loop_dir.name)
        if loop_number == 10**9:
            continue
        csv_path = loop_dir / "final_accepted.csv"
        if csv_path.exists():
            loop_files.append((loop_number, csv_path))
    return sorted(loop_files, key=lambda x: x[0])


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Sample PRs from Layers/loop_*/final_accepted.csv and export two CSV files."
        )
    )
    parser.add_argument("--layers-dir", default="Layers", help="Path to Layers directory.")
    parser.add_argument("--samples-per-loop", type=int, default=8, help="Number of random PRs per loop.")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed.")
    parser.add_argument(
        "--output-with-labels",
        default="manual_random_samples_with_labels.csv",
        help="Output CSV path containing PR, loop, and labels.",
    )
    parser.add_argument(
        "--output-pr-loop",
        default="manual_random_samples_pr_loop.csv",
        help="Output CSV path containing only PR and loop.",
    )
    args = parser.parse_args()

    layers_dir = Path(args.layers_dir)
    if not layers_dir.exists():
        raise FileNotFoundError(f"Layers directory not found: {layers_dir}")

    loop_files = _discover_loop_files(layers_dir)
    if not loop_files:
        raise FileNotFoundError(
            f"No final_accepted.csv files found under loop_* folders in {layers_dir}"
        )

    rows_with_labels = []
    rows_pr_loop = []

    for loop_number, csv_path in loop_files:
        df = pd.read_csv(csv_path)
        if "pr_number" not in df.columns:
            print(f"Skipping loop {loop_number}: missing pr_number column in {csv_path}")
            continue

        missing_labels = [col for col in LABEL_COLUMNS if col not in df.columns]
        if missing_labels:
            print(
                f"Skipping loop {loop_number}: missing label columns {missing_labels} in {csv_path}"
            )
            continue

        if df.empty:
            print(f"Skipping loop {loop_number}: empty file {csv_path}")
            continue

        sample_size = min(args.samples_per_loop, len(df))
        sampled_df = df.sample(n=sample_size, random_state=args.seed + loop_number)

        for _, row in sampled_df.iterrows():
            pr_number = row["pr_number"]
            rows_pr_loop.append(
                {
                    "pr_number": pr_number,
                    "loop_number": loop_number,
                }
            )
            rows_with_labels.append(
                {
                    "pr_number": pr_number,
                    "loop_number": loop_number,
                    "bug_label": row["bug"],
                    "security_label": row["security"],
                    "performance_label": row["performance"],
                    "maintainability_label": row["maintainability"],
                }
            )

        if sample_size < args.samples_per_loop:
            print(
                f"Loop {loop_number}: requested {args.samples_per_loop}, "
                f"but only {sample_size} available in {csv_path}"
            )

    with_labels_df = pd.DataFrame(
        rows_with_labels,
        columns=[
            "pr_number",
            "loop_number",
            "bug_label",
            "security_label",
            "performance_label",
            "maintainability_label",
        ],
    )
    pr_loop_df = pd.DataFrame(rows_pr_loop, columns=["pr_number", "loop_number"])

    with_labels_path = Path(args.output_with_labels)
    pr_loop_path = Path(args.output_pr_loop)
    with_labels_df.to_csv(with_labels_path, index=False)
    pr_loop_df.to_csv(pr_loop_path, index=False)

    print(f"Saved with-label samples: {with_labels_path} ({len(with_labels_df)} rows)")
    print(f"Saved PR/loop samples: {pr_loop_path} ({len(pr_loop_df)} rows)")


if __name__ == "__main__":
    main()

