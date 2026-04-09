#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import re
from pathlib import Path


DECISION_COLUMNS = ["bug", "security", "performance", "maintainability"]
MODEL_FILE_PATTERN = re.compile(r"^Layer(?P<layer>\d+)_(?P<model>.+)\.csv$")


def read_model_rows(path: Path) -> dict[int, dict[str, str]]:
    rows: dict[int, dict[str, str]] = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pr_number_raw = row.get("pr_number", "").strip()
            if not pr_number_raw:
                continue
            pr_number = int(pr_number_raw)
            rows[pr_number] = row
    return rows


def discover_layers(layers_dir: Path) -> dict[int, dict[str, Path]]:
    out: dict[int, dict[str, Path]] = {}
    for child in layers_dir.iterdir():
        if not child.is_dir():
            continue
        if not child.name.isdigit():
            continue

        layer_number = int(child.name)
        model_files: dict[str, Path] = {}
        for f in child.glob("Layer*_*.csv"):
            m = MODEL_FILE_PATTERN.match(f.name)
            if not m:
                continue
            if int(m.group("layer")) != layer_number:
                continue
            model_files[m.group("model")] = f

        if model_files:
            out[layer_number] = model_files
    return dict(sorted(out.items(), key=lambda x: x[0]))


def layer_consensus_accepted(layer_number: int, model_files: dict[str, Path]) -> list[dict[str, str]]:
    model_rows = {model: read_model_rows(path) for model, path in model_files.items()}
    model_names = sorted(model_rows.keys())

    if len(model_names) < 3:
        print(
            f"Layer {layer_number}: expected 3 model files, found {len(model_names)} "
            f"({', '.join(model_names)}). Skipping."
        )
        return []

    all_prs: set[int] = set()
    for rows in model_rows.values():
        all_prs.update(rows.keys())

    accepted: list[dict[str, str]] = []
    for pr_number in sorted(all_prs):
        if any(pr_number not in model_rows[m] for m in model_names):
            continue

        anchor_model = model_names[0]
        anchor = model_rows[anchor_model][pr_number]
        disagree = False

        for col in DECISION_COLUMNS:
            anchor_val = (anchor.get(col) or "").strip()
            for m in model_names[1:]:
                val = (model_rows[m][pr_number].get(col) or "").strip()
                if val != anchor_val:
                    disagree = True
                    break
            if disagree:
                break

        if disagree:
            continue

        reason_by_col: dict[str, str] = {}
        for col in DECISION_COLUMNS:
            reason_col = f"{col}_reason"
            reason_by_col[reason_col] = json.dumps(
                {m: (model_rows[m][pr_number].get(reason_col) or "") for m in model_names},
                ensure_ascii=False,
            )

        model_outputs = json.dumps(
            {
                m: {col: (model_rows[m][pr_number].get(col) or "") for col in DECISION_COLUMNS}
                for m in model_names
            },
            ensure_ascii=False,
        )

        accepted.append(
            {
                "pr_number": str(pr_number),
                "consensus_models": ",".join(model_names),
                "consensus_layer": str(layer_number),
                "bug": (anchor.get("bug") or "").strip(),
                "bug_reason": reason_by_col["bug_reason"],
                "security": (anchor.get("security") or "").strip(),
                "security_reason": reason_by_col["security_reason"],
                "performance": (anchor.get("performance") or "").strip(),
                "performance_reason": reason_by_col["performance_reason"],
                "maintainability": (anchor.get("maintainability") or "").strip(),
                "maintainability_reason": reason_by_col["maintainability_reason"],
                "model_outputs": model_outputs,
            }
        )

    return accepted


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    fieldnames = [
        "pr_number",
        "consensus_models",
        "consensus_layer",
        "bug",
        "bug_reason",
        "security",
        "security_reason",
        "performance",
        "performance_reason",
        "maintainability",
        "maintainability_reason",
        "model_outputs",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    layers_dir = Path("Layers")
    layer_map = discover_layers(layers_dir)

    all_rows: list[dict[str, str]] = []
    for layer_number, model_files in layer_map.items():
        accepted_rows = layer_consensus_accepted(layer_number, model_files)
        layer_output_path = layers_dir / str(layer_number) / f"Layer_{layer_number}_accepted.csv"
        write_csv(layer_output_path, accepted_rows)
        print(f"Wrote {len(accepted_rows)} accepted rows -> {layer_output_path}")
        all_rows.extend(accepted_rows)

    # Keep a consolidated accepted view under Layers/0/accepted.csv for compatibility.
    consolidated_zero_path = layers_dir / "0" / "accepted.csv"
    write_csv(consolidated_zero_path, all_rows)
    print(f"Wrote {len(all_rows)} total accepted rows -> {consolidated_zero_path}")

    final_output_path = layers_dir / "final_accepted.csv"
    write_csv(final_output_path, all_rows)
    print(f"Wrote {len(all_rows)} total accepted rows -> {final_output_path}")


if __name__ == "__main__":
    main()
