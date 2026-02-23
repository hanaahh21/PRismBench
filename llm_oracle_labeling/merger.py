import json
from typing import Dict, Tuple

import pandas as pd
from tqdm import tqdm

from .config import LABEL_COLUMN_MAP


DECISION_COLUMNS = list(LABEL_COLUMN_MAP.values())


def _norm_decision(value) -> str:
    text = str(value).strip().lower()
    return 'Yes' if text in {'yes', 'y', 'true', '1'} else 'No'


def _collect_model_row(df: pd.DataFrame, pr_number: int) -> pd.Series:
    match = df[df['pr_number'] == pr_number]
    if match.empty:
        return pd.Series(dtype=object)
    return match.iloc[0]


def merge_model_results_and_apply_consensus(
    model_results: Dict[str, pd.DataFrame],
    consensus_threshold: int = 3
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Strict consensus for unary pipeline:
    PR is accepted only if all decision columns match across all participating models.
    """
    print('\n' + '=' * 80)
    print('Merging unary model outputs and applying strict full-label consensus...')
    print('=' * 80)

    if not model_results:
        return pd.DataFrame(), pd.DataFrame()

    model_names = sorted(model_results.keys())
    all_pr_numbers = set()
    for df in model_results.values():
        all_pr_numbers.update(df['pr_number'].tolist())

    accepted_rows = []
    unlabeled_rows = []

    for pr_number in tqdm(sorted(all_pr_numbers), desc='Consensus'):
        per_model_rows = {}
        for model_name, df in model_results.items():
            per_model_rows[model_name] = _collect_model_row(df, pr_number)

        if any(row.empty for row in per_model_rows.values()):
            unlabeled_rows.append({
                'pr_number': pr_number,
                'reason': 'Missing model output for at least one model',
                'model_outputs': json.dumps({
                    m: {'present': not per_model_rows[m].empty}
                    for m in model_names
                })
            })
            continue

        # Use first model as anchor for exact per-label agreement.
        anchor_model = model_names[0]
        anchor_row = per_model_rows[anchor_model]

        agreed = True
        disagreement_labels = []
        for col in DECISION_COLUMNS:
            anchor_decision = _norm_decision(anchor_row.get(col, 'No'))
            for model_name in model_names[1:]:
                other_decision = _norm_decision(per_model_rows[model_name].get(col, 'No'))
                if anchor_decision != other_decision:
                    agreed = False
                    disagreement_labels.append(col)
                    break

        model_summary = {}
        for model_name, row in per_model_rows.items():
            model_summary[model_name] = {
                col: _norm_decision(row.get(col, 'No'))
                for col in DECISION_COLUMNS
            }

        if agreed and len(model_names) >= consensus_threshold:
            accepted_row = {
                'pr_number': pr_number,
                'consensus_models': ','.join(model_names),
                'consensus_layer': int(anchor_row.get('layer', 0))
            }
            for col in DECISION_COLUMNS:
                accepted_row[col] = _norm_decision(anchor_row.get(col, 'No'))

                reasons = {
                    model_name: str(per_model_rows[model_name].get(f'{col}_reason', ''))
                    for model_name in model_names
                }
                accepted_row[f'{col}_reason'] = json.dumps(reasons)

            accepted_row['model_outputs'] = json.dumps(model_summary)
            accepted_rows.append(accepted_row)
        else:
            unlabeled_rows.append({
                'pr_number': pr_number,
                'reason': f'Disagreement on labels: {sorted(set(disagreement_labels))}',
                'model_outputs': json.dumps(model_summary)
            })

    accepted_df = pd.DataFrame(accepted_rows)
    unlabeled_df = pd.DataFrame(unlabeled_rows)

    total = len(accepted_df) + len(unlabeled_df)
    print(f'Consensus complete. Accepted={len(accepted_df)}, Unlabeled={len(unlabeled_df)}, Total={total}')
    return accepted_df, unlabeled_df
