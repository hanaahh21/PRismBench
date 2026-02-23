import json
import time
import traceback
from typing import Callable, Dict, List, Optional

import pandas as pd
from tqdm import tqdm

from .config import LABEL_COLUMN_MAP, LLM_SLEEP, UNARY_LABELS
from .prompt_builder import build_unary_convergence_prompt, build_unary_layer0_prompt
from .response_handler import label_to_csv_column, parse_unary_decision


def _build_empty_row(pr_number: int, model_name: str, layer_number: int) -> Dict:
    row = {
        'pr_number': pr_number,
        'model': model_name,
        'layer': layer_number,
        'timestamp': time.time(),
        'success': True,
        'error_count': 0,
        'errors': '[]'
    }
    for _, col in LABEL_COLUMN_MAP.items():
        row[col] = 'No'
        row[f'{col}_reason'] = ''
    return row


def _get_prior_value(
    prior_layer_outputs: Optional[Dict[str, pd.DataFrame]],
    model_name: str,
    pr_number: int,
    col: str,
    default: str
) -> str:
    if not prior_layer_outputs or model_name not in prior_layer_outputs:
        return default
    df = prior_layer_outputs[model_name]
    match = df[df['pr_number'] == pr_number]
    if match.empty:
        return default
    value = match.iloc[0].get(col, default)
    return str(value) if value is not None else default


def _get_peer_feedback(
    prior_layer_outputs: Optional[Dict[str, pd.DataFrame]],
    current_model_name: str,
    pr_number: int,
    label_col: str
) -> List[Dict]:
    if not prior_layer_outputs:
        return []

    feedback = []
    for peer_model, peer_df in prior_layer_outputs.items():
        if peer_model == current_model_name:
            continue

        match = peer_df[peer_df['pr_number'] == pr_number]
        if match.empty:
            continue

        row = match.iloc[0]
        # Pass ONLY peer reasons, NOT their decisions (avoid sycophancy)
        reason = str(row.get(f'{label_col}_reason', ''))
        if reason.strip():  # Only include if there's actual reasoning
            feedback.append({
                'model': peer_model,
                'reasoning': reason
            })

    return feedback


def process_pr_with_unary_model(
    pr: Dict,
    model_name: str,
    query_function: Callable,
    layer_number: int,
    prior_layer_outputs: Optional[Dict[str, pd.DataFrame]] = None
) -> Dict:
    pr_number = int(pr.get('pr_number'))
    row = _build_empty_row(pr_number, model_name, layer_number)
    errors: List[Dict] = []

    for risk_label in UNARY_LABELS:
        label_col = label_to_csv_column(risk_label)

        try:
            if layer_number == 0:
                prompt = build_unary_layer0_prompt(pr, risk_label)
            else:
                previous_decision = _get_prior_value(
                    prior_layer_outputs,
                    model_name,
                    pr_number,
                    label_col,
                    'No'
                )
                previous_reason = _get_prior_value(
                    prior_layer_outputs,
                    model_name,
                    pr_number,
                    f'{label_col}_reason',
                    ''
                )
                peer_feedback = _get_peer_feedback(
                    prior_layer_outputs,
                    model_name,
                    pr_number,
                    label_col
                )
                prompt = build_unary_convergence_prompt(
                    pr=pr,
                    risk_label=risk_label,
                    current_model=model_name,
                    previous_decision=previous_decision,
                    previous_reason=previous_reason,
                    peer_feedback=peer_feedback,
                    layer_number=layer_number
                )

            response = query_function(prompt, pr)
            decision, reason = parse_unary_decision(response, risk_label)
            row[label_col] = decision
            row[f'{label_col}_reason'] = reason

        except Exception as exc:
            row['success'] = False
            row['error_count'] += 1
            row[label_col] = 'No'
            row[f'{label_col}_reason'] = f'Error: {exc}'
            errors.append({'label': risk_label, 'error': str(exc)})
            print(f'ERROR: {model_name} failed for PR {pr_number}, label {risk_label}: {exc}')
            traceback.print_exc()

        time.sleep(LLM_SLEEP)

    row['timestamp'] = time.time()
    row['errors'] = json.dumps(errors)
    return row


def process_all_prs_with_model_unary(
    pr_list: List[Dict],
    model_name: str,
    query_function: Callable,
    layer_number: int,
    output_csv_path: str,
    prior_layer_outputs: Optional[Dict[str, pd.DataFrame]] = None,
    save_output: bool = True
) -> pd.DataFrame:
    print('\n' + '=' * 80)
    print(f'Processing {len(pr_list)} PRs with {model_name} at Layer {layer_number}...')
    print('=' * 80)

    rows = []
    for pr in tqdm(pr_list, desc=f'{model_name} L{layer_number}'):
        row = process_pr_with_unary_model(
            pr=pr,
            model_name=model_name,
            query_function=query_function,
            layer_number=layer_number,
            prior_layer_outputs=prior_layer_outputs
        )
        rows.append(row)

    df = pd.DataFrame(rows)
    if save_output:
        df.to_csv(output_csv_path, index=False)
        print(f'Saved {model_name} Layer {layer_number} output: {output_csv_path}')

    return df
