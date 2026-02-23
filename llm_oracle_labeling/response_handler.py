import json
from typing import Dict, List, Tuple

from .config import LABEL_COLUMN_MAP, UNARY_LABELS


def parse_llm_json(response_text: str) -> Dict:
    try:
        return json.loads(response_text)
    except Exception as e1:
        text = response_text.strip()
        if text.startswith('```'):
            lines = text.split('\n')
            if lines[0].startswith('```'):
                lines = lines[1:]
            if lines and lines[-1].strip() == '```':
                lines = lines[:-1]
            text = '\n'.join(lines)

        try:
            return json.loads(text)
        except Exception:
            pass

        start = text.find('{')
        end = text.rfind('}')
        if start != -1 and end != -1 and end > start:
            try:
                json_str = text[start:end + 1]
                return json.loads(json_str)
            except Exception as e3:
                print('Could not parse JSON.')
                print(f'Response length: {len(response_text)} chars')
                print(f'First 800 chars: {response_text[:800]}')
                print(f'Last 200 chars: {response_text[-200:]}')
                print(f'Parse errors: {str(e1)[:100]}, {str(e3)[:100]}')

    raise ValueError('Could not parse JSON from LLM response')


def label_to_csv_column(label: str) -> str:
    if label not in LABEL_COLUMN_MAP:
        raise ValueError(f'Unknown label for CSV mapping: {label}')
    return LABEL_COLUMN_MAP[label]


def parse_unary_decision(response: Dict, risk_label: str) -> Tuple[str, str]:
    """Parse unary response and return canonical ('Yes'|'No', reason)."""
    if risk_label not in UNARY_LABELS:
        raise ValueError(f'Unsupported unary risk label: {risk_label}')

    if not isinstance(response, dict):
        return 'No', 'Invalid response format; defaulted to No.'

    decision_raw = (
        response.get('decision')
        or response.get('answer')
        or response.get('label_decision')
        or response.get('is_risk')
    )

    reason = str(response.get('reason') or response.get('rationale') or response.get('explanation') or '').strip()

    if isinstance(decision_raw, bool):
        decision = 'Yes' if decision_raw else 'No'
    else:
        decision_text = str(decision_raw or '').strip().lower()
        if decision_text in {'yes', 'y', 'true', '1'}:
            decision = 'Yes'
        elif decision_text in {'no', 'n', 'false', '0'}:
            decision = 'No'
        else:
            # Defensive fallback: infer from text when model misses schema.
            if 'yes' in decision_text:
                decision = 'Yes'
            else:
                decision = 'No'

    if not reason:
        reason = 'No reason provided by model.'

    return decision, reason
