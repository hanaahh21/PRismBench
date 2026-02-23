import json
from typing import Dict, List

from .config import MAX_PROMPT_CHARS, UNARY_LABELS

LABEL_DESCRIPTIONS = {
    'Bug Risk': (
        'Observable functional failures such as incorrect behavior, crashes, exceptions, '
        'deadlocks, race conditions, broken features, or failing tests.'
    ),
    'Security Risk': (
        'Observable security failures such as auth/authz bypass, data leakage, '
        'privilege escalation, or policy violations.'
    ),
    'Performance Risk': (
        'Observable performance degradation such as increased latency, lower throughput, '
        'resource exhaustion, memory/CPU regressions, or scalability issues.'
    ),
    'Maintainability Risk': (
        'Observed fragility or design debt that directly led to regressions or made bug '
        'fixes difficult in follow-up issues.'
    )
}


def _parse_szz_issues(pr: Dict) -> List[Dict]:
    szz_issues_raw = pr.get('szz_origin_issues', '[]')
    if isinstance(szz_issues_raw, str):
        try:
            szz_issues = json.loads(szz_issues_raw)
        except Exception:
            szz_issues = []
    elif isinstance(szz_issues_raw, list):
        szz_issues = szz_issues_raw
    else:
        szz_issues = []

    allowed_issue_fields = {'issue_key', 'key', 'title', 'summary', 'description', 'priority'}
    cleaned = []
    for issue in szz_issues:
        if isinstance(issue, dict):
            cleaned.append({k: issue[k] for k in allowed_issue_fields if k in issue})
    return cleaned


def _build_common_pr_payload(pr: Dict) -> Dict:
    payload = {
        'pr_number': pr.get('pr_number'),
        'pr_title': pr.get('pr_title', ''),
        'pr_description': pr.get('pr_description', ''),
        'szz_origin_issues': _parse_szz_issues(pr)
    }
    prompt_str = json.dumps(payload)
    if len(prompt_str) > MAX_PROMPT_CHARS:
        payload['truncation_notice'] = 'PR content was truncated to fit the context window.'
    return payload


def _stringify_prompt(prompt_dict: Dict) -> str:
    prompt_str = json.dumps(prompt_dict, indent=2)
    if len(prompt_str) > MAX_PROMPT_CHARS:
        prompt_dict['truncation_notice'] = 'Prompt was truncated to fit the LLM context window.'
        prompt_str = json.dumps(prompt_dict, indent=2)
    return prompt_str


def build_unary_layer0_prompt(pr: Dict, risk_label: str) -> str:
    if risk_label not in UNARY_LABELS:
        raise ValueError(f'Unsupported unary risk label: {risk_label}')

    prompt = {
        'task': 'Unary risk decision for one PR and one risk label',
        'decision_label': risk_label,
        'decision_label_definition': LABEL_DESCRIPTIONS[risk_label],
        'instructions': [
            'Decide ONLY for the selected decision_label.',
            'Use only observable evidence from SZZ origin issues and PR details.',
            'Do not speculate or infer hypothetical failures.',
            "Return JSON only with keys: decision, reason.",
            "decision must be exactly 'Yes' or 'No'."
        ],
        'output_schema': {
            'decision': 'Yes|No',
            'reason': 'Short evidence-grounded justification.'
        },
        'pr_context': _build_common_pr_payload(pr)
    }
    return _stringify_prompt(prompt)


def build_unary_convergence_prompt(
    pr: Dict,
    risk_label: str,
    current_model: str,
    previous_decision: str,
    previous_reason: str,
    peer_feedback: List[Dict],
    layer_number: int
) -> str:
    if risk_label not in UNARY_LABELS:
        raise ValueError(f'Unsupported unary risk label: {risk_label}')

    prompt = {
        'task': 'Convergence unary risk reconsideration for one PR and one risk label',
        'layer': layer_number,
        'decision_label': risk_label,
        'decision_label_definition': LABEL_DESCRIPTIONS[risk_label],
        'instructions': [
            f'You are {current_model}. Re-evaluate your previous decision for this same label.',
            'Read your own prior decision and peer reasoning observations (NOT their conclusions).',
            'Make your own independent decision. Keep if evidence supports it; change only if peer observations reveal new evidence.',
            'Use only observable evidence from PR and SZZ origin issues.',
            "Return JSON only with keys: decision, reason.",
            "decision must be exactly 'Yes' or 'No'."
        ],
        'your_previous_decision': {
            'decision': previous_decision,
            'reasoning': previous_reason
        },
        'peer_observations_and_reasoning': peer_feedback,
        'output_schema': {
            'decision': 'Yes|No',
            'reason': 'Short evidence-grounded justification.'
        },
        'pr_context': _build_common_pr_payload(pr)
    }
    return _stringify_prompt(prompt)


# Backward-compatibility wrapper
SYSTEM_INSTRUCTIONS = {
    'task': 'Unary per-label PR risk labeling. Return JSON only.'
}


def build_prompt(pr: Dict) -> str:
    return build_unary_layer0_prompt(pr, 'Bug Risk')
