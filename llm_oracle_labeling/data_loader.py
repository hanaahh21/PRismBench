from typing import Dict, List
import pandas as pd

_ml_features_cache = None


def load_ml_features_dataset(ml_csv_path: str) -> pd.DataFrame:
    global _ml_features_cache

    if _ml_features_cache is None:
        print(f'Loading ML features dataset from: {ml_csv_path}')
        _ml_features_cache = pd.read_csv(ml_csv_path)
        print(f'Loaded {len(_ml_features_cache)} PRs from ML features dataset')

    return _ml_features_cache


def get_pr_numbers_from_csv(csv_path: str) -> List[int]:
    df = pd.read_csv(csv_path)
    print(f'Read {len(df)} PRs from CSV: {csv_path}')
    pr_list = [int(pr) for pr in df['pr_number'].tolist()]
    return pr_list


def lookup_pr_details(pr_number: int, ml_features_df: pd.DataFrame) -> Dict:
    pr_row = ml_features_df[ml_features_df['pr_number'] == pr_number]

    if len(pr_row) == 0:
        print(f'WARNING: PR #{pr_number} not found in ML features dataset')
        return {
            'pr_number': pr_number,
            'pr_title': f'PR #{pr_number} (not found in dataset)',
            'pr_description': '',
            'cleaned_diff': '',
            'szz_origin_issues': '[]',
            'github_pr_url': ''
        }

    pr_row = pr_row.iloc[0]

    return {
        'pr_number': pr_number,
        'pr_title': pr_row.get('pr_title') or '',
        'pr_description': pr_row.get('pr_description') or '',
        'cleaned_diff': pr_row.get('cleaned_diff') or '',
        'szz_origin_issues': pr_row.get('szz_origin_issues') or '[]',
        'github_pr_url': pr_row.get('github_pr_url') or ''
    }


def save_df(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False)
    print('Saved %s (%d rows)' % (path, len(df)))

