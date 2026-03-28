import os
from typing import Dict, List, Tuple, Optional

import pandas as pd

from .config import (
    BATCH_SIZE,
    GLOBAL_ACCEPTED_FILE,
    GLOBAL_UNLABELED_FILE,
    LAYER_ACCEPTED_FILE_PATTERN,
    LAYER_CSV_SUBDIR,
    LAYER_UNLABELED_FILE_PATTERN,
    LAYERS_BASE_DIR,
    MAX_CONVERGENCE_LAYERS,
    TARGET_ACCEPTED_COUNT,
)
from .data_loader import get_pr_numbers_from_csv, load_ml_features_dataset, lookup_pr_details
from .llm_client import query_gemma, query_llama, query_mistral
from .merger import merge_model_results_and_apply_consensus
from .model_processor import process_all_prs_with_model_unary

MODELS_CONFIG = [
    {'name': 'Qwen3', 'query_fn': query_gemma},
    {'name': 'Gemma2', 'query_fn': query_llama},
    {'name': 'Llama3.1', 'query_fn': query_mistral},
]

# Global state for pipeline execution
_PIPELINE_STATE = {
    'pr_numbers': None,
    'pr_details_map': None,
    'ml_features_df': None,
}


def _chunk_list(items: List[int], chunk_size: int) -> List[List[int]]:
    return [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _build_pr_details_map(pr_numbers: List[int], ml_features_df: pd.DataFrame) -> Dict[int, Dict]:
    pr_map = {}
    for pr_number in pr_numbers:
        details = lookup_pr_details(int(pr_number), ml_features_df)
        if not details.get('pr_title') or str(details['pr_title']).endswith('(not found in dataset)'):
            continue
        pr_map[int(pr_number)] = details
    return pr_map


def _load_prior_layer_outputs(layer_number: int) -> Optional[Dict[str, pd.DataFrame]]:
    """Load model outputs from previous layer for convergence."""
    if layer_number == 0:
        return None
    
    prior_layer_dir = os.path.join(LAYERS_BASE_DIR, str(layer_number - 1))
    prior_outputs = {}
    
    for model_cfg in MODELS_CONFIG:
        model_name = model_cfg['name']
        # In previous layers, outputs are in the layer directory (merged from all batches)
        prior_file = os.path.join(prior_layer_dir, f'Layer{layer_number - 1}_{model_name}.csv')
        if os.path.exists(prior_file):
            prior_outputs[model_name] = pd.read_csv(prior_file)
            print(f'Loaded prior outputs for {model_name} from {prior_file}')
        else:
            print(f'WARNING: Prior outputs not found for {model_name} at {prior_file}')
    
    return prior_outputs if prior_outputs else None


def _save_global_state(base_dir: str, accepted_df: pd.DataFrame, unlabeled_df: pd.DataFrame):
    """Save global accepted and unlabeled CSVs (overwrites previous)."""
    accepted_path = os.path.join(base_dir, GLOBAL_ACCEPTED_FILE)
    unlabeled_path = os.path.join(base_dir, GLOBAL_UNLABELED_FILE)
    accepted_df.to_csv(accepted_path, index=False)
    unlabeled_df.to_csv(unlabeled_path, index=False)
    print(f'\n✅ Updated global accepted: {accepted_path} ({len(accepted_df)})')
    print(f'✅ Updated global unlabeled: {unlabeled_path} ({len(unlabeled_df)})\n')


def _run_single_batch(
    layer_number: int,
    batch_number: int,
    batch_pr_numbers: List[int],
    pr_details_map: Dict[int, Dict],
    prior_layer_outputs: Optional[Dict[str, pd.DataFrame]] = None
) -> Tuple[Dict[str, pd.DataFrame], str]:
    """Run a single batch of PRs through all 3 models."""
    batch_dir = os.path.join(LAYERS_BASE_DIR, str(layer_number), f'Batch{batch_number}', LAYER_CSV_SUBDIR)
    _ensure_dir(batch_dir)

    model_outputs: Dict[str, pd.DataFrame] = {}
    batch_prs = [pr_details_map[pn] for pn in batch_pr_numbers if pn in pr_details_map]

    print(f'\n{"="*80}')
    print(f'Layer {layer_number} | Batch {batch_number} | Processing {len(batch_prs)} PRs')
    print(f'{"="*80}\n')

    for model_cfg in MODELS_CONFIG:
        model_name = model_cfg['name']
        query_fn = model_cfg['query_fn']

        print(f'Layer {layer_number} | Batch {batch_number} | {model_name}')
        batch_df = process_all_prs_with_model_unary(
            pr_list=batch_prs,
            model_name=model_name,
            query_function=query_fn,
            layer_number=layer_number,
            output_csv_path='',
            prior_layer_outputs=prior_layer_outputs,
            save_output=False,
        )
        
        batch_model_path = os.path.join(batch_dir, f'Layer{layer_number}_{model_name}.csv')
        batch_df.to_csv(batch_model_path, index=False)
        print(f'  ✓ Saved batch results: {batch_model_path}\n')
        model_outputs[model_name] = batch_df

    return model_outputs, batch_dir


def _run_layer_0_with_batches(
    pr_numbers: List[int],
    pr_details_map: Dict[int, Dict],
    num_batches: int = 4,
    batch_size: int = 25
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    """
    Run Layer 0 explicitly in 4 batches.
    Each batch is processed separately and results are saved per-batch.
    Final merged results are saved to Layers/Layer0/
    """
    print('\n' + '='*80)
    print('LAYER 0: INITIAL EVALUATION (4 Batches of 25 PRs)')
    print('='*80)

    # Split into 4 explicit batches
    batches = _chunk_list(pr_numbers[:num_batches * batch_size], batch_size)
    all_model_dataframes: Dict[str, List[pd.DataFrame]] = {cfg['name']: [] for cfg in MODELS_CONFIG}
    
    for batch_idx, batch_pr_numbers in enumerate(batches):
        batch_model_outputs, _ = _run_single_batch(
            layer_number=0,
            batch_number=batch_idx,
            batch_pr_numbers=batch_pr_numbers,
            pr_details_map=pr_details_map,
            prior_layer_outputs=None
        )
        
        for model_name, df in batch_model_outputs.items():
            all_model_dataframes[model_name].append(df)

    # Merge all batch results per model
    merged_model_outputs = {}
    layer0_dir = os.path.join(LAYERS_BASE_DIR, '0')
    _ensure_dir(layer0_dir)
    
    for model_name in all_model_dataframes.keys():
        if all_model_dataframes[model_name]:
            merged_df = pd.concat(all_model_dataframes[model_name], ignore_index=True)
            final_path = os.path.join(layer0_dir, f'Layer0_{model_name}.csv')
            merged_df.to_csv(final_path, index=False)
            print(f'✓ Saved merged Layer0 results for {model_name}: {final_path}')
            merged_model_outputs[model_name] = merged_df

    # Apply consensus across all Layer 0 results
    accepted_df, unlabeled_df = merge_model_results_and_apply_consensus(merged_model_outputs, consensus_threshold=3)

    # Save global accepted and unlabeled
    _save_global_state(LAYERS_BASE_DIR, accepted_df, unlabeled_df)

    print(f'\n{"="*80}')
    print(f'Layer 0 Complete: {len(accepted_df)} accepted, {len(unlabeled_df)} unlabeled')
    print(f'{"="*80}\n')

    return merged_model_outputs, accepted_df, unlabeled_df


def _run_convergence_layer(
    layer_number: int,
    pr_numbers: List[int],
    pr_details_map: Dict[int, Dict],
    batch_size: int = 25
) -> Tuple[Dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame]:
    """
    Run a convergence layer (Layer 1+).
    Unlabeled PRs are re-evaluated with prior decisions and peer feedback.
    """
    print('\n' + '='*80)
    print(f'CONVERGENCE LAYER {layer_number}')
    print(f'Processing {len(pr_numbers)} unlabeled PRs')
    print('='*80)

    # Load prior layer outputs for context
    prior_layer_outputs = _load_prior_layer_outputs(layer_number)
    
    batches = _chunk_list(pr_numbers, batch_size)
    all_model_dataframes: Dict[str, List[pd.DataFrame]] = {cfg['name']: [] for cfg in MODELS_CONFIG}
    
    for batch_idx, batch_pr_numbers in enumerate(batches):
        batch_model_outputs, _ = _run_single_batch(
            layer_number=layer_number,
            batch_number=batch_idx,
            batch_pr_numbers=batch_pr_numbers,
            pr_details_map=pr_details_map,
            prior_layer_outputs=prior_layer_outputs
        )
        
        for model_name, df in batch_model_outputs.items():
            all_model_dataframes[model_name].append(df)

    # Merge all batch results per model
    merged_model_outputs = {}
    layer_dir = os.path.join(LAYERS_BASE_DIR, str(layer_number))
    _ensure_dir(layer_dir)
    
    for model_name in all_model_dataframes.keys():
        if all_model_dataframes[model_name]:
            merged_df = pd.concat(all_model_dataframes[model_name], ignore_index=True)
            final_path = os.path.join(layer_dir, f'Layer{layer_number}_{model_name}.csv')
            merged_df.to_csv(final_path, index=False)
            print(f'✓ Saved merged Layer{layer_number} results for {model_name}: {final_path}')
            merged_model_outputs[model_name] = merged_df

    # Apply consensus
    layer_accepted_df, layer_unlabeled_df = merge_model_results_and_apply_consensus(merged_model_outputs, consensus_threshold=3)

    # Update global state
    _save_global_state(LAYERS_BASE_DIR, layer_accepted_df, layer_unlabeled_df)

    print(f'\n{"="*80}')
    print(f'Layer {layer_number} Complete: {len(layer_accepted_df)} accepted, {len(layer_unlabeled_df)} unlabeled')
    print(f'{"="*80}\n')

    return merged_model_outputs, layer_accepted_df, layer_unlabeled_df


def initialize_pipeline(
    ml_features_csv: str,
    pr_list_csv: str,
    max_items: int = None,
) -> None:
    """Initialize pipeline with data. Call this once before running layers."""
    global _PIPELINE_STATE
    
    print(f'\n{"="*80}')
    print('PIPELINE INITIALIZATION')
    print(f'{"="*80}\n')
    
    pr_numbers = get_pr_numbers_from_csv(pr_list_csv)
    if max_items:
        pr_numbers = pr_numbers[:max_items]

    ml_features_df = load_ml_features_dataset(ml_features_csv)
    pr_details_map = _build_pr_details_map(pr_numbers, ml_features_df)

    valid_pr_numbers = [pn for pn in pr_numbers if pn in pr_details_map]
    
    _PIPELINE_STATE['pr_numbers'] = valid_pr_numbers
    _PIPELINE_STATE['pr_details_map'] = pr_details_map
    _PIPELINE_STATE['ml_features_df'] = ml_features_df
    
    print(f'✓ Pipeline initialized with {len(valid_pr_numbers)} PRs')
    print(f'{"="*80}\n')


def run_layer(layer_number: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run a specific layer manually.
    
    Args:
        layer_number: 0 for initial evaluation, 1+ for convergence layers
        
    Returns:
        (accepted_df, unlabeled_df) from this layer
    """
    global _PIPELINE_STATE
    
    if not _PIPELINE_STATE['pr_numbers']:
        raise ValueError('Pipeline not initialized. Call initialize_pipeline() first.')
    
    pr_details_map = _PIPELINE_STATE['pr_details_map']
    
    if layer_number == 0:
        # Layer 0: Process all PRs in 4 explicit batches
        _, accepted_df, unlabeled_df = _run_layer_0_with_batches(
            pr_numbers=_PIPELINE_STATE['pr_numbers'],
            pr_details_map=pr_details_map,
            num_batches=4,
            batch_size=25
        )
    else:
        # Convergence layers: Load unlabeled PRs and re-evaluate
        unlabeled_path = os.path.join(LAYERS_BASE_DIR, GLOBAL_UNLABELED_FILE)
        if not os.path.exists(unlabeled_path):
            raise FileNotFoundError(f'Unlabeled PRs file not found at {unlabeled_path}. Run Layer 0 first.')
        
        unlabeled_df_prev = pd.read_csv(unlabeled_path)
        unlabeled_pr_numbers = unlabeled_df_prev['pr_number'].tolist()
        
        if not unlabeled_pr_numbers:
            print(f'\n⚠️  No unlabeled PRs remaining. All PRs have converged.')
            return pd.DataFrame(), pd.DataFrame()
        
        _, accepted_df, unlabeled_df = _run_convergence_layer(
            layer_number=layer_number,
            pr_numbers=unlabeled_pr_numbers,
            pr_details_map=pr_details_map,
            batch_size=25
        )
    
    return accepted_df, unlabeled_df
