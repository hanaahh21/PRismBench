import json
import sys
from collections import Counter
import pandas as pd

from .config import LABEL_COLUMN_MAP


def analyze_labeling_quality(accepted_df: pd.DataFrame, unlabeled_df: pd.DataFrame):
    """
    Analyze labeling quality from the new unary pipeline structure.
    
    Args:
        accepted_df: DataFrame of PRs with consensus (all 3 models agreed on all labels)
        unlabeled_df: DataFrame of PRs without consensus (disagreement on at least 1 label)
    """
    print('='*80)
    print('LABELING QUALITY ANALYSIS')
    print('='*80)

    total = len(accepted_df) + len(unlabeled_df)

    print(f'\n📊 Overall Statistics:')
    print(f'  Total PRs processed: {total}')
    print(f'  ✅ Accepted (unanimous consensus): {len(accepted_df)} ({len(accepted_df)/total*100:.1f}%)')
    print(f'  ❓ Unlabeled (need convergence): {len(unlabeled_df)} ({len(unlabeled_df)/total*100:.1f}%)')

    if len(accepted_df) > 0:
        print(f'\n🎯 Consensus Details:')
        print(f'  All 3 models agreed on all 4 labels for these {len(accepted_df)} PRs')
        if 'consensus_models' in accepted_df.columns:
            print(f'  Consensus Models: {accepted_df["consensus_models"].iloc[0]}')
        if 'consensus_layer' in accepted_df.columns:
            layer_dist = accepted_df['consensus_layer'].value_counts().sort_index()
            for layer, count in layer_dist.items():
                print(f'    Layer {int(layer)}: {count} PRs accepted')

        print(f'\n🏷️  Label Distribution (Accepted PRs):')
        label_counts = {}
        for col_display, col_name in LABEL_COLUMN_MAP.items():
            if col_name in accepted_df.columns:
                yes_count = (accepted_df[col_name] == 'Yes').sum()
                label_counts[col_display] = yes_count
                pct = yes_count / len(accepted_df) * 100
                print(f'  {col_display}: {yes_count} ({pct:.1f}%)')
        
        # Show multi-label combinations
        print(f'\n  Label Combinations (Top 5):')
        if all(col in accepted_df.columns for col in LABEL_COLUMN_MAP.values()):
            label_combos = []
            for _, row in accepted_df.iterrows():
                combo = tuple(row[col] == 'Yes' for col in LABEL_COLUMN_MAP.values())
                label_combos.append(combo)
            
            combo_counts = Counter(label_combos)
            label_names = list(LABEL_COLUMN_MAP.keys())
            
            for combo, count in combo_counts.most_common(5):
                combo_str = ', '.join(
                    f'{label}=Yes' if val else f'{label}=No'
                    for label, val in zip(label_names, combo)
                )
                print(f'    {combo_str}: {count} PRs ({count/len(accepted_df)*100:.1f}%)')

        print(f'\n📈 Reason Quality Analysis:')
        # Analyze reason column availability
        reason_cols = [f'{col}_reason' for col in LABEL_COLUMN_MAP.values()]
        available_reasons = [col for col in reason_cols if col in accepted_df.columns]
        
        if available_reasons:
            total_reasons = 0
            filled_reasons = 0
            avg_reason_length = []
            
            for reason_col in available_reasons:
                for reason_str in accepted_df[reason_col]:
                    total_reasons += 1
                    if reason_str and str(reason_str).strip():
                        filled_reasons += 1
                        avg_reason_length.append(len(str(reason_str)))
            
            if total_reasons > 0:
                print(f'  Reasons provided: {filled_reasons}/{total_reasons} ({filled_reasons/total_reasons*100:.1f}%)')
                if avg_reason_length:
                    print(f'  Average reason length: {sum(avg_reason_length)/len(avg_reason_length):.0f} chars')
                    print(f'  Min reason length: {min(avg_reason_length)} chars')
                    print(f'  Max reason length: {max(avg_reason_length)} chars')
        
        # Check for model outputs stored in metadata
        if 'model_outputs' in accepted_df.columns:
            print(f'\n  Model Agreement Details:')
            for _, row in accepted_df.head(3).iterrows():
                try:
                    model_outputs = json.loads(row['model_outputs'])
                    pr_num = row['pr_number']
                    models = list(model_outputs.keys())
                    print(f'    PR #{int(pr_num)}: {len(models)} models in agreement')
                except:
                    pass

    if len(unlabeled_df) > 0:
        print(f'\n❓ Unlabeled Cases (Disagreement on at least 1 label):')
        print(f'  Total: {len(unlabeled_df)}')
        print(f'\n  Disagreement Patterns (Top 5):')
        
        if 'reason' in unlabeled_df.columns:
            # Extract disagreement patterns
            disagreement_labels = []
            for reason_str in unlabeled_df['reason']:
                try:
                    reason = str(reason_str)
                    # Extract label names from reason string like "Disagreement on labels: ['bug', 'performance']"
                    if 'Disagreement on labels:' in reason:
                        # Parse the label list from string
                        start = reason.find('[')
                        end = reason.rfind(']')
                        if start != -1 and end != -1:
                            labels_str = reason[start:end+1]
                            labels = eval(labels_str)
                            disagreement_labels.append(tuple(sorted(labels)))
                except:
                    pass
            
            if disagreement_labels:
                pattern_counts = Counter(disagreement_labels)
                label_display_map = {v: k for k, v in LABEL_COLUMN_MAP.items()}
                
                for pattern, count in pattern_counts.most_common(5):
                    pattern_display = ', '.join(
                        label_display_map.get(label, label) for label in pattern
                    )
                    print(f'    Disagreement on {pattern_display}: {count} PRs ({count/len(unlabeled_df)*100:.1f}%)')
        
        print(f'\n  Sample Unlabeled PRs:')
        for idx, row in unlabeled_df.head(5).iterrows():
            pr_num = int(row['pr_number'])
            reason = str(row.get('reason', 'Unknown')).strip()
            print(f'    PR #{pr_num}: {reason[:70]}...' if len(reason) > 70 else f'    PR #{pr_num}: {reason}')

    print('\n' + '='*80)

    print('\n💡 Recommendations:')
    if total > 0:
        consensus_rate = len(accepted_df) / total
        if consensus_rate < 0.5:
            print('  ⚠️  Low consensus rate (<50%). Consider running more convergence layers:')
            print(f'     - You have {len(unlabeled_df)} unlabeled PRs')
            print('     - Run Layer 1 or higher to re-evaluate with peer feedback')
            print('     - Models can reconsider decisions with other models\' reasoning')
        elif consensus_rate < 0.8:
            print(f'  ℹ️  Moderate consensus rate ({consensus_rate*100:.1f}%). You can:')
            print('     - Continue with more convergence layers for refinement')
            print('     - Or proceed to extraction if you have enough labeled data')
        else:
            print(f'  ✨ Good consensus rate ({consensus_rate*100:.1f}%)!')
            if len(unlabeled_df) > 0:
                print(f'     - Still have {len(unlabeled_df)} unlabeled PRs')
                print('     - Run more layers to improve coverage')

    print('\n✅ Convergence Strategy:')
    print('  1. Check consensus rate and label distribution')
    print('  2. If consensus_rate < 50%, run convergence layers (Layer 1, 2, 3...)')
    print('  3. Use show_status() to monitor progress')
    print('  4. Continue until:')
    print('     - You reach 50+ labeled PRs, OR')
    print('     - All PRs converge (no unlabeled remaining), OR')
    print('     - You decide to stop')
    print('  5. Once satisfied, run extract_and_assign_labels()')

    # Build return dictionary
    result = {
        'total': total,
        'accepted': len(accepted_df),
        'unlabeled': len(unlabeled_df),
        'consensus_rate': len(accepted_df) / total if total > 0 else 0,
    }
    
    if len(accepted_df) > 0 and label_counts:
        result['label_distribution'] = label_counts
    
    return result

