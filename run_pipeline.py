from pathlib import Path
import sys
import os
import pandas as pd
from uncertainty_with_dnn.run_unceratinty_selection import run_uncertainty_selection
from llm_oracle_labeling import initialize_pipeline, run_layer, analyze_labeling_quality
from llm_oracle_labeling.config import GLOBAL_ACCEPTED_FILE, GLOBAL_UNLABELED_FILE, LAYERS_BASE_DIR
from extract_and_assign_labels import extract_and_assign_labels


def show_menu():
    """Display the main menu."""
    print("\n" + "="*80)
    print("LLM ORACLE LABELING PIPELINE - MANUAL LAYER EXECUTION")
    print("="*80)
    print("\nOptions:")
    print("  0  - Initialize pipeline and run Layer 0 (4 batches of 25 PRs)")
    print("  1+ - Run convergence layer N (with unlabeled PRs from previous layer)")
    print("  s  - Show current status (accepted/unlabeled counts)")
    print("  a  - Analyze labeling quality")
    print("  e  - Extract and assign labels to training data")
    print("  q  - Quit")
    print("="*80 + "\n")


def show_status():
    """Show current accepted/unlabeled counts."""
    
    accepted_file = os.path.join(LAYERS_BASE_DIR, GLOBAL_ACCEPTED_FILE)
    unlabeled_file = os.path.join(LAYERS_BASE_DIR, GLOBAL_UNLABELED_FILE)
    
    if os.path.exists(accepted_file):
        accepted_df = pd.read_csv(accepted_file)
        print(f"\n✅ Accepted PRs: {len(accepted_df)}")
    else:
        print(f"\n⚠️  No accepted PRs yet")
    
    if os.path.exists(unlabeled_file):
        unlabeled_df = pd.read_csv(unlabeled_file)
        print(f"❓ Unlabeled PRs: {len(unlabeled_df)}")
    else:
        print(f"❓ No unlabeled PRs found")
    
    print()


def main():
    """Main interactive pipeline runner."""
    print("\n" + "="*80)
    print("STARTING LLM ORACLE LABELING PIPELINE")
    print("="*80 + "\n")
    
    # Configuration
    ml_features_csv = 'ML_Label_Input_apache_beam.csv'
    uncertain_output_dir = Path("UncertainPoint")
    
    # Initialize
    loop_number = 1
    pr_list_csv = uncertain_output_dir / f"loop_{loop_number}_selected.csv"
    
    # Run uncertainty selection to get the 100 uncertain PRs
    print("Step 1: Running uncertainty selection...")
    selected = run_uncertainty_selection(
        ml_features_csv=ml_features_csv,
        loop_number=loop_number,
        data_root=Path("SamplingLoopData"),
        output_dir=Path("UncertainPoint"),
        model_monitor_dir=Path("ModelMonitoring"),
        n_top_uncertain=100,
        k_diverse=100,
        metric="euclidean",
        verbose=True
    )
    
    print(f"\nStep 2: Initializing pipeline with {pr_list_csv}...")
    initialize_pipeline(
        ml_features_csv=ml_features_csv,
        pr_list_csv=str(pr_list_csv),
        max_items=None
    )
    
    # Interactive layer execution
    print("\nNow you can run layers manually.\n")
    
    while True:
        show_menu()
        user_input = input("Enter your choice: ").strip().lower()
        
        if user_input == 'q':
            print("\n✅ Exiting pipeline. Goodbye!")
            break
        
        elif user_input == 's':
            show_status()
        
        elif user_input == 'a':
            print("\nAnalyzing labeling quality...")
            
            accepted_file = os.path.join(LAYERS_BASE_DIR, GLOBAL_ACCEPTED_FILE)
            unlabeled_file = os.path.join(LAYERS_BASE_DIR, GLOBAL_UNLABELED_FILE)
            
            if os.path.exists(accepted_file) and os.path.exists(unlabeled_file):
                accepted_df = pd.read_csv(accepted_file)
                unlabeled_df = pd.read_csv(unlabeled_file)
                analysis_results = analyze_labeling_quality(accepted_df, unlabeled_df)
                
                if analysis_results:
                    print("\n" + "="*80)
                    print("LABELING QUALITY ANALYSIS")
                    print("="*80)
                    print(f"Total PRs: {analysis_results.get('total', 0)}")
                    print(f"Accepted PRs: {analysis_results.get('accepted', 0)}")
                    print(f"Unlabeled PRs: {analysis_results.get('unlabeled', 0)}")
                    consensus_rate = analysis_results.get('consensus_rate', 0)
                    print(f"Consensus Rate: {consensus_rate * 100:.1f}%")
                    
                    if 'label_distribution' in analysis_results:
                        print(f"\nLabel Distribution (in accepted PRs):")
                        for label, count in analysis_results['label_distribution'].items():
                            print(f"  {label}: {count}")
                    print("="*80 + "\n")
            else:
                print("⚠️  Not enough data to analyze. Run Layer 0 first.")
        
        elif user_input == 'e':
            print("\nExtracting and assigning labels...")
            extract_and_assign_labels(loop_number)
            print("✅ Labels extracted and assigned.")
        
        else:
            try:
                layer_num = int(user_input)
                if layer_num < 0:
                    print("❌ Invalid layer number. Please enter 0 or higher.")
                    continue
                
                # Validate that previous layer exists before running convergence layers
                if layer_num > 0:
                    import os
                    prev_layer_dir = os.path.join(LAYERS_BASE_DIR, str(layer_num - 1))
                    if not os.path.exists(prev_layer_dir):
                        print(f"❌ Cannot run Layer {layer_num}: Previous Layer {layer_num - 1} not found.")
                        print(f"   Please run Layer {layer_num - 1} first.")
                        continue
                
                print(f"\n🔄 Running Layer {layer_num}...")
                accepted_df, unlabeled_df = run_layer(layer_num)
                
                if layer_num == 0:
                    print(f"\n✅ Layer 0 complete!")
                else:
                    print(f"\n✅ Layer {layer_num} complete!")
                
                show_status()
                
                # Check if we've reached the target
                if len(accepted_df) >= 50:
                    print("🎉 TARGET REACHED: 50+ PRs accepted!")
                    print("You can continue to refine or proceed to extract labels.")
                
                elif unlabeled_df.empty:
                    print("✅ All PRs have converged!")
                    show_status()
                
            except ValueError:
                print("❌ Invalid input. Please enter a valid layer number, or 'q' to quit.")
            except Exception as e:
                print(f"❌ Error: {e}")
                import traceback
                traceback.print_exc()


if __name__ == "__main__":
    main()
