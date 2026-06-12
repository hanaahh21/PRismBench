from pathlib import Path
import sys
import os
import pandas as pd
from uncertainty_with_dnn.run_unceratinty_selection import run_uncertainty_selection
from llm_oracle_labeling import initialize_pipeline, run_layer, analyze_labeling_quality
from llm_oracle_labeling.config import GLOBAL_ACCEPTED_FILE, GLOBAL_UNLABELED_FILE
from extract_and_assign_labels import extract_and_assign_labels

TARGET_ACCEPTED_PRS = 32
MAX_AUTO_LAYERS = 20


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


def show_status(layers_base_dir: Path):
    """Show current accepted/unlabeled counts."""
    
    accepted_file = os.path.join(str(layers_base_dir), GLOBAL_ACCEPTED_FILE)
    unlabeled_file = os.path.join(str(layers_base_dir), GLOBAL_UNLABELED_FILE)
    
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


def update_unlabeled_after_llm_convergence(
    loop_number: int,
    ml_features_csv: str,
    accepted_df: pd.DataFrame,
) -> None:
    """
    Build next-loop unlabeled pool as:
      (previous loop unlabeled ∩ PRs with SZZ issues) - (LLM accepted PRs in current loop)
    """
    sampling_root = Path("SamplingLoopData")
    prev_unlabeled_path = sampling_root / f"loop_{loop_number - 1}_data" / "unlabeled_data.csv"
    current_loop_dir = sampling_root / f"loop_{loop_number}_data"
    current_loop_dir.mkdir(parents=True, exist_ok=True)
    current_unlabeled_path = current_loop_dir / "unlabeled_data.csv"

    if not prev_unlabeled_path.exists():
        raise FileNotFoundError(f"Missing previous loop unlabeled data: {prev_unlabeled_path}")

    prev_unlabeled_df = pd.read_csv(prev_unlabeled_path)
    if "pr_number" not in prev_unlabeled_df.columns:
        raise ValueError(f"'pr_number' column missing in {prev_unlabeled_path}")

    ml_df = pd.read_csv(ml_features_csv)
    if "pr_number" not in ml_df.columns or "szz_origin_issues" not in ml_df.columns:
        raise ValueError(
            f"Required columns not found in {ml_features_csv}. Need: pr_number, szz_origin_issues"
        )

    prs_with_szz = set(ml_df.loc[~ml_df["szz_origin_issues"].isna(), "pr_number"].astype(str))
    szz_filtered_pool = prev_unlabeled_df[
        prev_unlabeled_df["pr_number"].astype(str).isin(prs_with_szz)
    ].copy()

    if accepted_df.empty or "pr_number" not in accepted_df.columns:
        remaining_unlabeled_df = szz_filtered_pool
        accepted_prs_count = 0
    else:
        accepted_prs = set(accepted_df["pr_number"].astype(str))
        remaining_unlabeled_df = szz_filtered_pool[
            ~szz_filtered_pool["pr_number"].astype(str).isin(accepted_prs)
        ].copy()
        accepted_prs_count = len(accepted_prs)

    remaining_unlabeled_df.to_csv(current_unlabeled_path, index=False)
    print(
        "✅ Updated post-LLM unlabeled pool: "
        f"{current_unlabeled_path} (prev={len(prev_unlabeled_df)}, "
        f"szz_pool={len(szz_filtered_pool)}, accepted={accepted_prs_count}, "
        f"remaining={len(remaining_unlabeled_df)})"
    )


def main():
    """Main pipeline runner with continuous active-learning loops until interrupted."""
    print("\n" + "="*80)
    print("STARTING LLM ORACLE LABELING PIPELINE")
    print("="*80 + "\n")
    
    # Configuration
    ml_features_csv = 'ML_Label_Input_apache_kafka.csv'
    uncertain_output_dir = Path("UncertainPoint")
    
    # Initialize start loop once; execution continues loop+1, loop+2, ... until interrupted.
    current_loop = 20
    print(f"Auto-run starting from active-learning loop {current_loop}. Press Ctrl+C to stop.\n")

    try:
        while True:
            loop_number = current_loop
            print("\n" + "=" * 80)
            print(f"ACTIVE LEARNING LOOP {loop_number}")
            print("=" * 80 + "\n")

            loop_layers_dir = Path("Layers") / f"loop_{loop_number}"
            loop_layers_dir.mkdir(parents=True, exist_ok=True)
            final_accepted_path = loop_layers_dir / "final_accepted.csv"
            pr_list_csv = uncertain_output_dir / f"loop_{loop_number}_selected.csv"

            # Run uncertainty selection to get uncertain PRs for this loop.
            print("Step 1: Running uncertainty selection...")
            _, gnn_auto_labeled_all = run_uncertainty_selection(
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

            if gnn_auto_labeled_all:
                print(
                    "\n✅ GNN auto-labeling activated based on F1 threshold. "
                    "All remaining unlabeled PRs were labeled directly by the GNN."
                )
                print("Skipping LLM convergence layers for this loop.")
                current_loop += 1
                continue

            print(f"\nStep 2: Initializing pipeline with {pr_list_csv}...")
            initialize_pipeline(
                ml_features_csv=ml_features_csv,
                pr_list_csv=str(pr_list_csv),
                max_items=None,
                layers_base_dir=str(loop_layers_dir),
            )

            print("\nRunning layers automatically in order: 0, 1, 2, ...")
            print(f"Target accepted PRs: {TARGET_ACCEPTED_PRS}\n")

            if final_accepted_path.exists():
                final_accepted_df = pd.read_csv(final_accepted_path)
                print(f"Loaded existing cumulative accepted: {len(final_accepted_df)} rows from {final_accepted_path}")
            else:
                final_accepted_df = pd.DataFrame()
            final_unlabeled_df = pd.DataFrame()

            for layer_num in range(0, MAX_AUTO_LAYERS + 1):
                try:
                    print(f"\n🔄 Running Layer {layer_num}...")
                    layer_accepted_df, unlabeled_df = run_layer(layer_num)

                    final_unlabeled_df = unlabeled_df
                    if not layer_accepted_df.empty:
                        final_accepted_df = pd.concat([final_accepted_df, layer_accepted_df], ignore_index=True)
                        if "pr_number" in final_accepted_df.columns:
                            final_accepted_df = final_accepted_df.drop_duplicates(subset=["pr_number"], keep="first")
                        final_accepted_df.to_csv(final_accepted_path, index=False)

                    print(f"\n✅ Layer {layer_num} complete!")
                    show_status(loop_layers_dir)

                    accepted_count = len(final_accepted_df)
                    unlabeled_count = len(unlabeled_df)
                    print(
                        f"Cumulative accepted in {final_accepted_path}: {accepted_count} "
                        f"(this layer added {len(layer_accepted_df)})"
                    )

                    if accepted_count >= TARGET_ACCEPTED_PRS:
                        print(
                            f"🎉 TARGET REACHED: {accepted_count} cumulative accepted PRs "
                            f"(target was {TARGET_ACCEPTED_PRS})."
                        )
                        break

                    if unlabeled_count == 0:
                        print("✅ All PRs have converged before hitting the target.")
                        break
                except Exception as e:
                    print(f"❌ Error while running Layer {layer_num}: {e}")
                    import traceback
                    traceback.print_exc()
                    break
            else:
                print(
                    f"⚠️  Reached max auto layers ({MAX_AUTO_LAYERS}) before hitting "
                    f"{TARGET_ACCEPTED_PRS} accepted PRs."
                )

            if not final_accepted_df.empty or not final_unlabeled_df.empty:
                print("\nAnalyzing labeling quality...")
                analysis_results = analyze_labeling_quality(final_accepted_df, final_unlabeled_df)

                if analysis_results:
                    print("\n" + "="*80)
                    print("LABELING QUALITY ANALYSIS")
                    print("="*80)
                    print(f"Total PRs: {analysis_results.get('total', 0)}")
                    print(f"Accepted PRs: {analysis_results.get('accepted', 0)}")
                    print(f"Unlabeled PRs: {analysis_results.get('unlabeled', 0)}")
                    consensus_rate = analysis_results.get('consensus_rate', 0)
                    print(f"Consensus Rate: {consensus_rate * 100:.1f}%")
                    print("="*80 + "\n")

            if not final_accepted_df.empty:
                final_accepted_df.to_csv(final_accepted_path, index=False)
                print(f"✅ Updated cumulative accepted file: {final_accepted_path}")

            update_unlabeled_after_llm_convergence(
                loop_number=loop_number,
                ml_features_csv=ml_features_csv,
                accepted_df=final_accepted_df,
            )

            if len(final_accepted_df) >= TARGET_ACCEPTED_PRS:
                print("Extracting and assigning labels...")
                extract_and_assign_labels(loop_number, accepted_labels_path=str(final_accepted_path))
                print("✅ Labels extracted and assigned.")
            else:
                print(
                    f"⚠️  Skipping label extraction because accepted PRs "
                    f"({len(final_accepted_df)}) are below target ({TARGET_ACCEPTED_PRS})."
                )

            print(f"\n✅ Completed active-learning loop {loop_number}. Moving to loop {loop_number + 1}...")
            current_loop += 1
    except KeyboardInterrupt:
        print(f"\n\n🛑 Stopped by user at loop {current_loop}. Exiting gracefully.")


if __name__ == "__main__":
    main()
