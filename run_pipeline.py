"""
run_pipeline.py — WM-811K Wafer Defect Analysis Pipeline
Single entry point. Each stage activates as src/ modules are completed.

Usage:
    python run_pipeline.py --config config.yaml --stage all
    python run_pipeline.py --config config.yaml --stage binary
    python run_pipeline.py --config config.yaml --stage multiclass
"""
import argparse
import yaml
import sys


def print_status():
    """Print current pipeline completion status."""
    status = [
        ("Phase 1", "EDA + Feature Engineering (20 features)", "COMPLETE"),
        ("Phase 2", "Binary + 7-class + SHAP (Hybrid F1=0.870)", "COMPLETE"),
        ("Phase 3", "src/ modularization + run_pipeline integration", "IN PROGRESS"),
        ("Phase 4", "SPC + Autoencoder + Streamlit dashboard", "PENDING"),
    ]
    print("\n WM-811K Pipeline Status")
    print("=" * 55)
    for phase, desc, state in status:
        icon = "✓" if state == "COMPLETE" else ("→" if state == "IN PROGRESS" else "·")
        print(f"  {icon} {phase}: {desc}")
        print(f"       [{state}]")
    print("=" * 55)


def main():
    parser = argparse.ArgumentParser(
        description="WM-811K Wafer Defect Classification Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Stages:
  all         Run all available stages (default)
  features    Feature engineering (20 manual features)
  binary      Binary classification: Normal vs Defect
  multiclass  7-class pattern classification (ML / CNN / Hybrid)
  shap        SHAP + Grad-CAM interpretation
  spc         SPC monitoring + Autoencoder anomaly detection

Examples:
  python run_pipeline.py --stage binary
  python run_pipeline.py --config config.yaml --stage multiclass
        """
    )
    parser.add_argument("--config", default="config.yaml",
                        help="Path to config YAML (default: config.yaml)")
    parser.add_argument("--stage", default="all",
                        choices=["all", "features", "binary", "multiclass", "shap", "spc"],
                        help="Pipeline stage to run (default: all)")
    args = parser.parse_args()

    # Load config
    try:
        with open(args.config) as f:
            cfg = yaml.safe_load(f)
        print(f"[Pipeline] Config loaded: {args.config}")
    except FileNotFoundError:
        print(f"[ERROR] Config file not found: {args.config}")
        sys.exit(1)

    print_status()

    # Stage routing — populated progressively as Phase 3 completes
    print(f"\n[Pipeline] Requested stage: {args.stage}")

    if args.stage in ("all", "features"):
        print("[features] → src/features.py :: extract_features()")
        print("  Status: IMPLEMENTED. Run notebooks/02_feature_engineering.ipynb")

    if args.stage in ("all", "binary"):
        print("[binary]   → src/train.py :: train_binary()  [Phase 3]")
        print("  Status: PENDING src/ integration (Phase 3)")

    if args.stage in ("all", "multiclass"):
        print("[multiclass] → src/train.py :: train_multiclass()  [Phase 3]")
        print("  Status: PENDING src/ integration (Phase 3)")
        print(f"  Best result: Hybrid macro F1 = {cfg.get('results', {}).get('hybrid_f1', 0.870):.3f}")

    if args.stage in ("all", "shap"):
        print("[shap]     → src/visualize.py :: plot_shap(), plot_gradcam()  [Phase 3]")
        print("  Status: PENDING src/ integration (Phase 3)")

    if args.stage in ("all", "spc"):
        print("[spc]      → src/spc.py + src/autoencoder.py  [Phase 4]")
        print("  Status: PENDING (Phase 4)")

    print("\n[Pipeline] Phase 3 will wire all stages into this entry point.")
    print("[Pipeline] See plans/phase3_plan.md for implementation schedule.\n")


if __name__ == "__main__":
    main()
