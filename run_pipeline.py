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
import numpy as np
import pandas as pd
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score

from src.features import extract_features
from src.train import train_binary, train_multiclass_ml, train_cnn, train_hybrid
from src.evaluate import check_gate2_binary, check_gate2_full, check_gate3
from src.visualize import plot_confusion_matrix, plot_3way_comparison

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
        icon = "[V]" if state == "COMPLETE" else (">>" if state == "IN PROGRESS" else " -")
        print(f"  {icon} {phase}: {desc}")
        print(f"       [{state}]")
    print("=" * 55)

def run_binary(cfg):
    print("\n--- Running Binary Stage ---")
    df = pd.read_pickle('data/features_labeled_v2.pkl')
    # Binary: Normal (0) vs Defect (1)
    df['is_defect'] = (df['failureType'] != 'Normal').astype(int)
    
    FEATURE_COLS = [c for c in df.columns if c not in ['failureType', 'is_defect']]
    X = df[FEATURE_COLS].values
    y = df['is_defect'].values
    
    idx_tr, idx_te = train_test_split(range(len(X)), test_size=0.2, stratify=y, random_state=cfg['seed'])
    
    res_bin = train_binary(cfg, X[idx_tr], y[idx_tr], X[idx_te], y[idx_te])
    
    # Gate 2 binary
    check_gate2_binary(res_bin['recall'], cfg)
    
    # plot confusion matrix
    y_te = y[idx_te]
    y_pr = res_bin['model'].predict(X[idx_te])
    plot_confusion_matrix(y_te, y_pr, labels=['Normal', 'Defect'], save_path='results/figures/binary_cm.png')
    print("Binary stage complete.\n")

def run_multiclass(cfg):
    print("\n--- Running Multiclass Stage ---")
    # Load dataset for ML & Hybrid
    df_hybrid = pd.read_pickle('data/features_labeled_v2.pkl')
    df_hybrid['failureType'] = df_hybrid['failureType'].replace({'Near-full': 'Normal', 'none': 'Normal'})
    FEATURE_COLS = [c for c in df_hybrid.columns if c != 'failureType']
    X_manual = df_hybrid[FEATURE_COLS].values
    y_hybrid = df_hybrid['failureType'].values
    
    print("Loading raw images for CNN...")
    raw_df = pd.read_pickle('data/LSWMD.pkl')
    raw_df['failureType'] = raw_df['failureType'].apply(lambda x: x[0][0] if len(x) > 0 else '')
    raw_df['failureType'] = raw_df['failureType'].replace({'Near-full': 'Normal', 'none': 'Normal'})
    df_raw_valid = raw_df[raw_df['failureType'].notna() & (raw_df['failureType'] != '')].copy()
    
    wafer_maps = df_raw_valid['waferMap'].values
    
    print("Pre-resizing wafer maps to 64x64...")
    resized_maps = []
    for wmap in wafer_maps:
        if wmap.shape != (64, 64):
            H, W = wmap.shape
            row_indices = np.floor(np.arange(64) * H / 64).astype(int)
            col_indices = np.floor(np.arange(64) * W / 64).astype(int)
            wmap = wmap[row_indices, :][:, col_indices]
        resized_maps.append(wmap)
    wafer_maps = np.array(resized_maps)
    
    idx_tr, idx_te = train_test_split(range(len(wafer_maps)), test_size=0.2, stratify=y_hybrid, random_state=cfg['seed'])
    
    classes = sorted(np.unique(y_hybrid).tolist())
    
    print("Training ML Multiclass...")
    res_ml = train_multiclass_ml(cfg, X_manual[idx_tr], y_hybrid[idx_tr], X_manual[idx_te], y_hybrid[idx_te], classes)
    
    print("Training CNN Multiclass...")
    res_cnn = train_cnn(cfg, wafer_maps[idx_tr], y_hybrid[idx_tr], wafer_maps[idx_te], y_hybrid[idx_te])
    
    print("Loading CNN embeddings for Hybrid...")
    import os
    if not os.path.exists(cfg['data']['cnn_embeddings']):
        print(f"Error: {cfg['data']['cnn_embeddings']} not found. Run Task 3 first.")
        sys.exit(1)
        
    cnn_embs = np.load(cfg['data']['cnn_embeddings'])
    X_hybrid = np.hstack([X_manual, cnn_embs])
    
    print("Training Hybrid Multiclass...")
    res_hybrid = train_hybrid(cfg, X_hybrid[idx_tr], y_hybrid[idx_tr], X_hybrid[idx_te], y_hybrid[idx_te], cfg['data']['cnn_embeddings'])
    
    print("Evaluating Hybrid on Gate 2 and Gate 3...")
    y_pred_h = res_hybrid['model'].predict(X_hybrid[idx_te])
    
    y_pred_h_str = [classes[i] for i in y_pred_h]
    
    recalls = recall_score(y_hybrid[idx_te], y_pred_h_str, average=None, labels=classes)
    recall_dict = dict(zip(classes, recalls))
    
    scratch_recall = recall_dict.get('Scratch', 0)
    donut_recall = recall_dict.get('Donut', 0)
    
    check_gate2_full(scratch_recall, donut_recall, cfg)
    check_gate3(y_hybrid[idx_te], y_pred_h_str, classes, cfg)
    
    plot_3way_comparison('results/metrics.csv', save_path='results/figures/3way_comparison.png')
    print("Multiclass stage complete.\n")

def run_validate(cfg):
    print("\n--- Running Validate Stage ---")
    df_hybrid = pd.read_pickle('data/features_augmented.pkl')
    df_hybrid['failureType'] = df_hybrid['failureType'].replace({'Near-full': 'Normal', 'none': 'Normal'})
    FEATURE_COLS = [c for c in df_hybrid.columns if c != 'failureType']
    X_manual = df_hybrid[FEATURE_COLS].values
    y_hybrid = df_hybrid['failureType'].values
    
    print("Loading existing CNN embeddings...")
    import os
    emb_path = 'data/cnn_embeddings_augmented.npy'
    if not os.path.exists(emb_path):
        print(f"Error: {emb_path} not found. Run Task 3 first.")
        sys.exit(1)
        
    cnn_embs = np.load(emb_path)
    X_hybrid = np.hstack([X_manual, cnn_embs])
    
    idx_tr, idx_te = train_test_split(range(len(X_hybrid)), test_size=0.2, stratify=y_hybrid, random_state=cfg['seed'])
    
    print("Training Hybrid Multiclass with existing embeddings...")
    res_hybrid = train_hybrid(cfg, X_hybrid[idx_tr], y_hybrid[idx_tr], X_hybrid[idx_te], y_hybrid[idx_te], emb_path)
    
    print("Evaluating Hybrid on Gate 2 and Gate 3...")
    y_pred_h = res_hybrid['model'].predict(X_hybrid[idx_te])
    
    classes = sorted(np.unique(y_hybrid).tolist())
    y_pred_h_str = [classes[i] for i in y_pred_h]
    
    recalls = recall_score(y_hybrid[idx_te], y_pred_h_str, average=None, labels=classes)
    recall_dict = dict(zip(classes, recalls))
    
    scratch_recall = recall_dict.get('Scratch', 0)
    donut_recall = recall_dict.get('Donut', 0)
    
    check_gate2_full(scratch_recall, donut_recall, cfg)
    check_gate3(y_hybrid[idx_te], y_pred_h_str, classes, cfg)
    
    print("Updating metrics.csv...")
    metrics_path = 'results/metrics.csv'
    if os.path.exists(metrics_path):
        metrics_df = pd.read_csv(metrics_path)
        if 'Hybrid' in metrics_df['Model'].values:
            metrics_df.loc[metrics_df['Model'] == 'Hybrid', 'macro_F1'] = res_hybrid['macro_f1']
            metrics_df.loc[metrics_df['Model'] == 'Hybrid', 'Scratch_recall'] = scratch_recall
            metrics_df.loc[metrics_df['Model'] == 'Hybrid', 'Donut_recall'] = donut_recall
        metrics_df.to_csv(metrics_path, index=False)
        
    plot_3way_comparison('results/metrics.csv', save_path='results/figures/3way_comparison.png')
    print("Validate stage complete.\n")

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
  validate    Load existing CNN embeddings, train Hybrid, check Gates
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
                        choices=["all", "features", "binary", "multiclass", "validate", "shap", "spc"],
                        help="Pipeline stage to run (default: all)")
    args = parser.parse_args()

    try:
        with open(args.config, encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        print(f"[Pipeline] Config loaded: {args.config}")
    except FileNotFoundError:
        print(f"[ERROR] Config file not found: {args.config}")
        sys.exit(1)

    print_status()
    print(f"\n[Pipeline] Requested stage: {args.stage}")

    if args.stage in ("all", "binary"):
        run_binary(cfg)

    if args.stage in ("all", "multiclass"):
        run_multiclass(cfg)
        
    if args.stage in ("all", "validate"):
        run_validate(cfg)

    if args.stage in ("all", "features"):
        print("[features] → src/features.py :: extract_features()")
        print("  Status: IMPLEMENTED. Run notebooks/02_feature_engineering.ipynb")

    if args.stage in ("all", "shap"):
        print("[shap]     → src/visualize.py :: plot_shap(), plot_gradcam()  [Phase 3]")
        print("  Status: PENDING src/ integration (Phase 3)")

    if args.stage in ("all", "spc"):
        print("[spc]      → src/spc.py + src/autoencoder.py  [Phase 4]")
        print("  Status: PENDING (Phase 4)")

    print("\n[Pipeline] Complete.\n")

if __name__ == "__main__":
    main()
