"""
Task 6 -- Gate 2 Full Check
Task 7 -- SHAP (ML/Hybrid) + Grad-CAM (CNN, manual hook-based)
Branch: agent/stage5-shap
"""
import yaml, json, os, sys
import sys as _sys

# Force UTF-8 stdout to avoid cp949 issues on Windows
if hasattr(_sys.stdout, 'reconfigure'):
    _sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath('.'))
from src.evaluate import check_gate2_full

cfg = yaml.safe_load(open('config.yaml'))

# -----------------------------------------------------------
# TASK 6 -- Gate 2 Full Check
# -----------------------------------------------------------
print("=" * 60)
print("TASK 6 -- Gate 2 Full Check")
print("=" * 60)
gate2_result = check_gate2_full(
    scratch_recall=0.7950,
    donut_recall=0.7838,
    cfg=cfg
)
print(json.dumps(gate2_result, indent=2))
assert gate2_result['passed'], "Gate 2 FAILED -- stopping execution."
print("Gate 2 PASSED. Proceeding to Task 7.")

# -----------------------------------------------------------
# TASK 7 -- SHAP + Grad-CAM
# -----------------------------------------------------------
import mlflow
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb
import shap
import torch
from src.models import WaferCNN, WaferMapDataset

mlflow.set_experiment("wafer_defect_phase2")
os.makedirs('results/figures', exist_ok=True)

print("\n" + "=" * 60)
print("TASK 7 Part A -- SHAP (Hybrid XGBoost)")
print("=" * 60)

# -- Data --
print("Loading data...")
df = pd.read_pickle('data/features_labeled_v2.pkl')
df['failureType'] = df['failureType'].replace('Near-full', 'Normal')
FEATURE_COLS = [c for c in df.columns if c != 'failureType']

le = LabelEncoder()
y_multi = le.fit_transform(df['failureType'])
CLASS_NAMES = le.classes_.tolist()
manual_feats = df[FEATURE_COLS].values

cnn_embs = np.load('data/cnn_embeddings.npy')
X_hybrid = np.hstack([manual_feats, cnn_embs])

X_tr_h, X_te_h, y_tr_h, y_te_h = train_test_split(
    X_hybrid, y_multi, test_size=0.2, stratify=y_multi, random_state=cfg['seed']
)

# -- Re-train Hybrid model (best params from Task 5 trial 9) --
print("Re-training Hybrid XGBoost for SHAP (best known params)...")
best_hybrid_params = {
    'n_estimators': 56,
    'max_depth': 8,
    'learning_rate': 0.19635838367901684,
    'subsample': 0.7393797971382635,
    'objective': 'multi:softmax',
    'num_class': len(CLASS_NAMES),
    'random_state': cfg['seed'],
    'n_jobs': -1,
    'tree_method': 'hist'
}
xgb_hybrid = xgb.XGBClassifier(**best_hybrid_params)
xgb_hybrid.fit(X_tr_h, y_tr_h, sample_weight=compute_sample_weight('balanced', y_tr_h))
print("Hybrid model trained.")

# -- SHAP --
print("Computing SHAP values (on 500 test samples)...")
X_shap = X_te_h[:500]

explainer = shap.TreeExplainer(xgb_hybrid, feature_perturbation='tree_path_dependent')
shap_values = explainer.shap_values(X_shap)

# Handle old list API vs new ndarray API
if isinstance(shap_values, list):
    shap_arr = np.stack(shap_values, axis=2)   # (500, n_feats, n_classes)
else:
    shap_arr = shap_values

# -- Per-class SHAP bar plots (manual features only) --
n_manual = len(FEATURE_COLS)
print(f"Saving per-class SHAP bar plots ({len(CLASS_NAMES)} classes)...")
for cls_idx, cls_name in enumerate(CLASS_NAMES):
    if shap_arr.ndim == 3:
        cls_shap = shap_arr[:, :n_manual, cls_idx]
    else:
        cls_shap = shap_arr[:, :n_manual]

    mean_abs = np.abs(cls_shap).mean(axis=0)
    top_k = min(10, n_manual)
    top_idx = np.argsort(mean_abs)[-top_k:][::-1]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(range(top_k), mean_abs[top_idx][::-1], color='steelblue')
    ax.set_yticks(range(top_k))
    ax.set_yticklabels([FEATURE_COLS[i] for i in top_idx[::-1]], fontsize=9)
    ax.set_xlabel('Mean |SHAP value|')
    ax.set_title(f'SHAP Feature Importance -- {cls_name}')
    plt.tight_layout()
    save_path = f'results/figures/shap_{cls_name.replace("-","_").replace(" ","_")}.png'
    plt.savefig(save_path, dpi=100)
    plt.close()
    print(f"  Saved {save_path}")

# -- SHAP global summary --
print("Saving SHAP summary bar plot (all classes)...")
if shap_arr.ndim == 3:
    mean_abs_global = np.abs(shap_arr[:, :n_manual, :]).mean(axis=(0, 2))
else:
    mean_abs_global = np.abs(shap_arr[:, :n_manual]).mean(axis=0)

top20_idx = np.argsort(mean_abs_global)[-20:][::-1]
fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(range(20), mean_abs_global[top20_idx][::-1], color='darkorange')
ax.set_yticks(range(20))
ax.set_yticklabels([FEATURE_COLS[i] for i in top20_idx[::-1]], fontsize=8)
ax.set_xlabel('Mean |SHAP value| (across all classes)')
ax.set_title('SHAP Summary -- Top 20 Manual Features (Hybrid Model)')
plt.tight_layout()
plt.savefig('results/figures/shap_summary_table.png', dpi=100)
plt.close()
print("  Saved results/figures/shap_summary_table.png")

with mlflow.start_run(run_name="stage5_shap"):
    mlflow.log_artifact('results/figures/shap_summary_table.png')
    for cls_name in CLASS_NAMES:
        p = f'results/figures/shap_{cls_name.replace("-","_").replace(" ","_")}.png'
        if os.path.exists(p):
            mlflow.log_artifact(p)

print("\n" + "=" * 60)
print("TASK 7 Part B -- Grad-CAM (manual hook, WaferCNN)")
print("=" * 60)

# -- Load raw wafer maps first to derive CNN class names (includes 'none') --
print("Loading wafer maps...")
raw_df = pd.read_pickle('data/LSWMD.pkl')
raw_df['failureType'] = raw_df['failureType'].apply(
    lambda x: x[0][0] if hasattr(x, '__len__') and len(x) > 0 else ''
)
raw_df['failureType'] = raw_df['failureType'].replace('Near-full', 'Normal')
labeled_df = raw_df[raw_df['failureType'].notna() & (raw_df['failureType'] != '')].copy()

# CNN was trained with its own LabelEncoder that includes 'none'
le_cnn = LabelEncoder()
le_cnn.fit(labeled_df['failureType'])
CNN_CLASS_NAMES = le_cnn.classes_.tolist()
print(f"CNN classes ({len(CNN_CLASS_NAMES)}): {CNN_CLASS_NAMES}")

labeled_df['encoded_label'] = le_cnn.transform(labeled_df['failureType'])

# -- Load CNN model with correct num_classes from saved checkpoint --
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_cnn = WaferCNN(num_classes=len(CNN_CLASS_NAMES)).to(device)
model_cnn.load_state_dict(torch.load('results/cnn_weights.pth', map_location=device))
model_cnn.eval()
print(f"CNN loaded successfully with num_classes={len(CNN_CLASS_NAMES)}")

# -- Manual Grad-CAM via hooks --
def compute_gradcam(model, input_tensor, target_class, target_layer):
    """Compute Grad-CAM for a single input using forward/backward hooks."""
    activations = {}
    gradients = {}

    def fwd_hook(module, inp, out):
        activations['feat'] = out.detach()

    def bwd_hook(module, grad_in, grad_out):
        gradients['grad'] = grad_out[0].detach()

    target_conv = list(target_layer.children())[0]  # Conv2d inside Sequential
    h_fwd = target_conv.register_forward_hook(fwd_hook)
    h_bwd = target_conv.register_full_backward_hook(bwd_hook)

    model.zero_grad()
    output = model(input_tensor)
    score = output[0, target_class]
    score.backward()

    h_fwd.remove()
    h_bwd.remove()

    feat = activations['feat'][0]      # (C, H, W)
    grad = gradients['grad'][0]        # (C, H, W)
    weights = grad.mean(dim=(1, 2))    # (C,)
    cam = (weights[:, None, None] * feat).sum(0)  # (H, W)
    cam = torch.clamp(cam, min=0)
    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()
    return cam.cpu().numpy()

print("Generating Grad-CAM heatmaps (1 representative sample per class)...")
with mlflow.start_run(run_name="stage5_gradcam"):
    for cls_idx, cls_name in enumerate(CNN_CLASS_NAMES):
        cls_rows = labeled_df[labeled_df['encoded_label'] == cls_idx]
        if len(cls_rows) == 0:
            print(f"  Skipping {cls_name} -- no samples")
            continue

        sample_row = cls_rows.iloc[len(cls_rows) // 2]
        wmap = sample_row['waferMap'].astype(np.float32)

        if wmap.shape != (64, 64):
            H, W = wmap.shape
            ri = np.floor(np.arange(64) * H / 64).astype(int)
            ci = np.floor(np.arange(64) * W / 64).astype(int)
            wmap = wmap[ri, :][:, ci]

        wmap_3c = np.stack([wmap] * 3, axis=0).astype(np.float32) / 2.0
        inp = torch.tensor(wmap_3c).unsqueeze(0).to(device)
        inp.requires_grad_(True)

        cam = compute_gradcam(model_cnn, inp, cls_idx, model_cnn.conv3)

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        axes[0].imshow(wmap, cmap='Blues', vmin=0, vmax=2)
        axes[0].set_title(f'Wafer Map ({cls_name})')
        axes[0].axis('off')

        # conv3 output is 8x8 -> upsample to 64x64
        cam_up = np.repeat(np.repeat(cam, 8, axis=0), 8, axis=1)[:64, :64]
        axes[1].imshow(wmap, cmap='Blues', vmin=0, vmax=2, alpha=0.6)
        axes[1].imshow(cam_up, cmap='jet', alpha=0.4)
        axes[1].set_title(f'Grad-CAM ({cls_name})')
        axes[1].axis('off')

        plt.suptitle(f'Grad-CAM -- {cls_name}', fontsize=12)
        plt.tight_layout()
        save_path = f'results/figures/gradcam_{cls_name.replace("-","_").replace(" ","_")}.png'
        plt.savefig(save_path, dpi=100)
        plt.close()
        mlflow.log_artifact(save_path)
        print(f"  Saved {save_path}")

print("\nAll SHAP + Grad-CAM figures saved.")
print("Task 7 complete.")
