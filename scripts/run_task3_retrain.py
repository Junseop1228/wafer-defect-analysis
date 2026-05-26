import yaml
import mlflow
import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
import os
import sys

sys.path.append(os.path.abspath('.'))
from src.train import train_cnn, train_hybrid
from src.models import WaferCNN, WaferMapDataset
from sklearn.model_selection import train_test_split

cfg = yaml.safe_load(open('config.yaml', encoding='utf-8'))
mlflow.set_experiment("wafer_defect_phase3")

print("1. LSWMD.pkl 로드 후 전처리 적용...")
raw_df = pd.read_pickle('data/LSWMD.pkl')
raw_df['failureType'] = raw_df['failureType'].apply(lambda x: x[0][0] if len(x) > 0 else '')

# Plan correction: none -> Normal to match 172950 rows (otherwise it drops to 25k)
raw_df['failureType'] = raw_df['failureType'].replace({'Near-full': 'Normal', 'none': 'Normal'})
df = raw_df[raw_df['failureType'].notna() & (raw_df['failureType'] != '')].copy()

print(f"Resulting labeled dataset shape: {df.shape}")
print(f"Unique classes: {df['failureType'].unique()} (Total: {len(df['failureType'].unique())})")

wafer_maps = df['waferMap'].values
labels = df['failureType'].values

print(f"Pre-resizing {len(wafer_maps)} wafer maps to 64x64...")
resized_maps = []
for wmap in wafer_maps:
    if wmap.shape != (64, 64):
        H, W = wmap.shape
        row_indices = np.floor(np.arange(64) * H / 64).astype(int)
        col_indices = np.floor(np.arange(64) * W / 64).astype(int)
        wmap = wmap[row_indices, :][:, col_indices]
    resized_maps.append(wmap)
wafer_maps = np.array(resized_maps)
print("Finished pre-resizing.")

idx_tr, idx_te = train_test_split(range(len(wafer_maps)), test_size=0.2, stratify=labels, random_state=cfg['seed'])

print("2. train_cnn 실행 (8클래스)...")
res_cnn = train_cnn(cfg, wafer_maps[idx_tr], labels[idx_tr], wafer_maps[idx_te], labels[idx_te])
print(f"CNN Macro F1: {res_cnn['macro_f1']:.4f}")

print("3. data/cnn_embeddings.npy 재생성 (172950 × 128)...")
model_cnn = res_cnn['model']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_cnn.to(device)
model_cnn.eval()

# We need integers for labels to create dataset, but WaferMapDataset takes labels so we need to encode them
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
labels_enc = le.fit_transform(labels)

emb_ds = WaferMapDataset(wafer_maps, labels_enc)
emb_dl = DataLoader(emb_ds, batch_size=256, shuffle=False)

embeddings = []
with torch.no_grad():
    for xb, _ in emb_dl:
        emb = model_cnn.extract_features(xb.to(device)).cpu().numpy()
        embeddings.append(emb)
cnn_embeddings = np.vstack(embeddings)

os.makedirs(os.path.dirname(cfg['data']['cnn_embeddings']), exist_ok=True)
np.save(cfg['data']['cnn_embeddings'], cnn_embeddings)
print(f"CNN embeddings saved. Shape: {cnn_embeddings.shape}")

print("4. Hybrid 새 embedding으로 재학습...")
df_hybrid = pd.read_pickle('data/features_labeled_v2.pkl')
FEATURE_COLS = [c for c in df_hybrid.columns if c != 'failureType']
X_manual = df_hybrid[FEATURE_COLS].values
y_hybrid = df_hybrid['failureType'].values

X_hybrid = np.hstack([X_manual, cnn_embeddings])
print(f"Hybrid Matrix shape: {X_hybrid.shape}")

idx_tr_h, idx_te_h = train_test_split(range(len(X_hybrid)), test_size=0.2, stratify=y_hybrid, random_state=cfg['seed'])

res_hybrid = train_hybrid(cfg, X_hybrid[idx_tr_h], y_hybrid[idx_tr_h], X_hybrid[idx_te_h], y_hybrid[idx_te_h], cfg['data']['cnn_embeddings'])
print(f"Hybrid Macro F1: {res_hybrid['macro_f1']:.4f}")

# Update metrics.csv
metrics_path = 'results/metrics.csv'
if os.path.exists(metrics_path):
    metrics_df = pd.read_csv(metrics_path)
    # Update CNN row
    if 'CNN' in metrics_df['Model'].values:
        metrics_df.loc[metrics_df['Model'] == 'CNN', 'macro_F1'] = res_cnn['macro_f1']
    # Update Hybrid row
    if 'Hybrid' in metrics_df['Model'].values:
        metrics_df.loc[metrics_df['Model'] == 'Hybrid', 'macro_F1'] = res_hybrid['macro_f1']
    metrics_df.to_csv(metrics_path, index=False)
    print("metrics.csv updated with new F1 scores.")

print("Task 3 Retraining Completed.")
