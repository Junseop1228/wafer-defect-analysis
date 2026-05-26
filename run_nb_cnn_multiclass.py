import yaml, mlflow, pandas as pd, numpy as np, time
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, recall_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
import torch
from torch.utils.data import DataLoader
import os, sys

sys.path.append(os.path.abspath('.'))
from src.models import WaferCNN, WaferMapDataset
from src.visualize import plot_confusion_matrix

cfg = yaml.safe_load(open('config.yaml'))
mlflow.set_experiment("wafer_defect_phase2")

start_time = time.time()
print(f"[{time.time() - start_time:.1f}s] Loading data...")
raw_df = pd.read_pickle('data/LSWMD.pkl')
print(f"[{time.time() - start_time:.1f}s] Data loaded.")

# Apply lambda function safely to extract the actual string from the nested list
raw_df['failureType'] = raw_df['failureType'].apply(lambda x: x[0][0] if len(x) > 0 else '')
# failureType 병합 (Near-full -> Normal)
raw_df['failureType'] = raw_df['failureType'].replace('Near-full', 'Normal')
# labeled only
print(f"[{time.time() - start_time:.1f}s] Extracting labeled_df...")
labeled_df = raw_df[raw_df['failureType'].notna() & (raw_df['failureType'] != '')].copy()
wafer_maps = labeled_df['waferMap'].values

print(f"[{time.time() - start_time:.1f}s] Pre-resizing {len(wafer_maps)} wafer maps to 64x64...")
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

le = LabelEncoder()
labels_cnn = le.fit_transform(labeled_df['failureType'].values)
CLASS_NAMES = le.classes_.tolist()

idx_tr, idx_te = train_test_split(range(len(wafer_maps)), test_size=0.2,
                                   stratify=labels_cnn, random_state=cfg['seed'])
train_ds = WaferMapDataset(wafer_maps[idx_tr], labels_cnn[idx_tr])
test_ds  = WaferMapDataset(wafer_maps[idx_te], labels_cnn[idx_te])

train_dl = DataLoader(train_ds, batch_size=cfg['models']['cnn']['batch_size'],
                      shuffle=True, num_workers=0)
test_dl  = DataLoader(test_ds,  batch_size=cfg['models']['cnn']['batch_size'],
                      shuffle=False, num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model_cnn = WaferCNN(num_classes=len(CLASS_NAMES)).to(device)
cw = compute_class_weight('balanced', classes=np.unique(labels_cnn), y=labels_cnn)
criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32).to(device))
optimizer = torch.optim.Adam(model_cnn.parameters(),
                              lr=cfg['models']['cnn']['lr'],
                              weight_decay=cfg['models']['cnn']['weight_decay'])

print("Starting CNN Multiclass Training...")
with mlflow.start_run(run_name="stage42_cnn"):
    for epoch in range(cfg['models']['cnn']['epochs']):
        model_cnn.train()
        train_loss = 0.0
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model_cnn(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(xb)
        
        train_loss /= len(train_ds)
        print(f"Epoch {epoch+1}/{cfg['models']['cnn']['epochs']} - Loss: {train_loss:.4f}")

    # Evaluation
    model_cnn.eval()
    all_preds, all_labels = [], []
    print("Evaluating on test set...")
    with torch.no_grad():
        for xb, yb in test_dl:
            preds = model_cnn(xb.to(device)).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(yb.numpy())
            
    macro_f1_cnn = f1_score(all_labels, all_preds, average='macro')
    per_class_cnn = dict(zip(CLASS_NAMES, f1_score(all_labels, all_preds, average=None)))
    
    scratch_idx = CLASS_NAMES.index('Scratch')
    donut_idx = CLASS_NAMES.index('Donut')
    scratch_recall_cnn = recall_score(all_labels, all_preds, labels=[scratch_idx], average=None)[0]
    donut_recall_cnn = recall_score(all_labels, all_preds, labels=[donut_idx], average=None)[0]
    
    mlflow.log_params(cfg['models']['cnn'])
    mlflow.log_metrics({'macro_f1': macro_f1_cnn, 
                        'scratch_recall': scratch_recall_cnn, 
                        'donut_recall': donut_recall_cnn,
                        **{f'f1_{k}': v for k,v in per_class_cnn.items()}})
    
    print("\nCNN Results:")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))
    
    plot_confusion_matrix(all_labels, all_preds, labels=CLASS_NAMES,
                          save_path='results/figures/cm_cnn_multiclass.png')
    
    os.makedirs('results', exist_ok=True)
    torch.save(model_cnn.state_dict(), 'results/cnn_weights.pth')
    print("Model weights saved to results/cnn_weights.pth")
    
print("Extracting 128-dim embeddings for all labeled data...")
embeddings = []
model_cnn.eval()
with torch.no_grad():
    for xb, _ in DataLoader(WaferMapDataset(wafer_maps, labels_cnn),
                             batch_size=256, shuffle=False):
        emb = model_cnn.extract_features(xb.to(device)).cpu().numpy()
        embeddings.append(emb)
cnn_embeddings = np.vstack(embeddings)
np.save('data/cnn_embeddings.npy', cnn_embeddings)
print(f"CNN embeddings saved to data/cnn_embeddings.npy with shape {cnn_embeddings.shape}")

import json
if os.path.exists('results/ml_metrics.json'):
    with open('results/ml_metrics.json', 'r') as f:
        metrics = json.load(f)
else:
    metrics = {}

metrics['macro_f1_cnn'] = float(macro_f1_cnn)
metrics['scratch_cnn'] = float(scratch_recall_cnn)
metrics['donut_cnn'] = float(donut_recall_cnn)

with open('results/ml_metrics.json', 'w') as f:
    json.dump(metrics, f)
