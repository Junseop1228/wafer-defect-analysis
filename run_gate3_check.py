"""
Task 8 -- Gate 3 Check
Runs Hybrid XGBoost on test set and checks confusion pair F1 gaps.
Pairs: Edge-Ring<->Edge-Loc, Center<->Donut, Loc<->Random
"""
import sys, os
import sys as _sys
if hasattr(_sys.stdout, 'reconfigure'):
    _sys.stdout.reconfigure(encoding='utf-8')

import yaml, json, numpy as np, pandas as pd
sys.path.append(os.path.abspath('.'))
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import classification_report, f1_score
import xgboost as xgb
from src.evaluate import check_gate3

cfg = yaml.safe_load(open('config.yaml'))

print("Loading features + CNN embeddings...")
df = pd.read_pickle('data/features_labeled_v2.pkl')
df['failureType'] = df['failureType'].replace('Near-full', 'Normal')
FEATURE_COLS = [c for c in df.columns if c != 'failureType']

le = LabelEncoder()
y_multi = le.fit_transform(df['failureType'])
CLASS_NAMES = le.classes_.tolist()
print(f"Classes ({len(CLASS_NAMES)}): {CLASS_NAMES}")

manual_feats = df[FEATURE_COLS].values
cnn_embs = np.load('data/cnn_embeddings.npy')
X_hybrid = np.hstack([manual_feats, cnn_embs])

X_tr_h, X_te_h, y_tr_h, y_te_h = train_test_split(
    X_hybrid, y_multi, test_size=0.2, stratify=y_multi, random_state=cfg['seed']
)

# Re-train Hybrid with best known params (trial 9 from Task 5)
print("Re-training Hybrid XGBoost...")
best_params = {
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
model = xgb.XGBClassifier(**best_params)
model.fit(X_tr_h, y_tr_h, sample_weight=compute_sample_weight('balanced', y_tr_h))

print("Predicting on test set...")
y_pred = model.predict(X_te_h)

print("\nClassification Report:")
print(classification_report(y_te_h, y_pred, target_names=CLASS_NAMES))

# Gate 3
print("=" * 60)
print("GATE 3 CHECK")
print("=" * 60)
confusion_pairs = [
    ("Edge-Ring", "Edge-Loc"),
    ("Center", "Donut"),
    ("Loc", "Random"),
]
gate3_result = check_gate3(
    y_true=y_te_h,
    y_pred=y_pred,
    class_names=CLASS_NAMES,
    cfg=cfg,
    confusion_pairs=confusion_pairs,
)
print(json.dumps(gate3_result, indent=2))
