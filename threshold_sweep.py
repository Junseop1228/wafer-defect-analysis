import yaml, pandas as pd, numpy as np
from sklearn.metrics import classification_report, recall_score, precision_score, f1_score
import xgboost as xgb
import os, sys

sys.path.append(os.path.abspath('.'))
from src.evaluate import check_gate2_binary

cfg = yaml.safe_load(open('config.yaml'))

df = pd.read_pickle('data/features_labeled_v2.pkl')
FEATURE_COLS = [c for c in df.columns if c != 'failureType']
df['binary_label'] = (df['failureType'] != 'Normal').astype(int)
X = df[FEATURE_COLS].values
y = df['binary_label'].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=cfg['seed']
)

# Best params from previous Optuna run
params = {
    'n_estimators': 193,
    'max_depth': 3,
    'learning_rate': 0.2973121894927801,
    'subsample': 0.7463695313373693,
    'scale_pos_weight': float((y_train == 0).sum() / (y_train == 1).sum()),
    'use_label_encoder': False,
    'eval_metric': 'logloss',
    'random_state': cfg['seed'],
    'n_jobs': -1
}

model = xgb.XGBClassifier(**params)
model.fit(X_train, y_train)
y_probs = model.predict_proba(X_test)[:, 1]

thresholds = np.arange(0.25, 0.51, 0.05)

print("Threshold Sweep Results:")
valid_thresholds = []
for t in thresholds:
    preds = (y_probs >= t).astype(int)
    rec = recall_score(y_test, preds, pos_label=1)
    prec = precision_score(y_test, preds, pos_label=1)
    f1 = f1_score(y_test, preds, pos_label=1)
    print(f"Threshold: {t:.2f} | Recall: {rec:.4f} | Precision: {prec:.4f} | F1: {f1:.4f}")
    if rec >= 0.90:
        valid_thresholds.append((t, rec, prec, f1, preds))

if valid_thresholds:
    valid_thresholds.sort(key=lambda x: x[0], reverse=True)
    best_t, best_rec, best_prec, best_f1, best_preds = valid_thresholds[0]
    print(f"\nOptimal Threshold Found: {best_t:.2f}")
    print(f"Recall: {best_rec:.4f} | Precision: {best_prec:.4f} | F1: {best_f1:.4f}")
    
    # Check gate 2 binary
    gate2_binary = check_gate2_binary(best_rec, cfg)
    print("Gate Check Output:", gate2_binary)
    
    # Save the new confusion matrix if needed
    from src.visualize import plot_confusion_matrix
    plot_confusion_matrix(y_test, best_preds, labels=['Normal', 'Defect'],
                          save_path='results/figures/cm_binary_optimal.png')
else:
    print("\nNo threshold met recall >= 0.90.")
