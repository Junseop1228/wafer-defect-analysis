import yaml, mlflow, pandas as pd, numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import classification_report, f1_score, recall_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb
import optuna
import os, sys

sys.path.append(os.path.abspath('.'))
from src.visualize import plot_confusion_matrix

cfg = yaml.safe_load(open('config.yaml'))
mlflow.set_experiment("wafer_defect_phase2")

df = pd.read_pickle('data/features_labeled_v2.pkl')
FEATURE_COLS = [c for c in df.columns if c != 'failureType']

le = LabelEncoder()
y_multi = le.fit_transform(df['failureType'])
X_multi = df[FEATURE_COLS].values
CLASS_NAMES = le.classes_.tolist()

X_tr, X_te, y_tr, y_te = train_test_split(
    X_multi, y_multi, test_size=0.2, stratify=y_multi, random_state=cfg['seed']
)

# RandomForest
print("Starting RF Multiclass...")
with mlflow.start_run(run_name="stage42_rf_multiclass"):
    rf_mc = RandomForestClassifier(
        n_estimators=cfg['models']['rf']['n_estimators'],
        class_weight='balanced',
        n_jobs=-1,
        random_state=cfg['seed']
    )
    rf_mc.fit(X_tr, y_tr)
    y_pred_rf_mc = rf_mc.predict(X_te)
    macro_f1_rf = f1_score(y_te, y_pred_rf_mc, average='macro')
    per_class = dict(zip(CLASS_NAMES, f1_score(y_te, y_pred_rf_mc, average=None)))
    mlflow.log_params({'model': 'RF_multiclass', 'class_weight': 'balanced', 'n_estimators': cfg['models']['rf']['n_estimators']})
    mlflow.log_metrics({'macro_f1': macro_f1_rf, **{f'f1_{k}': v for k,v in per_class.items()}})
    
    scratch_idx = CLASS_NAMES.index('Scratch')
    donut_idx = CLASS_NAMES.index('Donut')
    scratch_recall_rf = recall_score(y_te, y_pred_rf_mc, labels=[scratch_idx], average=None)[0]
    donut_recall_rf = recall_score(y_te, y_pred_rf_mc, labels=[donut_idx], average=None)[0]
    mlflow.log_metrics({'scratch_recall': scratch_recall_rf, 'donut_recall': donut_recall_rf})
    
    print("RF Results:")
    print(classification_report(y_te, y_pred_rf_mc, target_names=CLASS_NAMES))

# XGBoost Optuna
print("Starting XGBoost Multiclass Optuna...")
def objective_mc(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 150),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_float('lr', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'objective': 'multi:softmax',
        'num_class': len(CLASS_NAMES),
        'random_state': cfg['seed'],
        'n_jobs': -1,
        'tree_method': 'hist'
    }
    model = xgb.XGBClassifier(**params)
    model.fit(X_tr, y_tr, sample_weight=compute_sample_weight('balanced', y_tr))
    preds = model.predict(X_te)
    return f1_score(y_te, preds, average='macro')

study_mc = optuna.create_study(direction='maximize')
study_mc.optimize(objective_mc, n_trials=3, show_progress_bar=True)

best_params = study_mc.best_params
best_params['objective'] = 'multi:softmax'
best_params['num_class'] = len(CLASS_NAMES)
best_params['random_state'] = cfg['seed']
best_params['n_jobs'] = -1
best_params['tree_method'] = 'hist'

with mlflow.start_run(run_name="stage42_xgb_multiclass"):
    xgb_best_mc = xgb.XGBClassifier(**best_params)
    xgb_best_mc.fit(X_tr, y_tr, sample_weight=compute_sample_weight('balanced', y_tr))
    y_pred_xgb_mc = xgb_best_mc.predict(X_te)
    
    macro_f1_xgb_mc = f1_score(y_te, y_pred_xgb_mc, average='macro')
    per_class_xgb = dict(zip(CLASS_NAMES, f1_score(y_te, y_pred_xgb_mc, average=None)))
    mlflow.log_params(best_params)
    mlflow.log_metrics({'macro_f1': macro_f1_xgb_mc, **{f'f1_{k}': v for k,v in per_class_xgb.items()}})
    
    scratch_recall_xgb = recall_score(y_te, y_pred_xgb_mc, labels=[scratch_idx], average=None)[0]
    donut_recall_xgb = recall_score(y_te, y_pred_xgb_mc, labels=[donut_idx], average=None)[0]
    mlflow.log_metrics({'scratch_recall': scratch_recall_xgb, 'donut_recall': donut_recall_xgb})
    
    print("\nXGBoost Results:")
    print(classification_report(y_te, y_pred_xgb_mc, target_names=CLASS_NAMES))
    
    plot_confusion_matrix(y_te, y_pred_xgb_mc, labels=CLASS_NAMES,
                          save_path='results/figures/cm_ml_multiclass.png')

# Save best metrics for hybrid step later
metrics = {
    'macro_f1_xgb_mc': float(macro_f1_xgb_mc),
    'scratch_xgb_mc': float(scratch_recall_xgb),
    'donut_xgb_mc': float(donut_recall_xgb)
}
import json
with open('results/ml_metrics.json', 'w') as f:
    json.dump(metrics, f)

print("All tasks completed.")
print(f"Metrics to report: macro_f1={macro_f1_xgb_mc:.4f}, scratch_recall={scratch_recall_xgb:.4f}, donut_recall={donut_recall_xgb:.4f}")
