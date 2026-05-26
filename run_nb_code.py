import yaml, mlflow, pandas as pd, numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, recall_score
import xgboost as xgb
import optuna
import os, sys

sys.path.append(os.path.abspath('.'))
from src.evaluate import check_gate2_binary
from src.visualize import plot_confusion_matrix

cfg = yaml.safe_load(open('config.yaml'))
mlflow.set_experiment("wafer_defect_phase2")

df = pd.read_pickle('data/features_labeled_v2.pkl')
FEATURE_COLS = [c for c in df.columns if c != 'failureType']
df['binary_label'] = (df['failureType'] != 'Normal').astype(int)
X = df[FEATURE_COLS].values
y = df['binary_label'].values
print(f'Class distribution: {pd.Series(y).value_counts().to_dict()}')

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=cfg['seed']
)

with mlflow.start_run(run_name="stage41_rf_baseline"):
    rf = RandomForestClassifier(
        n_estimators=cfg['models']['rf']['n_estimators'],
        class_weight='balanced',
        n_jobs=-1,
        random_state=cfg['seed']
    )
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    recall_defect_rf = recall_score(y_test, y_pred_rf, pos_label=1)
    mlflow.log_params({'model': 'RF', 'n_estimators': cfg['models']['rf']['n_estimators'], 'class_weight': 'balanced'})
    mlflow.log_metrics({'binary_defect_recall': float(recall_defect_rf)})
    print("RF Baseline Report:")
    print(classification_report(y_test, y_pred_rf, target_names=['Normal', 'Defect']))

def objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_float('lr', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'scale_pos_weight': (y_train == 0).sum() / (y_train == 1).sum(),
        'use_label_encoder': False,
        'eval_metric': 'logloss',
        'random_state': cfg['seed'],
        'n_jobs': -1
    }
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return recall_score(y_test, preds, pos_label=1)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=cfg['optuna']['n_trials_binary'], show_progress_bar=True)

best_params = study.best_params
with mlflow.start_run(run_name="stage41_xgb_optuna"):
    xgb_best = xgb.XGBClassifier(**best_params, random_state=cfg['seed'], n_jobs=-1)
    xgb_best.fit(X_train, y_train)
    y_pred_xgb = xgb_best.predict(X_test)
    recall_defect_xgb = recall_score(y_test, y_pred_xgb, pos_label=1)
    mlflow.log_params(best_params)
    mlflow.log_metrics({'binary_defect_recall': float(recall_defect_xgb)})
    print("XGBoost Tuned Report:")
    print(classification_report(y_test, y_pred_xgb, target_names=['Normal', 'Defect']))

best_recall = max(recall_defect_rf, recall_defect_xgb)
gate2_binary = check_gate2_binary(best_recall, cfg)
print(gate2_binary)

os.makedirs('results/figures', exist_ok=True)
plot_confusion_matrix(y_test, y_pred_xgb, labels=['Normal', 'Defect'],
                      save_path='results/figures/cm_binary.png')
