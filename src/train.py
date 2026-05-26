import mlflow
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, recall_score
from sklearn.utils.class_weight import compute_sample_weight, compute_class_weight
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
import optuna
import torch
import os
from torch.utils.data import DataLoader
from src.models import WaferCNN, WaferMapDataset


def _preprocess_labels(y_train, y_test):
    """Applies Near-full -> Normal merging if data contains strings."""
    if isinstance(y_train, pd.Series):
        y_train = y_train.replace({'Near-full': 'Normal'}).values
        y_test = y_test.replace({'Near-full': 'Normal'}).values
    elif isinstance(y_train, np.ndarray) and y_train.dtype.kind in {'U', 'S', 'O'}:
        y_train = np.where(y_train == 'Near-full', 'Normal', y_train)
        y_test = np.where(y_test == 'Near-full', 'Normal', y_test)
    return y_train, y_test


def train_binary(cfg: dict, X_train, y_train, X_test, y_test) -> dict:
    y_train, y_test = _preprocess_labels(y_train, y_test)

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
        mlflow.log_params(
            {'model': 'RF', 'n_estimators': cfg['models']['rf']['n_estimators'], 'class_weight': 'balanced'}
        )
        mlflow.log_metrics({'binary_defect_recall': float(recall_defect_rf)})

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
            'n_jobs': -1,
            'tree_method': cfg['models']['xgb']['tree_method']
        }
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        return recall_score(y_test, preds, pos_label=1)

    study = optuna.create_study(direction=cfg['optuna']['direction'])
    study.optimize(objective, n_trials=cfg['optuna']['n_trials_binary'], show_progress_bar=False)

    best_params = study.best_params
    best_params['scale_pos_weight'] = (y_train == 0).sum() / (y_train == 1).sum()
    best_params['use_label_encoder'] = False
    best_params['eval_metric'] = 'logloss'
    best_params['random_state'] = cfg['seed']
    best_params['n_jobs'] = -1
    best_params['tree_method'] = cfg['models']['xgb']['tree_method']

    with mlflow.start_run(run_name="stage41_xgb_optuna"):
        xgb_best = xgb.XGBClassifier(**best_params)
        xgb_best.fit(X_train, y_train)
        y_pred_xgb = xgb_best.predict(X_test)
        recall_defect_xgb = recall_score(y_test, y_pred_xgb, pos_label=1)
        mlflow.log_params(best_params)
        mlflow.log_metrics({'binary_defect_recall': float(recall_defect_xgb)})

    return {
        'model': xgb_best if recall_defect_xgb >= recall_defect_rf else rf,
        'recall': float(max(recall_defect_rf, recall_defect_xgb)),
        'threshold': 0.5
    }


def train_multiclass_ml(cfg: dict, X_train, y_train, X_test, y_test, class_names: list) -> dict:
    y_train, y_test = _preprocess_labels(y_train, y_test)

    # Needs label encoding if not numerical
    le = LabelEncoder()
    if y_train.dtype.kind in {'U', 'S', 'O'}:
        y_train = le.fit_transform(y_train)
        y_test = le.transform(y_test)
        encoded_class_names = le.classes_.tolist()
    else:
        encoded_class_names = class_names

    def objective_mc(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 150),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('lr', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'objective': 'multi:softmax',
            'num_class': len(encoded_class_names),
            'random_state': cfg['seed'],
            'n_jobs': -1,
            'tree_method': cfg['models']['xgb']['tree_method']
        }
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, sample_weight=compute_sample_weight('balanced', y_train))
        preds = model.predict(X_test)
        return f1_score(y_test, preds, average='macro')

    study_mc = optuna.create_study(direction=cfg['optuna']['direction'])
    study_mc.optimize(objective_mc, n_trials=cfg['optuna']['n_trials_multiclass'], show_progress_bar=False)

    best_params = study_mc.best_params
    best_params['objective'] = 'multi:softmax'
    best_params['num_class'] = len(encoded_class_names)
    best_params['random_state'] = cfg['seed']
    best_params['n_jobs'] = -1
    best_params['tree_method'] = cfg['models']['xgb']['tree_method']

    with mlflow.start_run(run_name="stage42_xgb_multiclass"):
        xgb_best_mc = xgb.XGBClassifier(**best_params)
        xgb_best_mc.fit(X_train, y_train, sample_weight=compute_sample_weight('balanced', y_train))
        y_pred_xgb_mc = xgb_best_mc.predict(X_test)

        macro_f1_xgb_mc = f1_score(y_test, y_pred_xgb_mc, average='macro')
        per_class_xgb = dict(zip(encoded_class_names, f1_score(y_test, y_pred_xgb_mc, average=None)))
        mlflow.log_params(best_params)

        metrics_dict = {'macro_f1': float(macro_f1_xgb_mc)}
        for k, v in per_class_xgb.items():
            metrics_dict[f'f1_{k}'] = float(v)
        mlflow.log_metrics(metrics_dict)

    return {
        'model': xgb_best_mc,
        'macro_f1': float(macro_f1_xgb_mc),
        'per_class_f1': per_class_xgb
    }


def train_cnn(cfg: dict, wafer_maps_train, labels_train, wafer_maps_test, labels_test) -> dict:
    labels_train, labels_test = _preprocess_labels(labels_train, labels_test)

    le = LabelEncoder()
    if labels_train.dtype.kind in {'U', 'S', 'O'}:
        labels_train = le.fit_transform(labels_train)
        labels_test = le.transform(labels_test)

    train_ds = WaferMapDataset(wafer_maps_train, labels_train)
    test_ds = WaferMapDataset(wafer_maps_test, labels_test)

    train_dl = DataLoader(train_ds, batch_size=cfg['models']['cnn']['batch_size'], shuffle=True, num_workers=0)
    test_dl = DataLoader(test_ds, batch_size=cfg['models']['cnn']['batch_size'], shuffle=False, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_classes = len(np.unique(labels_train))

    model_cnn = WaferCNN(num_classes=num_classes).to(device)
    cw = compute_class_weight('balanced', classes=np.unique(labels_train), y=labels_train)
    criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32).to(device))
    optimizer = torch.optim.Adam(
        model_cnn.parameters(),
        lr=cfg['models']['cnn']['lr'],
        weight_decay=cfg['models']['cnn']['weight_decay']
    )

    with mlflow.start_run(run_name="stage42_cnn"):
        for epoch in range(cfg['models']['cnn']['epochs']):
            model_cnn.train()
            for xb, yb in train_dl:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                out = model_cnn(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()

        model_cnn.eval()
        all_preds, all_labels = [], []
        with torch.no_grad():
            for xb, yb in test_dl:
                preds = model_cnn(xb.to(device)).argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(yb.numpy())

        macro_f1_cnn = f1_score(all_labels, all_preds, average='macro')

        mlflow.log_params(cfg['models']['cnn'])
        mlflow.log_metrics({'macro_f1': float(macro_f1_cnn)})

        os.makedirs(os.path.dirname(cfg['data']['cnn_weights']), exist_ok=True)
        torch.save(model_cnn.state_dict(), cfg['data']['cnn_weights'])

    embeddings_path = cfg['data']['cnn_embeddings']
    return {
        'model': model_cnn,
        'macro_f1': float(macro_f1_cnn),
        'embeddings_path': embeddings_path
    }


def train_hybrid(cfg: dict, X_train, y_train, X_test, y_test, cnn_embeddings_path: str) -> dict:
    y_train, y_test = _preprocess_labels(y_train, y_test)

    le = LabelEncoder()
    if y_train.dtype.kind in {'U', 'S', 'O'}:
        y_train = le.fit_transform(y_train)
        y_test = le.transform(y_test)

    def objective_hybrid(trial):
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 50, 150),
            'max_depth': trial.suggest_int('max_depth', 3, 8),
            'learning_rate': trial.suggest_float('lr', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'objective': 'multi:softmax',
            'num_class': len(np.unique(y_train)),
            'random_state': cfg['seed'],
            'n_jobs': -1,
            'tree_method': cfg['models']['xgb']['tree_method']
        }
        model = xgb.XGBClassifier(**params)
        model.fit(X_train, y_train, sample_weight=compute_sample_weight('balanced', y_train))
        preds = model.predict(X_test)
        return f1_score(y_test, preds, average='macro')

    study_hybrid = optuna.create_study(direction=cfg['optuna']['direction'])
    study_hybrid.optimize(objective_hybrid, n_trials=cfg['optuna']['n_trials_multiclass'], show_progress_bar=False)

    best_params = study_hybrid.best_params
    best_params['objective'] = 'multi:softmax'
    best_params['num_class'] = len(np.unique(y_train))
    best_params['random_state'] = cfg['seed']
    best_params['n_jobs'] = -1
    best_params['tree_method'] = cfg['models']['xgb']['tree_method']

    with mlflow.start_run(run_name="stage42_hybrid"):
        xgb_best_hybrid = xgb.XGBClassifier(**best_params)
        xgb_best_hybrid.fit(X_train, y_train, sample_weight=compute_sample_weight('balanced', y_train))
        y_pred_hybrid = xgb_best_hybrid.predict(X_test)

        macro_f1_hybrid = f1_score(y_test, y_pred_hybrid, average='macro')
        mlflow.log_params(best_params)
        mlflow.log_metrics({'macro_f1': float(macro_f1_hybrid)})

    return {
        'model': xgb_best_hybrid,
        'macro_f1': float(macro_f1_hybrid)
    }
