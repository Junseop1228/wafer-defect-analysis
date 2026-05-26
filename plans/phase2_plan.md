# Phase 2 Execution Plan — Modeling Notebooks

> Claude.ai generated this plan. Antigravity executes Tasks in order.
> Read context/stage_status.md and context/handoff.md before starting.
> Do NOT skip tasks or combine tasks.

---

## Phase 2 Scope

- Task 0: src/features.py 정합 수정 (21개 피처로 drop 반영)
- Stage 4-1: Binary Classification (03_binary_classification.ipynb)
- Stage 4-2: 7-class Pattern Classification (04_pattern_classification.ipynb)
  - Part A: ML (RF / XGBoost)
  - Part B: CNN
  - Part C: Hybrid (28 features → 21 features + CNN 128 = 149 dim)
- Stage 5: SHAP + Grad-CAM (05_comparison_shap.ipynb)
- Stage 6: Domain Interpretation (06_domain_interpretation.ipynb)
- Gate 2: binary_defect_recall ≥ 0.90 / Scratch recall ≥ 0.70 / Donut recall ≥ 0.75
- Gate 3: Confusion pair F1 gap ≤ 0.20

---

## Input Data

- Primary: `data/features_labeled_v2.pkl`
  - Shape: (172950, 22) — 21 features + failureType
  - failureType classes: Normal / Edge-Ring / Edge-Loc / Center / Loc / Scratch / Random / Donut (8종, Near-full은 Normal에 병합 완료)
- Raw wafer maps: `data/LSWMD.pkl` — CNN 입력용 (64×64 리사이즈)
- Config: `config.yaml`

---

## Design Decisions

### 이진 레이블 정의
- Normal → 0 (정상)
- Edge-Ring / Edge-Loc / Center / Loc / Scratch / Random / Donut → 1 (불량)
- 적용: features_labeled_v2.pkl의 failureType 컬럼 기반 파생

### CNN 이미지 전처리
- waferMap 값 그대로 사용: 0=외부, 1=정상, 2=결함
- 64×64 리사이즈 (PIL NEAREST 또는 cv2 INTER_NEAREST — 정수값 보존)
- 3채널로 복제 (1ch → 3ch, RGB 동일값)
- 정규화: 0~2 → [0, 1] 범위 (÷ 2.0)

### Hybrid 입력 차원
- 21개 수동 피처 + CNN GlobalAvgPool 출력 128차원 = 149차원 (설계서의 156은 28기준, 실제는 149)
- CNN은 Task 4 학습 완료 후 FC 직전 레이어에서 128차원 추출

### 불균형 처리 전략 (단계적 적용)
1단계: class_weight='balanced' (RF/XGBoost)
2단계: Gate 2 실패 시 → 사이클 ③ 발동 (SMOTE → Pseudo-labeling → threshold 조정)

### MLflow 실험 구조
- Experiment: "wafer_defect_phase2"
- Run 이름 규칙: "stage{N}_{model}_{timestamp}"
- 필수 log: params(model_type, n_estimators/lr, class_weight), metrics(macro_f1, binary_defect_recall, scratch_recall, donut_recall)

---

## Task List

---

### Task 0 — src/features.py 정합 수정
**Branch**: `agent/phase2-features-sync`

**목적**: features_labeled_v2.pkl의 21개 피처와 src/features.py extract_features() 출력을 일치시킴.
현재 extract_features()는 28개를 반환하지만, v2에서 7개가 제거된 상태.

**Steps**:
1. features_labeled_v2.pkl 컬럼 목록 출력하여 drop된 피처명 확인
   ```powershell
   conda activate wm811k; python -c "
   import pandas as pd
   df = pd.read_pickle('data/features_labeled_v2.pkl')
   print('Shape:', df.shape)
   print('Columns:', df.columns.tolist())
   "
   ```
2. 확인된 drop 목록을 `config.yaml`에 추가
   - 위치: `features.dropped_features` 키 아래 리스트로 저장
3. `src/features.py` `extract_features()` 함수 수정
   - 함수 끝에서 config.yaml의 dropped_features 리스트에 있는 키를 결과 dict에서 제거
   - 단, config.yaml을 인자로 받지 않는 경우 → 함수 시그니처에 `drop_list=None` 파라미터 추가

**Verification**:
```powershell
conda activate wm811k; python -c "
import numpy as np, yaml
from src.features import extract_features
cfg = yaml.safe_load(open('config.yaml'))
test_map = np.random.choice([0,1,2], size=(64,64), p=[0.3,0.5,0.2])
feats = extract_features(test_map, drop_list=cfg['features']['dropped_features'])
print(f'Feature count: {len(feats)} (expected 21)')
assert len(feats) == 21
print('PASS')
"
```

**Completion**: extract_features() 반환 키 수 == 21, config.yaml에 dropped_features 리스트 존재

---

### Task 1 — Stage 4-1: Binary Classification Notebook
**Branch**: `agent/stage41-binary`
**File**: `notebooks/03_binary_classification.ipynb`

**Cells 구조**:

Cell 1 — Setup
```python
import yaml, mlflow, pandas as pd, numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report, recall_score
import xgboost as xgb
import optuna
from src.evaluate import check_gate2_binary
from src.visualize import plot_confusion_matrix

cfg = yaml.safe_load(open('../config.yaml'))
mlflow.set_experiment("wafer_defect_phase2")
```

Cell 2 — Data Load & Binary Label
```python
df = pd.read_pickle('../data/features_labeled_v2.pkl')
FEATURE_COLS = [c for c in df.columns if c != 'failureType']
df['binary_label'] = (df['failureType'] != 'Normal').astype(int)
X = df[FEATURE_COLS].values
y = df['binary_label'].values
print(f'Class distribution: {pd.Series(y).value_counts().to_dict()}')
```

Cell 3 — Stratified Split
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=cfg['seed']
)
```

Cell 4 — RF Baseline (class_weight='balanced')
```python
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
    mlflow.log_metrics({'binary_defect_recall': recall_defect_rf})
    print(classification_report(y_test, y_pred_rf, target_names=['Normal', 'Defect']))
```

Cell 5 — XGBoost + Optuna Tuning
```python
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
    mlflow.log_metrics({'binary_defect_recall': recall_defect_xgb})
    print(classification_report(y_test, y_pred_xgb, target_names=['Normal', 'Defect']))
```

Cell 6 — Gate 2 Binary Check
```python
from src.evaluate import check_gate2_binary
best_recall = max(recall_defect_rf, recall_defect_xgb)
gate2_binary = check_gate2_binary(best_recall, cfg)
print(gate2_binary)
# {'passed': True/False, 'details': {'binary_defect_recall': X.XX, 'threshold': 0.90}}
```

Cell 7 — Confusion Matrix Visualization
```python
from src.visualize import plot_confusion_matrix
plot_confusion_matrix(y_test, y_pred_xgb, labels=['Normal', 'Defect'],
                      save_path='../results/figures/cm_binary.png')
```

**config.yaml에 추가 필요**:
```yaml
seed: 42
models:
  rf:
    n_estimators: 200
optuna:
  n_trials_binary: 30
  n_trials_multiclass: 50
```

**src/evaluate.py에 추가 필요**:
- `check_gate2_binary(recall, cfg)` → `{'passed': bool, 'details': dict}`

**Verification**:
```powershell
conda activate wm811k; jupyter nbconvert --to notebook --execute notebooks/03_binary_classification.ipynb --ExecutePreprocessor.timeout=600
```

**Completion criteria**:
- 노트북 실행 완료 (에러 없음)
- results/figures/cm_binary.png 생성
- experiment_log.md에 binary_defect_recall 수치 기록
- Gate 2 binary check 결과 출력

---

### Task 2 — Gate 2 Binary Check & Decision
**담당**: Antigravity 실행 → Claude.ai 해석

Antigravity는 Task 1 완료 후 experiment_log.md에 결과 append.
Claude.ai가 "결과 해석해줘" 트리거로 Gate 2 binary 판단.

- PASSED (recall ≥ 0.90) → Task 3으로 진행
- FAILED → 사이클 ③ 발동 (class_weight → SMOTE → threshold 순서)

---

### Task 3 — Stage 4-2 Part A: ML 7-class Classification
**Branch**: `agent/stage42-ml`
**File**: `notebooks/04_pattern_classification.ipynb` (Part A)

**Cells 구조**:

Cell 1 — Setup (Task 1과 동일 imports + 추가)
```python
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, classification_report
```

Cell 2 — Data Load & Encode
```python
df = pd.read_pickle('../data/features_labeled_v2.pkl')
FEATURE_COLS = [c for c in df.columns if c != 'failureType']
le = LabelEncoder()
y_multi = le.fit_transform(df['failureType'])
X_multi = df[FEATURE_COLS].values
CLASS_NAMES = le.classes_.tolist()

X_tr, X_te, y_tr, y_te = train_test_split(
    X_multi, y_multi, test_size=0.2, stratify=y_multi, random_state=cfg['seed']
)
```

Cell 3 — RF Multiclass
```python
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
    mlflow.log_params({'model': 'RF_multiclass', 'class_weight': 'balanced'})
    mlflow.log_metrics({'macro_f1': macro_f1_rf, **{f'f1_{k}': v for k,v in per_class.items()}})
    print(classification_report(y_te, y_pred_rf_mc, target_names=CLASS_NAMES))
```

Cell 4 — XGBoost Multiclass + Optuna
```python
def objective_mc(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_float('lr', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'objective': 'multi:softmax',
        'num_class': len(CLASS_NAMES),
        'random_state': cfg['seed'],
        'n_jobs': -1
    }
    model = xgb.XGBClassifier(**params)
    model.fit(X_tr, y_tr, sample_weight=compute_sample_weight('balanced', y_tr))
    preds = model.predict(X_te)
    return f1_score(y_te, preds, average='macro')

study_mc = optuna.create_study(direction='maximize')
study_mc.optimize(objective_mc, n_trials=cfg['optuna']['n_trials_multiclass'], show_progress_bar=True)
# best model 저장 및 MLflow 기록 동일 패턴
```

Cell 5 — Confusion Matrix (3방향 비교용 저장)
```python
plot_confusion_matrix(y_te, y_pred_xgb_mc, labels=CLASS_NAMES,
                      save_path='../results/figures/cm_ml_multiclass.png')
```

**Completion criteria**:
- macro_f1 (ML), scratch_recall, donut_recall 수치 기록
- cm_ml_multiclass.png 생성

---

### Task 4 — Stage 4-2 Part B: CNN 7-class Classification
**Branch**: `agent/stage42-cnn`
**File**: `notebooks/04_pattern_classification.ipynb` (Part B — 별도 섹션으로 append)

**CNN 아키텍처 (src/models.py에 구현)**:
```
WaferCNN:
  Conv2d(3→32, k=3, p=1) → BN → ReLU → MaxPool(2)    # 64→32
  Conv2d(32→64, k=3, p=1) → BN → ReLU → MaxPool(2)   # 32→16
  Conv2d(64→128, k=3, p=1) → BN → ReLU → MaxPool(2)  # 16→8
  AdaptiveAvgPool2d(1)                                  # 8→1 (GlobalAvgPool)
  Flatten → Linear(128→64) → ReLU → Linear(64→7)
  (중간 128차원 벡터 = Hybrid 입력용)
```

**Dataset 클래스 (src/models.py)**:
- `WaferMapDataset(wafer_maps, labels, transform=None)`
- wafer_maps: numpy array of variable-size maps → 64×64 리사이즈 → 3ch 복제 → ÷2.0
- LSWMD.pkl에서 waferMap 컬럼 추출

**학습 설정**:
```yaml
# config.yaml에 추가
models:
  cnn:
    epochs: 30
    batch_size: 64
    lr: 0.001
    weight_decay: 0.0001
```

**Cells 구조**:

Cell B-1 — CNN Data Prep
```python
import torch
from torch.utils.data import DataLoader
from src.models import WaferCNN, WaferMapDataset

raw_df = pd.read_pickle('../data/LSWMD.pkl')
# failureType 병합 (Near-full → Normal)
raw_df['failureType'] = raw_df['failureType'].replace('Near-full', 'Normal')
# labeled only
labeled_df = raw_df[raw_df['failureType'].notna() & (raw_df['failureType'] != '')]
wafer_maps = labeled_df['waferMap'].values
labels_cnn = le.transform(labeled_df['failureType'].values)  # Task 3에서 fit한 le 재사용
```

Cell B-2 — Train/Val Split & DataLoader
```python
from sklearn.model_selection import train_test_split
idx_tr, idx_te = train_test_split(range(len(wafer_maps)), test_size=0.2,
                                   stratify=labels_cnn, random_state=cfg['seed'])
train_ds = WaferMapDataset(wafer_maps[idx_tr], labels_cnn[idx_tr])
test_ds  = WaferMapDataset(wafer_maps[idx_te], labels_cnn[idx_te])
train_dl = DataLoader(train_ds, batch_size=cfg['models']['cnn']['batch_size'],
                      shuffle=True, num_workers=0)
test_dl  = DataLoader(test_ds,  batch_size=cfg['models']['cnn']['batch_size'],
                      shuffle=False, num_workers=0)
```

Cell B-3 — Training Loop
```python
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_cnn = WaferCNN(num_classes=len(CLASS_NAMES)).to(device)
# class_weight for CrossEntropyLoss
from sklearn.utils.class_weight import compute_class_weight
cw = compute_class_weight('balanced', classes=np.unique(labels_cnn), y=labels_cnn)
criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor(cw, dtype=torch.float32).to(device))
optimizer = torch.optim.Adam(model_cnn.parameters(),
                              lr=cfg['models']['cnn']['lr'],
                              weight_decay=cfg['models']['cnn']['weight_decay'])

with mlflow.start_run(run_name="stage42_cnn"):
    for epoch in range(cfg['models']['cnn']['epochs']):
        model_cnn.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model_cnn(xb), yb)
            loss.backward()
            optimizer.step()
    # Evaluation
    model_cnn.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for xb, yb in test_dl:
            preds = model_cnn(xb.to(device)).argmax(dim=1).cpu().numpy()
            all_preds.extend(preds); all_labels.extend(yb.numpy())
    macro_f1_cnn = f1_score(all_labels, all_preds, average='macro')
    mlflow.log_metrics({'macro_f1': macro_f1_cnn})
    # Save model
    torch.save(model_cnn.state_dict(), '../results/cnn_weights.pth')
```

Cell B-4 — Extract CNN 128-dim Embeddings (Hybrid 입력용)
```python
# WaferCNN에 extract_features() 메서드 추가 필요 (GlobalAvgPool 출력)
embeddings = []
with torch.no_grad():
    for xb, _ in DataLoader(WaferMapDataset(wafer_maps, labels_cnn),
                             batch_size=256, shuffle=False):
        emb = model_cnn.extract_features(xb.to(device)).cpu().numpy()
        embeddings.append(emb)
cnn_embeddings = np.vstack(embeddings)  # (172950, 128)
np.save('../data/cnn_embeddings.npy', cnn_embeddings)
```

**Completion criteria**:
- results/cnn_weights.pth 저장
- data/cnn_embeddings.npy 저장 (172950, 128)
- macro_f1_cnn 기록

---

### Task 5 — Stage 4-2 Part C: Hybrid Classification
**Branch**: `agent/stage42-hybrid`
**File**: `notebooks/04_pattern_classification.ipynb` (Part C — append)

**Cells 구조**:

Cell C-1 — Build Hybrid Feature Matrix
```python
manual_feats = df[FEATURE_COLS].values           # (172950, 21) — Task 3 기준
cnn_embs = np.load('../data/cnn_embeddings.npy')  # (172950, 128)
X_hybrid = np.hstack([manual_feats, cnn_embs])    # (172950, 149)
print(f'Hybrid dim: {X_hybrid.shape}')  # expected (172950, 149)
```

Cell C-2 — XGBoost on Hybrid
```python
X_tr_h, X_te_h, y_tr_h, y_te_h = train_test_split(
    X_hybrid, y_multi, test_size=0.2, stratify=y_multi, random_state=cfg['seed']
)
# Optuna 동일 패턴 (n_trials=cfg['optuna']['n_trials_multiclass'])
# best Hybrid model → MLflow run_name="stage42_hybrid"
```

Cell C-3 — 3방향 비교 요약 테이블
```python
comparison = pd.DataFrame({
    'Model': ['ML (XGBoost)', 'CNN', 'Hybrid'],
    'macro_F1': [macro_f1_xgb_mc, macro_f1_cnn, macro_f1_hybrid],
    'Scratch_recall': [scratch_rf_mc, scratch_cnn, scratch_hybrid],
    'Donut_recall': [donut_rf_mc, donut_cnn, donut_hybrid]
})
comparison.to_csv('../results/metrics.csv', index=False)
print(comparison)
```

**Completion criteria**:
- results/metrics.csv 생성 (3행 × 4열)
- macro_f1_hybrid 기록

---

### Task 6 — Gate 2 Full Check
**담당**: Antigravity 실행 → Claude.ai 해석

```python
# src/evaluate.py에 check_gate2_full() 추가
# 입력: scratch_recall, donut_recall, cfg
# 반환: {'passed': bool, 'details': dict}
```

Antigravity는 결과를 experiment_log.md에 append.
Claude.ai가 Gate 2 최종 판단:
- PASSED → Task 7 진행
- FAILED → 사이클 ③ 발동 순서: class_weight 강화 → SMOTE → Pseudo-labeling → threshold tuning

---

### Task 7 — Stage 5: SHAP + Grad-CAM
**Branch**: `agent/stage5-shap`
**File**: `notebooks/05_comparison_shap.ipynb`

**Part A — SHAP (ML/Hybrid)**:
```python
import shap
# XGBoost multiclass SHAP (wm811k env 패치 적용 완료)
explainer = shap.TreeExplainer(xgb_best_mc,
                                feature_perturbation='tree_path_dependent')
shap_values = explainer.shap_values(X_te[:500])  # (500, 21, 8)
# 클래스별 SHAP 막대 그래프 → results/figures/shap_{class}.png
shap.summary_plot(shap_values[cls_idx], X_te[:500],
                  feature_names=FEATURE_COLS, show=False)
```

**Part B — Grad-CAM (CNN)**:
```python
# pytorch-grad-cam 설치 확인: pip install grad-cam
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
target_layer = model_cnn.conv3  # 마지막 Conv 레이어 (WaferCNN에 속성명 명시 필요)
cam = GradCAM(model=model_cnn, target_layers=[target_layer])
# 패턴별 대표 샘플 → Grad-CAM 히트맵 → results/figures/gradcam_{class}.png
```

**Part C — Feature 도메인 해석 테이블**
```python
# SHAP 기여 Top-5 피처를 각 클래스별로 정리
# → results/figures/shap_summary_table.png
```

**Completion criteria**:
- results/figures/shap_{class}.png × 8개 생성
- results/figures/gradcam_{class}.png × 8개 생성

---

### Task 8 — Gate 3 Check
**담당**: Antigravity 실행 → Claude.ai 해석

```python
# src/evaluate.py에 check_gate3() 추가
# 입력: confusion matrix, cfg
# 로직: 혼동 쌍 (Edge-Ring↔Edge-Loc, Center↔Donut, Scratch↔Random 등)의 F1 gap 계산
# 반환: {'passed': bool, 'details': {'worst_pair': str, 'f1_gap': float}}
```

- PASSED (F1 gap ≤ 0.20) → Task 9 진행
- FAILED → 사이클 ② 발동 (혼동 쌍 판별 피처 추가 + 재학습)

---

### Task 9 — Stage 6: Domain Interpretation Notebook
**Branch**: `agent/stage6-domain`
**File**: `notebooks/06_domain_interpretation.ipynb`

**구조**:
- SHAP 결과 → 패턴별 핵심 피처 → 공정 원인 매핑 테이블 (config에 있는 도메인 해석 참조)
- 패턴별 대표 웨이퍼맵 시각화 (5개 샘플 × 8클래스)
- 도메인 해석 요약: 피처 → 공정 → 실무 액션 텍스트 셀로 정리
- 결과: notebooks에 마크다운 셀로 작성 (README에도 반영 예정)

**Completion criteria**:
- 각 클래스별 대표 웨이퍼맵 그리드 이미지 저장
- 도메인 해석 마크다운 정리 완료

---

## Phase 2 Complete Criteria

All of the following must be true:
- [ ] src/features.py extract_features() → 21개 피처 반환
- [ ] Gate 2: binary_defect_recall ≥ 0.90 PASSED
- [ ] Gate 2: Scratch recall ≥ 0.70 PASSED
- [ ] Gate 2: Donut recall ≥ 0.75 PASSED
- [ ] results/metrics.csv 존재 (3방향 비교표)
- [ ] Gate 3: confusion pair F1 gap ≤ 0.20 PASSED
- [ ] results/figures/shap_{class}.png × 8 존재
- [ ] results/figures/gradcam_{class}.png × 8 존재
- [ ] notebooks/06_domain_interpretation.ipynb 완료

When all complete → context/stage_status.md 업데이트 → Phase 3 진행
