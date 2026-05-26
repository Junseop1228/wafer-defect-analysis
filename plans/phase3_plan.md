# Phase 3 Execution Plan — Modularization + Automation

> Claude.ai generated this plan. Antigravity executes Tasks in order.
> Read context/stage_status.md and context/handoff.md before starting.
> Reference: GIT_AND_DIRECTORY.md for branch/commit rules.

---

## Phase 3 Scope

Phase 2에서 scripts/ 또는 notebook 형태로 실행된 코드를 src/ 모듈로 이전.
run_pipeline.py가 단일 진입점으로 전체 파이프라인을 실행할 수 있게 한다.

- Task 0: 06_domain_interpretation.ipynb 작성 (Phase 2 마무리)
- Task 1: src/train.py 구현 (binary + multiclass ML + CNN + Hybrid)
- Task 2: src/visualize.py 확장 (SHAP plot, Grad-CAM, confusion matrix)
- Task 3: CNN 8클래스 재학습 (C-4 수정)
- Task 4: run_pipeline.py 스테이지 배선
- Task 5: 전체 파이프라인 end-to-end 실행 검증

---

## Input

- data/features_labeled_v2.pkl (172950 × 21)
- data/LSWMD.pkl (raw wafer maps, CNN용)
- data/cnn_embeddings.npy (Phase 2 산출, Task 3 이후 교체 예정)
- results/cnn_weights.pth (Phase 2 산출, Task 3에서 교체)
- scripts/*.py (Phase 2 임시 스크립트 — 참조용)
- config.yaml

---

## Task 0 — 06_domain_interpretation.ipynb 작성
**Branch**: `agent/task9-domain-interp`
**File**: `notebooks/06_domain_interpretation.ipynb`

현재 193 bytes (빈 scaffold). SHAP 결과와 도메인 해석을 연결하는 핵심 스토리라인.

**Notebook 구조**:

Cell 1 — Setup
```python
import yaml, pandas as pd, numpy as np, matplotlib.pyplot as plt
import shap
cfg = yaml.safe_load(open('../config.yaml'))
CLASS_NAMES = cfg['classes']['names']
FEATURE_COLS = [...]  # config에서 로드
```

Cell 2 — 대표 웨이퍼맵 시각화 (8클래스 × 5샘플 그리드)
```python
# LSWMD.pkl에서 각 클래스별 대표 5장 추출
# results/figures/sample_wafers_grid.png 저장
```

Cell 3 — SHAP 결과 로드 + 클래스별 Top-5 피처 테이블
```python
# results/figures/shap_*.png 로드 불필요
# Hybrid XGBoost model에서 직접 shap_values 계산
# 각 클래스별 Top-5 피처 → pandas DataFrame으로 출력
```

Cell 4 — 패턴-공정-액션 매핑 (마크다운 셀)
아래 테이블을 노트북 마크다운 셀로 작성:

| 패턴 | Top SHAP 피처 | 의심 공정 | 실무 액션 |
|------|--------------|----------|----------|
| Edge-Ring | edge_sector_std, radial_cdf_7 | CMP 패드 마모 / CVD 균일도 저하 | 패드 교체 주기 단축, 가장자리 레시피 점검 |
| Center | radial_cdf_0~2, cx/cy | CVD/스퍼터링 중앙 가스 과잉 | 샤워헤드 노즐 청소, 가스 비율 점검 |
| Donut | radial_cdf_3~5 | RTP/Annealing 히터 존 불균일 | 히터 존별 온도 프로파일 점검 |
| Scratch | compactness, linearity_pca | 이송 로봇 미끄러짐 | 이송 경로 로그, radon_max_angle 방향 특정 |
| Local | morans_i, n_clusters, lot_consistency | 챔버 파티클 오염 | lot_consistency로 일시적/구조적 판단 |
| Random | defect_ratio, morans_i | 공정 전반 불안정 | Cpk 모니터링, SPC 전체 지표 확인 |
| Edge-Loc | cx, cy, edge_sector_0/5 | 척(Chuck) 특정 부위 손상 | cx/cy 치우침 방향으로 척 핀 특정 |

Cell 5 — 3방향 성능 비교 테이블 (results/metrics.csv 로드)
```python
metrics = pd.read_csv('../results/metrics.csv')
print(metrics.to_markdown(index=False))
```

Cell 6 — Hybrid 선택 근거 (마크다운 셀)
수치 피처 20개: 밀도·반경·클러스터 구조 포착
CNN 128차원: 공간 텍스처·선형성·경계 형태 포착
Hybrid concat: 두 표현을 동시에 활용 → macro F1 0.870

**Verification**:
```powershell
conda activate wm811k; jupyter nbconvert --to notebook --execute notebooks/06_domain_interpretation.ipynb --ExecutePreprocessor.timeout=300
```

ipykernel 없으면 python script로 대체 (Phase 2 패턴 동일)

**Completion criteria**:
- 노트북 실행 완료 (에러 없음)
- 8클래스 대표 웨이퍼맵 그리드 이미지 생성
- 패턴-공정-액션 테이블 마크다운 셀 존재

---

## Task 1 — src/train.py 구현
**Branch**: `agent/phase3-train`

현재 2줄 stub → 4개 학습 함수 구현.
scripts/ 의 run_nb_*.py 파일들을 함수로 추출하되, config.yaml 주입 구조로 리팩토링.

**구현할 함수**:

```python
def train_binary(cfg: dict, X_train, y_train, X_test, y_test) -> dict:
    """
    RF baseline + XGBoost Optuna tuning for binary classification.
    Returns: {'model': best_model, 'recall': float, 'threshold': float}
    """

def train_multiclass_ml(cfg: dict, X_train, y_train, X_test, y_test,
                        class_names: list) -> dict:
    """
    XGBoost Optuna tuning for 7-class classification.
    Returns: {'model': best_model, 'macro_f1': float, 'per_class_f1': dict}
    """

def train_cnn(cfg: dict, wafer_maps_train, labels_train,
              wafer_maps_test, labels_test) -> dict:
    """
    WaferCNN training with class-weighted CrossEntropy.
    Returns: {'model': model, 'macro_f1': float, 'embeddings_path': str}
    """

def train_hybrid(cfg: dict, X_train, y_train, X_test, y_test,
                 cnn_embeddings_path: str) -> dict:
    """
    Hybrid: concat(manual_features, cnn_embeddings) → XGBoost Optuna.
    Returns: {'model': best_model, 'macro_f1': float}
    """
```

**설계 원칙**:
- 모든 하이퍼파라미터: cfg에서 로드. 함수 내 하드코딩 금지.
- MLflow 로깅: 각 함수 내부에서 `with mlflow.start_run()` 블록으로 처리.
- Near-full → Normal 병합: 각 함수 진입 직후 적용.
- class_weight / scale_pos_weight: cfg['models'] 또는 sklearn compute_class_weight로 계산.

**Verification**:
```powershell
conda activate wm811k; python -c "
from src.train import train_binary, train_multiclass_ml, train_cnn, train_hybrid
print('All train functions importable')
"
```

**Completion criteria**:
- 4개 함수 import 가능
- MLflow 로그 포함 (mock 실행 불필요 — import 테스트만)
- flake8 통과

---

## Task 2 — src/visualize.py 확장
**Branch**: `agent/phase3-visualize`

현재 19줄 → confusion matrix + SHAP summary + Grad-CAM 함수 추가.

**추가할 함수**:

```python
def plot_confusion_matrix(y_true, y_pred, labels: list, save_path: str = None):
    """Normalized confusion matrix heatmap."""

def plot_shap_summary(shap_values, X, feature_names: list,
                      class_name: str, save_path: str = None):
    """SHAP bar plot for one class."""

def plot_3way_comparison(metrics_csv_path: str, save_path: str = None):
    """Bar chart comparing ML / CNN / Hybrid across macro_f1, scratch, donut."""

def plot_gradcam(model, input_tensor, target_class: int,
                 class_name: str, save_path: str = None):
    """Grad-CAM heatmap using manual backward hook (no external library)."""
```

**Verification**:
```powershell
conda activate wm811k; python -c "
from src.visualize import plot_confusion_matrix, plot_shap_summary, plot_3way_comparison, plot_gradcam
print('All visualize functions importable')
"
```

---

## Task 3 — CNN 8클래스 재학습 (C-4 수정)
**Branch**: `agent/phase3-cnn-retrain`

**문제**: 현재 cnn_weights.pth가 9클래스(none 포함)로 학습됨.
Near-full → Normal 병합 + none 클래스 제거 후 8클래스로 재학습.

**Steps**:
1. LSWMD.pkl 로드 후 아래 전처리 적용
   ```python
   df['failureType'] = df['failureType'].replace({'Near-full': 'Normal', 'none': None})
   df = df[df['failureType'].notna() & (df['failureType'] != '')]
   # 결과: 8클래스 172950장 (Near-full 149장 → Normal, none 제거)
   ```
2. `src/train.py`의 `train_cnn(cfg, ...)` 사용 (Task 1 완료 후)
3. 학습 완료 후:
   - `results/cnn_weights.pth` 덮어쓰기 (8클래스)
   - `data/cnn_embeddings.npy` 재생성 (172950 × 128)
4. Hybrid도 새 embedding으로 재학습 (macro_f1 변동 가능)

**Verification**:
```powershell
conda activate wm811k; python -c "
import torch
from src.models import WaferCNN
model = WaferCNN(num_classes=8)
state = torch.load('results/cnn_weights.pth', map_location='cpu')
model.load_state_dict(state)
print('CNN loaded successfully — num_classes=8')
"
```

**Completion criteria**:
- cnn_weights.pth가 8클래스 출력층으로 로드 가능
- cnn_embeddings.npy shape: (172950, 128)
- experiment_log에 새 macro_f1 기록
- Gate 2 재확인 (Scratch/Donut recall 재계산)

---

## Task 4 — run_pipeline.py 스테이지 배선
**Branch**: `agent/phase3-pipeline`

**의존**: Task 1, 2, 3 완료 후 실행.

현재 run_pipeline.py는 status 출력만 함. 각 stage 블록에 실제 함수를 연결한다.

```python
from src.features import extract_features
from src.train import train_binary, train_multiclass_ml, train_cnn, train_hybrid
from src.evaluate import check_gate2_binary, check_gate2_full, check_gate3
from src.visualize import plot_confusion_matrix, plot_3way_comparison

def run_binary(cfg):
    # 1. data load
    # 2. train_binary()
    # 3. check_gate2_binary()
    # 4. plot_confusion_matrix()

def run_multiclass(cfg):
    # 1. data load
    # 2. train_multiclass_ml() + train_cnn() + train_hybrid()
    # 3. check_gate2_full() + check_gate3()
    # 4. plot_3way_comparison()
```

**Verification**:
```powershell
conda activate wm811k; python run_pipeline.py --stage binary --config config.yaml
```

**Completion criteria**:
- `--stage binary` 실행 시 실제 학습 + Gate 체크 수행
- `--stage multiclass` 실행 시 3방향 비교 수행
- MLflow에 모든 실행 기록

---

## Task 5 — End-to-End 검증
**Branch**: `agent/phase3-e2e`

```powershell
conda activate wm811k; python run_pipeline.py --stage all --config config.yaml
```

- 에러 없이 실행 완료
- Gate 2, 3 PASSED 재확인
- results/metrics.csv 갱신
- CI lint_test.yml 통과

---

## Phase 3 Complete Criteria

- [ ] 06_domain_interpretation.ipynb 완성
- [ ] src/train.py: 4개 함수 구현 완료
- [ ] src/visualize.py: 4개 함수 추가
- [ ] CNN 8클래스 재학습 완료 (cnn_weights.pth 교체)
- [ ] run_pipeline.py --stage all 에러 없이 실행
- [ ] Gate 2 + Gate 3 재확인 PASSED
- [ ] CI lint_test.yml 13+ tests passed
- [ ] dev → main merge + v3.0-phase3 태그

완료 후 → plans/phase4_plan.md 작성 → Phase 4 진입
