# Phase 1 Execution Plan — EDA + Feature Engineering

> Claude.ai generated this plan. Antigravity executes Tasks in order.
> Read AGENTS.md and context/stage_status.md before starting.
> Do NOT skip tasks or combine tasks.

---

## Phase 1 Scope

- Stage 2: EDA (01_EDA.ipynb)
- Stage 3: Feature Engineering (02_feature_engineering.ipynb → src/features.py)
- Gate 1: Feature validity check

---

## Design Decisions

### Near-full 클래스 처리 (2026-05-26 확정)
- Near-full(149개)은 None에 병합한다.
- 근거: 149개로 단독 학습 불가, 공정 관점에서 특정 패턴이 아닌 심각도 개념
- 적용 시점: Task 6 Gate 1 체크 전 features_labeled.pkl에서 전처리
- 최종 클래스: Normal / Edge-Ring / Edge-Loc / Center / Loc / Scratch / Random / Donut (8 → 7클래스, Near-full → Normal 병합)

---

## Task List

### Task 1 — Data Loading and Basic EDA ✅ DONE
**File**: `notebooks/01_EDA.ipynb`

---

### Task 2 — t-SNE Visualization ✅ DONE
**File**: `notebooks/01_EDA.ipynb` (append)

---

### Task 3 — Feature Engineering Core (Priority 1) ✅ DONE
**File**: `src/features.py`
18 features implemented and verified.

---

### Task 4 — Feature Engineering Extended (Priority 2) ✅ DONE
**File**: `src/features.py`
28 features verified.

---

### Task 5 — Feature Matrix Construction ✅ DONE
**File**: `notebooks/02_feature_engineering.ipynb`
Result: shape=(172950, 29), 0 nulls. Near-full 149개 포함된 상태.

---

### Task 6 — Gate 1 Check
**File**: `src/evaluate.py`
**Model**: Gemini 3.1 Pro (High)
**Reference**: Read `skills/gate_check.md` first

**[IMPORTANT] Near-full 전처리 선행**:
- Gate 1 체크 전에 features_labeled.pkl 로드 후 Near-full → Normal로 병합
- `df['failureType'] = df['failureType'].replace('Near-full', 'Normal')`
- 최종 클래스 수: 8종 (Normal / Edge-Ring / Edge-Loc / Center / Loc / Scratch / Random / Donut)

Implementation steps:
1. Load `data/features_labeled.pkl`, apply Near-full → Normal 병합
2. Implement `check_gate1(shap_values, corr_matrix, cfg)` in `src/evaluate.py`
   - shap_contribution ≥ 0.01 체크 (config.yaml gates.gate1.min_shap_contribution)
   - feature correlation ≤ 0.90 체크 (config.yaml gates.gate1.max_correlation)
   - 반환: `{'passed': bool, 'details': dict}`
3. Run quick XGBoost (n_estimators=100) to get SHAP values
4. Compute correlation matrix
5. Call `check_gate1()`, log result to `context/experiment_log.md`
6. MLflow log: params(n_estimators, gate1 thresholds) + metrics(n_features_passed, n_corr_pairs_failed)

Verification command:
```powershell
conda activate wm811k; python -c "
import yaml, pandas as pd
from src.evaluate import check_gate1
cfg = yaml.safe_load(open('config.yaml'))
result = check_gate1(shap_values, corr_matrix, cfg)
print(result)
"
```
Completion criteria: `result['passed'] == True` OR cycle ① triggered with failure details logged

---

## Phase 1 Complete Criteria

All of the following must be true:
- [x] `results/figures/class_distribution.png` exists
- [x] `results/figures/sample_wafers.png` exists
- [x] `results/figures/tsne_raw.png` exists
- [x] `results/figures/feature_correlation.png` exists
- [x] `data/features_labeled.pkl` exists with shape (172950, 29)
- [ ] Gate 1: PASSED (logged in context/experiment_log.md)
- [x] `src/features.py` has all 28 features implemented and tested

When all complete → update `context/stage_status.md` and proceed to Phase 2.
