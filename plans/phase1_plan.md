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

## Task List

### Task 1 — Data Loading and Basic EDA
**File**: `notebooks/01_EDA.ipynb`
**Model**: Gemini 3 Flash (simple implementation)

Implementation steps:
1. Load `data/LSWMD.pkl` with `pd.read_pickle()`
2. Separate labeled (172,950) and unlabeled (638,507) samples
3. Print class distribution for labeled data
4. Plot class distribution bar chart → save to `results/figures/class_distribution.png`
5. Plot 3 sample wafer maps per class → save to `results/figures/sample_wafers.png`

Verification command:
```powershell
conda activate wm811k; jupyter nbconvert --to notebook --execute notebooks/01_EDA.ipynb
```
Completion criteria: No errors, figures saved to results/figures/

---

### Task 2 — t-SNE Visualization
**File**: `notebooks/01_EDA.ipynb` (append)
**Model**: Gemini 3 Flash

Implementation steps:
1. Flatten each wafer map to 1D vector (subsample 500 per class for speed)
2. Run t-SNE (n_components=2, perplexity=30)
3. Plot colored by class label
4. Save to `results/figures/tsne_raw.png`

Verification: Figure exists and shows visible cluster separation for Edge-Ring and Center

---

### Task 3 — Feature Engineering Core (Priority 1)
**File**: `src/features.py`
**Model**: Gemini 3.1 Pro (High) — complex logic
**Reference**: Read `skills/feature_engineering.md` first

Implement these functions in order:
1. `extract_defect_coords(wafer_map)` — returns np.array of (row, col) pairs
2. `calc_defect_ratio(wafer_map)` — Group A
3. `calc_radial_cdf(wafer_map, n_bins=10)` — Group B, returns list of 10 floats
4. `calc_angular_features(wafer_map)` — Group C, returns edge_sector_std only first
5. `calc_geometry(wafer_map)` — Group D, returns cx, cy
6. `calc_morphology(wafer_map)` — Group E, returns compactness, aspect_ratio
7. `calc_morans_i(wafer_map)` — Group F, use lattice approximation
8. `calc_n_clusters(wafer_map)` — Group G, DBSCAN
9. `extract_features(wafer_map, lot_df=None)` — master function, returns dict of all features

Verification command:
```powershell
conda activate wm811k; python -c "
import numpy as np
from src.features import extract_features
test_map = np.random.choice([0,1,2], size=(64,64), p=[0.3,0.5,0.2])
feats = extract_features(test_map)
print(f'Feature count: {len(feats)}')
print(feats)
"
```
Completion criteria: Returns dict with ≥18 keys, no errors, runs under 2 seconds

---

### Task 4 — Feature Engineering Extended (Priority 2)
**File**: `src/features.py` (extend Task 3)
**Model**: Gemini 3.1 Pro (High)

Add to existing functions:
1. `calc_edge_sectors(wafer_map)` — returns edge_sector[0..7] (8 values)
2. `calc_lot_consistency(wafer_map, lot_df)` — Group H
3. `calc_radon(wafer_map)` — Group I

Update `extract_features()` to include all 28 features.

Verification command:
```powershell
conda activate wm811k; python -c "
from src.features import extract_features
import numpy as np
test_map = np.random.choice([0,1,2], size=(64,64), p=[0.3,0.5,0.2])
feats = extract_features(test_map)
assert len(feats) == 28, f'Expected 28, got {len(feats)}'
print('All 28 features verified.')
"
```
Completion criteria: Exactly 28 features, no errors

---

### Task 5 — Feature Matrix Construction
**File**: `notebooks/02_feature_engineering.ipynb`
**Model**: Gemini 3 Flash

1. Apply `extract_features()` to all 172,950 labeled wafers
2. Build DataFrame with 28 feature columns + `failureType` label
3. Save to `data/features_labeled.pkl`
4. Print feature summary statistics (mean, std per feature)
5. Plot correlation heatmap → save to `results/figures/feature_correlation.png`

Verification command:
```powershell
conda activate wm811k; python -c "
import pandas as pd
df = pd.read_pickle('data/features_labeled.pkl')
print(df.shape)
print(df.isnull().sum().sum(), 'nulls')
"
```
Completion criteria: shape=(172950, 29), 0 nulls

---

### Task 6 — Gate 1 Check
**File**: `src/evaluate.py`
**Model**: Gemini 3.1 Pro
**Reference**: Read `skills/gate_check.md` first

1. Implement `check_gate1(shap_values, corr_matrix, cfg)`
2. Run quick XGBoost (100 trees) on features_labeled.pkl to get SHAP values
3. Compute correlation matrix
4. Call `check_gate1()` and print result

Verification command:
```powershell
conda activate wm811k; python -c "
import yaml, pandas as pd
from src.evaluate import check_gate1
cfg = yaml.safe_load(open('config.yaml'))
# Load precomputed shap_values and corr_matrix
result = check_gate1(shap_values, corr_matrix, cfg)
print(result)
"
```
Completion criteria: `result['passed'] == True` OR cycle ① triggered with failure details logged

---

## Phase 1 Complete Criteria

All of the following must be true:
- [ ] `results/figures/class_distribution.png` exists
- [ ] `results/figures/sample_wafers.png` exists
- [ ] `results/figures/tsne_raw.png` exists
- [ ] `results/figures/feature_correlation.png` exists
- [ ] `data/features_labeled.pkl` exists with shape (172950, 29)
- [ ] Gate 1: PASSED (logged in context/experiment_log.md)
- [ ] `src/features.py` has all 28 features implemented and tested

When all complete → update `context/stage_status.md` and proceed to Phase 2.
