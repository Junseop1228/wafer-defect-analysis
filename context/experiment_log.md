# Experiment Log — WM-811K Pipeline

> Antigravity appends to this file after every execution.
> Claude.ai reads this for result interpretation and Gate decisions.
> Format: newest entry at the TOP.

---

## [2026-05-22 17:24] Stage 3 — Feature Engineering Core (Priority 1)
- Branch: agent/stage3-features-core
- Command: `conda activate wm811k; python -c "import numpy as np; from src.features import extract_features; test_map = np.random.choice([0,1,2], size=(64,64), p=[0.3,0.5,0.2]); feats = extract_features(test_map); print(f'Feature count: {len(feats)}'); print(feats)"`
- Result: SUCCESS
- Metrics:
  - macro_f1: N/A
  - scratch_recall: N/A
  - donut_recall: N/A
  - binary_defect_recall: N/A
- Gate: N/A
- MLflow run_id: N/A
- Notes: Implemented 18 Priority 1 features (defect_ratio, radial_cdf, edge_sector_std, cx, cy, compactness, aspect_ratio, morans_i, n_clusters). Verification returns exactly 18 features successfully.

## [2026-05-22 17:27] Stage 3 — Feature Engineering Extended (Priority 2)
- Branch: agent/stage3-features-extended
- Command: `conda activate wm811k; python -c "from src.features import extract_features; import numpy as np; test_map = np.random.choice([0,1,2], size=(64,64), p=[0.3,0.5,0.2]); feats = extract_features(test_map); assert len(feats) == 28, f'Expected 28, got {len(feats)}'; print('All 28 features verified.')"`
- Result: SUCCESS
- Metrics:
  - macro_f1: N/A
  - scratch_recall: N/A
  - donut_recall: N/A
  - binary_defect_recall: N/A
- Gate: N/A
- MLflow run_id: N/A
- Notes: Implemented calc_edge_sectors, calc_lot_consistency, and calc_radon. extract_features now successfully returns all 28 features.

## [2026-05-26 16:00] Stage 3 — Feature Matrix Construction (Task 5)
- Branch: agent/stage3-feature-matrix
- Command: `conda activate wm811k; jupyter nbconvert --to notebook --execute notebooks/02_feature_engineering.ipynb --ExecutePreprocessor.timeout=-1`
- Result: SUCCESS
- Metrics:
  - macro_f1: N/A
  - scratch_recall: N/A
  - donut_recall: N/A
  - binary_defect_recall: N/A
- Gate: N/A
- MLflow run_id: N/A
- Notes: Replaced radon with PCA linearity for speed. Used joblib Parallel(n_jobs=-1) with loky backend and early returns for None class (0 defects) to optimize processing. Generated data/features_labeled.pkl with shape (172950, 29) and 0 nulls. Saved feature correlation heatmap.

<!-- APPEND NEW ENTRIES ABOVE THIS LINE -->

---

## Log Template (copy for each entry)

```
## [YYYY-MM-DD HH:MM] Stage X — [description]
- Command: `conda activate wm811k; python ...`
- Result: SUCCESS / FAILED
- Metrics:
  - macro_f1: 0.XX
  - scratch_recall: 0.XX
  - donut_recall: 0.XX
  - binary_defect_recall: 0.XX
- Gate: Gate N — PASSED / FAILED / N/A
- MLflow run_id: [id]
- Notes: ...
```
