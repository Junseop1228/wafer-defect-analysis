# Stage Status — WM-811K Pipeline

> This file is the single source of truth for pipeline progress.
> Claude.ai reads this at session start. Antigravity reads this before every task.
> Update this file at the END of every session.

---

## Current Status

- **Active Phase**: Phase 1 — Notebook Prototype
- **Active Stage**: Stage 2 — EDA (not started)
- **Last Completed**: Project scaffold created
- **Last Updated**: 2026-05-22

---

## Phase Progress

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | EDA + Feature Engineering prototype | 🔄 In Progress |
| Phase 2 | Modeling notebooks (binary + 7-class + SHAP) | ⏳ Pending |
| Phase 3 | Modularization + automation (src/ + Gates + Optuna + MLflow) | ⏳ Pending |
| Phase 4 | Pseudo-label + AE + SPC + Streamlit + deploy | ⏳ Pending |

---

## Stage Progress

| Stage | Description | Status | Gate |
|-------|-------------|--------|------|
| Stage 2 | EDA (distribution, t-SNE, correlation) | ⏳ Not Started | — |
| Stage 3 | Feature Engineering (28 features) | ⏳ Not Started | Gate 1 |
| Stage 4-1 | Binary classification (Normal vs Defect) | ⏳ Not Started | Gate 2 |
| Stage 4-2 | 7-class pattern classification (ML/CNN/Hybrid) | ⏳ Not Started | Gate 2 |
| Stage 5 | SHAP + Grad-CAM interpretation | ⏳ Not Started | Gate 3 |
| Stage 6 | Domain interpretation (pattern → process → action) | ⏳ Not Started | — |
| Stage 7 | SPC + Autoencoder | ⏳ Not Started | Gate 4 |
| Stage 8 | Streamlit dashboard + Cloud deploy | ⏳ Not Started | — |

---

## Gate Status

| Gate | Condition | Threshold | Result |
|------|-----------|-----------|--------|
| Gate 1 | SHAP contribution + feature correlation | ≥0.01 / ≤0.90 | ⏳ Pending |
| Gate 2 | Binary recall + Scratch/Donut recall | ≥0.90 / ≥0.70 / ≥0.75 | ⏳ Pending |
| Gate 3 | Confusion pair F1 gap | ≤0.20 | ⏳ Pending |
| Gate 4 | SPC ARL | ≥370 | ⏳ Pending |

---

## Environment

- Conda env: `wm811k`
- Python: 3.10
- Data file: `data/LSWMD.pkl` (not yet downloaded)
- MLflow tracking: `mlruns/`

---

## Next Action

1. Download `LSWMD.pkl` to `data/`
2. Create `wm811k` conda environment: `conda create -n wm811k python=3.10`
3. `pip install -r requirements.txt`
4. Open `notebooks/01_EDA.ipynb` and begin Stage 2
