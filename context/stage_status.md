# Stage Status — WM-811K Pipeline

> This file is the single source of truth for pipeline progress.
> Claude.ai reads this at session start. Antigravity reads this before every task.
> Update this file at the END of every session.

---

## Current Status

- **Active Phase**: Phase 2 — Modeling
- **Active Stage**: Stage 4-1 — Binary Classification
- **Last Completed**: Stage 3 Feature Engineering & Gate 1 Checks
- **Last Updated**: 2026-05-26

---

## Phase Progress

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | EDA + Feature Engineering prototype | ✅ Completed |
| Phase 2 | Modeling notebooks (binary + 7-class + SHAP) | 🔄 In Progress |
| Phase 3 | Modularization + automation (src/ + Gates + Optuna + MLflow) | ⏳ Pending |
| Phase 4 | Pseudo-label + AE + SPC + Streamlit + deploy | ⏳ Pending |

---

## Stage Progress

| Stage | Description | Status | Gate |
|-------|-------------|--------|------|
| Stage 2 | EDA (distribution, t-SNE, correlation) | ✅ Completed | — |
| Stage 3 | Feature Engineering (28 features) | ✅ Completed | Gate 1 |
| Stage 4-1 | Binary classification (Normal vs Defect) | 🔄 In Progress | Gate 2 |
| Stage 4-2 | 7-class pattern classification (ML/CNN/Hybrid) | ⏳ Not Started | Gate 2 |
| Stage 5 | SHAP + Grad-CAM interpretation | ⏳ Not Started | Gate 3 |
| Stage 6 | Domain interpretation (pattern → process → action) | ⏳ Not Started | — |
| Stage 7 | SPC + Autoencoder | ⏳ Not Started | Gate 4 |
| Stage 8 | Streamlit dashboard + Cloud deploy | ⏳ Not Started | — |

---

## Gate Status

| Gate | Condition | Threshold | Result |
|------|-----------|-----------|--------|
| Gate 1 | SHAP contribution + feature correlation | ≥0.01 / ≤0.90 | ✅ PASSED |
| Gate 2 | Binary recall + Scratch/Donut recall | ≥0.90 / ≥0.70 / ≥0.75 | ⏳ Pending |
| Gate 3 | Confusion pair F1 gap | ≤0.20 | ⏳ Pending |
| Gate 4 | SPC ARL | ≥370 | ⏳ Pending |

---

## Environment

- Conda env: `wm811k` ✅ (Python 3.10, all packages installed)
- Torch: 2.12.0+cpu ✅
- Data file: `data/LSWMD.pkl` ✅ (2.0 GB, ready)
- MLflow tracking: `mlruns/`

---

## Next Action

1. Antigravity: "plans/phase1_plan.md 읽고 Task 1 실행해줘"
   - notebooks/01_EDA.ipynb 실행
   - results/figures/class_distribution.png, sample_wafers.png 생성
