# Stage Status ??WM-811K Pipeline

> This file is the single source of truth for pipeline progress.
> Claude.ai reads this at session start. Antigravity reads this before every task.
> Update this file at the END of every session.

---

## Current Status

- **Active Phase**: Phase 2 ??Modeling
- **Active Stage**: Stage 4-1 ??Binary Classification
- **Last Completed**: Stage 3 Feature Engineering & Gate 1 PASSED
- **Last Updated**: 2026-05-26

---

## Phase Progress

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | EDA + Feature Engineering prototype | ??Completed |
| Phase 2 | Modeling notebooks (binary + 7-class + SHAP) | ?봽 In Progress |
| Phase 3 | Modularization + automation (src/ + Gates + Optuna + MLflow) | ??Pending |
| Phase 4 | Pseudo-label + AE + SPC + Streamlit + deploy | ??Pending |

---

## Stage Progress

| Stage | Description | Status | Gate |
|-------|-------------|--------|------|
| Stage 2 | EDA (distribution, t-SNE, correlation) | ??Completed | ??|
| Stage 3 | Feature Engineering (21 features confirmed) | ??Completed | Gate 1 ??|
| Stage 4-1 | Binary classification (Normal vs Defect) | ?봽 In Progress | Gate 2 |
| Stage 4-2 | 7-class pattern classification (ML/CNN/Hybrid) | ??Not Started | Gate 2 |
| Stage 5 | SHAP + Grad-CAM interpretation | ??Not Started | Gate 3 |
| Stage 6 | Domain interpretation (pattern ??process ??action) | ??Not Started | ??|
| Stage 7 | SPC + Autoencoder | ??Not Started | Gate 4 |
| Stage 8 | Streamlit dashboard + Cloud deploy | ??Not Started | ??|

---

## Gate Status

| Gate | Condition | Threshold | Result |
|------|-----------|-----------|--------|
| Gate 1 | SHAP contribution + feature correlation | ??.01 / ??.90 | ??PASSED |
| Gate 2 | Binary recall + Scratch/Donut recall | ??.90 / ??.70 / ??.75 | ??Pending |
| Gate 3 | Confusion pair F1 gap | ??.20 | ??Pending |
| Gate 4 | SPC ARL | ??70 | ??Pending |

---

## Environment

- Conda env: `wm811k` ??(Python 3.10, all packages installed)
- Torch: 2.12.0+cpu ??
- Data file: `data/LSWMD.pkl` ??(2.0 GB, ready)
- Feature data: `data/features_labeled_v2.pkl` ??(172950 횞 22, 21 features)
- MLflow tracking: `mlruns/`

---

## Next Action

Phase 2 ?쒖옉 ??plans/phase2_plan.md 湲곗?

1. **Task 0**: src/features.py ?뺥빀 ?섏젙 (21媛??쇱쿂 drop 諛섏쁺)
2. **Task 1**: notebooks/03_binary_classification.ipynb ?묒꽦 諛??ㅽ뻾
3. Gate 2 binary check ??Claude.ai ?댁꽍

Antigravity ?낅젰: "plans/phase2_plan.md ?쎄퀬 Task 0 ?ㅽ뻾?댁쨾"

