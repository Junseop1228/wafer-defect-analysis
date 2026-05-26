# Stage Status — WM-811K Pipeline

> Single source of truth for pipeline progress.
> Claude.ai reads this at session start. Antigravity reads this before every task.
> Last Updated: 2026-05-26

---

## Current Status

- **Active Phase**: Phase 3 — Modularization + Automation
- **Active Stage**: Task 0 — Domain Interpretation Notebook
- **Last Completed**: Phase 2 완료 + 레포 정리 (12개 문제 처리)
- **Blocking Item**: 없음. Phase 3 즉시 시작 가능.

---

## Phase Progress

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | EDA + Feature Engineering | ✅ Completed |
| Phase 2 | Modeling (binary + 7-class + SHAP) | ✅ Completed |
| Phase 3 | src/ 모듈화 + run_pipeline.py 완성 | 🔄 Active |
| Phase 4 | Pseudo-label + AE + SPC + Streamlit | ⏳ Pending |

---

## Gate Status

| Gate | Condition | Threshold | Result |
|------|-----------|-----------|--------|
| Gate 1 | SHAP contribution + correlation | ≥0.01 / ≤0.90 | ✅ PASSED |
| Gate 2 Binary | binary_defect_recall | ≥0.90 | ✅ 0.9146 |
| Gate 2 Scratch | scratch_recall (Hybrid) | ≥0.70 | ✅ 0.7950 |
| Gate 2 Donut | donut_recall (Hybrid) | ≥0.75 | ✅ 0.7838 |
| Gate 3 | worst confusion pair F1 gap | ≤0.20 | ✅ 0.168 (Loc↔Random) |
| Gate 4 | SPC ARL | ≥370 | ⏳ Phase 4 |

*Note: Gate 2/3 will be re-verified after CNN 8-class retraining (Phase 3 Task 3)*

---

## Phase 3 Task Checklist

| Task | Description | Status |
|------|-------------|--------|
| Task 0 | 06_domain_interpretation.ipynb 작성 | ⏳ |
| Task 1 | src/train.py 구현 (4개 함수) | ⏳ |
| Task 2 | src/visualize.py 확장 (4개 함수) | ⏳ |
| Task 3 | CNN 8클래스 재학습 + embeddings 재생성 | ⏳ |
| Task 4 | run_pipeline.py 스테이지 배선 | ⏳ |
| Task 5 | End-to-end 실행 검증 | ⏳ |

---

## Phase 2 Final Model (Reference)

| Model | macro F1 | Scratch recall | Donut recall |
|-------|----------|----------------|--------------|
| ML (XGBoost) | 0.705 | 0.285 | 0.766 |
| CNN | 0.787 | 0.774 | 0.838 |
| **Hybrid (채택)** | **0.870** | **0.795** | **0.784** |

---

## Known Issues (Being Fixed in Phase 3)

| # | 문제 | Phase 3 Task |
|---|------|-------------|
| 1 | CNN 9클래스로 학습됨 (none 포함) | Task 3 |
| 2 | src/train.py, spc.py, autoencoder.py, pseudo_label.py stub | Task 1 (train), Phase 4 (나머지) |
| 3 | notebooks 04~07 빈 scaffold | Task 0 + Phase 4 |
| 4 | config n_trials=50 (실제 20) | config.yaml 수정 완료 ✅ |

---

## Environment

- Conda env: `wm811k` ✅ (Python 3.10)
- PyTorch: CUDA 활성화 ✅ (GTX 1650 Max-Q)
- Data: data/LSWMD.pkl ✅, data/features_labeled_v2.pkl ✅
- CNN weights: results/cnn_weights.pth ⚠️ (9클래스, Task 3에서 교체)
- CNN embeddings: data/cnn_embeddings.npy ⚠️ (9클래스 기반, Task 3에서 교체)

---

## Next Action

Antigravity 입력:
> "plans/phase3_plan.md 읽고 Task 0 실행해줘. dev 기준으로 agent/task9-domain-interp 브랜치 만들어서 시작해줘."
