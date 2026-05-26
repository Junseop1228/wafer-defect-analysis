# Stage Status — WM-811K Pipeline

> Single source of truth for pipeline progress.
> Claude.ai reads this at session start. Antigravity reads this before every task.
> Last Updated: 2026-05-26

---

## Current Status

- **Active Phase**: Phase 3 — Modularization + Automation
- **Active Stage**: Task 4 — run_pipeline.py 스테이지 배선
- **Last Completed**: Task 3 — CNN 8클래스 재학습 + embeddings 재생성 (SUCCESS)
- **Blocking Item**: 없음.

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
| Gate 2 Scratch | scratch_recall (Hybrid, 8-class) | ≥0.70 | ✅ 0.7950 |
| Gate 2 Donut | donut_recall (Hybrid, 8-class) | ≥0.75 | ✅ 0.7838 |
| Gate 3 | worst confusion pair F1 gap | ≤0.20 | ✅ 0.168 — re-verify at Task 5 |
| Gate 4 | SPC ARL | ≥370 | ⏳ Phase 4 |

---

## Phase 3 Task Checklist

| Task | Description | Status |
|------|-------------|--------|
| Task 0 | 06_domain_interpretation.ipynb 작성 | ✅ Completed |
| Task 1 | src/train.py 구현 (4개 함수) | ✅ Completed |
| Task 2 | src/visualize.py 확장 (4개 함수) | ✅ Completed |
| Task 3 | CNN 8클래스 재학습 + embeddings 재생성 | ✅ Completed |
| Task 4 | run_pipeline.py 스테이지 배선 | 🔄 Active |
| Task 5 | End-to-end 실행 검증 | ⏳ |

---

## Phase 3 Final Model (Task 3 기준)

| Model | macro F1 | Scratch recall | Donut recall |
|-------|----------|----------------|--------------|
| ML (XGBoost) | 0.705 | 0.285 | 0.766 |
| CNN (8-class) | 0.757 | 0.774 | 0.838 |
| **Hybrid (8-class, 채택)** | **0.860** | **0.795** | **0.784** |

---

## Known Issues

| # | 문제 | Phase 3 Task |
|---|------|-------------|
| 1 | data/cnn_embeddings.npy .gitignore 추가 완료 | ✅ |
| 2 | src/spc.py, autoencoder.py, pseudo_label.py stub | Phase 4 |
| 3 | Gate 3 공식 재확인 | Task 5 |

---

## Environment

- Conda env: `wm811k` ✅ (Python 3.10)
- PyTorch: CUDA 활성화 ✅ (GTX 1650 Max-Q)
- CNN weights: results/cnn_weights.pth ✅ (8클래스 교체 완료)
- CNN embeddings: data/cnn_embeddings.npy ✅ (172950×128, .gitignore 적용)

---

## Next Action

Antigravity 입력:
> "plans/phase3_plan.md 읽고 Task 4 실행해줘. dev 기준으로 agent/phase3-pipeline 브랜치 만들어서 시작해줘."
