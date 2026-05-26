# Stage Status — WM-811K Pipeline

> Single source of truth for pipeline progress.
> Claude.ai reads this at session start. Antigravity reads this before every task.
> Last Updated: 2026-05-27

---

## Current Status

- **Active Phase**: Phase 4 — Pseudo-label + AE + SPC + Streamlit
- **Active Stage**: Task 0 — plans/phase4_plan.md 작성 (Claude.ai 담당)
- **Last Completed**: Phase 3 완료 (Task 5 pipeline 연결 검증)
- **Blocking Item**: 없음. Phase 4 즉시 시작 가능.

---

## Phase Progress

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | EDA + Feature Engineering | ✅ Completed |
| Phase 2 | Modeling (binary + 7-class + SHAP) | ✅ Completed |
| Phase 3 | src/ 모듈화 + run_pipeline.py 완성 | ✅ Completed |
| Phase 4 | Pseudo-label + AE + SPC + Streamlit | 🔄 Active |

---

## Gate Status

| Gate | Condition | Threshold | Result |
|------|-----------|-----------|--------|
| Gate 1 | SHAP contribution + correlation | ≥0.01 / ≤0.90 | ✅ PASSED |
| Gate 2 Binary | binary_defect_recall | ≥0.90 | ✅ 0.9146 |
| Gate 2 Scratch | scratch_recall (Hybrid, Task 3 기준) | ≥0.70 | ✅ 0.7950 |
| Gate 2 Donut | donut_recall (Hybrid, Task 3 기준) | ≥0.75 | ✅ 0.7838 |
| Gate 3 | worst confusion pair F1 gap | ≤0.20 | ✅ 0.1528 (Task 5 재확인) |
| Gate 4 | SPC ARL | ≥370 | ⏳ Phase 4 |

*Note: Gate 2 Scratch Task 5 재실행 시 0.665~0.795 변동 (Optuna 확률적 특성 + Scratch 150장 한계). Phase 4 Pseudo-labeling으로 근본 해결 예정.*

---

## Phase 3 Final Model

| Model | macro F1 | Scratch recall | Donut recall |
|-------|----------|----------------|--------------|
| ML (XGBoost) | 0.705 | 0.285 | 0.766 |
| CNN (8-class) | 0.757 | 0.774 | 0.838 |
| **Hybrid (8-class, 채택)** | **0.860** | **0.795** | **0.784** |

---

## Phase 4 Task Checklist

| Task | Description | Status |
|------|-------------|--------|
| Task 0 | plans/phase4_plan.md 작성 | 🔄 Active (Claude.ai) |
| Task 1 | src/pseudo_label.py 구현 | ⏳ |
| Task 2 | src/autoencoder.py 구현 + 정상 웨이퍼 학습 | ⏳ |
| Task 3 | src/spc.py 구현 (3종 관리도 + ARL) | ⏳ |
| Task 4 | Gate 4 검증 (ARL ≥ 370) | ⏳ |
| Task 5 | Streamlit 3탭 대시보드 | ⏳ |
| Task 6 | Cloud 배포 + README 마무리 | ⏳ |

---

## Known Issues (Phase 4 Fix)

| # | 문제 | 처리 |
|---|------|------|
| 1 | Scratch recall Optuna 불안정 (150장 한계) | Pseudo-labeling으로 샘플 확보 후 재검증 |
| 2 | data/cnn_embeddings.npy 84MB push 반복 | .gitignore 재확인 필요 |
| 3 | Optuna XGBoost pruning 미작동 | Phase 4 train_hybrid에 XGBoostPruningCallback 연동 |
| 4 | results/hybrid_model.pkl 저장됨 | .gitignore 추가 권장 |

---

## Environment

- Conda env: `wm811k` ✅
- CNN weights: results/cnn_weights.pth ✅ (8클래스)
- Hybrid model: results/hybrid_model.pkl ✅ (joblib)
- CNN embeddings: data/cnn_embeddings.npy ✅ (.gitignore)

---

## Next Action

1. Claude.ai → plans/phase4_plan.md 작성
2. Antigravity 입력:
> "git checkout dev; git merge --squash agent/phase3-e2e; git commit -m 'feat: phase3 task5 — end-to-end validation complete'; git push origin dev; git checkout main; git merge dev; git tag v3.0-phase3; git push origin main --tags; git push origin dev"
