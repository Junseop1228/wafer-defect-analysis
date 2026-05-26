# Stage Status — WM-811K Pipeline

> Single source of truth for pipeline progress.
> Claude.ai reads this at session start. Antigravity reads this before every task.
> Last Updated: 2026-05-26

---

## Current Status

- **Active Phase**: Phase 3 — Modularization + Automation
- **Active Stage**: 준비 중 (plans/phase3_plan.md 미작성)
- **Last Completed**: Phase 2 — All Gates PASSED
- **Blocking Item**: plans/phase3_plan.md 작성 필요 (Claude.ai 담당)

---

## Phase Progress

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | EDA + Feature Engineering | ✅ Completed |
| Phase 2 | Modeling notebooks (binary + 7-class + SHAP) | ✅ Completed |
| Phase 3 | src/ 모듈화 + run_pipeline.py + Gate 자동화 + MLflow | 🔄 Next |
| Phase 4 | Pseudo-label + AE + SPC + Streamlit + deploy | ⏳ Pending |

---

## Gate Status (All Phase 1 & 2 Gates)

| Gate | Condition | Threshold | Result |
|------|-----------|-----------|--------|
| Gate 1 | SHAP contribution + correlation | ≥0.01 / ≤0.90 | ✅ PASSED |
| Gate 2 Binary | binary_defect_recall | ≥0.90 | ✅ 0.9146 |
| Gate 2 Scratch | scratch_recall (Hybrid) | ≥0.70 | ✅ 0.7950 |
| Gate 2 Donut | donut_recall (Hybrid) | ≥0.75 | ✅ 0.7838 |
| Gate 3 | worst confusion pair F1 gap | ≤0.20 | ✅ 0.168 (Loc↔Random) |
| Gate 4 | SPC ARL | ≥370 | ⏳ Phase 4 |

---

## Phase 2 Final Model

| Model | macro F1 | Scratch recall | Donut recall |
|-------|----------|----------------|--------------|
| ML (XGBoost) | 0.705 | 0.285 | 0.766 |
| CNN | 0.787 | 0.774 | 0.838 |
| **Hybrid (채택)** | **0.870** | **0.795** | **0.784** |

**Adopted model**: Hybrid (21 manual features + 128 CNN embeddings → XGBoost)

---

## Known Issues → Phase 3 Fix List

1. CNN 클래스 불일치: CNN이 9클래스(none 포함)로 학습됨 → 8클래스로 재학습 필요
2. experiment_log.md append 순서: newest-at-top 규칙이 일부 미준수 → Phase 3에서 정리
3. mlflow.db가 git에 포함됨 → .gitignore에 추가 필요
4. run_nb_*.py 임시 스크립트들 root에 산재 → Phase 3 src/ 이전 후 삭제
5. Optuna study 재현성: n_trials=20 (계획 50) → Phase 3에서 config.yaml 조정

---

## Environment

- Conda env: `wm811k` ✅ (Python 3.10)
- PyTorch: CUDA 활성화 완료 (GTX 1650 Max-Q)
- Data: `data/LSWMD.pkl` ✅, `data/features_labeled_v2.pkl` ✅ (172950×21)
- CNN weights: `results/cnn_weights.pth` ✅
- CNN embeddings: `data/cnn_embeddings.npy` ✅ (172950×128)
- MLflow: `mlruns/`

---

## Next Action

Phase 3 시작 전 처리:
1. dev → main merge + v2.0-phase2 태그
2. plans/phase3_plan.md 작성 (Claude.ai 담당 — 다음 세션)
3. Task 9 (Domain Interpretation notebook) — Phase 3 시작 전 완료 권장

Antigravity 입력:
> "git checkout main; git merge dev; git tag v2.0-phase2; git push origin main --tags; git push origin dev"
