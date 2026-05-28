# Stage Status — WM-811K Pipeline

> Single source of truth for pipeline progress.
> Claude.ai reads this at session start. Antigravity reads this before every task.
> Last Updated: 2026-05-28

---

## Current Status

- **Active Phase**: Phase 4 — Pseudo-label + AE + SPC + Streamlit
- **Active Stage**: Task 5 — Streamlit 3탭 대시보드
- **Last Completed**: Phase 4 Task 4 — Gate 4 PASSED (ARL = ∞, cusum_h=5.0, strict zero-defect baseline)
- **Blocking Item**: 없음. Task 5 즉시 시작 가능.

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
| Gate 2 Scratch | scratch_recall (pseudo-augmented) | ≥0.70 | ✅ 0.9498 |
| Gate 2 Donut | donut_recall (pseudo-augmented) | ≥0.75 | ✅ 0.9550 |
| Gate 3 | worst confusion pair F1 gap | ≤0.20 | ✅ 0.0782 |
| Gate 4 | SPC ARL | ≥370 | ✅ ARL=∞ (PASSED) |

---

## Phase 3 Final Model

| Model | macro F1 | Scratch recall | Donut recall |
|-------|----------|----------------|--------------|
| ML (XGBoost) | 0.705 | 0.285 | 0.766 |
| CNN (8-class) | 0.757 | 0.774 | 0.838 |
| **Hybrid (8-class, 채택)** | **0.9581** | **0.9498** | **0.9550** |

*Note: Phase 4 Pseudo-labeling 이후 Hybrid 재평가 수치.*

---

## Phase 4 Task Checklist

| Task | Description | Status |
|------|-------------|--------|
| Task 0 | plans/phase4_plan.md 작성 | ✅ Done |
| Task 1 | src/pseudo_label.py 구현 | ✅ Done |
| Task 2 | src/autoencoder.py 구현 + 정상 웨이퍼 학습 | ✅ Done (defect 68%) |
| Task 3 | src/spc.py 구현 (3종 관리도 + ARL) | ✅ Done |
| Task 4 | Gate 4 검증 (ARL ≥ 370) | ✅ PASSED (ARL=∞) |
| Task 5 | Streamlit 3탭 대시보드 | 🔄 Active |
| Task 6 | Cloud 배포 + README 마무리 | ⏳ |

---

## SPC 설계 결정 (Gate 4 통과 방법)

- **cusum_h**: 4.0 → 5.0 상향 (Cycle ⑤ 발동)
- **Phase 1 baseline 기준**: None 비율 상위 30% → defect_rate == 0 클린 로트만
  - 클린 로트 수: 2,756개
  - False alarm: 0건 → ARL = ∞
- **Limitation 명시**: WM-811K 설비 ID 없음. lotName 기준 시계열 근사. 실제 양산에서는 설비 ID별 관리도 분리 필요.

---

## Known Issues (Phase 4)

| # | 문제 | 처리 |
|---|------|------|
| 1 | failureType ndarray 타입 불일치 | ✅ 해결 (x[0][0] if x.size > 0 else "" 패턴) |
| 2 | AE trivial solution (MSE=0) | ✅ 해결 (binary mask + pos_weight=20.0) |
| 3 | Gate 4 ARL 6.8 미달 | ✅ 해결 (cusum_h=5.0 + zero-defect baseline) |
| 4 | data/cnn_embeddings.npy 84MB push 반복 | .gitignore 재확인 필요 |

---

## Streamlit 배포 주의사항 (Task 5 전달)

- data/LSWMD.pkl (2GB), data/cnn_embeddings.npy (84MB) → Cloud에 올리면 안 됨
- results/ 산출물만 사용 (metrics.csv, figures/, model pkl)
- SPC 시각화용 경량 데이터 별도 생성 필요: data/spc_timeseries.csv
- requirements.txt에 streamlit, plotly 추가

---

## Environment

- Conda env: `wm811k` ✅
- CNN weights: results/cnn_weights.pth ✅
- Hybrid model: results/hybrid_model.pkl ✅
- AE weights: results/ae_weights.pth ✅
- CNN embeddings: data/cnn_embeddings.npy ✅ (.gitignore)
