# Design Decisions Log — WM-811K Pipeline

> Record WHY, not just WHAT.
> Used for portfolio README, self-introduction, and interview prep.
> Claude.ai writes here when a significant design choice is made.

---

## Decision Template

```
## [YYYY-MM-DD] Decision: [short title]
- **Context**: What situation triggered this decision
- **Options considered**: A / B / C
- **Choice**: A
- **Rationale**: Why A, including domain + technical reasoning
- **Trade-off**: What we gave up
- **Interview talking point**: One-line version for verbal explanation
```

---

## Decisions

### [2026-05-22] Decision: 3-way Model Comparison (ML / CNN / Hybrid)
- **Context**: Need to classify 7 wafer defect patterns
- **Options considered**: ML-only / CNN-only / Hybrid
- **Choice**: All three, compared
- **Rationale**: Manual features encode domain knowledge (radial CDF, Moran's I). CNN learns spatial features automatically. Hybrid combines both. Comparison shows which signal matters more — domain knowledge or learned representation
- **Trade-off**: 3x implementation cost vs single-model approach
- **Interview talking point**: "28개 수동 피처가 CNN 자동 피처보다 얼마나 경쟁력 있는지 직접 검증했습니다"

### [2026-05-22] Decision: Moran's I for Local vs Random separation
- **Context**: Local and Random patterns are hardest confusion pair. Need spatial clustering signal
- **Options considered**: DBSCAN only / Moran's I only / Both
- **Choice**: Moran's I (primary) + n_clusters from DBSCAN (secondary)
- **Rationale**: Moran's I directly measures spatial autocorrelation — Local clusters have high positive I, Random has near-zero I. DBSCAN adds cluster count as confirmatory signal
- **Trade-off**: Moran's I on full grid is O(n²). Using lattice-approximate weight matrix to keep under 1s per wafer
- **Interview talking point**: "공간통계 지표로 Local과 Random을 정량적으로 분리했습니다"

### [2026-05-22] Decision: CUSUM as primary SPC chart
- **Context**: 3 chart types to implement: Shewhart p-chart, CUSUM, EWMA
- **Options considered**: Shewhart only / CUSUM primary / EWMA primary
- **Choice**: CUSUM primary, others supporting
- **Rationale**: Real Fab SPC runs CUSUM as workhorse because it detects sustained drift (e.g. CMP pad wear). Shewhart catches spikes. EWMA bridges both. This mirrors actual production usage
- **Trade-off**: CUSUM requires Phase 1 baseline; approximated with top-30% None-ratio lots
- **Interview talking point**: "현업 SPC 운영 방식 그대로 CUSUM 중심으로 설계했습니다"

---

## Known Issues & Root Cause Log

> 재발 방지 목적. 같은 버그를 다른 Task에서 반복하지 않기 위해 기록.

### [2026-05-28] Bug: failureType ndarray 타입 불일치 (Task 2 autoencoder.py)
- **증상**: `Unlabeled samples: 0` — 레이블 없는 샘플을 하나도 찾지 못해 AE 학습 데이터 0장
- **근본 원인**: LSWMD.pkl의 `failureType` 컬럼이 문자열이 아닌 `np.ndarray`로 저장되어 있음
  - 레이블 없는 샘플: `array([], shape=(0,0))` (빈 배열)
  - 레이블 있는 샘플: `array([['Scratch']])` (2D 배열)
  - autoencoder.py의 `_is_unlabeled`가 `str` / `list` 타입만 체크 → ndarray 전부 필터 미통과
- **수정**: `x[0][0] if x.size > 0 else ""` 패턴으로 통일 (pseudo_label.py와 동일)
- **왜 반복됐나**: Task 1(pseudo_label.py)에서 동일 문제를 이미 해결했으나, Task 2 코드 작성 시 해당 패턴을 참조하지 않고 독립적으로 재작성 → 동일 버그 재발. import 테스트는 통과하지만 실제 데이터 실행 전까지 드러나지 않는 종류의 버그
- **재발 방지**: LSWMD.pkl을 다루는 모든 신규 코드는 반드시 pseudo_label.py의 `failureType` 파싱 패턴을 참조할 것. `x.size > 0` 체크 후 `x[0][0]` 추출이 표준 패턴.
