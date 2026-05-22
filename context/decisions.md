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
