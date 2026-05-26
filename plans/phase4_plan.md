# Phase 4 Execution Plan — Pseudo-label + AE + SPC + Streamlit

> Claude.ai generated this plan. Antigravity executes Tasks in order.
> Read context/stage_status.md and context/handoff.md before starting.

---

## Phase 4 Scope

Phase 3까지는 레이블 있는 172,950장만 사용한 분류 파이프라인이었다.
Phase 4는 레이블 없는 638,507장을 활용해 두 가지를 추가한다.

1. **Pseudo-labeling**: Hybrid 모델로 고신뢰도 예측 → 데이터 확장 → Scratch/Donut 재검증
2. **SPC + Autoencoder**: lotName 기준 시계열 공정 이상 사전 감지
3. **Streamlit 대시보드**: 3탭 Cloud 배포

---

## Input

- data/LSWMD.pkl (전체 811,457장 — 레이블 있음 172,950 + 없음 638,507)
- data/features_labeled_v2.pkl (레이블 있는 피처 행렬)
- results/hybrid_model.pkl (Phase 3 학습 완료 Hybrid XGBoost)
- results/cnn_weights.pth (8클래스 CNN)
- data/cnn_embeddings.npy (172,950 × 128)
- config.yaml

---

## ⚠️ RAM 주의사항

로컬 환경: 7.8GB RAM (현재 6.3GB 사용 중)
638,507장 전체를 한 번에 메모리에 올리면 OOM 위험.
모든 대용량 처리는 **chunk 단위(기본 10,000장)** 로 수행할 것.

---

## Task 1 — src/pseudo_label.py 구현
**Branch**: `agent/phase4-pseudo`

### 목적
레이블 없는 638,507장에 Hybrid 모델로 예측 → confidence ≥ 0.95인 샘플만 pseudo label 부여.
Scratch/Donut 샘플 확보 → features_pseudo.pkl 생성 → Gate 2 재검증.

### 구현할 함수

```python
def generate_pseudo_labels(cfg: dict) -> dict:
    """
    1. LSWMD.pkl에서 레이블 없는 샘플 추출 (failureType == '' or [])
    2. src/features.py extract_features()로 피처 추출 (chunk 10000장씩)
    3. results/hybrid_model.pkl 로드
    4. predict_proba → confidence = max(proba) ≥ 0.95 필터링
    5. data/features_pseudo.pkl 저장
    Returns: {'n_total': int, 'n_accepted': int, 'class_dist': dict}
    """

def merge_pseudo_with_labeled(cfg: dict) -> str:
    """
    features_labeled_v2.pkl + features_pseudo.pkl 합치기
    → data/features_augmented.pkl 저장
    Returns: save_path
    """
```

### 설계 원칙
- chunk 처리: `pd.read_pickle` 후 배열 슬라이싱으로 10,000장씩
- confidence 기준: cfg['cycles']['pseudo_label_confidence'] = 0.95
- Near-full → Normal 병합: extract_features 전에 적용
- MLflow 로깅: n_accepted, class_dist 기록

### Verification
```powershell
conda activate wm811k; python -c "
from src.pseudo_label import generate_pseudo_labels, merge_pseudo_with_labeled
print('pseudo_label functions importable')
"
```

### Completion criteria
- data/features_pseudo.pkl 생성 (최소 1,000장 이상 pseudo label 확보)
- Scratch pseudo label 최소 50장 이상 확보 (현재 150장의 33%)
- experiment_log에 class_dist 기록

---

## Task 2 — src/autoencoder.py 구현 + 정상 웨이퍼 학습
**Branch**: `agent/phase4-autoencoder`

### 목적
레이블 없는 정상 웨이퍼(none 클래스)로 AE 학습.
학습된 AE의 reconstruction error = 이상 점수.
정상 웨이퍼에서는 error 낮음, 결함 패턴에서는 error 높음.

### AE 아키텍처

```
Encoder: Conv(1→16) → Conv(16→32) → Conv(32→64) → Flatten → FC(→128)
Decoder: FC(128→) → Reshape → ConvT(64→32) → ConvT(32→16) → ConvT(16→1)
입력/출력: 64×64×1 (이진 마스크: 결함=1, 정상=0, 외부=0)
```

### 구현할 함수

```python
class WaferAutoencoder(nn.Module):
    """Convolutional AE for wafer map anomaly detection."""

def train_autoencoder(cfg: dict) -> dict:
    """
    1. LSWMD.pkl에서 failureType=='' (레이블 없는) 샘플 추출
    2. waferMap을 이진 마스크로 변환 (결함 다이=1, 나머지=0)
    3. WaferAutoencoder 학습 (MSE loss, 50 epochs)
    4. results/ae_weights.pth 저장
    Returns: {'model': ae, 'mean_recon_error': float, 'threshold': float}
    """

def compute_anomaly_scores(cfg: dict, wafer_maps: np.ndarray) -> np.ndarray:
    """
    학습된 AE로 각 웨이퍼의 reconstruction error 계산.
    Returns: anomaly_scores shape (N,)
    """
```

### AE threshold 설정
정상 웨이퍼 reconstruction error의 95 percentile → threshold.
이 값을 초과하면 이상 판정.

### Verification
```powershell
conda activate wm811k; python -c "
from src.autoencoder import WaferAutoencoder, train_autoencoder, compute_anomaly_scores
print('autoencoder functions importable')
"
```

### Completion criteria
- results/ae_weights.pth 생성
- 레이블 있는 결함 웨이퍼 anomaly score가 정상 웨이퍼보다 유의미하게 높음 확인
- experiment_log에 mean_recon_error, threshold 기록

---

## Task 3 — src/spc.py 구현 (3종 관리도 + ARL)
**Branch**: `agent/phase4-spc`

### 목적
lotName 기준 시계열로 공정 이상을 사전 감지.
3가지 관리도 + AE 이상 점수를 조합해 이중 감지.

### SPC 구조

**Phase 1 (기준 구간 설정)**:
레이블 있는 데이터에서 None 비율 상위 30% 로트 → 안정 공정 기준

**3종 관리도**:
- Shewhart p-chart: 급격한 스파이크 감지
- CUSUM (k=0.5, h=4.0): 점진적 drift 감지 (현업 주력)
- EWMA (λ=0.2): 완만한 추세 감지

### 구현할 함수

```python
def setup_phase1_baseline(cfg: dict, df_labeled: pd.DataFrame) -> dict:
    """
    lotName 기준 None 비율 계산 → 상위 30% 로트 추출 → 기준 μ, σ 계산
    Returns: {'mean': float, 'std': float, 'ucl': float, 'lcl': float}
    """

def run_shewhart(defect_rates: np.ndarray, baseline: dict) -> dict:
    """Shewhart p-chart. Returns violations index array."""

def run_cusum(defect_rates: np.ndarray, baseline: dict, cfg: dict) -> dict:
    """CUSUM chart. Returns C_plus, C_minus arrays + violation indices."""

def run_ewma(defect_rates: np.ndarray, baseline: dict, cfg: dict) -> dict:
    """EWMA chart. Returns ewma array + violation indices."""

def compute_arl(violations: np.ndarray, n_total: int) -> float:
    """
    ARL = 1 / (violation_rate).
    정상 상태에서 ARL ≥ 370 목표 (Shewhart 3σ 기준과 동일).
    """

def run_dual_detection(cfg: dict, defect_rates: np.ndarray,
                       anomaly_scores: np.ndarray, baseline: dict) -> dict:
    """
    SPC 신호 + AE 신호 동시 초과 시 최고 신뢰도 이상 판정.
    Returns: {'alerts': list, 'arl': float}
    """
```

### Verification
```powershell
conda activate wm811k; python -c "
from src.spc import setup_phase1_baseline, run_cusum, run_ewma, compute_arl, run_dual_detection
print('spc functions importable')
"
```

### Completion criteria
- 3종 관리도 정상 동작
- compute_arl 반환값 ≥ 370 → Gate 4 PASSED
- experiment_log에 ARL 기록

---

## Task 4 — Gate 4 검증 (ARL ≥ 370)
**Branch**: `agent/phase4-gate4`

```powershell
conda activate wm811k; python -c "
import yaml, pandas as pd, numpy as np
from src.spc import setup_phase1_baseline, run_cusum, compute_arl
cfg = yaml.safe_load(open('config.yaml', encoding='utf-8'))
df = pd.read_pickle('data/features_labeled_v2.pkl')
baseline = setup_phase1_baseline(cfg, df)
# lotName 기준 defect_rate 시계열 생성
lot_stats = df.groupby('lotName')['failureType'].apply(
    lambda x: (x != 'Normal').mean()
).reset_index()
defect_rates = lot_stats['failureType'].values
cusum_result = run_cusum(defect_rates, baseline, cfg)
arl = compute_arl(cusum_result['violations'], len(defect_rates))
print(f'ARL: {arl:.1f} (threshold: 370)')
from src.evaluate import check_gate4
check_gate4(arl, cfg)
"
```

### check_gate4 구현 (src/evaluate.py에 추가)
```python
def check_gate4(arl: float, cfg: dict) -> dict:
    threshold = cfg['gates']['gate4']['min_arl']
    passed = arl >= threshold
    status = 'PASSED' if passed else 'FAILED'
    print(f'[GATE 4 SPC ARL] {status}: {arl:.1f} (threshold: {threshold})')
    return {'passed': passed, 'details': {'arl': arl, 'threshold': threshold}}
```

### ARL이 370 미달이면
CUSUM h 파라미터 조정 (config.yaml cusum_h 상향) → Cycle ⑤ 발동.
h=4.0이 이론적으로 ARL≈370을 달성하므로 정상 데이터에서는 통과 가능.

---

## Task 5 — Streamlit 3탭 대시보드
**Branch**: `agent/phase4-streamlit`

### 파일 구조
```
app/
├── streamlit_app.py     # 메인 진입점
├── components/
│   ├── tab1_pipeline.py # 파이프라인 현황
│   ├── tab2_model.py    # 모델 분석
│   └── tab3_spc.py      # SPC 모니터링
└── utils.py             # 공통 로드 함수
```

### 탭 1 — 파이프라인 현황
- Gate 상태 테이블 (Gate 1~4)
- Phase별 완료 현황
- 3방향 모델 비교 bar chart (results/metrics.csv)
- MLflow 최신 실험 결과 요약

### 탭 2 — 모델 분석
- 패턴 선택 드롭다운 (8클래스)
- 대표 웨이퍼맵 5장 그리드
- SHAP bar plot (results/figures/shap_*.png 로드)
- Grad-CAM heatmap (results/figures/gradcam_*.png 로드)
- Confusion Matrix (Hybrid 기준)
- 도메인 해석 사이드바 (패턴 → 공정 → 액션)

### 탭 3 — SPC 모니터링
- 패턴 선택 → 해당 패턴 로트별 defect rate 시계열
- 3종 관리도 (Shewhart / CUSUM / EWMA) 동시 표시
- AE anomaly score 오버레이
- UCL 초과 시점 강조 표시
- 관찰 구간 슬라이더

### Streamlit Cloud 배포 주의사항
- data/LSWMD.pkl (2GB), data/cnn_embeddings.npy (84MB) Cloud에 올리면 안 됨
- 대신 results/ 산출물만 사용 (metrics.csv, figures/, model pkl)
- SPC 시각화용 경량 시계열 데이터 별도 생성: data/spc_timeseries.csv (로트별 통계)
- requirements.txt에 streamlit, plotly 추가

### Verification
```powershell
conda activate wm811k; streamlit run app/streamlit_app.py
```

### Completion criteria
- 3탭 에러 없이 로컬 실행
- 모든 그래프 정상 렌더링
- 대용량 파일 미포함 상태로 Cloud 배포 가능

---

## Task 6 — Cloud 배포 + README 마무리
**Branch**: `agent/phase4-deploy`

### Streamlit Cloud 배포
1. GitHub main 브랜치 기준 배포
2. requirements.txt 최종 확인
3. app/streamlit_app.py가 진입점
4. README에 Live Demo 링크 추가

### README 업데이트 항목
- 프로젝트 핵심 메시지 (영/한)
- 파이프라인 다이어그램 (ASCII)
- Gate 결과 테이블 (Gate 1~4 최종 수치)
- 3방향 비교 결과 테이블
- 도메인 해석 테이블 (패턴 → 공정 → 액션)
- Live Demo 링크
- 로컬 실행 방법

### Completion criteria
- Streamlit Cloud 배포 URL 생성
- README Live Demo 링크 정상 작동
- v4.0-phase4 태그 push
- dev → main merge

---

## Phase 4 Complete Criteria

- [ ] src/pseudo_label.py: Scratch pseudo label ≥ 50장 확보
- [ ] src/autoencoder.py: 결함 웨이퍼 anomaly score 유의미하게 높음
- [ ] src/spc.py: 3종 관리도 + ARL 계산 완료
- [ ] Gate 4: ARL ≥ 370 PASSED
- [ ] Gate 2 Scratch 재검증: pseudo-augmented 데이터로 ≥ 0.70 재확인
- [ ] Streamlit 3탭 로컬 실행 완료
- [ ] Streamlit Cloud 배포 완료
- [ ] README Live Demo 링크 추가
- [ ] v4.0-phase4 태그

완료 후 → 포트폴리오 최종 점검 → 자소서 연결 표현 작성
