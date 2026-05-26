# WM-811K Wafer Defect Classification — Agent Rules
# Read by ALL AI agents (Antigravity, Claude Code, Cursor, etc.)
# Last updated: 2026-05-22

---

## Project Identity

- **Goal**: Classify wafer map defect patterns (7 classes), trace to root-cause process, detect SPC anomalies
- **Target**: Samsung Electronics TSP 평가및분석 인턴 → SK Hynix P&T
- **Stack**: Python 3.10, PyTorch, scikit-learn, SHAP, Optuna, MLflow, Streamlit
- **Conda env**: `wm811k`
- **Root**: `C:\Users\userPC\Desktop\Workspace\01_Projects\WM811K_Portfolio\`
- **GitHub**: https://github.com/Junseop1228/wafer-defect-analysis

---

## Two-Tool Workflow Overview

This project runs on a two-tool split:

```
Claude.ai (claude.ai)              Antigravity (local IDE)
──────────────────────             ──────────────────────
역할: 설계 / 알고리즘 / 해석        역할: 파일수정 / 코드실행 / 디버깅
접근: Windows MCP로 로컬파일 읽기   접근: 파일시스템 직접 접근
입력: 자연어 지시만                  입력: plans/ 파일 읽고 실행
출력: plans/ 파일 + context/ 업데이트  출력: context/experiment_log.md 업데이트
```

**Bridge files**: `context/` 디렉토리가 두 도구 사이의 유일한 통신 채널

---

## Directory Map

```
WM811K_Portfolio/
├── AGENTS.md                  ← 모든 에이전트 공통 규칙 (지금 이 파일)
├── GEMINI.md                  ← Antigravity 전용 오버라이드
├── config.yaml                ← 모든 하이퍼파라미터 (절대 하드코딩 금지)
├── run_pipeline.py            ← 단일 진입점
├── context/                   ← Claude.ai ↔ Antigravity 브리지
│   ├── stage_status.md        ← 현재 Phase/Stage/Gate 상태 ★ 항상 먼저 읽기
│   ├── experiment_log.md      ← 실험 결과 누적 (Antigravity가 append)
│   ├── decisions.md           ← 설계 결정 + 면접 talking point
│   └── handoff.md             ← 세션 인계 메모 (gitignore됨, 로컬전용)
├── skills/                    ← 반복 작업 표준 (Antigravity가 읽음)
│   ├── feature_engineering.md ← 28개 피처 구현 스펙
│   ├── mlflow_logging.md      ← MLflow 로깅 표준
│   ├── gate_check.md          ← Gate 함수 구현 스펙
│   └── spc_pattern.md         ← SPC 3종 + AE 구조
├── plans/                     ← Claude.ai 생성 → Antigravity 실행
│   ├── phase1_plan.md         ← EDA + 피처 엔지니어링
│   ├── phase2_plan.md         ← 모델링 (생성 예정)
│   ├── phase3_plan.md         ← 모듈화 + 자동화 (생성 예정)
│   └── phase4_plan.md         ← SPC + 대시보드 (생성 예정)
├── src/                       ← 모든 구현 모듈
│   ├── features.py            ← 28개 피처 추출
│   ├── models.py              ← ML / CNN / Hybrid 모델 정의
│   ├── train.py               ← 학습 루프
│   ├── evaluate.py            ← Gate 체크 함수
│   ├── spc.py                 ← SPC 3종 관리도
│   ├── autoencoder.py         ← 이상 감지 AE
│   ├── pseudo_label.py        ← Pseudo-labeling
│   └── visualize.py           ← 시각화 함수
├── notebooks/                 ← 프로토타입 (src/ import해서 사용)
├── app/streamlit_app.py       ← 대시보드 (3탭)
├── results/                   ← figures/ + metrics.csv
└── .github/workflows/         ← GitHub Actions CI/CD
    ├── gate_check.yml         ← src/ 변경시 Gate 자동 검증
    └── update_metrics.yml     ← experiment_log 변경시 metrics.csv 자동갱신
```

---

## Full Development Workflow

### PHASE A — 세션 시작 (Claude.ai 담당)

Claude.ai가 Windows MCP로 아래 3개 파일을 자동으로 읽는다:
1. `context/stage_status.md` — 현재 Phase, Stage, Gate 상태
2. `context/experiment_log.md` — 최근 실험 결과
3. `context/handoff.md` — 이전 세션 인계 사항

트리거 문구: **"세션 시작. 현재 상태 읽어줘"**
→ Claude.ai가 파일 3개 읽고 오늘 할 Task 제시. 붙여넣기 불필요.

### PHASE B — 설계 및 계획 수립 (Claude.ai 담당)

Claude.ai가 담당하는 작업:
- 알고리즘 방향 결정 + 도메인 해석
- `plans/phase[N]_plan.md` 생성 또는 업데이트
- `context/decisions.md` 업데이트 (설계 결정 기록)
- Gate 통과 여부 판단 + 다음 사이클 결정

plans/ 파일 형식:
```markdown
### Task N — [task name]
- 목적: ...
- 구현 파일: src/X.py
- 모델 추천: Gemini Flash / Gemini Pro / Claude Sonnet
- 구현 스펙: ...
- 검증 명령어: `conda activate wm811k; python ...`
- 완료 조건: ...
```

### PHASE C — 코드 실행 (Antigravity 담당)

Antigravity가 담당하는 작업:
1. `context/stage_status.md` 읽기
2. `plans/phase[N]_plan.md` 읽고 현재 Task 확인
3. 해당 `skills/*.md` 읽기 (feature_engineering, mlflow_logging 등)
4. `agent/[task-name]` 브랜치 생성
5. 코드 작성 / 파일 수정
6. 검증 명령어 실행
7. `context/experiment_log.md` 에 결과 append
8. 작업 완료 후 정지 (머지는 사람이 함)

### PHASE D — 결과 해석 (Claude.ai 담당)

트리거 문구: **"Gate [N] 판단해줘"** 또는 **"결과 해석해줘"**
→ Claude.ai가 `context/experiment_log.md` Windows MCP로 읽기
→ 통계 해석 + 도메인 해석 + Gate 판단 + 다음 액션 제시

### PHASE E — 세션 종료 (Claude.ai 담당)

트리거 문구: **"세션 마무리해줘"**
→ Claude.ai가 Windows MCP로 아래 파일 직접 업데이트:
- `context/stage_status.md` — 완료된 Task, Gate 상태 갱신
- `context/handoff.md` — 다음 세션 시작점 기록

### PHASE F — Git 동기화 (자동)

`context/experiment_log.md` push →
- `update_metrics.yml` 자동 트리거
- `results/metrics.csv` 자동 갱신
- `README.md` Key Results 테이블 자동 업데이트

---

## Coding Rules — Non-Negotiable

**Config**
- NEVER hardcode hyperparameters — always `yaml.safe_load("config.yaml")`
- Gate thresholds, model params, SPC params — all from `config.yaml`

**Module structure**
- All logic in `src/` — notebooks only do `from src.X import Y`
- Gate checks → `src/evaluate.py`, return `{"passed": bool, "details": dict}`
- Visualizations → `src/visualize.py`

**MLflow — mandatory on every training run**
```python
with mlflow.start_run(run_name="stage{N}_{model}"):
    mlflow.log_params(cfg["..."])
    mlflow.log_metrics({"macro_f1": ..., "scratch_recall": ..., "donut_recall": ...})
    mlflow.log_artifact("results/figures/confusion_matrix.png")
```

**Output**: minimize print(). MLflow or Streamlit only.
**Comments**: English, concise, key logic only.

---

## Gate Thresholds

| Gate | Condition | Threshold | Fail → Cycle |
|------|-----------|-----------|-------------|
| Gate 1 | SHAP contribution | ≥ 0.01 | Cycle ① |
| Gate 1 | Feature correlation | ≤ 0.90 | Cycle ① |
| Gate 2 | Binary defect recall | ≥ 0.90 | Cycle ③ |
| Gate 2 | Scratch recall | ≥ 0.70 | Cycle ③ |
| Gate 2 | Donut recall | ≥ 0.75 | Cycle ③ |
| Gate 3 | Confusion pair F1 gap | ≤ 0.20 | Cycle ② |
| Gate 4 | SPC ARL | ≥ 370 | Cycle ⑤ |

Gate 실패 시: `context/experiment_log.md`에 실패 상세 기록 후 정지. 다음 Stage 진행 금지.

---

## Branch Strategy

→ **See GIT_AND_DIRECTORY.md** for full git strategy, directory rules, and CI/CD trigger map.

---

## What NOT To Do

- config.yaml 값 직접 변경 금지 — Claude.ai 채팅에서 먼저 협의
- data/, mlruns/, results/ 파일 삭제 금지
- main 브랜치 직접 push 금지
- conda install 사용 금지 — pip install만 (wm811k env 안에서)
- plans/에 명시되지 않은 작업 범위 확장 금지
- context/ 파일을 plans/ 지시 없이 임의로 수정 금지

