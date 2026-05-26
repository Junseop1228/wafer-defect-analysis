# GIT_AND_DIRECTORY.md
# WM-811K 프로젝트 — Git 전략 & 디렉토리 규칙
# Claude.ai + Antigravity 공통 참조 파일
# Last updated: 2026-05-26

---

## 1. 브랜치 구조

```
origin/main   ← 포트폴리오 공개용. Gate PASSED 시점에만 머지.
    ↑
origin/dev    ← 통합 브랜치. Task SUCCESS마다 squash merge.
    ↑
agent/*       ← Antigravity 실행 단위. Task 1개 = 브랜치 1개.
```

### 브랜치별 역할

| 브랜치 | 역할 | 누가 커밋 | 머지 트리거 |
|--------|------|-----------|-------------|
| `main` | 포트폴리오 공개 상태. 항상 클린. | Claude.ai / 사람 | Gate PASSED + 사람 트리거 |
| `dev` | 일상 통합. Phase 진행 상황 반영. | Antigravity (자동) | Task SUCCESS |
| `agent/*` | 실행 단위. 1 Task = 1 브랜치. | Antigravity | — |

### 브랜치 네이밍 규칙

```
agent/stage{N}-{description}     # Stage 단위 Task
agent/gate{N}-{fix-description}  # Gate 실패 → 사이클 재실행
agent/phase{N}-{description}     # Phase 전환 작업

예시:
  agent/stage41-binary
  agent/stage42-cnn
  agent/gate2-smote-fix
  agent/phase2-features-sync
```

---

## 2. Git Sync 프로토콜 (Antigravity 자동 실행)

### 트리거 조건

| 상황 | 동작 |
|------|------|
| Task 검증 SUCCESS | Git Sync 자동 실행 (agent → dev) |
| Task 검증 FAILED | 머지 절대 금지. 로그만 기록 후 정지. |
| Gate PASSED (Claude.ai 판단 후 사람 트리거) | dev → main merge + 태그 |

### Task SUCCESS 시 Git Sync

```powershell
# Step 1. 현재 agent/ 브랜치 commit + push
git add -A
git commit -m "{type}: {branch-name} — {한 줄 요약}"
git push origin agent/{task-name}

# Step 2. dev squash merge
git checkout dev
git merge --squash agent/{task-name}
git commit -m "{type}: {branch-name} — {한 줄 요약}"
git push origin dev

# Step 3. agent/ 브랜치로 복귀 (정지 대기)
git checkout agent/{task-name}
```

### Gate PASSED 시 dev → main (사람이 트리거)

```powershell
git checkout main
git merge dev
git tag v{N}.{M}-gate{K}          # 예: v2.0-phase2, v1.0-gate1
git push origin main --tags
git checkout dev
```

### 태그 네이밍 규칙

| 태그 | 의미 | 예시 |
|------|------|------|
| `v{phase}.0-phase{N}` | Phase 완료 마일스톤 | `v2.0-phase2` |
| `v{N}.{M}-gate{K}` | 특정 Gate 통과 | `v1.0-gate1` |

현재 태그: `v1.0-gate1` (Gate 1 PASSED), `v2.0-phase2` (Phase 2 완료)

---

## 3. Commit 메시지 규칙

```
{type}: {branch-name} — {한 줄 요약}

타입:
  feat     새 기능/구현
  fix      버그 수정 / Gate 사이클 재실행
  docs     context/, plans/, README 업데이트
  refactor 리팩토링 (기능 변화 없음)
  chore    빌드/설정/의존성 변경
  test     테스트 추가/수정

예시:
  feat: stage42-cnn — CNN 7-class training + embedding extraction
  fix: gate2-threshold-sweep — binary recall 0.82→0.91 with proper scale_pos_weight
  docs: update stage_status — Phase 2 complete
```

---

## 4. .gitignore 전략

```gitignore
# 2GB+ 데이터 파일 — LFS 미사용, 로컬 전용
data/*.pkl
data/*.csv
data/*.npy          # CNN embeddings (172950×128, ~178MB)

# ML 실행 결과 — 재현 가능, repo에 불필요
mlruns/
optuna_studies/
mlflow.db           # 로컬 MLflow 트래킹 DB

# 생성 가능한 figures — 재실행으로 복원
results/figures/*.png

# Antigravity 임시 스크립트 — scripts/에 이동 후 보존
# (scripts/ 자체는 gitignore 아님 — 기록 보존 목적)

# 세션 노트 — 로컬 전용
context/handoff.md

# 표준 제외
__pycache__/, *.pyc, .ipynb_checkpoints/
.DS_Store, Thumbs.db, .vscode/, .idea/
```

### 포함시키는 것 (의도적)

| 파일/폴더 | 이유 |
|-----------|------|
| `results/metrics.csv` | 3방향 비교 핵심 수치 — README에서 참조 |
| `context/stage_status.md` | 파이프라인 상태 공유 |
| `context/experiment_log.md` | 실험 히스토리 공유 |
| `context/decisions.md` | 설계 결정 기록 |
| `scripts/*.py` | Antigravity 임시 스크립트 보존 |
| `src/evaluate.py` 등 | 모든 구현 모듈 |

---

## 5. 디렉토리 구조 & 파일 배치 규칙

```
WM811K_Portfolio/
│
├── 📋 프로젝트 루트 (항상 클린 유지)
│   ├── AGENTS.md              공통 규칙 (모든 AI 에이전트)
│   ├── GEMINI.md              Antigravity 전용 오버라이드
│   ├── GIT_AND_DIRECTORY.md   Git + 디렉토리 전략 (이 파일)
│   ├── config.yaml            모든 하이퍼파라미터 단일 관리
│   ├── run_pipeline.py        단일 실행 진입점
│   ├── README.md              포트폴리오 공개용
│   ├── requirements.txt       pip 의존성
│   └── setup_env.sh           환경 설정 스크립트
│
├── 📁 context/                Claude.ai ↔ Antigravity 브리지
│   ├── stage_status.md   ★   현재 Phase/Stage/Gate 상태 (세션 시작 필수)
│   ├── experiment_log.md ★   실험 결과 누적 (newest at TOP)
│   ├── decisions.md           설계 결정 + 면접 talking point
│   └── handoff.md        🔒  세션 인계 메모 (.gitignore)
│
├── 📁 plans/                  Claude.ai 생성 → Antigravity 실행
│   ├── phase1_plan.md    ✅   EDA + 피처 엔지니어링
│   ├── phase2_plan.md    ✅   모델링 (3방향 비교)
│   ├── phase3_plan.md    ⏳   src/ 모듈화 + 자동화
│   └── phase4_plan.md    ⏳   SPC + Streamlit
│
├── 📁 scripts/                Antigravity 임시 스크립트 보관소
│   └── run_nb_*.py 등         Task 실행용 임시 스크립트
│   (주의: src/에 올릴 수준이 아닌 1회성 실행 스크립트만)
│
├── 📁 src/                    모든 구현 모듈 (notebooks에서 import)
│   ├── features.py            피처 추출 (20개)
│   ├── models.py              ML / CNN / Hybrid 정의
│   ├── train.py               학습 루프
│   ├── evaluate.py            Gate 체크 함수
│   ├── spc.py                 SPC 3종 관리도
│   ├── autoencoder.py         이상 감지 AE
│   ├── pseudo_label.py        Pseudo-labeling
│   └── visualize.py           시각화 함수
│
├── 📁 notebooks/              프로토타입 (src/ import 전용)
│   ├── 01_EDA.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_binary_classification.ipynb
│   ├── 04_pattern_classification.ipynb
│   ├── 05_comparison_shap.ipynb
│   ├── 06_domain_interpretation.ipynb
│   └── 07_spc.ipynb
│
├── 📁 tests/                  단위 테스트 (CI 실행)
│   └── test_features.py       features.py + evaluate.py 테스트
│
├── 📁 data/                   .gitignore 전체 제외
│   ├── LSWMD.pkl         🔒  원본 (2GB)
│   ├── features_labeled_v2.pkl 🔒  피처 행렬 (172950×21)
│   └── cnn_embeddings.npy 🔒  CNN 임베딩 (172950×128)
│
├── 📁 results/
│   ├── metrics.csv       ✅  3방향 비교표 (git 포함)
│   └── figures/          🔒  .gitignore (재생성 가능)
│       ├── shap_*.png         SHAP bar plots × 8
│       ├── gradcam_*.png      Grad-CAM × 9
│       └── cm_*.png           Confusion matrices × 3
│
├── 📁 app/
│   └── streamlit_app.py       대시보드 (Phase 4)
│
├── 📁 skills/                 반복 작업 표준 (Antigravity 참조)
│
└── 📁 .github/workflows/
    ├── lint_test.yml          src/ 변경 시 flake8 + pytest
    └── update_metrics.yml     experiment_log 변경 시 metrics 갱신
```

---

## 6. 파일 배치 의사결정 트리

새 파일을 어디에 둘지 모를 때:

```
Q1. 재사용 가능한 함수/클래스인가?
  YES → src/{module}.py
  NO  → Q2

Q2. 노트북 프로토타입인가?
  YES → notebooks/0{N}_{name}.ipynb
  NO  → Q3

Q3. Task 실행용 1회성 스크립트인가?
  YES → scripts/{name}.py   (루트에 두지 말 것)
  NO  → Q4

Q4. 테스트 코드인가?
  YES → tests/test_{module}.py
  NO  → Q5

Q5. AI 에이전트 지시 파일인가?
  YES → plans/phase{N}_plan.md 또는 context/{name}.md
  NO  → 루트 (AGENTS.md, README.md, config.yaml 등)
```

---

## 7. CI/CD 트리거 맵

| 이벤트 | 트리거 워크플로 | 동작 |
|--------|----------------|------|
| `src/**` 또는 `tests/**` push | `lint_test.yml` | flake8 + pytest 13개 |
| `context/experiment_log.md` push (dev/agent/*) | `update_metrics.yml` | metrics.csv + README 갱신 |
| `main` push | (Phase 4) `deploy_streamlit.yml` | Streamlit Cloud 자동 배포 |
| PR to dev/main | `lint_test.yml` | 머지 전 검증 |

### 현재 CI 상태

- `lint_test.yml` — 13 tests, 0 skipped ✅
- `update_metrics.yml` — experiment_log 파싱 → metrics.csv ✅
- `deploy_streamlit.yml` — Phase 4에서 추가 예정 ⏳

### CI에서 할 수 없는 것 (데이터 없음)

실제 Gate 수치 체크 (recall, F1, ARL)는 2GB 데이터가 필요해서 CI 불가.
→ 로컬에서 Antigravity가 실행 → experiment_log 기록 → Claude.ai가 판단.
→ CI는 Gate 함수의 로직 정확성 (단위 테스트)만 검증.

---

## 8. 세션 간 컨텍스트 유지 전략

```
세션 시작
  → Claude.ai: Windows MCP로 context/4개 파일 읽기
  → Antigravity: stage_status.md + plans/phase[N]_plan.md 읽기

세션 중
  → context/experiment_log.md: Antigravity가 실행마다 append
  → context/stage_status.md: Claude.ai가 Gate 판단 후 업데이트

세션 종료
  → Claude.ai: handoff.md 업데이트 (다음 세션 시작점)
  → Antigravity: Git Sync 완료 확인 후 정지
```

### context/ 파일 업데이트 책임

| 파일 | 담당 | 업데이트 시점 |
|------|------|--------------|
| `stage_status.md` | Claude.ai | Gate 판단 후, Phase 전환 시 |
| `experiment_log.md` | Antigravity | 모든 실행 후 (newest at TOP) |
| `decisions.md` | Claude.ai | 설계 결정 시 |
| `handoff.md` | Claude.ai | 세션 종료 시 |
