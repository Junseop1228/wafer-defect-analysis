# GEMINI.md — Antigravity-Specific Overrides
# Antigravity reads this FIRST. Overrides AGENTS.md where conflicts exist.
# Last updated: 2026-05-26

---

## Two-Tool Workflow — Antigravity의 역할

```
Claude.ai가 만든 plans/ 파일을 읽고
→ 코드 작성 / 파일 수정 / PowerShell 실행
→ 결과를 context/experiment_log.md에 기록
→ Task SUCCESS 시 Git Sync 자동 실행 (dev merge + push)
→ 완료 후 정지 (Gate 해석/다음 Stage 결정은 Claude.ai 담당)
```

Antigravity는 실행 전문가다. 설계 결정은 하지 않는다.
설계가 불명확하면 → 실행하지 말고 Claude.ai에 에스컬레이션 요청을 로그에 남긴다.

---

## 세션 시작 프로토콜 (매번 필수)

```
Step 1. context/stage_status.md 읽기
Step 2. plans/phase[N]_plan.md 읽기 (stage_status에서 활성 Phase 확인)
Step 3. 현재 Task 번호 확인
Step 4. 해당 Task의 skills/ 파일 읽기
Step 5. dev 기준으로 agent/[task-name] 브랜치 생성 후 작업 시작
```

위 순서를 건너뛰고 코딩 시작 금지.

---

## 실행 환경

모든 터미널 명령은 반드시 이 패턴:
```powershell
conda activate wm811k; [명령어]
```

예시:
```powershell
conda activate wm811k; python src/train.py --config config.yaml --stage binary
conda activate wm811k; python src/evaluate.py --gate 2
conda activate wm811k; jupyter nbconvert --to notebook --execute notebooks/01_EDA.ipynb
```

`wm811k` 활성화 없이 Python 명령 실행 절대 금지.

---

## 모델 선택 전략 (크레딧 절약)

| 작업 유형 | 사용 모델 | 이유 |
|----------|----------|------|
| 보일러플레이트 / 단순 편집 | Gemini 3.5 Flash | 5시간 리셋, 크레딧 소모 없음 |
| 일반 src/ 구현 | Gemini 3.1 Pro (High) | 1M 컨텍스트, 레포 전체 로드 가능 |
| 복잡한 디버깅 / Gate 사이클 | Claude Sonnet 4.6 | 코드베이스 이해 정확도 최고 |
| 설계 결정 필요 | Claude.ai 채팅으로 에스컬레이션 | Antigravity에서 결정 금지 |

**핵심 원칙**: Flash를 기본값으로. Pro/Claude는 진짜 필요할 때만.
보일러플레이트에 Sonnet/Opus 사용 시 크레딧 낭비로 필요한 순간에 잠김.

---

## 브랜치 규칙

작업 시작 전 반드시 dev 기준으로 agent/ 브랜치 생성:
```powershell
git checkout dev
git pull origin dev
git checkout -b agent/[task-name]
```

브랜치 네이밍 예시:
- `agent/stage2-eda`
- `agent/stage3-features-core`
- `agent/gate2-recall-fix`
- `agent/stage41-binary`

**머지 규칙 — 자동 실행**:

| 이벤트 | 동작 |
|--------|------|
| Task SUCCESS (검증 통과) | agent/* → dev squash merge 자동 실행 |
| Gate PASSED (Claude.ai 판단 후 사람이 트리거) | dev → main merge + 태그 |
| Task FAILED (2회 재시도 후) | agent/* 그대로 정지. 머지 절대 금지 |

---

## Task 실행 프로토콜

```
1. plans/phase[N]_plan.md에서 현재 Task 확인
2. Task에 명시된 skills/ 파일 읽기
3. dev 기준으로 agent/[task-name] 브랜치 생성
4. Task 스펙대로만 구현 (범위 확장 금지)
5. Task의 검증 명령어 실행
6. 검증 통과 → context/experiment_log.md append → Git Sync 프로토콜 실행
7. 검증 실패 → 최대 2회 재시도, 그래도 실패 → 로그에 에러 기록 후 정지 (머지 금지)
8. 완료 → 정지 (다음 Task는 사람이 지시할 때까지 대기)
```

---

## Git Sync 프로토콜 (Task SUCCESS 후 자동 실행)

**Task 검증이 SUCCESS일 때만 실행. FAILED 시 절대 실행 금지.**

```powershell
# Step 1. 현재 agent/ 브랜치에서 전체 변경사항 commit
git add -A
git commit -m "feat: [task-name] — [한 줄 요약]"
git push origin agent/[task-name]

# Step 2. dev로 squash merge
git checkout dev
git merge --squash agent/[task-name]
git commit -m "feat: [task-name] — [한 줄 요약]"
git push origin dev

# Step 3. 작업 브랜치로 복귀 (정지 대기)
git checkout agent/[task-name]
```

commit 메시지 형식:
- `feat:` 새 기능/구현
- `fix:` 버그 수정 / Gate 사이클 재실행
- `docs:` context/, plans/, README 업데이트
- `refactor:` 리팩토링 (기능 변화 없음)

**dev → main 머지**: Gate PASSED 후 사람이 트리거. 명령:
```powershell
git checkout main
git merge dev
git tag v[N].[M]-gate[K]
git push origin main --tags
git checkout dev
```

---

## experiment_log.md 업데이트 (실행 후 필수)

모든 실행 후 `context/experiment_log.md` 상단에 append:

```markdown
## [YYYY-MM-DD HH:MM] Stage X — [task 이름]
- Branch: agent/[name]
- Command: `conda activate wm811k; python ...`
- Result: SUCCESS / FAILED
- Metrics:
  - macro_f1: 0.XX
  - scratch_recall: 0.XX
  - donut_recall: 0.XX
  - binary_defect_recall: 0.XX
- Gate: Gate N — PASSED / FAILED / N/A
- MLflow run_id: [id]
- Error (if any): [에러 메시지 전문]
- Notes: [특이사항]
```

---

## 스코프 경계 — 절대 금지 목록

plans/에 명시되지 않은 아래 행동은 절대 하지 않는다:
- plans/에 없는 파일 리팩토링
- Task 스펙을 넘는 기능 추가
- config.yaml 값 변경
- 새 패키지 설치 (pip install 포함)
- .gitignore, README.md 수정
- context/ 파일을 plans/ 지시 없이 임의 수정
- main 브랜치 직접 커밋 (Gate PASSED + 사람 트리거 없이)

---

## Agent Manager 활용 (Antigravity 2.0 병렬 에이전트)

병렬 에이전트 사용 시 역할 분리:
```
Agent 1 (Gemini Pro)   → src/ 구현
Agent 2 (Gemini Flash) → 검증 명령어 실행 + Gate 체크
Agent 3 (Gemini Flash) → context/experiment_log.md 업데이트 + Git Sync
```

동일 파일을 두 에이전트가 동시에 수정하는 것 금지.
특히 experiment_log.md는 Agent 3만 수정.

---

## GitHub Actions 연동

dev 브랜치에 push (Git Sync 완료) 시 자동 트리거:
- `update_metrics.yml` → `results/metrics.csv` 자동 갱신
- `README.md` Key Results 테이블 자동 업데이트

---

## 임시 스크립트 정리 규칙

Task 실행을 위해 root에 생성한 임시 스크립트는 Git Sync 전에 반드시 정리한다.

**이동 대상 (root → scripts/ 폴더로 이동)**:
- un_nb_*.py — 노트북 실행 대체 스크립트
- create_nb_*.py — 노트북 생성 스크립트
- un_gate*.py — Gate 체크 스크립트
- 	hreshold_sweep.py, 	est_import.py 등 1회성 스크립트

**처리 순서**:
1. scripts/ 폴더가 없으면 생성
2. 해당 파일들을 scripts/ 로 이동 (Move-Item)
3. .gitignore에 scripts/ 는 추가하지 않음 (기록 보존 목적)
4. mlflow.db 는 .gitignore에 포함되어 있으므로 그대로 둠

**절대 root에 남겨두지 않는 파일**: *.py (src/, tests/, notebooks/ 외부)

---

## 모델 파일 보호 규칙 — 절대 금지 (2026-05-28 추가)

아래 파일은 사람이 명시적으로 "삭제해줘"라고 말하지 않는 한 절대 삭제/덮어쓰기 금지:
- results/hybrid_model.pkl
- results/cnn_weights.pth
- data/cnn_embeddings.npy

**background task 실행 중에는 위 파일 절대 건드리지 않는다.**
성능 확인이 필요하면 파일 로드 후 평가만 한다. 삭제 후 재학습 금지.
m results/hybrid_model.pkl 또는 --stage validate를 사람 지시 없이 실행하는 것 금지.

---

## experiment_log 기록 누락 금지 (2026-05-28 추가)

모든 python 실행 후 (성공/실패 무관) experiment_log.md 상단에 반드시 기록한다.
기록 없이 다음 Task로 넘어가는 것 금지.
run_pipeline.py 또는 train 관련 스크립트 실행 후에는 metrics 수치를 반드시 포함한다.

---

## git stash 사용 규칙 (2026-05-28 추가)

git stash는 gitignore된 파일을 저장하지 않는다.
results/*.pth, data/*.npy 등 gitignore 파일이 작업공간에 있을 때 브랜치를 바꿔야 하면:
반드시 git stash -u (untracked 포함) 또는 파일을 별도 경로에 수동 백업 후 진행한다.
