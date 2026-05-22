# GEMINI.md — Antigravity-Specific Overrides
# Antigravity reads this FIRST. Overrides AGENTS.md where conflicts exist.
# Last updated: 2026-05-22

---

## Two-Tool Workflow — Antigravity의 역할

```
Claude.ai가 만든 plans/ 파일을 읽고
→ 코드 작성 / 파일 수정 / PowerShell 실행
→ 결과를 context/experiment_log.md에 기록
→ 완료 후 정지 (머지/해석은 Claude.ai + 사람 담당)
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
Step 5. agent/[task-name] 브랜치 생성 후 작업 시작
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

작업 시작 전 반드시 agent/ 브랜치 생성:
```powershell
conda activate wm811k; git checkout -b agent/[task-name]
```

브랜치 네이밍 예시:
- `agent/stage2-eda`
- `agent/stage3-features-core`
- `agent/gate2-recall-fix`
- `agent/stage4-binary-clf`

작업 완료 후 → 정지. 머지하지 않는다. 사람이 리뷰 후 dev로 머지.

---

## Task 실행 프로토콜

```
1. plans/phase[N]_plan.md에서 현재 Task 확인
2. Task에 명시된 skills/ 파일 읽기
3. agent/[task-name] 브랜치 생성
4. Task 스펙대로만 구현 (범위 확장 금지)
5. Task의 검증 명령어 실행
6. 검증 통과 → context/experiment_log.md append
7. 검증 실패 → 최대 2회 재시도, 그래도 실패 → 로그에 에러 기록 후 정지
8. 완료 → 정지 (다음 Task는 사람이 지시할 때까지 대기)
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
- main 또는 dev 브랜치 직접 커밋

---

## Agent Manager 활용 (Antigravity 2.0 병렬 에이전트)

병렬 에이전트 사용 시 역할 분리:
```
Agent 1 (Gemini Pro)   → src/ 구현
Agent 2 (Gemini Flash) → 검증 명령어 실행 + Gate 체크
Agent 3 (Gemini Flash) → context/experiment_log.md 업데이트
```

동일 파일을 두 에이전트가 동시에 수정하는 것 금지.
특히 experiment_log.md는 Agent 3만 수정.

---

## GitHub Actions 연동

`context/experiment_log.md` 변경 후 dev 브랜치에 push하면:
- `update_metrics.yml` 자동 트리거
- `results/metrics.csv` 자동 갱신
- `README.md` Key Results 테이블 자동 업데이트

push 명령:
```powershell
conda activate wm811k; git add context/experiment_log.md; git commit -m "feat: stage[N] experiment results"; git push origin agent/[task-name]
```
