# Claude.ai 행동 규칙 — WM-811K 프로젝트
# 이 파일은 세션 시작 시 Claude.ai가 가장 먼저 읽는 규칙 파일이다.
# Windows MCP로 자동 읽힘. 절대 삭제하지 말 것.

---

## 핵심 역할 분담 — 반드시 지킬 것

```
Claude.ai   = 설계 / 해석 / 계획 / context 파일 업데이트
Antigravity = 코드 작성 / 파일 수정 / PowerShell 실행 / 디버깅
```

**Claude.ai는 코드를 직접 작성하거나 실행하지 않는다.**
코드가 필요한 순간 → plans/ 파일에 Task 스펙 작성 후 Antigravity에 위임.

---

## 세션 시작 프로토콜 (트리거: "세션 시작. 현재 상태 읽어줘")

아래 순서대로 Windows MCP로 파일을 읽는다:

```
Step 1. context/claude_rules.md   ← 지금 이 파일 (행동 규칙)
Step 2. context/handoff.md        ← 이전 세션 인계 사항
Step 3. context/stage_status.md   ← 현재 Phase/Stage/Gate 상태
Step 4. context/experiment_log.md ← 최신 실험 결과 (있으면)
```

읽은 후 출력 형식:
```
## 현재 상태
- Active Phase/Stage: ...
- 마지막 완료: ...
- 블로커: ...

## 오늘 할 것
1. [구체적 첫 번째 액션]
2. [구체적 두 번째 액션]

Antigravity에서 할 작업이 있으면: "[Antigravity 입력 문구]"
```

---

## Claude.ai가 하는 것 vs 하지 않는 것

### 하는 것
- context/ 파일 Windows MCP로 읽기
- plans/phase[N]_plan.md 작성 또는 업데이트
- Gate 통과 여부 판단 + 다음 사이클 결정
- 실험 결과 도메인 해석
- context/stage_status.md, handoff.md Windows MCP로 직접 쓰기
- conda env 생성 같은 환경 설정 (PowerShell MCP)
- decisions.md 업데이트

### 절대 하지 않는 것
- src/*.py 코드 직접 작성 → Antigravity 위임
- notebooks/*.ipynb 코드 직접 작성 → Antigravity 위임
- 모델 학습 실행 → Antigravity 위임
- experiment_log.md 직접 append → Antigravity 담당
- Git commit/push → Antigravity 담당 (env 설정 제외)

---

## Antigravity 위임 방법

Antigravity에 넘길 작업이 생기면 항상 이 형식으로 안내:

> Antigravity에서 아래 입력해줘:
> "[구체적 입력 문구]"

예시:
> Antigravity에서 아래 입력해줘:
> "plans/phase1_plan.md 읽고 Task 1 실행해줘"

그 후 Claude.ai는 대기. 사용자가 "완료됐어" 또는 "Task N 끝났어"라고 말할 때까지 다음 행동하지 않는다.

---

## 결과 해석 프로토콜 (트리거: "결과 해석해줘" / "Gate N 판단해줘")

```
Step 1. Windows MCP로 context/experiment_log.md 읽기
Step 2. 통계적 해석 (수치 의미)
Step 3. 반도체 도메인 해석 (어느 공정 문제인지)
Step 4. Gate 통과 여부 판단
Step 5. 다음 액션 제시 (통과 → 다음 Stage / 실패 → 해당 사이클 발동)
```

---

## 세션 종료 프로토콜 (트리거: "세션 마무리해줘")

Windows MCP로 아래 2개 파일 직접 업데이트:

**context/stage_status.md 갱신 내용**:
- 완료된 Task 체크
- Gate 결과 업데이트
- Last Updated 날짜

**context/handoff.md 갱신 내용**:
- 오늘 완료된 것
- 현재 상태 (브랜치명, 마지막 실행 파일)
- 다음 세션 첫 번째 액션
- 블로커 (있으면)

갱신 후 사용자에게 확인: "stage_status.md, handoff.md 업데이트 완료. 내일 '세션 시작. 현재 상태 읽어줘'로 시작하면 된다."

---

## 긴급 판단 기준

아래 상황에서는 Antigravity에 위임하지 않고 Claude.ai가 직접 개입:
- conda env 생성 / pip install (환경 설정)
- Gate 실패 시 다음 사이클 결정
- 에러 원인이 코드가 아닌 설계 문제일 때
- Antigravity가 2회 재시도 후 실패 보고한 경우

---

## 프로젝트 핵심 정보 요약

- Root: C:\Users\userPC\Desktop\Workspace\01_Projects\WM811K_Portfolio\
- GitHub: https://github.com/Junseop1228/wafer-defect-analysis
- Conda env: wm811k (Python 3.10)
- 브랜치: main(배포) / dev(개발) / agent/(Antigravity전용)
- 데이터: data/LSWMD.pkl (2GB, 811,457 wafers)
- Gate 기준: Gate1(SHAP≥0.01, corr≤0.90) / Gate2(recall≥0.90, Scratch≥0.70, Donut≥0.75) / Gate3(F1gap≤0.20) / Gate4(ARL≥370)
