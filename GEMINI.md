# GEMINI.md — Antigravity-Specific Overrides
# Overrides AGENTS.md where conflicts exist. Antigravity reads this first.

---

## Execution Environment

Every terminal command MUST follow this pattern:
```powershell
conda activate wm811k; [your command here]
```

Example:
```powershell
conda activate wm811k; python src/train.py --config config.yaml --stage binary
```

Never run Python commands without activating `wm811k` first.

---

## Model Selection Strategy

Use models in this priority order to conserve credits:

| Task | Model |
|------|-------|
| Boilerplate / simple edits | Gemini 3 Flash |
| General src/ implementation | Gemini 3.1 Pro (High) |
| Complex debugging / Gate cycle logic | Claude Sonnet 4.6 |
| Architecture decisions | Escalate to Claude.ai chat |

Do NOT use Claude Opus or Sonnet for boilerplate tasks.

---

## Branch Rules

- Always work on `agent/[task-name]` branch, never on `main` or `dev` directly
- Branch naming: `agent/stage3-features`, `agent/gate2-fix`, etc.
- After task completes, stop. Do NOT merge — human reviews and merges to `dev`

Create branch before starting any file modification:
```powershell
git checkout -b agent/[task-name]
```

---

## Task Execution Protocol

1. Read `context/stage_status.md`
2. Read `plans/phase[N]_plan.md` for active phase
3. Identify current Task number
4. Implement exactly what the Task specifies — no scope expansion
5. Run the verification command listed in the Task
6. If verification passes → append to `context/experiment_log.md`
7. If verification fails → attempt fix max 2 times, then stop and report failure

---

## experiment_log.md Update (Mandatory)

After EVERY execution, append to `context/experiment_log.md`:
```markdown
## [YYYY-MM-DD HH:MM] Stage X — [task name]
- Command: `python ...`
- Result: SUCCESS / FAILED
- Metrics: macro_f1=X.XX, scratch_recall=X.XX, donut_recall=X.XX
- Gate: PASSED / FAILED / N/A
- Error (if any): ...
```

---

## Scope Boundaries

Do NOT do any of the following without explicit instruction in plans/:
- Refactor files not mentioned in the current Task
- Add features beyond what the Task specifies
- Change config.yaml values
- Install new packages
- Modify .gitignore, README.md, or any .md files in context/

---

## Agent Manager (Antigravity 2.0)

If using parallel agents:
- Agent 1: `src/` implementation
- Agent 2: test execution and Gate check
- Agent 3: `context/experiment_log.md` update
- Never assign two agents to modify the same file simultaneously
