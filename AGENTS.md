# WM-811K Wafer Defect Classification — Agent Rules

> This file is read by ALL AI coding agents (Antigravity, Claude Code, Cursor, etc.)
> Keep under 150 lines. Tool-specific overrides go in GEMINI.md or CLAUDE.md.

---

## Project Identity

- **Goal**: Classify wafer map defect patterns (7 classes), trace to root-cause process, detect anomalies via SPC
- **Target**: Samsung Electronics TSP 평가및분석 인턴 / SK Hynix P&T
- **Stack**: Python 3.10, PyTorch, scikit-learn, SHAP, Optuna, MLflow, Streamlit
- **Conda env**: `wm811k`
- **Root**: `C:\Users\userPC\Desktop\Workspace\01_Projects\WM811K_Portfolio\`

---

## Directory Map

```
WM811K_Portfolio/
├── AGENTS.md            ← you are here
├── GEMINI.md            ← Antigravity overrides
├── config.yaml          ← ALL hyperparameters live here
├── run_pipeline.py      ← single entry point
├── context/             ← session state bridge (Claude.ai ↔ Antigravity)
│   ├── stage_status.md  ← current phase/stage/gate status  ← READ FIRST
│   ├── experiment_log.md← cumulative experiment results
│   ├── decisions.md     ← design decisions + rationale
│   └── handoff.md       ← end-of-session handoff notes
├── skills/              ← reusable task rules
├── plans/               ← Claude-generated execution plans  ← READ BEFORE CODING
├── src/                 ← all implementation modules
├── notebooks/           ← prototype notebooks (import from src/)
├── app/streamlit_app.py ← dashboard
├── results/             ← figures + metrics.csv
└── .github/workflows/   ← CI/CD automation
```

---

## Session Start Protocol

**Every new session, in this order:**
1. Read `context/stage_status.md` — current phase, gate status, last completed task
2. Read `plans/phase[N]_plan.md` for the active phase — task list and completion criteria
3. Do NOT start coding until both files are read

---

## Coding Rules — Non-Negotiable

**Config**
- NEVER hardcode hyperparameters. Load ALL values from `config.yaml` via `yaml.safe_load()`
- Gate thresholds, model params, SPC params — all in `config.yaml`

**Module structure**
- All functions go in `src/` modules. Notebooks only call `from src.X import Y`
- Gate check functions → `src/evaluate.py`, return `{"passed": bool, "details": dict}`
- Visualization functions → `src/visualize.py`
- Feature engineering → `src/features.py`
- Model training → `src/train.py`
- SPC → `src/spc.py`

**MLflow — mandatory on every training run**
```python
with mlflow.start_run(run_name="..."):
    mlflow.log_params(cfg["model_params"])
    mlflow.log_metrics({"macro_f1": ..., "scratch_recall": ..., "donut_recall": ...})
    mlflow.log_artifact("results/figures/confusion_matrix.png")
```

**Output**
- Minimize `print()`. Use MLflow or Streamlit for output
- Exceptions: startup messages and Gate pass/fail announcements are OK

**Code comments**: English only, concise, key logic only

---

## Gate Thresholds (from config.yaml)

| Gate | Condition | Threshold |
|------|-----------|-----------|
| Gate 1 | SHAP contribution per feature | ≥ 0.01 |
| Gate 1 | Feature correlation | ≤ 0.90 |
| Gate 2 | Binary defect recall | ≥ 0.90 |
| Gate 2 | Scratch recall (7-class) | ≥ 0.70 |
| Gate 2 | Donut recall (7-class) | ≥ 0.75 |
| Gate 3 | Confusion pair F1 gap | ≤ 0.20 |
| Gate 4 | SPC ARL | ≥ 370 |

If a Gate fails → log failure details to `context/experiment_log.md`, do NOT proceed to next stage

---

## After Every Execution

Append results to `context/experiment_log.md` in this format:
```
## [YYYY-MM-DD] Stage X — Model Name
- macro_f1: 0.XX
- scratch_recall: 0.XX
- donut_recall: 0.XX
- Gate N: PASSED / FAILED
- Notes: ...
```

---

## What NOT To Do

- Do NOT modify `config.yaml` values directly — propose changes in chat first
- Do NOT delete files in `data/`, `mlruns/`, `results/`
- Do NOT push to `main` branch — use `dev` or `stage/[N]` branches
- Do NOT run `conda install` — use `pip install` within `wm811k` env
- Do NOT create files outside the project root
