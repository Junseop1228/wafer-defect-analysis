# Experiment Log — WM-811K Pipeline

> Antigravity appends to this file after every execution.
> Claude.ai reads this for result interpretation and Gate decisions.
> Format: newest entry at the TOP.

---

## [2026-05-22 18:22] Stage 2 — t-SNE Visualization (Task 2)
- Branch: agent/stage2-eda
- Command: `conda activate wm811k; jupyter nbconvert --to notebook --execute notebooks/01_EDA.ipynb`
- Result: SUCCESS
- Metrics:
  - macro_f1: N/A
  - scratch_recall: N/A
  - donut_recall: N/A
  - binary_defect_recall: N/A
- Gate: N/A
- MLflow run_id: N/A
- Notes: Subsampled 500 samples per class, resized to 28x28, flattened, and executed t-SNE (perplexity=30). Saved visualization to results/figures/tsne_raw.png.

---

## [2026-05-22 17:11] Stage 2 — Data Loading and Basic EDA (Task 1)
- Branch: agent/stage2-eda
- Command: `conda activate wm811k; jupyter nbconvert --to notebook --execute notebooks/01_EDA.ipynb`
- Result: SUCCESS
- Metrics:
  - macro_f1: N/A
  - scratch_recall: N/A
  - donut_recall: N/A
  - binary_defect_recall: N/A
- Gate: N/A
- MLflow run_id: N/A
- Notes: Successfully separated labeled (172,950) and unlabeled (638,507) wafers. Generated class_distribution.png and sample_wafers.png figures under results/figures/.

<!-- APPEND NEW ENTRIES ABOVE THIS LINE -->

---

## Log Template (copy for each entry)

```
## [YYYY-MM-DD HH:MM] Stage X — [description]
- Command: `conda activate wm811k; python ...`
- Result: SUCCESS / FAILED
- Metrics:
  - macro_f1: 0.XX
  - scratch_recall: 0.XX
  - donut_recall: 0.XX
  - binary_defect_recall: 0.XX
- Gate: Gate N — PASSED / FAILED / N/A
- MLflow run_id: [id]
- Notes: ...
```
