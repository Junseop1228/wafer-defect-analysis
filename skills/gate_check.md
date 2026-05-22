# Skill: Gate Check Standard — WM-811K

> Antigravity reads this when implementing or calling src/evaluate.py
> Gate functions are the only way to decide stage progression.

---

## Function Contract

Every gate check function MUST:
- Accept `metrics: dict` and `cfg: dict` as arguments
- Return `{"passed": bool, "details": dict}`
- Never raise exceptions — catch all errors and return passed=False with error in details
- Log result to console: `[GATE N] PASSED` or `[GATE N] FAILED: reason`

---

## Gate Implementations

### Gate 1 — Feature Validity
```python
def check_gate1(shap_values: dict, corr_matrix: pd.DataFrame, cfg: dict) -> dict:
    """
    shap_values: {feature_name: mean_abs_shap_value}
    corr_matrix: feature correlation matrix
    """
    threshold_shap = cfg["gates"]["gate1"]["min_shap_contribution"]
    threshold_corr = cfg["gates"]["gate1"]["max_correlation"]

    weak_features = [f for f, v in shap_values.items() if v < threshold_shap]
    high_corr_pairs = [(i, j) for i in corr_matrix.columns
                       for j in corr_matrix.columns
                       if i < j and abs(corr_matrix[i][j]) > threshold_corr]
    passed = len(weak_features) == 0 and len(high_corr_pairs) == 0
    return {
        "passed": passed,
        "details": {"weak_features": weak_features, "high_corr_pairs": high_corr_pairs}
    }
```

### Gate 2 — Recall Thresholds
```python
def check_gate2(metrics: dict, cfg: dict, stage: str) -> dict:
    """stage: 'binary' or 'multiclass'"""
    g = cfg["gates"]["gate2"]
    if stage == "binary":
        passed = metrics["binary_defect_recall"] >= g["stage1_min_defect_recall"]
        return {"passed": passed, "details": metrics}
    else:
        scratch_ok = metrics["scratch_recall"] >= g["stage2_min_scratch_recall"]
        donut_ok = metrics["donut_recall"] >= g["stage2_min_donut_recall"]
        passed = scratch_ok and donut_ok
        return {"passed": passed, "details": {
            "scratch_recall": metrics["scratch_recall"],
            "donut_recall": metrics["donut_recall"],
            "scratch_ok": scratch_ok, "donut_ok": donut_ok
        }}
```

### Gate 3 — Confusion Pair
```python
def check_gate3(per_class_f1: dict, cfg: dict) -> dict:
    """
    Check worst confusion pair: Local vs Random, Edge-Ring vs Edge-Local
    """
    threshold = cfg["gates"]["gate3"]["max_confusion_pair_f1_gap"]
    confusion_pairs = [("Local", "Random"), ("Edge-Ring", "Edge-Local")]
    failures = []
    for a, b in confusion_pairs:
        gap = abs(per_class_f1.get(a, 0) - per_class_f1.get(b, 0))
        if gap > threshold:
            failures.append({"pair": f"{a}-{b}", "gap": gap})
    return {"passed": len(failures) == 0, "details": {"failures": failures}}
```

### Gate 4 — SPC ARL
```python
def check_gate4(arl: float, cfg: dict) -> dict:
    threshold = cfg["gates"]["gate4"]["min_arl"]
    passed = arl >= threshold
    return {"passed": passed, "details": {"arl": arl, "threshold": threshold}}
```

---

## Cycle Triggers (what to do on failure)

| Gate | Failure → Cycle |
|------|----------------|
| Gate 1 | Cycle ①: remove weak features, re-run SHAP |
| Gate 2 (binary) | Cycle ③: class_weight → SMOTE → threshold search |
| Gate 2 (multiclass) | Cycle ③: same, focus on Scratch and Donut |
| Gate 3 | Cycle ②: add discriminative features for confused pair |
| Gate 4 | Cycle ⑤: recalculate UCL with tighter Phase 1 baseline |

Max iterations per cycle: `cfg["cycles"]["max_iterations"]` (default 3)
