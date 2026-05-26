import pandas as pd

# Gate checks + Metric computation
# Gate1: SHAP / corr  Gate2: Recall  Gate3: confusion pair  Gate4: ARL

def check_gate1(shap_values: dict, corr_matrix: pd.DataFrame, cfg: dict) -> dict:
    """
    shap_values: {feature_name: mean_abs_shap_value}
    corr_matrix: feature correlation matrix
    """
    try:
        threshold_shap = cfg["gates"]["gate1"]["min_shap_contribution"]
        threshold_corr = cfg["gates"]["gate1"]["max_correlation"]

        total_shap = sum(shap_values.values())
        if total_shap == 0:
            weak_features = list(shap_values.keys())
        else:
            weak_features = [f for f, v in shap_values.items() if (v / total_shap) < threshold_shap]
            
        high_corr_pairs = [(i, j) for i in corr_matrix.columns
                           for j in corr_matrix.columns
                           if i < j and abs(corr_matrix.loc[i, j]) > threshold_corr]
        passed = len(weak_features) == 0 and len(high_corr_pairs) == 0
        
        result = {
            "passed": passed,
            "details": {"weak_features": weak_features, "high_corr_pairs": high_corr_pairs}
        }
        
        if passed:
            print("[GATE 1] PASSED")
        else:
            reasons = []
            if weak_features: reasons.append(f"{len(weak_features)} weak features")
            if high_corr_pairs: reasons.append(f"{len(high_corr_pairs)} highly correlated pairs")
            print(f"[GATE 1] FAILED: {', '.join(reasons)}")
            
        return result
    except Exception as e:
        print(f"[GATE 1] FAILED: Exception occurred - {str(e)}")
        return {"passed": False, "details": {"error": str(e)}}

def check_gate2_binary(recall: float, cfg: dict) -> dict:
    threshold = cfg["gates"]["gate2"]["stage1_min_defect_recall"]
    passed = bool(recall >= threshold)
    
    result = {
        "passed": passed,
        "details": {"binary_defect_recall": float(recall), "threshold": threshold}
    }
    
    if passed:
        print(f"[GATE 2 BINARY] PASSED (Recall: {recall:.4f} >= {threshold})")
    else:
        print(f"[GATE 2 BINARY] FAILED (Recall: {recall:.4f} < {threshold})")
        
    return result


def check_gate2_full(scratch_recall: float, donut_recall: float, cfg: dict) -> dict:
    """Gate 2 full check: scratch_recall >= 0.70, donut_recall >= 0.75."""
    thr_scratch = cfg["gates"]["gate2"]["stage2_min_scratch_recall"]
    thr_donut = cfg["gates"]["gate2"]["stage2_min_donut_recall"]

    pass_scratch = bool(scratch_recall >= thr_scratch)
    pass_donut = bool(donut_recall >= thr_donut)
    passed = pass_scratch and pass_donut

    result = {
        "passed": passed,
        "details": {
            "scratch_recall": float(scratch_recall),
            "scratch_threshold": thr_scratch,
            "scratch_passed": pass_scratch,
            "donut_recall": float(donut_recall),
            "donut_threshold": thr_donut,
            "donut_passed": pass_donut,
        }
    }

    if passed:
        print(f"[GATE 2 FULL] PASSED  Scratch: {scratch_recall:.4f}>={thr_scratch}  Donut: {donut_recall:.4f}>={thr_donut}")
    else:
        reasons = []
        if not pass_scratch:
            reasons.append(f"Scratch recall {scratch_recall:.4f} < {thr_scratch}")
        if not pass_donut:
            reasons.append(f"Donut recall {donut_recall:.4f} < {thr_donut}")
        print(f"[GATE 2 FULL] FAILED: {', '.join(reasons)}")

    return result


def check_gate3(y_true, y_pred, class_names: list, cfg: dict,
                confusion_pairs: list = None) -> dict:
    """
    Gate 3: Confusion pair F1 gap <= max_confusion_pair_f1_gap.
    confusion_pairs: list of (classA, classB) name tuples to check.
    For each pair, computes |F1(A) - F1(B)| and checks if all gaps <= threshold.
    Returns worst pair and its gap.
    """
    from sklearn.metrics import f1_score

    threshold = cfg["gates"]["gate3"]["max_confusion_pair_f1_gap"]

    if confusion_pairs is None:
        confusion_pairs = [
            ("Edge-Ring", "Edge-Loc"),
            ("Center", "Donut"),
            ("Loc", "Random"),
        ]

    per_class_f1 = dict(zip(class_names,
                            f1_score(y_true, y_pred, average=None,
                                     labels=list(range(len(class_names))))))

    pair_results = []
    for classA, classB in confusion_pairs:
        f1_a = per_class_f1.get(classA, None)
        f1_b = per_class_f1.get(classB, None)
        if f1_a is None or f1_b is None:
            print(f"  [GATE 3] WARNING: '{classA}' or '{classB}' not in class_names, skipping pair.")
            continue
        gap = abs(float(f1_a) - float(f1_b))
        pair_results.append({
            "pair": f"{classA}<->{classB}",
            "f1_A": round(float(f1_a), 4),
            "f1_B": round(float(f1_b), 4),
            "f1_gap": round(gap, 4),
            "passed": bool(gap <= threshold),
        })

    if not pair_results:
        return {"passed": False, "details": {"error": "No valid pairs found"}}

    worst = max(pair_results, key=lambda x: x["f1_gap"])
    all_passed = all(p["passed"] for p in pair_results)

    result = {
        "passed": all_passed,
        "details": {
            "threshold": threshold,
            "worst_pair": worst["pair"],
            "worst_f1_gap": worst["f1_gap"],
            "pairs": pair_results,
        }
    }

    if all_passed:
        print(f"[GATE 3] PASSED  Worst pair: {worst['pair']} gap={worst['f1_gap']:.4f} <= {threshold}")
    else:
        failed = [p for p in pair_results if not p["passed"]]
        print(f"[GATE 3] FAILED  {len(failed)} pair(s) exceed threshold {threshold}:")
        for p in failed:
            print(f"  {p['pair']}: F1_gap={p['f1_gap']:.4f} (A={p['f1_A']}, B={p['f1_B']})")

    return result

