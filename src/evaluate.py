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
