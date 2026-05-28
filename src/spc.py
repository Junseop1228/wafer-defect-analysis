import numpy as np
import pandas as pd

def setup_phase1_baseline(cfg: dict, df_labeled: pd.DataFrame) -> dict:
    """
    Select stable process reference lots (Phase 1).
    Rank lots by None-wafer ratio and take top cfg['spc']['phase1_percentile']%.
    """
    spc_cfg = cfg.get("spc", {})
    percentile = spc_cfg.get("phase1_percentile", 70)
    
    # Calculate defect rate per lot
    df = df_labeled.copy()
    if 'failureType' in df.columns:
        # Convert ndarray to string if needed
        if df['failureType'].dtype == object and len(df) > 0 and hasattr(df['failureType'].iloc[0], 'size'):
            df['_ft_str'] = df['failureType'].apply(
                lambda x: x[0][0] if hasattr(x, 'size') and x.size > 0 else ''
            )
        else:
            df['_ft_str'] = df['failureType'].astype(str)
            
        # Standardize normal names
        merge_targets = cfg.get("classes", {}).get("merge_to_normal", [])
        df['_ft_str'] = df['_ft_str'].apply(
            lambda x: 'Normal' if x in merge_targets or x == 'none' else x
        )
    else:
        raise ValueError("df_labeled must contain 'failureType' column")

    df['is_defect'] = (df['_ft_str'] != 'Normal') & (df['_ft_str'] != '')

    # Aggregate by lot
    lot_stats = df.groupby('lotName').agg(
        total=('is_defect', 'count'),
        defects=('is_defect', 'sum')
    )
    lot_stats['defect_rate'] = lot_stats['defects'] / lot_stats['total']
    
    # Filter lots with at least some minimum wafers (e.g., 10)
    lot_stats = lot_stats[lot_stats['total'] >= 10]

    # Sort ascending by defect_rate (top stable lots have lowest defect rate)
    lot_stats = lot_stats.sort_values('defect_rate', ascending=True)
    
    if spc_cfg.get("strict_phase1_zero_defect", False):
        phase1_lots = lot_stats[lot_stats['defect_rate'] == 0.0]
        if len(phase1_lots) == 0:
            print("[SPC] WARNING: No zero-defect lots found, falling back to percentile.")
            n_phase1 = int(len(lot_stats) * (percentile / 100.0))
            phase1_lots = lot_stats.iloc[:max(1, n_phase1)]
    else:
        n_phase1 = int(len(lot_stats) * (percentile / 100.0))
        if n_phase1 == 0:
            n_phase1 = len(lot_stats)
        phase1_lots = lot_stats.iloc[:n_phase1]
    
    p_bar = phase1_lots['defects'].sum() / phase1_lots['total'].sum()
    n_avg = phase1_lots['total'].mean()
    
    sigma = np.sqrt(p_bar * (1 - p_bar) / n_avg)
    if sigma == 0.0:
        # Give a very tiny sigma to prevent H=0. 
        # For a standard control chart, if historical sigma is 0, we can use 1 / n_avg as a minimum bound.
        sigma = 1e-4
        
    ucl = p_bar + 3 * sigma
    lcl = max(0.0, p_bar - 3 * sigma)
    
    return {
        "mean": float(p_bar),
        "sigma": float(sigma),
        "ucl": float(ucl),
        "lcl": float(lcl),
        "n_avg": float(n_avg)
    }

def run_shewhart(defect_rates: np.ndarray, baseline: dict) -> dict:
    """Shewhart p-chart (spike detection)."""
    p_bar = baseline["mean"]
    ucl = baseline["ucl"]
    lcl = baseline["lcl"]
    
    violations = np.where(defect_rates > ucl)[0]
    return {
        "p_bar": p_bar,
        "ucl": ucl,
        "lcl": lcl,
        "violations": violations.tolist()
    }

def run_cusum(defect_rates: np.ndarray, baseline: dict, cfg: dict) -> dict:
    """CUSUM chart (drift detection)."""
    spc_cfg = cfg.get("spc", {})
    k_factor = spc_cfg.get("cusum_k", 0.5)
    h_factor = spc_cfg.get("cusum_h", 4.0)
    
    mu = baseline["mean"]
    sigma = baseline["sigma"]
    
    k = k_factor * sigma
    H = h_factor * sigma
    
    c_plus = np.zeros_like(defect_rates)
    c_minus = np.zeros_like(defect_rates)
    
    violations = []
    
    for i in range(1, len(defect_rates)):
        x = defect_rates[i]
        c_plus[i] = max(0, c_plus[i-1] + (x - mu - k))
        c_minus[i] = max(0, c_minus[i-1] - (x - mu + k))
        
        if c_plus[i] > H or c_minus[i] > H:
            violations.append(i)
            
    return {
        "c_plus": c_plus,
        "c_minus": c_minus,
        "h": H,
        "violations": violations
    }

def run_ewma(defect_rates: np.ndarray, baseline: dict, cfg: dict) -> dict:
    """EWMA chart (trend detection)."""
    spc_cfg = cfg.get("spc", {})
    lam = spc_cfg.get("ewma_lambda", 0.2)
    
    mu = baseline["mean"]
    sigma = baseline["sigma"]
    
    z = np.zeros_like(defect_rates)
    z[0] = mu
    
    ucl = np.zeros_like(defect_rates)
    lcl = np.zeros_like(defect_rates)
    
    violations = []
    
    for i in range(1, len(defect_rates)):
        z[i] = lam * defect_rates[i] + (1 - lam) * z[i-1]
        
        limit_term = 3 * sigma * np.sqrt((lam / (2 - lam)) * (1 - (1 - lam)**(2*i)))
        ucl[i] = mu + limit_term
        lcl[i] = max(0, mu - limit_term)
        
        if z[i] > ucl[i]:
            violations.append(i)
            
    return {
        "z": z,
        "ucl": ucl,
        "lcl": lcl,
        "violations": violations
    }

def compute_arl(violations: list, n_total: int) -> float:
    """
    Compute Average Run Length (ARL) = 1 / violation_rate.
    In normal process state, ARL should ideally be >= 370.
    """
    if len(violations) == 0:
        return float('inf')
    
    violation_rate = len(violations) / n_total
    return 1.0 / violation_rate

def run_dual_detection(cfg: dict, defect_rates: np.ndarray, 
                       anomaly_scores: np.ndarray, baseline: dict) -> dict:
    """
    SPC + AE dual detection logic.
    Alert = HIGH when BOTH SPC chart violates AND AE anomaly score exceeds threshold.
    """
    cusum_result = run_cusum(defect_rates, baseline, cfg)
    spc_violations = set(cusum_result["violations"])
    
    # In practice anomaly_scores passed here should be per lot (e.g. max or 95th pct AE score of wafers in lot)
    ae_threshold = cfg.get("autoencoder", {}).get("threshold", 0.0)
    
    ae_violations = set(np.where(anomaly_scores > ae_threshold)[0])
    
    high_confidence_alerts = sorted(list(spc_violations.intersection(ae_violations)))
    medium_confidence_alerts = sorted(list(spc_violations.symmetric_difference(ae_violations)))
    
    return {
        "high_alerts": high_confidence_alerts,
        "medium_alerts": medium_confidence_alerts,
        "arl": compute_arl(cusum_result["violations"], len(defect_rates))
    }
