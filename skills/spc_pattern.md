# Skill: SPC Pattern — WM-811K

> Antigravity reads this when implementing src/spc.py and src/autoencoder.py

---

## Three Chart Types

### 1. Shewhart p-chart (spike detection)
```python
def shewhart_pchart(defect_rates: pd.Series, cfg: dict) -> dict:
    """
    defect_rates: time-ordered series of defect rate per lot
    Phase 1: top cfg['spc']['phase1_percentile']% lots by None ratio
    """
    p_bar = phase1_mean
    sigma = np.sqrt(p_bar * (1 - p_bar) / n)
    ucl = p_bar + 3 * sigma
    lcl = max(0, p_bar - 3 * sigma)
    violations = defect_rates[defect_rates > ucl].index.tolist()
    return {"ucl": ucl, "lcl": lcl, "violations": violations, "p_bar": p_bar}
```

### 2. CUSUM (drift detection) — PRIMARY CHART
```python
def cusum_chart(defect_rates: pd.Series, cfg: dict) -> dict:
    """
    k = cfg['spc']['cusum_k'] (default 0.5 sigma)
    H = 5 * sigma (decision interval)
    Detects sustained upward drift — maps to CMP pad wear, CVD degradation
    """
    k = cfg["spc"]["cusum_k"]
    # C+[i] = max(0, C+[i-1] + (x[i] - mu - k*sigma))
    # Signal when C+ > H
```

### 3. EWMA (λ=0.2, bridges Shewhart and CUSUM)
```python
def ewma_chart(defect_rates: pd.Series, cfg: dict) -> dict:
    lam = cfg["spc"]["ewma_lambda"]  # 0.2
    # z[i] = lam * x[i] + (1 - lam) * z[i-1]
    # UCL = mu + 3*sigma * sqrt(lam/(2-lam))
```

---

## Autoencoder Anomaly Score

- Train ONLY on unlabeled wafers with low reconstruction error (normal baseline)
- Input: 64×64 wafer map (1 channel, values 0/1/2 → normalize to [0,1])
- Architecture: Conv encoder → bottleneck → Conv decoder
- Anomaly score: MSE reconstruction error per wafer
- Threshold: 95th percentile of Phase 1 reconstruction errors

```python
class WaferAutoencoder(nn.Module):
    # Encoder: Conv(1→16) → Conv(16→32) → flatten → FC(bottleneck=64)
    # Decoder: FC → reshape → ConvTranspose(32→16) → ConvTranspose(16→1)
```

---

## Dual Detection Rule

Flag as HIGH CONFIDENCE anomaly ONLY when BOTH fire:
1. SPC chart (any of 3) shows UCL violation
2. AE reconstruction error > threshold

Single-signal violations → MEDIUM confidence alert only

---

## Phase 1 Baseline Setup

```python
def setup_phase1_baseline(df_labeled: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Select stable process reference lots:
    - Filter to None-pattern wafers
    - Rank lots by None-wafer ratio
    - Take top cfg['spc']['phase1_percentile']% (default 70)
    """
```

---

## Limitation Statement (include in Streamlit tab 3)

> WM-811K does not include equipment IDs. The time axis is approximated using
> lotName sequential ordering. In actual production, control charts must be
> separated per equipment ID. This implementation demonstrates the methodology;
> real deployment requires equipment-level data.
