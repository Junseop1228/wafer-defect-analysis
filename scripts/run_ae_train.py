"""
scripts/run_ae_train.py
Task 2 verification: train AE + compare anomaly scores defect vs normal.
"""
import sys, os
# Ensure project root is on path when running from scripts/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yaml
import numpy as np
import pandas as pd
import torch

# Load config
cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))

from src.autoencoder import train_autoencoder, compute_anomaly_scores

# Step 1: Train autoencoder on unlabeled data
print("=" * 60)
print("STEP 1: Training Autoencoder on unlabeled wafer maps")
print("=" * 60)
result = train_autoencoder(cfg)
model = result["model"]
mean_err = result["mean_recon_error"]
threshold = result["threshold"]

print(f"\n[RESULT] mean_recon_error (normal unlabeled): {mean_err:.6f}")
print(f"[RESULT] threshold (95th pct): {threshold:.6f}")

# Step 2: Evaluate on labeled data -- compare defect vs normal scores
print("\n" + "=" * 60)
print("STEP 2: Comparing anomaly scores -- defect vs normal wafers")
print("=" * 60)

df_labeled = pd.read_pickle(cfg["data"]["features_v2"])
print(f"Labeled samples: {len(df_labeled):,}")

# We need the raw LSWMD data to get waferMap for labeled samples
df_raw = pd.read_pickle(cfg["data"]["raw_path"])

# Convert ndarray failureType → str (same pattern as pseudo_label.py)
df_raw["_ft_str"] = df_raw["failureType"].apply(
    lambda x: x[0][0] if hasattr(x, "size") and x.size > 0 else ""
)
df_raw_labeled = df_raw[df_raw["_ft_str"] != ""].reset_index(drop=True)
print(f"Raw labeled samples: {len(df_raw_labeled):,}")

# Merge Near-full to Normal
merge_targets = cfg["classes"]["merge_to_normal"] or []
df_raw_labeled["_ft_str"] = df_raw_labeled["_ft_str"].apply(
    lambda x: "Normal" if x in (merge_targets or []) else x
)
# 'none' → 'Normal'
df_raw_labeled["_ft_str"] = df_raw_labeled["_ft_str"].replace("none", "Normal")

# Sample for speed: 500 Normal + 500 Defect
df_normal = df_raw_labeled[df_raw_labeled["_ft_str"] == "Normal"].sample(
    min(500, (df_raw_labeled["_ft_str"] == "Normal").sum()), random_state=42
)
df_defect = df_raw_labeled[df_raw_labeled["_ft_str"] != "Normal"].sample(
    min(500, (df_raw_labeled["_ft_str"] != "Normal").sum()), random_state=42
)


wm_normal = np.array(df_normal["waferMap"].tolist(), dtype=object)
wm_defect = np.array(df_defect["waferMap"].tolist(), dtype=object)

print(f"Scoring {len(wm_normal)} Normal, {len(wm_defect)} Defect wafers ...")
scores_normal = compute_anomaly_scores(cfg, wm_normal)
scores_defect = compute_anomaly_scores(cfg, wm_defect)

print(f"\n[ANOMALY SCORES]")
print(f"  Normal  -- mean={scores_normal.mean():.6f}, std={scores_normal.std():.6f}")
print(f"  Defect  -- mean={scores_defect.mean():.6f}, std={scores_defect.std():.6f}")
print(f"  Threshold: {threshold:.6f}")

normal_above = (scores_normal > threshold).mean() * 100
defect_above = (scores_defect > threshold).mean() * 100
print(f"\n  Normal above threshold: {normal_above:.1f}% (expected ~5%)")
print(f"  Defect above threshold: {defect_above:.1f}% (expected >> 5%)")

if scores_defect.mean() > scores_normal.mean():
    print("\n[PASS] Defect anomaly score > Normal anomaly score [OK]")
else:
    print("\n[WARN] Defect score not higher than Normal -- check AE training")
