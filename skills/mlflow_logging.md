# Skill: MLflow Logging Standard — WM-811K

> Antigravity reads this when writing any training or evaluation code.
> Every experiment MUST follow this standard. No exceptions.

---

## Mandatory Pattern

```python
import mlflow
import yaml

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

mlflow.set_tracking_uri(cfg["mlflow"]["tracking_uri"])
mlflow.set_experiment(cfg["mlflow"]["experiment_name"])

with mlflow.start_run(run_name=f"stage{STAGE}_{MODEL_NAME}"):
    # 1. Log all relevant config params
    mlflow.log_params(cfg["gates"])
    mlflow.log_params({"model": MODEL_NAME, "stage": STAGE})

    # 2. Train / evaluate here
    # ...

    # 3. Log metrics — these three are ALWAYS required
    mlflow.log_metrics({
        "macro_f1": macro_f1,
        "scratch_recall": scratch_recall,
        "donut_recall": donut_recall,
    })

    # 4. Log artifacts
    mlflow.log_artifact("results/figures/confusion_matrix.png")
    mlflow.sklearn.log_model(model, "model")  # or mlflow.pytorch.log_model
```

---

## Stage-Specific Required Metrics

| Stage | Required metrics |
|-------|-----------------|
| Stage 4-1 (binary) | binary_defect_recall, binary_f1, threshold |
| Stage 4-2 ML | macro_f1, per_class_f1 (all 7), scratch_recall, donut_recall |
| Stage 4-2 CNN | same as ML + train_loss, val_loss, epochs |
| Stage 4-2 Hybrid | same as ML + cnn_embed_dim |
| Stage 5 (SHAP) | top_feature_1..5, shap_coverage_top10 |
| Stage 7 (SPC) | arl_shewhart, arl_cusum, arl_ewma, ae_threshold |

---

## Run Naming Convention

```
stage{N}_{model}_{variant}
```
Examples:
- `stage4_rf_baseline`
- `stage4_xgb_optuna_trial_12`
- `stage4_cnn_v2`
- `stage4_hybrid_final`
- `stage7_spc_cusum`

---

## What NOT to Do

- Do NOT use `mlflow.log_param()` one at a time — use `mlflow.log_params(dict)`
- Do NOT log inside a loop without a run context
- Do NOT hardcode experiment names — always use `cfg["mlflow"]["experiment_name"]`
- Do NOT forget to log the confusion matrix image as artifact
