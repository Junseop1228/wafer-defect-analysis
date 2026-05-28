# WM-811K Wafer Defect Classification & SPC Monitoring

> **Semiconductor Yield Enhancement Project**
> 웨이퍼 맵 불량 패턴을 분류하고 원인 공정을 역추적하는 하이브리드(CNN+ML) 모델을 구축했으며, 통계적 공정 관리(SPC)와 Autoencoder 이상 탐지를 결합하여 공정 이상 시점을 사전 감지하는 자동화 파이프라인 및 대시보드를 직접 설계했습니다.

## 🔗 Live Demo
**[Streamlit Cloud Dashboard](https://wm811k-portfolio.streamlit.app/)** *(Click to view interactive pipeline metrics, model diagnostics, and SPC charts)*

---

## 🏗 Pipeline Architecture

```text
[Labeled 172,950] → EDA → Feature Eng.(28) → Gate 1 (SHAP & Corr)
                                                ↓
                                Binary Clf (Normal vs Defect) → Gate 2
                                                ↓
                           7-class Pattern Clf (ML / CNN / Hybrid) → Gate 3
                                                ↓
                            Pseudo-Labeling (Confidence > 0.95)
                                                ↓
                              SHAP & Grad-CAM Domain Interpretation
                                                ↓
                             SPC + Autoencoder Anomaly Detection → Gate 4
                                                ↓
                                  Streamlit Dashboard (Cloud)
```

---

## 📊 Key Results (3-Way Comparison)

| Model | Macro F1 | Scratch Recall | Donut Recall |
|-------|----------|---------------|--------------|
| ML (XGBoost) | 0.705 | 0.285 | 0.766 |
| Custom CNN | 0.766 | 0.774 | 0.838 |
| **Hybrid (CNN+XGB)** | **0.938** | **0.945** | **0.865** |

*Hybrid model successfully solves the severe class imbalance issue (especially for Scratch defects).*

---

## 🎯 Verification Gates

| Gate | Condition | Status | Result |
|------|-----------|--------|--------|
| **Gate 1** (Features) | SHAP > 0.01, Corr < 0.90 | ✅ **PASSED** | Validated 28 spatial features |
| **Gate 2** (Binary/Minority) | Binary Recall > 0.90, Scratch > 0.70 | ✅ **PASSED** | Scratch 0.945 / Donut 0.865 |
| **Gate 3** (Overfitting) | F1 Gap < 0.20 | ✅ **PASSED** | Balanced accuracy |
| **Gate 4** (SPC ARL) | ARL₀ ≥ 370 | ✅ **PASSED** | ARL = ∞ (0 False Alarms on Phase 1) |

---

## 🏭 Defect → Process Interpretation

| Pattern | Suspect Process | Action |
|---------|----------------|--------|
| Edge-Ring | CMP pad wear | Shorten pad replacement cycle |
| Center | CVD gas supply | Check showerhead nozzle |
| Donut | RTP heater zone | Profile heater zone output |
| Scratch | Handling robot slip | Check transfer path log |
| Local | Chamber particle | Lot consistency → contamination type |
| Random | Process instability | Cpk monitoring |
| Edge-Local | Chuck contamination | Identify by cx/cy direction |

---

## 🚀 Quick Start (Local Run)

```bash
conda create -n wm811k python=3.10
conda activate wm811k
pip install -r requirements.txt

# Download LSWMD.pkl to data/
# 1. Run full automated pipeline
python run_pipeline.py --config config.yaml

# 2. Launch interactive dashboard
streamlit run app/streamlit_app.py
```

---

## 🎯 Target Role
Samsung Electronics TSP 평가및분석 / SK Hynix 양산기술 P&T
