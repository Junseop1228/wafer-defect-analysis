# WM-811K Wafer Defect Classification

> 웨이퍼 맵 불량 패턴을 분류하고 원인 공정을 추정하는 모델을 구축했으며,
> SPC 관리도와 연결해 공정 이상 시점을 사전 감지하는 자동화 파이프라인을 직접 설계했습니다.

## Pipeline Overview

`
[Labeled 172,950] → EDA → Feature Eng.(28) → Gate1
                                                ↓
                              Binary Clf (Normal vs Defect) → Gate2
                                                ↓
                           7-class Pattern Clf (ML / CNN / Hybrid) → Gate3
                                                ↓
                              SHAP + Domain Interpretation
                                                ↓
                              SPC + Autoencoder Anomaly Detection
                                                ↓
                              Streamlit Dashboard (Cloud)
`

## Key Results

| Model | Macro F1 | Scratch Recall | Donut Recall |
|-------|----------|---------------|--------------|
| RF (28 features) | TBD | TBD | TBD |
| Custom CNN | TBD | TBD | TBD |
| Hybrid | TBD | TBD | TBD |

## Defect → Process Interpretation

| Pattern | Suspect Process | Action |
|---------|----------------|--------|
| Edge-Ring | CMP pad wear | Shorten pad replacement cycle |
| Center | CVD gas supply | Check showerhead nozzle |
| Donut | RTP heater zone | Profile heater zone output |
| Scratch | Handling robot slip | Check transfer path log |
| Local | Chamber particle | Lot consistency → contamination type |
| Random | Process instability | Cpk monitoring |
| Edge-Local | Chuck contamination | Identify by cx/cy direction |

## Quick Start

`ash
conda create -n wm811k python=3.10
conda activate wm811k
pip install -r requirements.txt
# Download LSWMD.pkl to data/
python run_pipeline.py --config config.yaml
`

## Live Demo
*Streamlit Cloud link — TBD after deployment*

## Target Role
Samsung Electronics TSP 평가및분석 / SK Hynix 양산기술 P&T
