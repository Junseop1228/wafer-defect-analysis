# Stage 8 - Streamlit Dashboard
# Tab 1: Pipeline Status
# Tab 2: Model Analysis (Confusion Matrix + SHAP Force Plot + Domain Sidebar)
# Tab 3: SPC Monitoring (p-chart / CUSUM / EWMA + AE anomaly overlay)

import streamlit as st

st.set_page_config(page_title="WM-811K Wafer Defect Analysis", layout="wide")
tab1, tab2, tab3 = st.tabs(["Pipeline Status", "Model Analysis", "SPC Monitoring"])
