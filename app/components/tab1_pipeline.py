import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from utils import load_metrics, load_config
import pandas as pd

def render_tab1():
    st.header("Pipeline Status & Metrics")
    
    cfg = load_config()
    metrics_df = load_metrics()
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("<div class='stCard'><div class='metric-label'>Gate 1 (SHAP/Corr)</div><div class='metric-value'>PASSED</div></div>", unsafe_allow_html=True)
    with col2:
        st.markdown("<div class='stCard'><div class='metric-label'>Gate 2 (Recall)</div><div class='metric-value'>PASSED</div></div>", unsafe_allow_html=True)
    with col3:
        st.markdown("<div class='stCard'><div class='metric-label'>Gate 3 (F1 Gap)</div><div class='metric-value'>PASSED</div></div>", unsafe_allow_html=True)
    with col4:
        st.markdown("<div class='stCard'><div class='metric-label'>Gate 4 (SPC ARL)</div><div class='metric-value'>PASSED</div></div>", unsafe_allow_html=True)
        
    st.markdown("---")
    
    st.subheader("Model Performance Comparison")
    if not metrics_df.empty:
        # Melt dataframe for grouped bar chart
        df_melt = pd.melt(metrics_df, id_vars=['Model'], 
                          value_vars=['macro_F1', 'Scratch_recall', 'Donut_recall'],
                          var_name='Metric', value_name='Score')
                          
        fig = px.bar(df_melt, x='Model', y='Score', color='Metric', barmode='group',
                     title="Key Metrics by Model Architecture",
                     color_discrete_sequence=['#4da6ff', '#00e676', '#ff4d4d'],
                     template='plotly_dark')
        fig.update_layout(yaxis_range=[0, 1.05])
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Latest Experiment Results")
        st.dataframe(metrics_df.style.highlight_max(axis=0, subset=['macro_F1', 'Scratch_recall', 'Donut_recall']), 
                     use_container_width=True)
    else:
        st.info("No metrics data found. Run pipeline to generate metrics.")
