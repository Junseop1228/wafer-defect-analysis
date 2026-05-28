import streamlit as st
import os

st.set_page_config(
    page_title="WM-811K Wafer Defect Analysis",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark theme overrides (Streamlit uses config.toml normally, but we inject CSS)
st.markdown("""
<style>
    /* Dark glassmorphism card effect */
    .stCard {
        background-color: rgba(30, 30, 40, 0.8);
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
        margin-bottom: 20px;
        border: 1px solid rgba(255,255,255,0.05);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #4da6ff;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #a0a0a0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    h1, h2, h3 {
        font-family: 'Inter', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

from components.tab1_pipeline import render_tab1
from components.tab2_model import render_tab2
from components.tab3_spc import render_tab3

def main():
    st.title("🔍 WM-811K Defect Analysis & SPC Dashboard")
    st.markdown("Automated pipeline for semiconductor wafer map classification and anomaly detection.")
    
    st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/2/25/Semiconductor_wafer.jpg", use_container_width=True)
    st.sidebar.title("Navigation")
    
    tab1, tab2, tab3 = st.tabs([
        "📊 Pipeline & Metrics", 
        "🧠 Model Diagnostics", 
        "📈 SPC & Anomaly Detection"
    ])
    
    with tab1:
        render_tab1()
        
    with tab2:
        render_tab2()
        
    with tab3:
        render_tab3()

if __name__ == "__main__":
    main()
