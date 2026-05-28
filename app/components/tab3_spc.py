import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from utils import load_spc_timeseries, load_config
import numpy as np

def render_tab3():
    st.header("Statistical Process Control (SPC) Monitoring")
    
    df_spc = load_spc_timeseries()
    cfg = load_config()
    
    if df_spc.empty:
        st.info("SPC timeseries data not found. Run data generation script.")
        return
        
    classes = cfg.get("classes", {}).get("names", [])
    
    col1, col2 = st.columns([1, 4])
    
    with col1:
        st.markdown("<div class='stCard'>", unsafe_allow_html=True)
        st.subheader("Control Panel")
        selected_class = st.selectbox("Monitor Defect Type", classes + ["Total_Defect"])
        
        rate_col = f"{selected_class}_rate"
        
        # Slider for lot observation window
        max_lots = len(df_spc)
        window = st.slider("Observation Window (Lots)", 100, max_lots, min(1000, max_lots))
        
        st.markdown("---")
        st.markdown("### SPC Parameters")
        cusum_h = st.number_input("CUSUM h (Threshold)", value=cfg.get('spc', {}).get('cusum_h', 5.0))
        ewma_lam = st.number_input("EWMA lambda", value=cfg.get('spc', {}).get('ewma_lambda', 0.2))
        
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col2:
        if rate_col not in df_spc.columns:
            st.error(f"Column {rate_col} not found in SPC data.")
            return
            
        # Get subset of data
        df_subset = df_spc.tail(window).reset_index(drop=True)
        defect_rates = df_subset[rate_col].values
        lots = df_subset['lotName'].values
        
        # Calculate Phase 1 Baseline on the oldest 70% of the entire dataset's zeros? 
        # For visualization, we will just use the subset's mean as a mock baseline or 0 if strict
        mu = 0.0 if cfg.get("spc", {}).get("strict_phase1_zero_defect", True) else np.mean(df_spc[rate_col])
        sigma = np.std(df_spc[rate_col]) if np.std(df_spc[rate_col]) > 0 else 1e-4
        
        # Precompute charts
        ucl_shewhart = mu + 3 * sigma
        
        # CUSUM
        k = 0.5 * sigma
        H = cusum_h * sigma
        c_plus = np.zeros_like(defect_rates)
        for i in range(1, len(defect_rates)):
            c_plus[i] = max(0, c_plus[i-1] + (defect_rates[i] - mu - k))
            
        # EWMA
        z = np.zeros_like(defect_rates)
        z[0] = mu
        ucl_ewma = np.zeros_like(defect_rates)
        for i in range(1, len(defect_rates)):
            z[i] = ewma_lam * defect_rates[i] + (1 - ewma_lam) * z[i-1]
            limit_term = 3 * sigma * np.sqrt((ewma_lam / (2 - ewma_lam)) * (1 - (1 - ewma_lam)**(2*i)))
            ucl_ewma[i] = mu + limit_term
            
        # Find violations
        v_shewhart = np.where(defect_rates > ucl_shewhart)[0]
        v_cusum = np.where(c_plus > H)[0]
        v_ewma = np.where(z > ucl_ewma)[0]
        
        # Build Plotly Subplots
        fig = make_subplots(rows=4, cols=1, shared_xaxes=True, 
                            vertical_spacing=0.05,
                            subplot_titles=(
                                "Shewhart p-Chart (Spike Detection)", 
                                "CUSUM Chart (Drift Detection)", 
                                "EWMA Chart (Trend Detection)",
                                "Autoencoder Anomaly Score (Dual Detection)"
                            ))
                            
        # 1. Shewhart
        fig.add_trace(go.Scatter(x=lots, y=defect_rates, mode='lines+markers', name='Defect Rate',
                                 marker=dict(size=4, color='#4da6ff')), row=1, col=1)
        fig.add_trace(go.Scatter(x=lots, y=[ucl_shewhart]*len(lots), mode='lines', name='UCL',
                                 line=dict(color='red', dash='dash')), row=1, col=1)
        if len(v_shewhart) > 0:
            fig.add_trace(go.Scatter(x=lots[v_shewhart], y=defect_rates[v_shewhart], mode='markers',
                                     marker=dict(size=10, color='red', symbol='circle-open', line_width=2),
                                     name='Violation (Shewhart)'), row=1, col=1)
                                     
        # 2. CUSUM
        fig.add_trace(go.Scatter(x=lots, y=c_plus, mode='lines', name='CUSUM C+', line=dict(color='#00e676')), row=2, col=1)
        fig.add_trace(go.Scatter(x=lots, y=[H]*len(lots), mode='lines', name='CUSUM H',
                                 line=dict(color='red', dash='dash')), row=2, col=1)
        if len(v_cusum) > 0:
            fig.add_trace(go.Scatter(x=lots[v_cusum], y=c_plus[v_cusum], mode='markers',
                                     marker=dict(size=10, color='red', symbol='circle-open', line_width=2),
                                     name='Violation (CUSUM)'), row=2, col=1)
                                     
        # 3. EWMA
        fig.add_trace(go.Scatter(x=lots, y=z, mode='lines', name='EWMA Z', line=dict(color='#ff9900')), row=3, col=1)
        fig.add_trace(go.Scatter(x=lots, y=ucl_ewma, mode='lines', name='EWMA UCL',
                                 line=dict(color='red', dash='dash')), row=3, col=1)
        if len(v_ewma) > 0:
            fig.add_trace(go.Scatter(x=lots[v_ewma], y=z[v_ewma], mode='markers',
                                     marker=dict(size=10, color='red', symbol='circle-open', line_width=2),
                                     name='Violation (EWMA)'), row=3, col=1)
                                     
        # 4. AE Score
        ae_scores = df_subset['AE_anomaly_score'].values
        ae_threshold = cfg.get("autoencoder", {}).get("threshold", 0.0)
        v_ae = np.where(ae_scores > ae_threshold)[0]
        
        fig.add_trace(go.Scatter(x=lots, y=ae_scores, mode='lines', name='AE Score', line=dict(color='#b366ff')), row=4, col=1)
        fig.add_trace(go.Scatter(x=lots, y=[ae_threshold]*len(lots), mode='lines', name='AE Threshold',
                                 line=dict(color='red', dash='dash')), row=4, col=1)
        if len(v_ae) > 0:
            fig.add_trace(go.Scatter(x=lots[v_ae], y=ae_scores[v_ae], mode='markers',
                                     marker=dict(size=10, color='red', symbol='circle-open', line_width=2),
                                     name='Violation (AE)'), row=4, col=1)
                                     
        fig.update_layout(height=1000, template='plotly_dark', hovermode='x unified',
                          margin=dict(l=20, r=20, t=40, b=20))
        
        st.plotly_chart(fig, use_container_width=True)
