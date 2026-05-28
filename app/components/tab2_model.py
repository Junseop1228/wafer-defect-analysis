import streamlit as st
import os
from utils import load_config

def render_tab2():
    st.header("Model Diagnostics (Hybrid CNN+ML)")
    cfg = load_config()
    classes = cfg.get("classes", {}).get("names", [])
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.markdown("<div class='stCard'>", unsafe_allow_html=True)
        st.subheader("Select Pattern")
        selected_class = st.selectbox("Defect Type", classes)
        
        st.markdown("### Domain Interpretation")
        if selected_class == "Scratch":
            st.info("Process: Handling / Robot arm\nAction: Check wafer handler calibration.")
        elif selected_class == "Donut":
            st.info("Process: Etch / Deposition\nAction: Check chamber gas flow uniformity.")
        elif selected_class == "Edge-Ring":
            st.info("Process: Etch / CMP\nAction: Check edge exclusion ring or CMP pad.")
        elif selected_class == "Center":
            st.info("Process: Spin Coating / Deposition\nAction: Check nozzle dispensing.")
        elif selected_class == "Loc":
            st.info("Process: Lithography / Particles\nAction: Inspect reticle and cleanroom particles.")
        elif selected_class == "Edge-Loc":
            st.info("Process: Handling / Clamp\nAction: Inspect wafer edge clamps.")
        elif selected_class == "Random":
            st.info("Process: Particles / Dust\nAction: Global cleanroom particle check.")
        else:
            st.info("Normal process variation.")
        st.markdown("</div>", unsafe_allow_html=True)
        
    with col2:
        st.subheader(f"Diagnostic Analysis: {selected_class}")
        
        tab_shap, tab_gradcam, tab_cm = st.tabs(["SHAP Values", "Grad-CAM", "Confusion Matrix"])
        
        with tab_shap:
            shap_path = f"results/figures/shap_{selected_class}.png"
            if os.path.exists(shap_path):
                st.image(shap_path, caption=f"SHAP Feature Importance for {selected_class}", use_container_width=True)
            else:
                st.warning(f"SHAP plot not found at {shap_path}. Run phase 3 evaluation to generate.")
                
        with tab_gradcam:
            gc_path = f"results/figures/gradcam_{selected_class}.png"
            if os.path.exists(gc_path):
                st.image(gc_path, caption=f"Grad-CAM Activation for {selected_class}", use_container_width=True)
            else:
                st.warning(f"Grad-CAM plot not found at {gc_path}. Run phase 3 evaluation to generate.")
                
        with tab_cm:
            cm_path = "results/figures/confusion_matrix.png"
            if os.path.exists(cm_path):
                st.image(cm_path, caption="Hybrid Model Confusion Matrix", use_container_width=True)
            else:
                st.warning(f"Confusion Matrix not found at {cm_path}. Run model evaluation to generate.")
