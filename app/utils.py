import streamlit as st
import pandas as pd
import yaml

@st.cache_data
def load_config():
    with open('config.yaml', 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

@st.cache_data
def load_metrics():
    try:
        return pd.read_csv('results/metrics.csv')
    except Exception as e:
        st.error(f"Failed to load metrics.csv: {e}")
        return pd.DataFrame()

@st.cache_data
def load_spc_timeseries():
    try:
        return pd.read_csv('data/spc_timeseries.csv')
    except Exception as e:
        st.error(f"Failed to load spc_timeseries.csv: {e}")
        return pd.DataFrame()
