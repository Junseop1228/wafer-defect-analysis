import streamlit as st
import pandas as pd
import yaml

import os

@st.cache_data
def load_config():
    path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
    with open(path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

@st.cache_data
def load_metrics():
    path = os.path.join(os.path.dirname(__file__), '..', 'results', 'metrics.csv')
    try:
        return pd.read_csv(path)
    except Exception as e:
        st.error(f"Failed to load metrics.csv: {e}")
        return pd.DataFrame()

@st.cache_data
def load_spc_timeseries():
    path = os.path.join(os.path.dirname(__file__), '..', 'data', 'spc_timeseries.csv')
    try:
        return pd.read_csv(path)
    except Exception as e:
        st.error(f"Failed to load spc_timeseries.csv: {e}")
        return pd.DataFrame()
