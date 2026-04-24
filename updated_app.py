
# (Modified app.py with pyarrow-safe rendering)

import os
os.environ["STREAMLIT_DATAFRAME_USE_PYARROW"] = "false"
import sys
import streamlit as st
import pandas as pd
import numpy as np

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.dirname(__file__))

from utils.data_processor import load_data, preprocess_data, get_summary_stats
from models.ml_models import (
    train_regression_model,
    detect_anomalies,
    get_anomaly_summary,
    predict_maintenance,
)
from utils.charts import (
    fuel_consumption_bar,
    maintenance_cost_line,
    fuel_efficiency_scatter,
    driver_score_gauge,
    anomaly_scatter,
    feature_importance_bar,
    vehicle_type_pie,
)
from rag.rag_pipeline import build_documents, FleetVectorStore, generate_rag_answer

st.set_page_config(page_title="Fleet Management Insight Assistant", layout="wide")

# Load sample data
df = pd.read_csv("data/fleet_data.csv")

# Tabs
tab1, tab2 = st.tabs(["ML", "Data"])

with tab1:
    df_anomaly = detect_anomalies(df.copy(), contamination=0.1)
    if "anomaly" in df_anomaly.columns:
        ano_records = df_anomaly[df_anomaly["anomaly"]].copy()
        if not ano_records.empty:
            display_cols = [c for c in ano_records.columns]
            st.markdown("### Anomalous Records")

            # SAFE HTML RENDERING (FIXED)
            table_html = ano_records[display_cols].to_html(index=False)
            st.markdown(
                f'<div style="overflow-x:auto">{table_html}</div>',
                unsafe_allow_html=True
            )

with tab2:
    display_cols = df.columns.tolist()

    # SAFE HTML RENDERING (FIXED)
    table_html = df[display_cols].to_html(index=False)
    st.markdown(
        f'<div style="overflow-x:auto">{table_html}</div>',
        unsafe_allow_html=True
    )
