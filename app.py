"""
app.py
Fleet Management Insight Assistant
Main Streamlit Application
"""

import os
os.environ["STREAMLIT_DATAFRAME_USE_PYARROW"] = "false"
import sys
import streamlit as st
import pandas as pd
import numpy as np

from dotenv import load_dotenv
load_dotenv()

# ── Path setup ──────────────────────────────
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

# ════════════════════════════════════════════
# PAGE CONFIG
# ════════════════════════════════════════════
st.set_page_config(
    page_title="Fleet Management Insight Assistant",
    page_icon="🚛",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ════════════════════════════════════════════
# CUSTOM CSS
# ════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&family=DM+Sans:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

h1, h2, h3 {
    font-family: 'Space Grotesk', sans-serif;
    color: #1B4F72;
}

/* Hero header */
.hero-banner {
    background: linear-gradient(135deg, #1B4F72 0%, #2E86C1 60%, #85C1E9 100%);
    border-radius: 16px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    color: white;
    box-shadow: 0 8px 32px rgba(27,79,114,0.25);
}
.hero-banner h1 { color: white; font-size: 2rem; margin: 0 0 0.4rem 0; }
.hero-banner p  { color: #D6EAF8; margin: 0; font-size: 1rem; }

/* KPI cards */
.kpi-card {
    background: white;
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    box-shadow: 0 4px 18px rgba(27,79,114,0.10);
    border-left: 5px solid #2E86C1;
    margin-bottom: 0.5rem;
}
.kpi-label { font-size: 0.78rem; color: #7F8C8D; text-transform: uppercase; letter-spacing: 0.08em; }
.kpi-value { font-size: 1.8rem; font-weight: 700; color: #1B4F72; margin: 0.1rem 0; }
.kpi-sub   { font-size: 0.8rem; color: #2E86C1; }

/* Section headers */
.section-header {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #1B4F72;
    border-bottom: 2px solid #D6EAF8;
    padding-bottom: 0.4rem;
    margin: 1.2rem 0 0.8rem;
}

/* Query box */
.query-container {
    background: linear-gradient(135deg, #EBF5FB, #FDFEFE);
    border-radius: 14px;
    padding: 1.5rem;
    border: 1px solid #AED6F1;
    margin-bottom: 1rem;
}

/* Answer box */
.answer-box {
    background: #EBF5FB;
    border-left: 4px solid #2E86C1;
    border-radius: 0 12px 12px 0;
    padding: 1rem 1.2rem;
    margin-top: 0.8rem;
    font-size: 0.97rem;
    color: #1B4F72;
}

/* Anomaly badge */
.badge-anomaly { background: #FADBD8; color: #C0392B; border-radius: 6px; padding: 2px 8px; font-size: 0.82rem; font-weight: 600; }
.badge-normal  { background: #D5F5E3; color: #1E8449; border-radius: 6px; padding: 2px 8px; font-size: 0.82rem; font-weight: 600; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1B4F72 0%, #154360 100%);
}
[data-testid="stSidebar"] * { color: white !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label { color: #AED6F1 !important; }
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { color: #85C1E9 !important; }

/* Hide default Streamlit footer */
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════
# HERO BANNER
# ════════════════════════════════════════════
st.markdown("""
<div class="hero-banner">
    <h1>🚛 Fleet Management Insight Assistant</h1>
    <p>AI-powered analytics for fleet operations · Fuel · Maintenance · Driver Behavior · Anomaly Detection</p>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════
# SIDEBAR
# ════════════════════════════════════════════
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("---")

    # --- Data source ---
    st.markdown("### 📂 Data Source")
    data_source = st.radio("Choose data source:", ["Use sample dataset", "Upload CSV"])

    uploaded_file = None
    if data_source == "Upload CSV":
        uploaded_file = st.file_uploader("Upload fleet CSV", type=["csv"])

    st.markdown("---")

    # --- ML Settings ---
    st.markdown("### 🤖 ML Settings")
    contamination = st.slider("Anomaly contamination (%)", 5, 30, 10, 1) / 100

    st.markdown("---")

    # --- LLM Settings (optional) ---
    st.markdown("### 🔑 LLM API (Optional)")
    gemini_key = os.getenv('GEMINI_API_KEY')
    use_llm = st.checkbox("Enable LLM-enhanced answers", value=bool(gemini_key))
    if use_llm and not gemini_key:
        st.warning("Gemini API key not found in environment variables")

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.caption("College Major Project | Fleet Management Insight Assistant | Built with Streamlit + scikit-learn + FAISS + Sentence Transformers")


# ════════════════════════════════════════════
# DATA LOADING (cached)
# ════════════════════════════════════════════
@st.cache_data(show_spinner=False)
def get_raw_data(source: str, file_bytes=None) -> pd.DataFrame:
    if source == "upload" and file_bytes is not None:
        import io
        return pd.read_csv(io.BytesIO(file_bytes))
    return pd.read_csv(
        os.path.join(os.path.dirname(__file__), "data", "fleet_data.csv"),
        parse_dates=["date"]
    )

@st.cache_data(show_spinner=False)
def get_processed_data(raw_df_json: str) -> pd.DataFrame:
    raw_df = pd.read_json(raw_df_json)
    return preprocess_data(raw_df)

@st.cache_resource(show_spinner=False)
def get_vector_store(docs_json: str) -> FleetVectorStore:
    import json
    docs = json.loads(docs_json)
    vs = FleetVectorStore()
    vs.build(docs)
    return vs


# ── Load data ──────────────────────────────
with st.spinner("Loading fleet data..."):
    if data_source == "Upload CSV" and uploaded_file is not None:
        file_bytes = uploaded_file.read()
        raw_df = get_raw_data("upload", file_bytes)
    else:
        raw_df = get_raw_data("sample")

    df = get_processed_data(raw_df.to_json())

if df.empty:
    st.error("Dataset is empty or could not be loaded.")
    st.stop()

# ════════════════════════════════════════════
# TABS
# ════════════════════════════════════════════
tab_dash, tab_ml, tab_query, tab_data = st.tabs([
    "📊 Dashboard", "🤖 ML Insights", "💬 Query Assistant", "🗂️ Raw Data"
])


# ──────────────────────────────────────────
# TAB 1: DASHBOARD
# ──────────────────────────────────────────
with tab_dash:
    stats = get_summary_stats(df)

    # KPI Row
    col1, col2, col3, col4, col5 = st.columns(5)
    kpis = [
        ("🚗 Total Vehicles",   stats.get("total_vehicles", "-"),           "Unique in fleet"),
        ("⛽ Avg Fuel (L)",     stats.get("avg_fuel_consumption", "-"),     "Per trip"),
        ("🔧 Avg Maint. Cost",  f"₹{stats.get('avg_maintenance_cost',0):.0f}", "Per record"),
        ("📍 Total Distance",   f"{stats.get('total_distance',0):,.0f} km", "All vehicles"),
        ("⭐ Avg Driver Score", stats.get("avg_driver_score", "-"),         "Out of 100"),
    ]
    for col, (label, value, sub) in zip([col1, col2, col3, col4, col5], kpis):
        with col:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{value}</div>
                <div class="kpi-sub">{sub}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("")

    # Charts row 1
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(fuel_consumption_bar(df), use_container_width=True)
    with c2:
        st.plotly_chart(maintenance_cost_line(df), use_container_width=True)

    # Charts row 2
    c3, c4, c5 = st.columns([2, 1, 1])
    with c3:
        st.plotly_chart(fuel_efficiency_scatter(df), use_container_width=True)
    with c4:
        st.plotly_chart(driver_score_gauge(df), use_container_width=True)
    with c5:
        if "vehicle_type" in df.columns:
            st.plotly_chart(vehicle_type_pie(df), use_container_width=True)


# ──────────────────────────────────────────
# TAB 2: ML INSIGHTS
# ──────────────────────────────────────────
with tab_ml:
    st.markdown('<div class="section-header">🤖 Machine Learning Models</div>', unsafe_allow_html=True)

    col_reg, col_ano = st.columns(2)

    # --- REGRESSION ---
    with col_reg:
        st.markdown("#### 📈 Regression: Predict Maintenance Cost")
        with st.spinner("Training model..."):
            model, metrics, features = train_regression_model(df)

        if model and "error" not in metrics:
            m1, m2, m3 = st.columns(3)
            m1.metric("MAE", f"₹{metrics['MAE']}")
            m2.metric("RMSE", f"₹{metrics['RMSE']}")
            m3.metric("R² Score", f"{metrics['R2 Score']:.3f}")

            if "Feature Importances" in metrics:
                st.plotly_chart(feature_importance_bar(metrics["Feature Importances"]), use_container_width=True)

            # Prediction form
            st.markdown("##### 🔮 Predict Maintenance Cost")
            with st.form("predict_form"):
                p1, p2 = st.columns(2)
                distance = p1.number_input("Distance Travelled (km)", 100, 600, 300)
                fuel = p2.number_input("Fuel Consumption (L)", 10.0, 100.0, 40.0)
                driver_score = p1.number_input("Driver Behavior Score", 0, 100, 80)
                fuel_eff = fuel and round(distance / fuel, 3) or 0.0
                p2.metric("Auto Fuel Efficiency", f"{fuel_eff:.2f} km/L")

                submit = st.form_submit_button("🔮 Predict", type="primary")
                if submit:
                    pred = predict_maintenance(model, features, {
                        "distance_travelled": distance,
                        "fuel_consumption": fuel,
                        "driver_behavior_score": driver_score,
                        "fuel_efficiency": fuel_eff
                    })
                    st.success(f"**Predicted Maintenance Cost: ₹{pred:,.2f}**")
        else:
            st.warning(metrics.get("error", "Model training failed."))

    # --- ANOMALY DETECTION ---
    with col_ano:
        st.markdown("#### 🚨 Anomaly Detection (Isolation Forest)")
        with st.spinner("Detecting anomalies..."):
            df_anomaly = detect_anomalies(df.copy(), contamination=contamination)

        summary = get_anomaly_summary(df_anomaly)
        a1, a2 = st.columns(2)
        a1.metric("⚠️ Anomalies Found", summary.get("total_anomalies", 0))
        a2.metric("✅ Normal Records", len(df_anomaly) - summary.get("total_anomalies", 0))

        if summary.get("vehicles_affected"):
            st.warning(f"Vehicles with anomalies: **{', '.join(summary['vehicles_affected'])}**")

        st.plotly_chart(anomaly_scatter(df_anomaly), use_container_width=True)

        # Anomaly table
        if "anomaly" in df_anomaly.columns:
            ano_records = df_anomaly[df_anomaly["anomaly"]].copy()
            if not ano_records.empty:
                display_cols = ["vehicle_id", "date", "distance_travelled", "fuel_consumption",
                                "maintenance_cost", "driver_behavior_score", "anomaly_score"]
                display_cols = [c for c in display_cols if c in ano_records.columns]
                st.markdown("##### Anomalous Records")
            table_html = ano_records[display_cols].to_html(index=False)

            st.markdown("##### Anomalous Records")
            st.markdown(
                f'<div style="overflow-x:auto">{table_html}</div>',
                unsafe_allow_html=True
            )


# ──────────────────────────────────────────
# TAB 3: QUERY ASSISTANT (RAG)
# ──────────────────────────────────────────
with tab_query:
    st.markdown('<div class="section-header">💬 Natural Language Query Assistant</div>', unsafe_allow_html=True)

    # Build vector store (cached on df shape)
    with st.spinner("Building knowledge base..."):
        docs = build_documents(df)
        import json
        vs_key = json.dumps([d["text"][:80] for d in docs[:5]])
        vector_store = get_vector_store(json.dumps([{"text": d["text"], "metadata": {}} for d in docs]))

    st.markdown('<div class="query-container">', unsafe_allow_html=True)

    # Example queries
    st.caption("💡 Try these example queries:")
    ex_cols = st.columns(4)
    example_queries = [
        "Which vehicle has highest fuel consumption?",
        "Show vehicles with high maintenance cost",
        "What is the average distance travelled?",
        "Which vehicle is most fuel efficient?",
    ]
    query_input = st.session_state.get("query_input", "")

    for i, (col, eq) in enumerate(zip(ex_cols, example_queries)):
        with col:
            if st.button(f"📌 {eq}", key=f"ex_{i}", use_container_width=True):
                st.session_state["query_input"] = eq

    st.markdown("</div>", unsafe_allow_html=True)

    user_query = st.text_input(
        "🔍 Ask a question about your fleet:",
        value=st.session_state.get("query_input", ""),
        placeholder="e.g. Which vehicle has highest fuel consumption?",
        key="query_box"
    )

    if user_query:
        with st.spinner("Searching fleet knowledge base..."):
            answer = generate_rag_answer(
                query=user_query,
                df=df,
                vector_store=vector_store,
                api_key=gemini_key if gemini_key else None,
                use_llm=use_llm and bool(gemini_key)
            )

        st.markdown('<div class="answer-box">', unsafe_allow_html=True)
        st.markdown(f"**Query:** _{user_query}_")
        st.markdown("**Answer:**")
        st.markdown(answer)
        st.markdown('</div>', unsafe_allow_html=True)

        # Show retrieved context
        with st.expander("🔍 Retrieved Context (FAISS top-5 results)", expanded=False):
            retrieved = vector_store.search(user_query, top_k=5)
            for i, r in enumerate(retrieved, 1):
                st.markdown(f"**Result {i}** (score: {r['score']:.3f})")
                st.caption(r["text"])
                st.divider()


# ──────────────────────────────────────────
# TAB 4: RAW DATA
# ──────────────────────────────────────────
with tab_data:
    st.markdown('<div class="section-header">🗂️ Fleet Dataset</div>', unsafe_allow_html=True)

    # Filters
    fc1, fc2 = st.columns(2)
    if "vehicle_id" in df.columns:
        vehicles = ["All"] + sorted(df["vehicle_id"].unique().tolist())
        selected_vehicle = fc1.selectbox("Filter by Vehicle", vehicles)
    else:
        selected_vehicle = "All"

    if "vehicle_type" in df.columns:
        types = ["All"] + sorted(df["vehicle_type"].unique().tolist())
        selected_type = fc2.selectbox("Filter by Vehicle Type", types)
    else:
        selected_type = "All"

    view_df = df.copy()
    if selected_vehicle != "All":
        view_df = view_df[view_df["vehicle_id"] == selected_vehicle]
    if selected_type != "All" and "vehicle_type" in view_df.columns:
        view_df = view_df[view_df["vehicle_type"] == selected_type]

    st.caption(f"Showing {len(view_df)} of {len(df)} records")

    # Display key columns
    display_cols = [c for c in [
        "vehicle_id", "date", "vehicle_type", "driver_id",
        "distance_travelled", "fuel_consumption", "fuel_efficiency",
        "maintenance_cost", "cost_per_km", "driver_behavior_score"
    ] if c in view_df.columns]

    table_html = view_df[display_cols].reset_index(drop=True).to_html(index=False)

    st.markdown(
        f'<div style="overflow-x:auto">{table_html}</div>',
        unsafe_allow_html=True
    )

    # Download
    csv_out = view_df[display_cols].to_csv(index=False)
    st.download_button(
        "⬇️ Download Filtered Data",
        data=csv_out,
        file_name="fleet_filtered.csv",
        mime="text/csv"
    )
