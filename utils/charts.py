"""
utils/charts.py
Plotly chart builders for the Fleet Management Dashboard
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PALETTE = {
    "primary": "#1B4F72",
    "secondary": "#2E86C1",
    "accent": "#F39C12",
    "danger": "#E74C3C",
    "success": "#27AE60",
    "light": "#D6EAF8",
}


def fuel_consumption_bar(df: pd.DataFrame) -> go.Figure:
    """Bar chart: Average fuel consumption per vehicle."""
    grouped = df.groupby("vehicle_id")["fuel_consumption"].mean().reset_index()
    grouped.columns = ["Vehicle ID", "Avg Fuel (L)"]
    fig = px.bar(
        grouped, x="Vehicle ID", y="Avg Fuel (L)",
        color="Avg Fuel (L)",
        color_continuous_scale=["#D6EAF8", "#1B4F72"],
        title="🔥 Average Fuel Consumption by Vehicle",
        text_auto=".1f"
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#1B4F72"),
        title_font_size=16,
        coloraxis_showscale=False
    )
    return fig


def maintenance_cost_line(df: pd.DataFrame) -> go.Figure:
    """Line chart: Maintenance cost over time per vehicle."""
    if "date" not in df.columns:
        return go.Figure()
    fig = px.line(
        df, x="date", y="maintenance_cost", color="vehicle_id",
        title="🔧 Maintenance Cost Over Time",
        labels={"maintenance_cost": "Maintenance Cost (₹)", "date": "Date", "vehicle_id": "Vehicle"}
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#1B4F72"), title_font_size=16
    )
    return fig


def fuel_efficiency_scatter(df: pd.DataFrame) -> go.Figure:
    """Scatter: Distance vs Fuel Efficiency, colored by driver score."""
    if "fuel_efficiency" not in df.columns:
        return go.Figure()
    fig = px.scatter(
        df, x="distance_travelled", y="fuel_efficiency",
        color="driver_behavior_score",
        size="maintenance_cost",
        hover_data=["vehicle_id", "date"] if "date" in df.columns else ["vehicle_id"],
        color_continuous_scale="Blues",
        title="⚡ Distance vs Fuel Efficiency (bubble = maintenance cost)",
        labels={
            "distance_travelled": "Distance Travelled (km)",
            "fuel_efficiency": "Fuel Efficiency (km/L)",
            "driver_behavior_score": "Driver Score"
        }
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#1B4F72"), title_font_size=16
    )
    return fig


def driver_score_gauge(df: pd.DataFrame) -> go.Figure:
    """Gauge chart: Average driver behavior score."""
    avg_score = df["driver_behavior_score"].mean() if "driver_behavior_score" in df.columns else 0
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=round(avg_score, 1),
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": "🚗 Avg Driver Behavior Score", "font": {"size": 16, "color": "#1B4F72"}},
        delta={"reference": 80, "increasing": {"color": "#27AE60"}, "decreasing": {"color": "#E74C3C"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#1B4F72"},
            "bar": {"color": "#2E86C1"},
            "steps": [
                {"range": [0, 60], "color": "#FADBD8"},
                {"range": [60, 80], "color": "#FDEBD0"},
                {"range": [80, 100], "color": "#D5F5E3"},
            ],
            "threshold": {"line": {"color": "#E74C3C", "width": 4}, "thickness": 0.75, "value": 60}
        }
    ))
    fig.update_layout(
        height=260,
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#1B4F72")
    )
    return fig


def anomaly_scatter(df: pd.DataFrame) -> go.Figure:
    """Scatter plot highlighting anomalous records."""
    if "anomaly" not in df.columns or "fuel_efficiency" not in df.columns:
        return go.Figure()
    fig = px.scatter(
        df,
        x="distance_travelled",
        y="fuel_consumption",
        color=df["anomaly"].map({True: "Anomaly 🔴", False: "Normal 🟢"}),
        symbol=df["anomaly"].map({True: "x", False: "circle"}),
        hover_data=["vehicle_id", "maintenance_cost", "driver_behavior_score"],
        color_discrete_map={"Anomaly 🔴": "#E74C3C", "Normal 🟢": "#2E86C1"},
        title="🚨 Anomaly Detection — Vehicle Records",
        labels={
            "distance_travelled": "Distance (km)",
            "fuel_consumption": "Fuel Consumption (L)",
            "color": "Status"
        }
    )
    fig.update_layout(
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#1B4F72"), title_font_size=16,
        legend_title_text="Status"
    )
    return fig


def feature_importance_bar(importances: dict) -> go.Figure:
    """Horizontal bar chart for feature importances."""
    items = sorted(importances.items(), key=lambda x: x[1])
    labels = [i[0].replace("_", " ").title() for i in items]
    values = [i[1] for i in items]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker_color="#2E86C1",
        text=[f"{v:.3f}" for v in values], textposition="auto"
    ))
    fig.update_layout(
        title="📊 Feature Importance (Regression Model)",
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#1B4F72"), title_font_size=16,
        xaxis_title="Importance Score", yaxis_title="Feature",
        height=280
    )
    return fig


def vehicle_type_pie(df: pd.DataFrame) -> go.Figure:
    """Pie chart of vehicle types in fleet."""
    if "vehicle_type" not in df.columns:
        return go.Figure()
    counts = df.groupby("vehicle_type")["vehicle_id"].nunique().reset_index()
    counts.columns = ["Vehicle Type", "Count"]
    fig = px.pie(
        counts, names="Vehicle Type", values="Count",
        title="🚌 Fleet Composition by Vehicle Type",
        color_discrete_sequence=px.colors.sequential.Blues_r,
        hole=0.4
    )
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#1B4F72"), title_font_size=16
    )
    return fig
