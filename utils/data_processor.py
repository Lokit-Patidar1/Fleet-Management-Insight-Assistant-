"""
utils/data_processor.py
Data loading, cleaning, and feature engineering for Fleet Management Insight Assistant
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def load_data(filepath: str = None, df: pd.DataFrame = None) -> pd.DataFrame:
    """Load dataset from file path or return a provided DataFrame."""
    if df is not None:
        return df.copy()
    if filepath:
        return pd.read_csv(filepath, parse_dates=["date"])
    raise ValueError("Either filepath or df must be provided.")


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full preprocessing pipeline:
    - Handle missing values
    - Parse dates
    - Feature engineering
    - Normalize numerical columns
    """
    df = df.copy()

    # --- 1. Parse dates ---
    if "date" in df.columns and not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # --- 2. Handle missing values ---
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()

    for col in num_cols:
        df[col] = df[col].fillna(df[col].median())
    for col in cat_cols:
        df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else "Unknown")

    # --- 3. Feature Engineering ---
    if "distance_travelled" in df.columns and "fuel_consumption" in df.columns:
        df["fuel_efficiency"] = df["distance_travelled"] / df["fuel_consumption"].replace(0, np.nan)
        df["fuel_efficiency"] = df["fuel_efficiency"].fillna(0).round(3)

    if "maintenance_cost" in df.columns and "distance_travelled" in df.columns:
        df["cost_per_km"] = df["maintenance_cost"] / df["distance_travelled"].replace(0, np.nan)
        df["cost_per_km"] = df["cost_per_km"].fillna(0).round(3)

    if "date" in df.columns:
        df["month"] = df["date"].dt.month
        df["week"] = df["date"].dt.isocalendar().week.astype(int)
        df["day_of_week"] = df["date"].dt.dayofweek

    # --- 4. Normalize numerical columns (add _norm suffix) ---
    cols_to_normalize = [
        "distance_travelled", "fuel_consumption",
        "maintenance_cost", "driver_behavior_score",
        "fuel_efficiency", "cost_per_km"
    ]
    cols_to_normalize = [c for c in cols_to_normalize if c in df.columns]

    if cols_to_normalize:
        scaler = MinMaxScaler()
        normalized = scaler.fit_transform(df[cols_to_normalize])
        norm_df = pd.DataFrame(normalized, columns=[f"{c}_norm" for c in cols_to_normalize], index=df.index)
        df = pd.concat([df, norm_df], axis=1)

    return df


def get_summary_stats(df: pd.DataFrame) -> dict:
    """Return key fleet summary statistics."""
    stats = {}
    if "vehicle_id" in df.columns:
        stats["total_vehicles"] = df["vehicle_id"].nunique()
    if "fuel_consumption" in df.columns:
        stats["avg_fuel_consumption"] = round(df["fuel_consumption"].mean(), 2)
        stats["max_fuel_consumption"] = round(df["fuel_consumption"].max(), 2)
        stats["min_fuel_consumption"] = round(df["fuel_consumption"].min(), 2)
    if "maintenance_cost" in df.columns:
        stats["avg_maintenance_cost"] = round(df["maintenance_cost"].mean(), 2)
        stats["total_maintenance_cost"] = round(df["maintenance_cost"].sum(), 2)
    if "distance_travelled" in df.columns:
        stats["avg_distance"] = round(df["distance_travelled"].mean(), 2)
        stats["total_distance"] = round(df["distance_travelled"].sum(), 2)
    if "fuel_efficiency" in df.columns:
        stats["avg_fuel_efficiency"] = round(df["fuel_efficiency"].mean(), 2)
    if "driver_behavior_score" in df.columns:
        stats["avg_driver_score"] = round(df["driver_behavior_score"].mean(), 2)
    return stats
