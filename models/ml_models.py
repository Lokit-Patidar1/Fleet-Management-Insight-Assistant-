"""
models/ml_models.py
Machine Learning models for Fleet Management:
- Regression: Predict maintenance cost
- Anomaly Detection: Isolation Forest for unusual vehicle behavior
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
# REGRESSION MODEL
# ─────────────────────────────────────────────

REGRESSION_FEATURES = [
    "distance_travelled", "fuel_consumption",
    "driver_behavior_score", "fuel_efficiency"
]
REGRESSION_TARGET = "maintenance_cost"


def prepare_regression_data(df: pd.DataFrame):
    """Extract features and target for regression."""
    available = [f for f in REGRESSION_FEATURES if f in df.columns]
    if REGRESSION_TARGET not in df.columns or not available:
        return None, None, available

    X = df[available].dropna()
    y = df.loc[X.index, REGRESSION_TARGET]
    return X, y, available


def train_regression_model(df: pd.DataFrame):
    """
    Train a Random Forest Regressor to predict maintenance cost.
    Returns: model, metrics dict, feature list
    """
    X, y, features = prepare_regression_data(df)
    if X is None or len(X) < 10:
        return None, {"error": "Not enough data to train regression model."}, features

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "MAE": round(mean_absolute_error(y_test, y_pred), 2),
        "RMSE": round(np.sqrt(mean_squared_error(y_test, y_pred)), 2),
        "R2 Score": round(r2_score(y_test, y_pred), 4),
        "Test Samples": len(y_test),
        "Feature Importances": dict(zip(features, model.feature_importances_.round(4).tolist()))
    }

    return model, metrics, features


def predict_maintenance(model, features: list, input_vals: dict) -> float:
    """Predict maintenance cost for given input values."""
    row = {f: input_vals.get(f, 0) for f in features}
    X_input = pd.DataFrame([row])
    return round(float(model.predict(X_input)[0]), 2)


# ─────────────────────────────────────────────
# ANOMALY DETECTION MODEL
# ─────────────────────────────────────────────

ANOMALY_FEATURES = [
    "distance_travelled", "fuel_consumption",
    "maintenance_cost", "driver_behavior_score",
    "fuel_efficiency"
]


def detect_anomalies(df: pd.DataFrame, contamination: float = 0.1) -> pd.DataFrame:
    """
    Use Isolation Forest to detect anomalous vehicle records.
    Returns DataFrame with 'anomaly' column (True = anomaly).
    """
    available = [f for f in ANOMALY_FEATURES if f in df.columns]
    if not available or len(df) < 5:
        df["anomaly"] = False
        df["anomaly_score"] = 0.0
        return df

    X = df[available].fillna(df[available].median())

    clf = IsolationForest(
        n_estimators=100,
        contamination=contamination,
        random_state=42
    )
    clf.fit(X)

    df = df.copy()
    df["anomaly"] = clf.predict(X) == -1
    df["anomaly_score"] = -clf.score_samples(X)  # higher = more anomalous
    df["anomaly_score"] = df["anomaly_score"].round(4)

    return df


def get_anomaly_summary(df: pd.DataFrame) -> dict:
    """Return summary of anomalies by vehicle."""
    if "anomaly" not in df.columns:
        return {}

    anomaly_df = df[df["anomaly"]]
    if anomaly_df.empty:
        return {"total_anomalies": 0, "vehicles_affected": []}

    summary = {
        "total_anomalies": int(anomaly_df["anomaly"].sum()),
        "vehicles_affected": anomaly_df["vehicle_id"].unique().tolist() if "vehicle_id" in anomaly_df.columns else []
    }
    return summary
