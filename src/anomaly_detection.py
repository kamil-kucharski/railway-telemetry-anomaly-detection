"""Anomaly detection helpers based on Isolation Forest."""

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


def fit_isolation_forest(data, feature_columns, contamination=0.02, random_state=42):
    """Scale features and train an Isolation Forest model."""
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(data[feature_columns])

    model = IsolationForest(
        contamination=contamination,
        random_state=random_state,
        n_estimators=100,
    )
    model.fit(scaled_features)

    return model, scaler


def add_anomaly_predictions(data, feature_columns, model, scaler):
    """Add anomaly labels and anomaly scores to a dataframe."""
    result = data.copy()
    scaled_features = scaler.transform(result[feature_columns])

    predictions = model.predict(scaled_features)
    result["anomaly_label"] = predictions
    result["is_anomaly"] = result["anomaly_label"] == -1
    result["anomaly_score"] = -model.decision_function(scaled_features)

    return result


def export_anomaly_events(data, output_path="outputs/reports/anomaly_events.csv"):
    """Save detected anomaly rows to a CSV report."""
    anomaly_events = data[data["is_anomaly"]].copy()
    anomaly_events.to_csv(output_path, index=False)

    return anomaly_events

