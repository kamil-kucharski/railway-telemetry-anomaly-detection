"""Simple baseline modeling helpers for high-risk state classification."""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def create_high_risk_label(data, score_column="anomaly_score", quantile=0.95):
    """Create an educational high-risk label from the highest anomaly scores."""
    labeled_data = data.copy()
    threshold = labeled_data[score_column].quantile(quantile)
    labeled_data["high_risk"] = labeled_data[score_column] >= threshold

    return labeled_data, threshold


def split_features_and_target(data, feature_columns, target_column="high_risk"):
    """Split selected features and target into train and test sets."""
    x = data[feature_columns]
    y = data[target_column]

    return train_test_split(x, y, test_size=0.2, random_state=42, stratify=y)


def train_logistic_regression(x_train, y_train):
    """Train a scaled Logistic Regression baseline."""
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)

    model = LogisticRegression(max_iter=1000, class_weight="balanced")
    model.fit(x_train_scaled, y_train)

    return model, scaler


def train_random_forest(x_train, y_train):
    """Train a Random Forest baseline."""
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(x_train, y_train)

    return model


def evaluate_classifier(model_name, y_true, y_pred):
    """Return common classification metrics as a one-row dataframe."""
    metrics = {
        "model": model_name,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
    }

    return pd.DataFrame([metrics])

