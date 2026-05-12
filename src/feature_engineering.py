"""Feature engineering helpers for compressor telemetry."""

import pandas as pd


def add_time_features(data, timestamp_column="timestamp"):
    """Add simple calendar features from the timestamp column."""
    feature_data = data.copy()

    feature_data["hour"] = feature_data[timestamp_column].dt.hour
    feature_data["day_of_week"] = feature_data[timestamp_column].dt.dayofweek
    feature_data["month"] = feature_data[timestamp_column].dt.month

    return feature_data


def add_rolling_features(data, sensor_columns, window=5):
    """Add rolling mean and standard deviation for selected sensors."""
    feature_data = data.copy()

    for column in sensor_columns:
        feature_data[f"{column}_rolling_mean"] = feature_data[column].rolling(window).mean()
        feature_data[f"{column}_rolling_std"] = feature_data[column].rolling(window).std()

    return feature_data


def add_difference_features(data, sensor_columns):
    """Add simple change-from-previous-row features for selected sensors."""
    feature_data = data.copy()

    for column in sensor_columns:
        feature_data[f"{column}_diff"] = feature_data[column].diff()

    return feature_data


def create_model_features(data):
    """Return a model-ready table by removing rows with missing feature values."""
    return data.dropna().reset_index(drop=True)

