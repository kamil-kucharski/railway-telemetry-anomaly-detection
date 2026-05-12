"""Simple data loading and cleaning helpers for MetroPT-3 telemetry."""

from pathlib import Path

import pandas as pd


def load_raw_data(file_path="data/MetroPT3.csv"):
    """Load the raw MetroPT-3 CSV file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}. Place the MetroPT-3 CSV file in the data folder."
        )

    return pd.read_csv(path)


def find_timestamp_column(data):
    """Find the timestamp column using a few common column names."""
    possible_names = ["timestamp", "Timestamp", "time", "Time", "datetime", "DateTime"]

    for column in possible_names:
        if column in data.columns:
            return column

    raise ValueError("No timestamp column was found. Please check the dataset columns.")


def prepare_telemetry_data(data, timestamp_column=None):
    """Parse timestamps, sort records, and remove rows without valid timestamps."""
    clean_data = data.copy()

    if timestamp_column is None:
        timestamp_column = find_timestamp_column(clean_data)

    clean_data[timestamp_column] = pd.to_datetime(clean_data[timestamp_column], errors="coerce")
    clean_data = clean_data.dropna(subset=[timestamp_column])
    clean_data = clean_data.sort_values(timestamp_column)
    clean_data = clean_data.reset_index(drop=True)

    return clean_data


def get_numeric_sensor_columns(data, timestamp_column="timestamp"):
    """Return numeric columns that can be used as sensor features."""
    numeric_columns = data.select_dtypes(include="number").columns.tolist()

    columns_to_skip = {"index", timestamp_column}
    sensor_columns = [column for column in numeric_columns if column not in columns_to_skip]

    return sensor_columns


def aggregate_to_minutes(data, timestamp_column="timestamp", sensor_columns=None):
    """Aggregate high-frequency telemetry into one-minute average values."""
    if sensor_columns is None:
        sensor_columns = get_numeric_sensor_columns(data, timestamp_column)

    minute_data = (
        data.set_index(timestamp_column)[sensor_columns]
        .resample("1min")
        .mean()
        .dropna(how="all")
        .reset_index()
    )

    return minute_data

