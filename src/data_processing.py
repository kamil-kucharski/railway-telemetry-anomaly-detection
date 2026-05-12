"""Simple data loading and cleaning helpers for MetroPT-3 telemetry."""

from pathlib import Path

import pandas as pd


TIMESTAMP_COLUMN = "timestamp"

ANALOG_SENSOR_COLUMNS = [
    "TP2",
    "TP3",
    "H1",
    "DV_pressure",
    "Reservoirs",
    "Oil_temperature",
    "Motor_current",
]

DIGITAL_SENSOR_COLUMNS = [
    "COMP",
    "DV_eletric",
    "Towers",
    "MPG",
    "LPS",
    "Pressure_switch",
    "Oil_level",
    "Caudal_impulses",
]

SENSOR_COLUMNS = ANALOG_SENSOR_COLUMNS + DIGITAL_SENSOR_COLUMNS


def load_raw_data(file_path="data/MetroPT3.csv"):
    """Load the raw MetroPT-3 CSV file."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {path}. Place the MetroPT-3 CSV file in the data folder."
        )

    return pd.read_csv(path)


def remove_index_columns(data):
    """Remove index columns that were saved into the CSV file."""
    clean_data = data.copy()
    index_columns = [column for column in clean_data.columns if column.startswith("Unnamed:")]

    if index_columns:
        clean_data = clean_data.drop(columns=index_columns)

    return clean_data


def check_required_columns(data, timestamp_column=TIMESTAMP_COLUMN):
    """Check that the dataset contains the timestamp and all expected sensors."""
    required_columns = [timestamp_column] + SENSOR_COLUMNS
    missing_columns = [column for column in required_columns if column not in data.columns]

    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"Missing required columns: {missing_text}")


def prepare_telemetry_data(data, timestamp_column=TIMESTAMP_COLUMN):
    """Clean raw telemetry and keep the 15 MetroPT-3 sensor columns."""
    clean_data = remove_index_columns(data)
    check_required_columns(clean_data, timestamp_column)

    clean_data = clean_data[[timestamp_column] + SENSOR_COLUMNS].copy()
    clean_data[timestamp_column] = pd.to_datetime(clean_data[timestamp_column], errors="coerce")
    clean_data = clean_data.dropna(subset=[timestamp_column])
    clean_data = clean_data.sort_values(timestamp_column)
    clean_data = clean_data.reset_index(drop=True)

    return clean_data


def get_sensor_columns(data):
    """Return the expected MetroPT-3 sensor columns that exist in the dataframe."""
    return [column for column in SENSOR_COLUMNS if column in data.columns]


def aggregate_to_minutes(data, timestamp_column=TIMESTAMP_COLUMN, sensor_columns=None):
    """Aggregate telemetry into one-minute average sensor values."""
    if sensor_columns is None:
        sensor_columns = get_sensor_columns(data)

    missing_columns = [column for column in sensor_columns if column not in data.columns]
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise ValueError(f"Cannot aggregate missing columns: {missing_text}")

    minute_data = (
        data.set_index(timestamp_column)[sensor_columns]
        .resample("1min")
        .mean()
        .dropna(how="all")
        .reset_index()
    )

    return minute_data


def load_clean_minute_data(file_path="data/MetroPT3.csv"):
    """Load, clean, and aggregate MetroPT-3 telemetry to one-minute records."""
    raw_data = load_raw_data(file_path)
    clean_data = prepare_telemetry_data(raw_data)
    minute_data = aggregate_to_minutes(clean_data)

    return minute_data
