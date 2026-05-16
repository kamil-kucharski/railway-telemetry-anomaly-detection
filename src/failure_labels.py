"""Helpers for adding known MetroPT-3 failure windows to telemetry data."""

import pandas as pd


FAILURE_WINDOWS = [
    {
        "failure_event": "failure_1_air_leak",
        "start": "2020-04-18 00:00:00",
        "end": "2020-04-18 23:59:59",
    },
    {
        "failure_event": "failure_2_air_leak",
        "start": "2020-05-29 23:30:00",
        "end": "2020-05-30 06:00:00",
    },
    {
        "failure_event": "failure_3_air_leak",
        "start": "2020-06-05 10:00:00",
        "end": "2020-06-07 14:30:00",
    },
    {
        "failure_event": "failure_4_air_leak",
        "start": "2020-07-15 14:30:00",
        "end": "2020-07-15 19:00:00",
    },
]


def add_failure_labels(data, timestamp_column="timestamp", pre_failure_hours=24):
    """Add failure and pre-failure labels based on known MetroPT-3 failure windows."""
    labeled_data = data.copy()

    labeled_data["failure_period"] = False
    labeled_data["pre_failure_24h"] = False
    labeled_data["failure_event"] = "none"
    labeled_data["operating_state"] = "normal"

    for failure_window in FAILURE_WINDOWS:
        event_name = failure_window["failure_event"]
        start_time = pd.Timestamp(failure_window["start"])
        end_time = pd.Timestamp(failure_window["end"])
        pre_failure_start = start_time - pd.Timedelta(hours=pre_failure_hours)

        failure_mask = labeled_data[timestamp_column].between(start_time, end_time)
        pre_failure_mask = (
            (labeled_data[timestamp_column] >= pre_failure_start)
            & (labeled_data[timestamp_column] < start_time)
        )

        labeled_data.loc[failure_mask, "failure_period"] = True
        labeled_data.loc[failure_mask, "failure_event"] = event_name
        labeled_data.loc[pre_failure_mask, "pre_failure_24h"] = True

    labeled_data.loc[labeled_data["pre_failure_24h"], "operating_state"] = "pre_failure"
    labeled_data.loc[labeled_data["failure_period"], "operating_state"] = "failure"

    return labeled_data


def summarize_operating_states(data):
    """Count records and percentages for each operating state."""
    summary = data["operating_state"].value_counts().rename_axis("operating_state")
    summary = summary.reset_index(name="records")
    summary["percent"] = (summary["records"] / len(data) * 100).round(2)

    return summary
