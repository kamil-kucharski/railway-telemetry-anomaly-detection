"""Reusable plotting helpers for the MetroGuard Insight notebook."""

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns


def save_current_figure(output_path):
    """Save the current matplotlib figure and create the folder if needed."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")


def plot_sensor_over_time(data, timestamp_column, sensor_column, title, output_path=None):
    """Plot one sensor as a time-series line chart."""
    plt.figure(figsize=(12, 4))
    sns.lineplot(data=data, x=timestamp_column, y=sensor_column, linewidth=1)
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel(sensor_column)

    if output_path:
        save_current_figure(output_path)

    plt.show()


def plot_multiple_sensors_over_time(
    data,
    timestamp_column,
    sensor_columns,
    title,
    output_path=None,
):
    """Plot several sensors on one time-series chart."""
    plt.figure(figsize=(12, 5))

    for column in sensor_columns:
        sns.lineplot(data=data, x=timestamp_column, y=column, label=column, linewidth=1)

    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Sensor value")
    plt.legend()

    if output_path:
        save_current_figure(output_path)

    plt.show()


def plot_correlation_heatmap(
    data,
    sensor_columns,
    title="Sensor Correlation Heatmap",
    output_path=None,
):
    """Plot a correlation heatmap for selected sensor columns."""
    plt.figure(figsize=(10, 8))
    correlation = data[sensor_columns].corr()
    sns.heatmap(correlation, cmap="coolwarm", center=0, annot=True, fmt=".2f")
    plt.title(title)

    if output_path:
        save_current_figure(output_path)

    plt.show()


def plot_sensor_distributions(
    data,
    sensor_columns,
    bins=40,
    title="Sensor Distributions",
    output_path=None,
):
    """Plot histograms for selected sensor columns."""
    number_of_sensors = len(sensor_columns)
    rows = (number_of_sensors + 2) // 3

    fig, axes = plt.subplots(rows, 3, figsize=(15, 4 * rows))
    axes = axes.flatten()

    for index, column in enumerate(sensor_columns):
        sns.histplot(data[column], bins=bins, kde=True, ax=axes[index])
        axes[index].set_title(column)
        axes[index].set_xlabel("Value")
        axes[index].set_ylabel("Count")

    for index in range(number_of_sensors, len(axes)):
        axes[index].set_visible(False)

    fig.suptitle(title, y=1.02)

    if output_path:
        save_current_figure(output_path)

    plt.show()


def plot_confusion_matrix(matrix, labels, title="Confusion Matrix", output_path=None):
    """Plot a confusion matrix with readable labels."""
    plt.figure(figsize=(5, 4))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    if output_path:
        save_current_figure(output_path)

    plt.show()
