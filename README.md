# MetroGuard Insight

Predictive maintenance and anomaly detection for metro train compressor telemetry.

## Project Overview

MetroGuard Insight is a portfolio data analysis and machine learning project based on industrial time-series telemetry from a metro train compressor system. The goal is to clean sensor records, explore operating patterns, detect unusual behavior, cluster operating modes, and build simple baseline models for high-risk state detection.

The implementation is intentionally beginner-friendly: clear functions, readable pandas workflows, and simple scikit-learn models.

## Tech Stack

- Python
- NumPy
- pandas
- scikit-learn
- matplotlib
- seaborn
- Jupyter Notebook

## Dataset

This project uses the [MetroPT-3 Dataset](https://archive.ics.uci.edu/dataset/791) from the UCI Machine Learning Repository.

The dataset contains multivariate time-series readings from pressure, oil temperature, motor current, and valve sensors installed on an Air Production Unit compressor in a metro train. According to UCI, the dataset contains 1,516,948 instances and 15 features.

The raw dataset is not included in this repository. Place the CSV file here:

```text
data/MetroPT3.csv
```

The raw data file is ignored by Git to keep the repository lightweight and suitable for public GitHub.

## Project Structure

```text
metroguard-insight/
|-- data/
|   |-- README.md
|   `-- MetroPT3.csv
|-- docs/
|   `-- Data Description_Metro.pdf
|-- notebooks/
|   `-- 01_metroguard_telemetry_analysis.ipynb
|-- outputs/
|   |-- figures/
|   `-- reports/
|-- src/
|   |-- data_processing.py
|   |-- feature_engineering.py
|   |-- anomaly_detection.py
|   |-- modeling.py
|   `-- visualization.py
|-- README.md
|-- requirements.txt
`-- LICENSE
```

## Workflow

1. Load raw compressor telemetry data.
2. Parse timestamps and sort records chronologically.
3. Select useful numeric sensor columns.
4. Aggregate high-frequency readings into one-minute intervals.
5. Create time-based, rolling, and difference features.
6. Explore sensor trends, distributions, and correlations.
7. Detect anomalies with Isolation Forest.
8. Cluster operating modes with KMeans.
9. Build baseline classifiers for an educational high-risk label.
10. Export figures and summary reports.

## Machine Learning Plan

The project will use:

- Isolation Forest for anomaly detection.
- KMeans for operating mode clustering.
- Logistic Regression and Random Forest as baseline classifiers.

If real failure labels are not available directly, the classification target will be engineered for educational purposes, for example from anomaly scores or known failure windows. This target should not be treated as a production-grade failure label.

## Planned Outputs

Figures will be saved in `outputs/figures/`, including sensor trends, anomaly score timelines, clustering plots, confusion matrices, and feature importance charts.

Reports will be saved in `outputs/reports/`, including detected anomaly events, model metrics, and operating mode summaries.

## How to Run

1. Create and activate a Python environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Download the MetroPT-3 dataset from UCI.
4. Rename the raw CSV file to `MetroPT3.csv`.
5. Place it in the `data/` directory.
6. Open and run:

```text
notebooks/01_metroguard_telemetry_analysis.ipynb
```

## Key Findings

This section will be completed after the first full analysis run.

## License

This project is released under the MIT License. The MetroPT-3 dataset has its own license and citation requirements from UCI.
