"""
<p align="center">
  <img src="https://raw.githubusercontent.com/Estrategia-e-innovacion-de-TI/SLA-Analisis-logs-python/refs/heads/docs/docs/images/sentinel%20icon%20with%20text.png" alt="Sentinel Logo" width="250"/>
</p>



A simple yet powerful tool to analyze logs and extract meaningful insights.

Sentinel is a Python library designed to facilitate the analysis of logs from systems,
applications, and services. It enables users to extract, process, and analyze log data
to detect anomalies, patterns, and trends that might indicate issues or interesting
behaviors in their systems. One of its main objectives is to quickly identify if your
data contains signals that could proactively indicate potential problems.

Installation
----------

To install Sentinel, simply run:

```bash
git clone https://github.com/Estrategia-e-innovacion-de-TI/SLA-Analisis-logs-python.git
cd SLA-Analisis-logs-python
pip install -e .
```

Modules
----------

1. ingestion
---------
Handles the transformation of raw, unstructured log files into structured pandas DataFrames.

It includes a base parser and specific parsers for:

- WAS (WebSphere Application Server)
- HSM (Hardware Security Module)
- HDC (High-Density Computing)
- IBMMQ (IBM Message Queue)

Each parser adapts to the structure and format of its respective log type.  
For unsupported log types, custom parsers can be built upon the base parser.

---

2. explorer
--------
Provides tools for initial dataset analysis focused on anomaly detection and data validation.

Functionalities include:

- **Anomaly Detection**:  
  Identifies anomalies using the Interquartile Range (IQR) method.

- **Data Quality Tests**:
  - Ensures minimum data entries in columns.
  - Verifies presence of a required `label` column.
  - Checks anomaly percentages.
  - Validates minimum non-null value percentages.
  - Tests minimum variance thresholds.

- **Correlation Analysis**:  
  Calculates point-biserial correlations between labels and features.

- **Model Evaluation**:  
  Measures the recall score of a logistic regression model using each feature individually.

These validations help determine dataset readiness for anomaly detection tasks.

---

3. transformer
-----------
Provides aggregation methods for structured dataframes, particularly useful for time-series and event-driven data.

Available tools:

- **StringAggregator**: Aggregates string values over a defined time window (e.g., 30 seconds).
- **RollingAggregator**: Applies rolling window aggregations.

---

4. detectors
---------
Offers anomaly detection models tailored for time-series data.

Available detectors:

- **AutoencoderDetector**: Autoencoder-based anomaly detection.
- **IsolationForestDetector**: Anomaly detection using the Isolation Forest algorithm.
- **RRCFDetector**: Robust Random Cut Forest-based detection.
- **LNNDetector**: Detection via a Learned Neural Network (LNN).

Each model identifies anomalies in temporal datasets.

---

5. simulation
----------
Initializes the simulation environment for SLA analysis logs and streaming anomaly detection.

Main class:

- **StreamingSimulation**: Simulates streaming-based anomaly detection scenarios.

---

6. visualization
-------------

Provides visualization utilities for:

- Anomaly detection results
- SHAP (SHapley Additive exPlanations) analysis

License
--------

This project is licensed under the terms of the MIT license.
"""

__version__ = "0.0.1"
__author__ = "ARQUITECTURA INNOVACIÓN TI"
__license__ = "MIT"