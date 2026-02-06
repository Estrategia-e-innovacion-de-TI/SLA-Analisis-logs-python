<p align="center">
  <img src="docs/images/sentinel icon with text.png" alt="Sentinel Logo" width="200"/>
</p>

# Sentinel

**Sentinel** is a simple yet powerful tool for analyzing logs and extracting meaningful information.

Sentinel is a Python library designed to simplify the analysis of logs from systems, applications, and services. It allows users to extract, process, and analyze log data to detect anomalies, patterns, and trends that could indicate issues or relevant behaviors.  
One of its main goals is to quickly identify whether the data contains signals that may proactively indicate potential problems.

---

## 🚀 Installation

To install Sentinel, run:

```bash
git clone https://github.com/Estrategia-e-innovacion-de-TI/SLA-Analisis-logs-python.git
cd SLA-Analisis-logs-python
pip install -e .
```

---

## 📚 Modules

### **Ingestion**
Transforms raw, unstructured log files into structured pandas DataFrames.

Includes a base parser and specific parsers for:

- WAS (WebSphere Application Server)
- HSM (Hardware Security Module)
- HDC (High-Density Computing)
- IBMMQ (IBM Message Queue)

Custom parsers can be created based on the base parser for unsupported log types.

---

### **Explorer**
Provides tools for initial dataset analysis focused on anomaly detection and data validation.

#### Main Features:
- **Anomaly Detection**: Detects anomalies using the Interquartile Range (IQR) method.
- **Data Quality Checks**:
    - Minimum number of records per column.
    - Presence of a label column.
    - Percentage of anomalies.
    - Minimum percentage of non-null values.
    - Minimum variance thresholds.
- **Correlation Analysis**: Calculates point-biserial correlation between labels and features.
- **Model Evaluation**: Measures recall of a logistic regression model using each feature individually.

---

### **Transformer**
Provides aggregation methods for structured DataFrames, useful for time series or event data.

Tools:
- **StringAggregator**: Aggregates string values within a defined time window (e.g., 30 seconds).
- **RollingAggregator**: Applies rolling window aggregations.

---

### **Detectors**
Includes anomaly detection models tailored for time series data.

Available detectors:
- **AutoencoderDetector**: Detection based on autoencoders.
- **IsolationForestDetector**: Detection using Isolation Forest.
- **RRCFDetector**: Detection using Robust Random Cut Forest.
- **LNNDetector**: Detection using Liquid Neural Networks (LNN).

---

### **Simulation**
Initializes the simulation environment for log analysis and streaming anomaly detection.

Main class:
- **StreamingSimulation**: Simulates streaming anomaly detection scenarios.

---

### **Visualization**
Provides tools to visualize:
- Anomaly detection results.
- SHAP (SHapley Additive exPlanations) analysis for interpreting model predictions.

---

## Development

### Run tests
```bash
pip install -e ".[dev]"
pytest -q
```

---

## 📦 Project Details
- **Version**: 0.0.1  
- **Author**: ARQUITECTURA INNOVACIÓN TI  
- **License**: APACHE 2.0  

---

## 📖 Documentation & Quickstarts

- [Full Documentation](https://estrategia-e-innovacion-de-ti.github.io/SLA-Analisis-logs-python/sla.html)  
- [Quickstart Guide](https://estrategia-e-innovacion-de-ti.github.io/SLA-Analisis-logs-python/sla.html#quickstart)