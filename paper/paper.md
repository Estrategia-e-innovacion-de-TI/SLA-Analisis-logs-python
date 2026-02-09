---
title: 'Sentinel: A Python Library for Log Analysis and Anomaly Detection'
tags:
  - Python
  - log analysis
  - anomaly detection
  - signal validation
  - enterprise logs
authors:
  - name: JM Vergara
    affiliation: 1
  - name: N Laverde
    affiliation: 1
  - name: JP Aguilar
    affiliation: 1
  - name: JV Niño
    affiliation: 1
  - name: JD Muñoz
    affiliation: 1
  - name: D Monsalve
    affiliation: 1
  - name: S Osorio
    affiliation: 1
affiliations:
  - name: Bancolombia, Colombia
    index: 1
bibliography: paper.bib
---

# Summary

Sentinel is a Python library designed to address a critical challenge in enterprise log analysis: determining whether unstructured log data contains meaningful signals before investing computational resources in complex anomaly detection pipelines. Many organizations generate massive volumes of logs from systems such as WebSphere Application Server (WAS), Hardware Security Modules (HSM), High-Density Computing (HDC) platforms, and IBM Message Queue (IBMMQ), but not all log data contains actionable patterns for anomaly detection.

The library's key innovation is the Explorer module, which performs early signal validation and data quality checks using techniques such as Interquartile Range (IQR) anomaly detection, correlation analysis, and variance thresholds. This fail-fast approach enables practitioners to quickly assess whether their log data is worth analyzing, potentially saving significant computational resources and development time. Sentinel provides a modular architecture spanning ingestion, transformation, exploration, detection, visualization, and simulation, with built-in parsers for common enterprise log formats and extensibility for custom implementations.

# Statement of Need

Enterprise environments generate vast quantities of unstructured log data from diverse systems, but a fundamental question often goes unaddressed: does this data contain meaningful signals for anomaly detection? Traditional approaches invest significant computational resources in building analysis pipelines before validating whether the underlying data exhibits patterns worth detecting. This can lead to wasted effort when log data lacks sufficient variance, contains predominantly null values, or shows no correlation with labeled anomalies.

Sentinel addresses this gap by providing upfront signal validation through its Explorer module. Before practitioners commit to training models or deploying detection algorithms, Sentinel evaluates data quality metrics including minimum record counts, label presence, anomaly percentages, non-null value thresholds, and variance levels. This early validation is particularly valuable for enterprise log formats such as WAS, HSM, HDC, and IBMMQ, which often require custom parsing and may not consistently produce signal-rich data.

By enabling a fail-fast approach to log analysis, Sentinel helps organizations avoid the computational and development costs associated with analyzing low-quality data. The library fills a gap between raw log collection and sophisticated anomaly detection tools, providing the critical validation step that determines whether further analysis is warranted.

# State of the Field

The anomaly detection ecosystem includes several mature libraries that provide diverse algorithms and robust implementations. PyOD [@zhao2019pyod] offers a comprehensive collection of outlier detection algorithms with a unified interface, while pySAD [@yilmaz2021pysad] focuses on streaming anomaly detection scenarios. ADTK [@adtk2019] provides specialized tools for time series anomaly detection, TODS [@lai2021tods] offers automated outlier detection with machine learning pipelines, and Anomalib [@akcay2022anomalib] delivers deep learning approaches for anomaly detection with a focus on computer vision applications.

These tools excel at detecting anomalies in structured, signal-rich data where patterns are present and meaningful. However, they generally assume that input data has already been validated for quality and signal presence. In enterprise environments dealing with unstructured logs, this assumption may not hold—log data may lack sufficient variance, contain excessive null values, or show no correlation with known anomalies.

Sentinel complements these existing tools by addressing the earlier stage of the analysis pipeline: determining whether log data is worth analyzing in the first place. Rather than competing with established anomaly detection libraries, Sentinel provides the validation layer that helps practitioners decide when to apply those tools. The library's built-in parsers for enterprise log formats (WAS, HSM, HDC, IBMMQ) and its extensible parser framework further differentiate it from general-purpose anomaly detection tools that assume pre-processed, structured input.

# Software Design

Sentinel implements a modular architecture that guides users through the complete log analysis workflow:

**Ingestion**: The Ingestion module transforms raw, unstructured log files into structured pandas DataFrames. It provides a base parser class for extensibility and includes specific parsers for WAS, HSM, HDC, and IBMMQ log formats. Custom parsers can be implemented by extending the base class to support additional log formats.

**Transformer**: The Transformer module provides aggregation methods for time series and event data. The StringAggregator consolidates string values within defined time windows, while the RollingAggregator applies rolling window operations to prepare data for detection algorithms.

**Explorer**: The Explorer module is Sentinel's key innovation, performing early signal validation and data quality checks. It uses IQR-based anomaly detection for initial assessment and evaluates multiple quality metrics: minimum records per column, label column presence, anomaly percentage thresholds, non-null value percentages, and variance thresholds. The module also performs point-biserial correlation analysis and evaluates logistic regression models on individual features to assess signal quality. Based on these checks, the Explorer module provides a fail-fast decision: proceed with analysis or reject the data.

**Detectors**: The Detectors module implements multiple anomaly detection algorithms with a consistent interface: AutoencoderDetector (neural network-based reconstruction), IsolationForestDetector (tree-based isolation), RRCFDetector (Robust Random Cut Forest), and LNNDetector (Liquid Neural Networks). This unified interface enables algorithm comparison and selection based on specific use cases.

**Visualization**: The Visualization module provides tools for displaying anomaly detection results and SHAP (SHapley Additive exPlanations) analysis, enabling interpretability of model decisions.

**Simulation**: The Simulation module includes the StreamingSimulation class for testing real-time anomaly detection scenarios, useful for validation and benchmarking.

The modular design enables practitioners to use individual components independently or combine them in custom workflows, supporting both exploratory analysis and production deployments.

# Research Impact Statement

Sentinel provides value to the research and practitioner communities through several engineering contributions:

**Reproducibility**: The library includes a comprehensive test suite with continuous integration, documented installation procedures, and example notebooks demonstrating usage patterns. This infrastructure enables researchers and practitioners to validate results and build upon the work.

**Modularity and Extensibility**: Sentinel's architecture separates concerns through distinct modules, each with clear responsibilities. The base parser class enables custom implementations for new log formats, and the unified detector interface allows algorithm comparison and extension. This design supports both research experimentation and production deployment.

**Enterprise Log Format Support**: By providing built-in parsers for WAS, HSM, HDC, and IBMMQ formats, Sentinel addresses real-world enterprise needs. These parsers handle the complexity of unstructured log formats, enabling practitioners to focus on analysis rather than data wrangling.

**Practical Value**: The fail-fast validation approach provides concrete benefits by identifying unsuitable data early in the analysis pipeline. This prevents wasted computational resources on low-quality data and helps organizations make informed decisions about where to invest analysis effort. While specific resource savings depend on individual use cases, the engineering principle of early validation is well-established in software development and applies directly to data analysis workflows.

The library's Apache 2.0 license, Code of Conduct, Contributing guidelines, and Security policy establish a foundation for open collaboration and responsible development, supporting both academic research and industrial applications.

# AI Usage Disclosure

Generative AI tools may have been used for language refinement; authors reviewed and validated the final content.

# References
