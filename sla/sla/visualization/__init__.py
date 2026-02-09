"""
This module provides visualization tools for anomaly detection and SHAP (SHapley Additive exPlanations) analysis.

Classes:

    AnomalyVisualizer: A class for creating plots to visualize anomalies detected in datasets.

    SHAPVisualizer: A class for generating SHAP plots to explain model predictions.

Usage:

    Import the desired visualizer class and use its methods to generate plots for anomaly detection or SHAP analysis.

Example:

    # Create an instance of AnomalyVisualizer
    anomaly_visualizer = AnomalyVisualizer()
    anomaly_visualizer.plot_anomalies(data)

    # Create an instance of SHAPVisualizer
    shap_visualizer = SHAPVisualizer()
    shap_visualizer.plot_shap_values(model, data)

    
"""
from .visualization import AnomalyVisualizer, SHAPVisualizer





__all__ = ['AnomalyVisualizer', 'SHAPVisualizer']