
"""Module of detectors.

This module provides a collection of anomaly detection classes for time series data.

Classes:

    AutoencoderDetector: Implements anomaly detection using an autoencoder model.

    IsolationForestDetector: Implements anomaly detection using the Isolation Forest algorithm.

    RRCFDetector: Implements anomaly detection using the Robust Random Cut Forest (RRCF) algorithm.
    
    LNNDetector: Implements anomaly detection using a Learned Neural Network (LNN) approach.

The detectors are designed to identify anomalous time points in time series data.
"""
from .autoencoder import AutoencoderDetector
from .isolation_forest import IsolationForestDetector
from .rrcf_detector import RRCFDetector
from .lnn import LNNDetector






__all__ = ['AutoencoderDetector', 'IsolationForestDetector', 'RRCFDetector', 'LNNDetector']