
"""Module of detectors.

This module provides a collection of anomaly detection classes for time series data.

Classes:

    AutoencoderDetector: Implements anomaly detection using an autoencoder model.

    IsolationForestDetector: Implements anomaly detection using the Isolation Forest algorithm.

    RRCFDetector: Implements anomaly detection using the Robust Random Cut Forest (RRCF) algorithm.
    
    LNNDetector: Implements anomaly detection using a Learned Neural Network (LNN) approach.

The detectors are designed to identify anomalous time points in time series data.
"""
from .isolation_forest import IsolationForestDetector
from .custom_detector import BaseCustomDetector


def _missing_optional_dependency(name, error):
    class _MissingDependency:
        def __init__(self, *args, **kwargs):
            raise ImportError(
                f"{name} requires optional dependencies that are not installed. "
                f"Original error: {error}"
            ) from error

    _MissingDependency.__name__ = name
    return _MissingDependency


try:
    from .autoencoder import AutoencoderDetector
except ImportError as exc:
    AutoencoderDetector = _missing_optional_dependency("AutoencoderDetector", exc)

try:
    from .rrcf_detector import RRCFDetector
except ImportError as exc:
    RRCFDetector = _missing_optional_dependency("RRCFDetector", exc)

try:
    from .lnn import LNNDetector
except ImportError as exc:
    LNNDetector = _missing_optional_dependency("LNNDetector", exc)


__all__ = [
    "AutoencoderDetector",
    "IsolationForestDetector",
    "RRCFDetector",
    "LNNDetector",
    "BaseCustomDetector",
]
