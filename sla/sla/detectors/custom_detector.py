"""
Custom detector base class module.

This module provides a template for users to implement their own
anomaly detection algorithms, following the interface used in
this library.
"""

from abc import ABC, abstractmethod
import numpy as np

class BaseCustomDetector(ABC):
    """
    Abstract base class for all custom anomaly detectors.
    
    Users must subclass this and implement the required methods.
    """

    def __init__(self):
        """
        Optional initialization.
        """
        super().__init__()

    @abstractmethod
    def fit(self, X, y=None):
        """
        Fit the model to training data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data.
        y : Ignored (included for compatibility).
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Predict binary labels (1 for anomaly, 0 for normal).

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to predict on.

        Returns
        -------
        array-like of shape (n_samples,)
            Binary predictions: 1 for anomaly, 0 for normal.
        """
        pass

    def anomaly_score(self, X):
        """
        Compute anomaly scores for input data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)

        Returns
        -------
        array-like of shape (n_samples,)
            Anomaly scores (higher = more anomalous).
        """
        # Optional: Override for detectors that support scoring
        return np.zeros(len(X))

    def save_model(self, path):
        """
        Optional method to save model to disk.
        """
        pass

    def load_model(self, path):
        """
        Optional method to load model from disk.
        """
        pass
