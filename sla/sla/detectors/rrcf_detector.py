"""Robust Random Cut Forest anomaly detection implementation."""
import numpy as np
import pandas as pd
import rrcf
from sklearn.base import BaseEstimator


class RRCFDetector(BaseEstimator):
    """Robust Random Cut Forest anomaly detector.

    Esta clase implementa la detección de anomalías utilizando el algoritmo
    Robust Random Cut Forest para la detección de anomalías en streaming.

    Parameters
    ----------
    shingle_size : int, default=15
        Tamaño de la ventana de shingling.
    num_trees : int, default=100
        Número de árboles en el bosque aleatorio.
    tree_size : int, default=500
        Tamaño máximo de cada árbol en el bosque.
    """

    def __init__(self, shingle_size=15, num_trees=100, tree_size=500):
        self.shingle_size = shingle_size
        self.num_trees = num_trees
        self.tree_size = tree_size
        self.forest = [rrcf.RCTree() for _ in range(self.num_trees)]
        self.index = None
        self.anomaly_scores = None

    def fit(self, X, y=None):
        """Fit the model to the input data.

        Parameters
        ----------
        X : array-like of shape (n_samples,)
            Input time series data.

        Returns
        -------
        self : object
            Returns self.
        """
        n = len(X)
        self.anomaly_scores = pd.Series(0.0, index=np.arange(n))
        
        for tree in self.forest:
            for i in range(n):
                point = X.iloc[i]
                unique_index = (i, id(tree))
                tree.insert_point(point, index=unique_index)
                
                if len(tree.leaves) > self.tree_size:
                    old_index = (i - self.tree_size, id(tree))
                    if old_index in tree.leaves:
                        tree.forget_point(old_index)
        
        return self

    def predict(self, X):
        """Calculate anomaly scores for input data.

        Parameters
        ----------
        X : array-like of shape (n_samples,)
            Input time series data.

        Returns
        -------
        scores : ndarray of shape (n_samples,)
            Anomaly scores for each input sample.
        """
        scores = np.zeros(len(X))
        
        for tree in self.forest:
            for i in range(len(X)):
                point = X.iloc[i]
                unique_index = (i, id(tree))
                
                if unique_index in tree.leaves:
                    leaf = tree.leaves[unique_index]
                    score = tree.codisp(leaf)
                    scores[i] += score
        
        return scores / self.num_trees  # Normalize by number of trees

    def fit_predict(self, X, y=None):
        """Fit the model and calculate anomaly scores.

        Parameters
        ----------
        X : array-like of shape (n_samples,)
            Input time series data.

        Returns
        -------
        anomaly_scores : Series of shape (n_samples,)
            Anomaly scores for each input sample.
        """
        self.fit(X)
        self.anomaly_scores = pd.Series(self.predict(X), index=np.arange(len(X)))
        return self.anomaly_scores

    def get_anomalies(self, threshold):
        """Identify anomalies based on a threshold.

        Parameters
        ----------
        threshold : float
            Threshold value for anomaly detection.

        Returns
        -------
        anomalies : Series
            Series containing anomaly scores above the threshold.
        """
        if self.anomaly_scores is None:
            raise ValueError("Model must be fit before getting anomalies.")
        return self.anomaly_scores[self.anomaly_scores > threshold]

    def predict_proba(self, X):
        """Calculate normalized probability-like anomaly scores.

        Parameters
        ----------
        X : array-like of shape (n_samples,)
            Input time series data.

        Returns
        -------
        proba : ndarray of shape (n_samples,)
            Normalized anomaly scores.
        """
        scores = self.predict(X)
        max_score = np.max(scores)
        
        # Avoid division by zero
        if max_score > 0:
            return scores / max_score
        return scores


# Example usage:
if __name__ == "__main__":
    # Generate some sample data
    np.random.seed(0)
    X = np.random.randn(1000)

    # Initialize the detector
    detector = RRCFDetector()

    # Get anomaly scores
    anomaly_scores = detector.fit_predict(X)

    # Get anomalies with a threshold
    anomalies = detector.get_anomalies(threshold=3.0)

    # Print the results
    print("Anomaly Scores:\n", anomaly_scores)
    print("Anomalies:\n", anomalies)