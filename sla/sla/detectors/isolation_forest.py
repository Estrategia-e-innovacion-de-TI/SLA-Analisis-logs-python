from sklearn.ensemble import IsolationForest
import numpy as np

class IsolationForestDetector:
    """
    A detector for identifying anomalies using the Isolation Forest algorithm.

    Parameters
    ----------
    n_estimators : int, optional (default=100)
        The number of  base estimators in the ensemble.
    max_samples : int or float, optional (default='auto')
        The number of samples to draw from X to train each base estimator.
        If 'auto', then max_samples=min(256, n_samples).
    contamination : 'auto' or float, optional (default='auto')
        The amount of contamination of the data set, i.e., the proportion of outliers in the data set.
        Used when fitting to define the threshold on the decision function.
        If 'auto', the threshold is determined as in the original paper.
    random_state : int, RandomState instance or None, optional (default=None)
        Controls the pseudo-randomness of the selection of the feature and split values for each branching step and each tree in the forest.
        Pass an int for reproducible results across multiple function calls.

    Methods
    -------
    fit(X)
        Fit the Isolation Forest model according to the given training data.
    predict(X)
        Predict if a particular sample is an outlier or not.
    decision_function(X)
        Compute the anomaly score of each sample.
    """
    def __init__(self, n_estimators=100, max_samples='auto', contamination='auto', random_state=None):
        self.model = IsolationForest(
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=contamination,
            random_state=random_state
        )

    def fit(self, X):
        self.model.fit(X)
        
    def predict(self, X):
        return self.model.predict(X)
        
    def decision_function(self, X):
        return self.model.decision_function(X)
        
    def fit_predict(self, X):
        return self.model.fit_predict(X)
        
    def get_anomalies(self, X):
        """
        Get the anomalies in the dataset X.
        
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        The input samples.
        
        Returns
        -------
        anomalies : array-like of shape (n_anomalies, n_features)
        The anomalies found in the dataset.
        """
        predictions = self.model.predict(X)
        anomalies = X[predictions == -1]
        return anomalies
    
    def predict_proba(self, X):
        """
        Predict the probability of each sample being an anomaly.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        proba : array-like of shape (n_samples,)
            The probability of each sample being an anomaly.
        """
        decision_scores = self.model.decision_function(X)
        proba = (decision_scores - decision_scores.min()) / (decision_scores.max() - decision_scores.min())
        return 1 - proba

if __name__ == "__main__":
    # Example usage
    X_train = np.random.rand(100, 2)
    X_test = np.random.rand(20, 2)

    detector = IsolationForestDetector(n_estimators=100, random_state=42)
    detector.fit(X_train)
    predictions = detector.predict(X_test)
    scores = detector.decision_function(X_test)
    anomalies = detector.get_anomalies(X_test)
    proba = detector.predict_proba(X_test)

    print("Predictions:", predictions)
    print("Scores:", scores)
    print("Anomalies:", anomalies)
    print("Anomaly Probabilities:", proba)