import numpy as np

from sentinel.detectors import IsolationForestDetector


def test_isolation_forest_smoke():
    rng = np.random.RandomState(0)
    inliers = rng.normal(0, 1, size=(50, 2))
    outliers = rng.normal(6, 0.1, size=(3, 2))
    X = np.vstack([inliers, outliers])

    detector = IsolationForestDetector(
        n_estimators=25,
        contamination=0.1,
        random_state=0,
    )

    detector.fit(X)
    predictions = detector.predict(X)

    assert predictions.shape[0] == X.shape[0]
    assert (predictions == -1).sum() >= 1
