import pandas as pd

from sentinel.explorer import detect_anomalies


def test_detect_anomalies_iqr_flags_outlier():
    df = pd.DataFrame({"value": [1, 1, 1, 1, 1, 1, 1, 100]})

    anomalies = detect_anomalies(df, "value")

    assert len(anomalies) == 1
    assert anomalies["value"].iloc[0] == 100
