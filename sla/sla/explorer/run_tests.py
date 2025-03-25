import pandas as pd
import pytest
from scipy.stats import pointbiserialr

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score

def detect_anomalies(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    anomalies = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return anomalies

def test_minimum_entries(df_and_columns):
    df, _ = df_and_columns
    
    num_entries = len(df)

    assert num_entries >= 25000, f"Number of entries {num_entries} is less than 25,000"

def test_column_names(df_and_columns):
    df, _ = df_and_columns

    assert 'label' in df.columns, "The DataFrame does not contain a column named 'label'"
        
def test_anomalies(df_and_columns):
    df, columns = df_and_columns
    failed_columns = []

    for column in columns:
        anomalies = detect_anomalies(df, column)
        
        anomaly_percentage = len(anomalies) / len(df) * 100
        
        print(f"------Anomaly percentage for column '{column}': {abs(anomaly_percentage):.2f}%")
        
        if anomaly_percentage < 5:
            failed_columns.append((column, anomaly_percentage))

    assert not failed_columns, f"Columns with anomaly percentage less than 5%: {', '.join([f'{col} ({perc:.2f}%)' for col, perc in failed_columns])}"

def test_non_null_percentage(df_and_columns):
    df, columns = df_and_columns

    min_non_null_percentage = 95

    for column in columns:
        non_null_count = df[column].notnull().sum()
        total_count = len(df)
        non_null_percentage = (non_null_count / total_count) * 100

        assert non_null_percentage >= min_non_null_percentage, f"Column '{column}' has less than 95% non-null values ({non_null_percentage}%)"

def test_column_variance(df_and_columns):
    df, columns = df_and_columns
    failed_columns = []

    min_acceptable_variance = 500

    for column in columns:
        column_variance = df[column].var()

        print(f"------Variance for column '{column}': {column_variance:.2f}")

        if column_variance < min_acceptable_variance:
            failed_columns.append((column, column_variance))

    assert not failed_columns, f"Columns with variance below the acceptable threshold: {', '.join([f'{col} ({var:.2f})' for col, var in failed_columns])}"
    
def test_value_label_correlation(df_and_columns):
    df, columns = df_and_columns
    failed_columns = []

    correlation_threshold = 0.4
    for column in columns:
        correlation_coefficient, p_value = pointbiserialr(df['label'], df[column])
        
        print(f"------Point-biserial coefficient for column '{column}': {correlation_coefficient:.2f}")

        if abs(correlation_coefficient) < correlation_threshold:
            failed_columns.append((column, correlation_coefficient))

    assert not failed_columns, f"Columns with point-biserial correlation coefficient below the threshold: {', '.join([f'{col} ({coef:.2f})' for col, coef in failed_columns])}"

def test_logistic_regression_recall(df_and_columns):
    df, columns = df_and_columns
    recall_scores = []

    for column in columns:
        X_train, X_test, y_train, y_test = train_test_split(df[[column]], df['label'], test_size=0.2, random_state=42)
        
        model = LogisticRegression()
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        recall = recall_score(y_test, y_pred)
        
        recall_scores.append((column, recall))

    recall_scores.sort(key=lambda x: x[1], reverse=True)

    for column, recall in recall_scores:
        print(f"------Recall Score for column '{column}': {recall:.2f}")

    failed_columns = [column for column, recall in recall_scores if recall <= 0.5]
    assert not failed_columns, f"Columns with recall below 0.5: {', '.join(failed_columns)}"