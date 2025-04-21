"""
Test suite for SLA data analysis.

This module provides test functions for validating data quality and relevance
for anomaly detection. Tests verify data completeness, presence of anomalies,
statistical properties, and predictive capability of features.
"""

import pandas as pd
import pytest
from scipy.stats import pointbiserialr
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score


def detect_anomalies(df, column):
    """
    Detect outliers in a DataFrame column using the IQR method.
    
    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the data to analyze
    column : str
        Name of the column to check for anomalies
        
    Returns
    -------
    pandas.DataFrame
        Subset of the original DataFrame containing only the anomalies
    """
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    anomalies = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return anomalies


def test_minimum_entries(df_and_columns):
    """
    Test that each column has a minimum number of non-null entries.
    
    Parameters
    ----------
    df_and_columns : tuple
        Tuple containing (DataFrame, list of column names)
    """
    df, columns = df_and_columns
    failed_columns = []
    min_required_entries = 25000
    
    total_entries = len(df)
    print(f"------Total entries in DataFrame: {total_entries}")
    
    for column in columns:
        column_entries = df[column].count()
        
        if column_entries < min_required_entries:
            print(f"------Column '{column}' has insufficient entries: {column_entries}")
            failed_columns.append((column, column_entries))
    
    assert not failed_columns, (f"Columns with fewer than {min_required_entries} entries: "
                               f"{', '.join([f'{col} ({entries})' for col, entries in failed_columns])}")
    

def test_column_names(df_and_columns):
    """
    Test that the DataFrame contains the required 'label' column.
    
    Parameters
    ----------
    df_and_columns : tuple
        Tuple containing (DataFrame, list of column names)
    """
    df, _ = df_and_columns
    assert 'label' in df.columns, "The DataFrame does not contain a column named 'label'"
        

def test_anomalies(df_and_columns):
    """
    Test that each column contains a minimum percentage of anomalies.
    
    Parameters
    ----------
    df_and_columns : tuple
        Tuple containing (DataFrame, list of column names)
    """
    df, columns = df_and_columns
    failed_columns = []

    for column in columns:
        anomalies = detect_anomalies(df, column)
        
        anomaly_percentage = len(anomalies) / len(df) * 100
        
        print(f"------Anomaly percentage for column '{column}': {abs(anomaly_percentage):.2f}%")
        
        if anomaly_percentage < 5:
            failed_columns.append((column, anomaly_percentage))

    assert not failed_columns, (f"Columns with anomaly percentage less than 5%: "
                               f"{', '.join([f'{col} ({perc:.2f}%)' for col, perc in failed_columns])}")


def test_non_null_percentage(df_and_columns):
    """
    Test that each column has a minimum percentage of non-null values.
    
    Parameters
    ----------
    df_and_columns : tuple
        Tuple containing (DataFrame, list of column names)
    """
    df, columns = df_and_columns
    failed_columns = []
    min_non_null_percentage = 95

    for column in columns:
        non_null_count = df[column].notnull().sum()
        total_count = len(df)
        non_null_percentage = (non_null_count / total_count) * 100
        
        if non_null_percentage < min_non_null_percentage:
            print(f"------Column '{column}' has {non_null_percentage:.2f}% non-null values")
            failed_columns.append((column, non_null_percentage))
    
    assert not failed_columns, (f"Columns with less than {min_non_null_percentage}% non-null values: "
                               f"{', '.join([f'{col} ({perc:.2f}%)' for col, perc in failed_columns])}")
    

def test_column_variance(df_and_columns):
    """
    Test that each column has sufficient variance.
    
    Parameters
    ----------
    df_and_columns : tuple
        Tuple containing (DataFrame, list of column names)
    """
    df, columns = df_and_columns
    failed_columns = []
    min_acceptable_variance = 500

    for column in columns:
        column_variance = df[column].var()

        if column_variance < min_acceptable_variance:
            print(f"------Variance for column '{column}': {column_variance:.2f}")
            failed_columns.append((column, column_variance))

    assert not failed_columns, (f"Columns with variance below the acceptable threshold: "
                               f"{', '.join([f'{col} ({var:.2f})' for col, var in failed_columns])}")
        

def test_value_label_correlation(df_and_columns):
    """
    Test correlation between column values and event labels.
    
    Parameters
    ----------
    df_and_columns : tuple
        Tuple containing (DataFrame, list of column names)
    """
    df, columns = df_and_columns
    failed_columns = []

    correlation_threshold = 0.4
    for column in columns:
        correlation_coefficient, p_value = pointbiserialr(df['label'], df[column])
        
        if abs(correlation_coefficient) < correlation_threshold:
            print(f"------Point-biserial coefficient for column '{column}': {correlation_coefficient:.2f}")
            failed_columns.append((column, correlation_coefficient))

    assert not failed_columns, (f"Columns with point-biserial correlation coefficient below the threshold: "
                               f"{', '.join([f'{col} ({coef:.2f})' for col, coef in failed_columns])}")
    

def test_logistic_regression_recall(df_and_columns):
    """
    Test predictive capability of each column using logistic regression.
    
    Parameters
    ----------
    df_and_columns : tuple
        Tuple containing (DataFrame, list of column names)
    """
    df, columns = df_and_columns
    recall_scores = []
    failed_columns = []
    recall_threshold = 0.5

    for column in columns:
        X_train, X_test, y_train, y_test = train_test_split(
            df[[column]], df['label'], test_size=0.2, random_state=42
        )
        
        model = LogisticRegression()
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        recall = recall_score(y_test, y_pred)
        
        recall_scores.append((column, recall))
        
        if recall <= recall_threshold:
            print(f"------Recall Score for column '{column}': {recall:.2f}")
            failed_columns.append((column, recall))

    assert not failed_columns, (f"Columns with recall below {recall_threshold}: "
                               f"{', '.join([f'{col} ({rec:.2f})' for col, rec in failed_columns])}")