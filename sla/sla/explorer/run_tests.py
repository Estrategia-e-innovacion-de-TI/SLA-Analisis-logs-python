import pandas as pd

import pytest
from scipy.stats import pointbiserialr

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score



def detect_anomalies(df, column):
    """
    Detects anomalies in a specified column of a DataFrame using the Interquartile Range (IQR) method.

    Parameters:

        - df (pandas.DataFrame): The input DataFrame containing the data.

        - column (str): The name of the column in the DataFrame to analyze for anomalies.

    Returns:

        pandas.DataFrame: A DataFrame containing the rows where anomalies were detected.
                          Anomalies are defined as values outside the range [Q1 - 1.5 * IQR, Q3 + 1.5 * IQR],
                          where Q1 and Q3 are the first and third quartiles of the column, respectively.
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
    Tests whether each specified column in a DataFrame has at least a minimum required number of entries.

    Args:

        df_and_columns (tuple): A tuple containing:

            - df (pandas.DataFrame): The DataFrame to be tested.

            - columns (list): A list of column names to check for the minimum required entries.

    Raises:

        AssertionError: If any column has fewer entries than the minimum required, 
                        an assertion error is raised with details of the failing columns.

    Prints:

        - The total number of entries in the DataFrame.

        - A message for each column that has insufficient entries, including the column name and its entry count.
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
    
    assert not failed_columns, f"Columns with fewer than {min_required_entries} entries: {', '.join([f'{col} ({entries})' for col, entries in failed_columns])}"
    
def test_column_names(df_and_columns):
    """
    Validates that the DataFrame contains a column named 'label'.

    Args:

        df_and_columns (tuple): A tuple where the first element is a pandas DataFrame
                                and the second element is a list of column names.

    Raises:

        AssertionError: If the DataFrame does not contain a column named 'label'.
    """
    df, _ = df_and_columns

    assert 'label' in df.columns, "The DataFrame does not contain a column named 'label'"
        
def test_anomalies(df_and_columns):
    """
    Tests for anomalies in the specified columns of a DataFrame.
    This function checks each column in the provided DataFrame for anomalies
    using the `detect_anomalies` function. It calculates the percentage of
    anomalies in each column and prints the anomaly percentage. If the anomaly
    percentage for any column is less than 5%, the column is considered to have
    failed the test.

    Args:

        df_and_columns (tuple): A tuple containing:

            - df (pd.DataFrame): The DataFrame to analyze.

            - columns (list): A list of column names to check for anomalies.

    Raises:

        AssertionError: If any column has an anomaly percentage less than 5%.
                        The error message includes the names of the failed
                        columns and their respective anomaly percentages.
    """
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
    """
    Tests whether the specified columns in a DataFrame have at least a minimum percentage of non-null values.

    Args:

        df_and_columns (tuple): A tuple containing:

            - df (pandas.DataFrame): The DataFrame to be tested.

            - columns (list of str): A list of column names to check for non-null values.

    Raises:

        AssertionError: If any column has less than the minimum required percentage (95%) of non-null values.
                        The error message will include the names of the failing columns and their respective percentages.

    Example:

        >>> df = pd.DataFrame({
        ...     'col1': [1, 2, None],
        ...     'col2': [None, 2, 3]
        ... })
        >>> test_non_null_percentage((df, ['col1', 'col2']))
        AssertionError: Columns with less than 95% non-null values: col1 (66.67%), col2 (66.67%)

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
    
    assert not failed_columns, f"Columns with less than {min_non_null_percentage}% non-null values: {', '.join([f'{col} ({perc:.2f}%)' for col, perc in failed_columns])}"
    
def test_column_variance(df_and_columns):
    """
    Tests the variance of specified columns in a DataFrame against a minimum acceptable threshold.

    Args:

        df_and_columns (tuple): A tuple containing a pandas DataFrame and a list of column names to check.

    Raises:

        AssertionError: If any column's variance is below the minimum acceptable threshold.

    Details:

        - The minimum acceptable variance is set to 500.

        - For each column in the provided list, the function calculates its variance.

        - If the variance of a column is below the threshold, it is added to a list of failed columns.

        - The function asserts that there are no failed columns, and raises an error if any are found.
    """
    df, columns = df_and_columns
    failed_columns = []
    min_acceptable_variance = 500

    for column in columns:
        column_variance = df[column].var()

        if column_variance < min_acceptable_variance:
            print(f"------Variance for column '{column}': {column_variance:.2f}")
            failed_columns.append((column, column_variance))

    assert not failed_columns, f"Columns with variance below the acceptable threshold: {', '.join([f'{col} ({var:.2f})' for col, var in failed_columns])}"
        
def test_value_label_correlation(df_and_columns):
    """
    Tests the correlation between a binary label column and a set of feature columns 
    using the point-biserial correlation coefficient.

    Args:

        df_and_columns (tuple): A tuple containing:

            - df (pandas.DataFrame): The DataFrame containing the data.

            - columns (list): A list of column names to test for correlation with the 'label' column.

    Raises:

        AssertionError: If any column has a point-biserial correlation coefficient 
                        with the 'label' column below the defined threshold (0.4). 
                        The assertion error message will include the names of the 
                        failing columns and their respective coefficients.

    Notes:

        - The function uses the `pointbiserialr` method to calculate the correlation 
          coefficient and p-value for each column.

        - A column is considered to fail the test if the absolute value of its 
          correlation coefficient is less than the threshold (0.4).
    """
    df, columns = df_and_columns
    failed_columns = []

    correlation_threshold = 0.4
    for column in columns:
        correlation_coefficient, p_value = pointbiserialr(df['label'], df[column])
        
        if abs(correlation_coefficient) < correlation_threshold:
            print(f"------Point-biserial coefficient for column '{column}': {correlation_coefficient:.2f}")
            failed_columns.append((column, correlation_coefficient))

    assert not failed_columns, f"Columns with point-biserial correlation coefficient below the threshold: {', '.join([f'{col} ({coef:.2f})' for col, coef in failed_columns])}"
    
def test_logistic_regression_recall(df_and_columns):
    """
    Tests the recall score of a logistic regression model for each specified column in the dataset.
    This function splits the dataset into training and testing sets, trains a logistic regression 
    model on each specified column, and evaluates its recall score. If the recall score for any 
    column is below the defined threshold, the test will fail and raise an assertion error.

    Args:

        df_and_columns (tuple): A tuple containing:

            - df (pandas.DataFrame): The dataset containing the features and the target label.

            - columns (list of str): A list of column names to be tested as features.
    Raises:

        AssertionError: If any column's recall score is below the defined threshold.

    Notes:

        - The target label column in the dataset must be named 'label'.

        - The recall threshold is set to 0.5 by default.

        - Prints the recall score for columns that fail the threshold.
    """
    df, columns = df_and_columns
    recall_scores = []
    failed_columns = []
    recall_threshold = 0.5

    for column in columns:
        X_train, X_test, y_train, y_test = train_test_split(df[[column]], df['label'], test_size=0.2, random_state=42)
        
        model = LogisticRegression()
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        
        recall = recall_score(y_test, y_pred)
        
        recall_scores.append((column, recall))
        
        if recall <= recall_threshold:
            print(f"------Recall Score for column '{column}': {recall:.2f}")
            failed_columns.append((column, recall))

    assert not failed_columns, f"Columns with recall below {recall_threshold}: {', '.join([f'{col} ({rec:.2f})' for col, rec in failed_columns])}"
