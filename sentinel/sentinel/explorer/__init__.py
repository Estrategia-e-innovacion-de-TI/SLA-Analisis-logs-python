"""Module: explorer

This module provides initial analysis tools for anomaly detection. It includes utility functions 
and test runners to ensure the dataset is prepared for anomaly detection techniques.

Initial Analysis for Anomaly Detection:

Before applying anomaly detection techniques, it is crucial to perform an initial analysis 
to ensure that the dataset has sufficient and appropriate data. This script provides a suite of utility functions and corresponding tests for analyzing datasets 
in the context of anomaly detection and binary classification. The functions are designed to 
validate the quality and characteristics of the data, ensuring it meets the requirements for 
further analysis or modeling.

The script includes the following functionalities:

    1. **Anomaly Detection**:

        - Detects anomalies in a specified column using the Interquartile Range (IQR) method.

    2. **Data Quality Tests**:

        - Validates the minimum number of entries in specified columns.

        - Ensures the presence of a required column named 'label'.

        - Checks for anomalies in specified columns and validates their percentage.

        - Verifies that specified columns have a minimum percentage of non-null values.

        - Tests the variance of specified columns against a minimum acceptable threshold.

    3. **Correlation Analysis**:

        - Evaluates the correlation between a binary label column and feature columns using 
        the point-biserial correlation coefficient.

    4. **Model Evaluation**:

        - Tests the recall score of a logistic regression model for each specified column 
        as a feature.

Each function is accompanied by detailed docstrings explaining its purpose, parameters, 
and expected behavior. The tests are designed to raise assertion errors if the data 
does not meet the specified criteria, providing clear feedback on the failing conditions.

These steps help determine the readiness of the data for anomaly detection and guide any 
necessary preprocessing steps.


"""

from .conftest import *
from .run_tests import * 

