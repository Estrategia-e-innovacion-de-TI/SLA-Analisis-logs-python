"""
Pytest configuration module for analyzer.

This module defines fixtures and hooks used for testing data analysis
with labeled events. It handles loading CSV data, applying event labels,
and improving test failure reporting.

Example usage:
    pytest run_tests.py --csvpath=data_file.csv --columns=column1,column2 --events_csvpath=events_file.csv --html=report.html
"""

import datetime
import pytest
import pandas as pd


def pytest_addoption(parser):
    """

    This function adds custom command-line options to pytest.

    Options:

        --csvpath (str): Path to the main CSV file to be used in testing.

        --columns (str): Comma-separated column names for testing.

        --events_csvpath (str): Path to the events CSV file containing begin and end dates.

        --timestamp_column (str, optional): Name of the timestamp column in the main CSV file. 
            Defaults to "timestamp".

        --event_begin_column (str, optional): Name of the event begin date column in the events CSV file. 
            Defaults to "begin_date".
            
        --event_end_column (str, optional): Name of the event end date column in the events CSV file. 
            Defaults to "end_date".
    
    Args:
    
      parser : _pytest.config.argparsing.Parser
          Pytest command line parser to which the options are added.

    """
    parser.addoption("--csvpath", action="store", help="Path to the CSV file")
    parser.addoption("--columns", action="store", help="Column names for testing")
    parser.addoption("--events_csvpath", action="store", 
                     help="Path to the events CSV file with begin and end dates")
    parser.addoption("--timestamp_column", action="store", default="timestamp", 
                     help="Name of the timestamp column in the main CSV")
    parser.addoption("--event_begin_column", action="store", default="begin_date", 
                     help="Name of the event begin date column in the events CSV")
    parser.addoption("--event_end_column", action="store", default="end_date", 
                     help="Name of the event end date column in the events CSV")


@pytest.fixture(scope="module")
def df_and_columns(request):
    """

    Fixture to provide a DataFrame and a list of columns for testing.
    This fixture reads a CSV file specified by the `--csvpath` command-line option
    and returns a DataFrame along with a list of column names specified by the
    `--columns` command-line option. Additionally, it processes an optional events
    CSV file to label rows in the DataFrame based on timestamp ranges.

    Args:

        request: pytest.FixtureRequest. A pytest `request` object that provides access to command-line options.

    Command-line options:

        --csvpath: Path to the main CSV file to be loaded into a DataFrame.

        --columns: Comma-separated list of column names to be returned.

        --events_csvpath: (Optional) Path to an events CSV file containing timestamp ranges.

        --timestamp_column: Name of the timestamp column in the main DataFrame.

        --event_begin_column: Name of the column in the events CSV file indicating the start of an event.

        --event_end_column: Name of the column in the events CSV file indicating the end of an event.

    Returns:

        tuple: A tuple containing:

            - df (pd.DataFrame): The processed DataFrame.

            - columns (list): A list of column names specified by the `--columns` option.

    Notes:

        - If the `--events_csvpath` option is provided, the fixture will label rows in the
        DataFrame with a new column `label` based on whether their timestamp falls within
        any of the event ranges specified in the events CSV file.

        - The `timestamp_column` in the main DataFrame is converted to a datetime type if it
        is not already in that format.

    """
    csv_path = request.config.getoption("--csvpath")
    columns = request.config.getoption("--columns").split(',')
    events_csv_path = request.config.getoption("--events_csvpath")
    timestamp_column = request.config.getoption("--timestamp_column")
    event_begin_column = request.config.getoption("--event_begin_column")
    event_end_column = request.config.getoption("--event_end_column")
    
    df = pd.read_csv(csv_path)
    
    if events_csv_path:
        events_df = pd.read_csv(events_csv_path)
        
        if timestamp_column in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df[timestamp_column]):
                df[timestamp_column] = pd.to_datetime(df[timestamp_column])
        
        df['label'] = 0
        
        for _, event in events_df.iterrows():
            begin_date = pd.to_datetime(event[event_begin_column])
            end_date = pd.to_datetime(event[event_end_column])
            
            mask = (df[timestamp_column] >= begin_date) & (df[timestamp_column] <= end_date)
            df.loc[mask, 'label'] = 1
                
    return df, columns


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """

    This pytest hook implementation modifies the test report generation process.
    The `pytest_runtest_makereport` function is executed during the test report creation
    phase and allows customization of the report for failed test cases.

    Hook Details:

    - `@pytest.hookimpl(tryfirst=True, hookwrapper=True)`:

        - `tryfirst=True`: Ensures this hook runs before other hooks of the same type.

        - `hookwrapper=True`: Allows wrapping the execution of the hook, enabling pre- and post-processing.

    Parameters:

        - `item`:pytest.Item. The test item being executed.

        - `call`: pytest.CallInfo. The test call object containing information about the test execution.

    Behavior:

        - Captures the test outcome using `yield` and retrieves the result.

        - Checks if the test phase (`rep.when`) is "call" and if the test has failed (`rep.failed`).

        - Attempts to capture standard output (`capstdout`) and standard error (`capstderr`) from the `call` object if available.

        - Updates the `rep.longrepr` attribute with the captured output or an empty string if no output is available.
        
    This hook is useful for customizing the representation of failed test cases in the test report.

    """
    outcome = yield
    rep = outcome.get_result()

    if rep.when == "call" and rep.failed:
        cap_stdout = call.capstdout if hasattr(call, 'capstdout') else ""
        cap_stderr = call.capstderr if hasattr(call, 'capstderr') else ""
        
        rep.longrepr = cap_stdout or cap_stderr or ""