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
    Add command line options to pytest.
    
    Parameters
    ----------
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
    Fixture to load and prepare dataframes with event labels.
    
    Reads a CSV file of time-series data and optionally applies event labels
    based on a second CSV containing event time periods.
    
    Parameters
    ----------
    request : pytest.FixtureRequest
        The pytest request object for accessing command line options.
        
    Returns
    -------
    tuple
        A tuple containing (dataframe, columns_list) with the loaded and
        labeled dataframe and list of column names for testing.
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
    Pytest hook for enhancing test failure reports.
    
    Captures stdout and stderr output during test execution and includes
    it in the test report when a test fails.
    
    Parameters
    ----------
    item : pytest.Item
        Test item being executed.
    call : pytest.CallInfo
        Information about the test execution.
    
    Yields
    ------
    _pytest.runner.CallInfo
        The call info outcome for further processing by pytest.
    """
    outcome = yield
    rep = outcome.get_result()

    if rep.when == "call" and rep.failed:
        cap_stdout = call.capstdout if hasattr(call, 'capstdout') else ""
        cap_stderr = call.capstderr if hasattr(call, 'capstderr') else ""
        
        rep.longrepr = cap_stdout or cap_stderr or ""