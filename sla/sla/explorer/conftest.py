import pytest
import pandas as pd
import datetime

def pytest_addoption(parser):
    parser.addoption("--csvpath", action="store", help="Path to the CSV file")
    parser.addoption("--columns", action="store", help="Column names for testing")
    parser.addoption("--events_csvpath", action="store", help="Path to the events CSV file with begin and end dates")
    parser.addoption("--timestamp_column", action="store", default="timestamp", 
                    help="Name of the timestamp column in the main CSV")
    parser.addoption("--event_begin_column", action="store", default="begin_date", 
                    help="Name of the event begin date column in the events CSV")
    parser.addoption("--event_end_column", action="store", default="end_date", 
                    help="Name of the event end date column in the events CSV")


@pytest.fixture(scope="module")
def df_and_columns(request):
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
    outcome = yield
    rep = outcome.get_result()

    if rep.when == "call" and rep.failed:
        cap_stdout = call.capstdout if hasattr(call, 'capstdout') else ""
        cap_stderr = call.capstderr if hasattr(call, 'capstderr') else ""
        
        rep.longrepr = cap_stdout or cap_stderr or ""
