import pytest
import pandas as pd

def pytest_addoption(parser):
    parser.addoption("--csvpath", action="store", help="Path to the CSV file")
    parser.addoption("--columns", action="store", help="Column names for testing")

@pytest.fixture(scope="module")
def df_and_columns(request):
    csv_path = request.config.getoption("--csvpath")
    columns = request.config.getoption("--columns").split(',')
    df = pd.read_csv(csv_path)
    return df, columns