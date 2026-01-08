import re
import pandas as pd
from .base_parser import BaseLogParser

LOG_PATTERN = re.compile(
    r"\[(\d{2}/\d{2}/\d{2} \d{2}:\d{2}:\d{2}:\d{3}) COT\]\s+(\w+)\s+(\w+)\s+(\w)\s+(.+)",
    re.DOTALL
)

def extract_error_code(description):
    match = re.search(r'([A-Z]+\d+[A-Z]+:)', description)
    return match.group(1) if match else None

class HDCParser(BaseLogParser):
    """
    HDCParser is a log parser class that extends the BaseLogParser to parse logs
    from a specified file path and return the data as a pandas DataFrame.
    """
    def parse(self) -> pd.DataFrame:
        """
            Reads the log file, processes each log line using a predefined pattern,
            and extracts relevant information into a structured pandas DataFrame.

            Returns:

                pd.DataFrame: A DataFrame containing the parsed log data with the following columns:

                    - timestamp: The timestamp of the log entry.

                    - thread_id: The thread ID associated with the log entry.

                    - log_source: The source of the log entry.

                    - log_type: The type of the log entry.

                    - message: The log message.
                    
                    - error_code: The extracted error code from the log message, if available.

            Raises:
            
                Exception: If the file cannot be opened or read, an empty DataFrame is returned.
        """
        try:
            with open(self.file_path, "r", encoding="utf-8") as file:
                raw_log = file.read()
        except Exception:
            return pd.DataFrame()

        log_lines = [line.strip() for line in raw_log.strip().split('\n') if line.strip()]
        data = []
        for line in log_lines:
            match = LOG_PATTERN.match(line)
            if match:
                data.append({
                    'timestamp': match.group(1),
                    'thread_id': match.group(2),
                    'log_source': match.group(3),
                    'log_type': match.group(4),
                    'message': match.group(5),
                    'error_code': extract_error_code(match.group(5))
                })
        return pd.DataFrame(data)