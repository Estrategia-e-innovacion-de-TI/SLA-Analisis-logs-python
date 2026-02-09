import re
import pandas as pd
from .base_parser import BaseLogParser

class HSMParser(BaseLogParser):
    """
    HSMParser is a log parser class that extends the BaseLogParser to parse logs 
    with a specific format and return the parsed data as a pandas DataFrame.

    Attributes:

        file_path (str): The path to the log file to be parsed.
            
    """
    def parse(self) -> pd.DataFrame:
        """
        Parses the log file and extracts structured data based on a predefined 
            regex pattern. Returns the parsed data as a pandas DataFrame.

        The log format expected is:

                <date> <time> [<level>] [<ip>] - [<category>] <message>

        Example log line:

                2023 Mar 15 12:34:56 [INFO] [192.168.1.1] - [CATEGORY] Some log message

        Returns:

                pd.DataFrame: A DataFrame containing the parsed log entries with 
                the following columns:

                    - date: The date of the log entry (e.g., "2023 Mar 15").

                    - time: The time of the log entry (e.g., "12:34:56").

                    - level: The log level (e.g., "INFO").

                    - ip: The IP address associated with the log entry.

                    - category: The category of the log entry.
                    
                    - message: The log message.
        """
        pattern = re.compile(
            r'(?P<date>\d{4}\s\w{3}\s+\d{1,2})\s(?P<time>\d{2}:\d{2}:\d{2})\s+\[(?P<level>[^\]]+)\]\s+\[(?P<ip>[\d.]+)\]\s+-\s+\[(?P<category>[^\]]+)\]\s+(?P<message>.+)'
        )
        entries = []
        with open(self.file_path, 'r') as file:
            for line in file:
                match = pattern.match(line)
                if match:
                    entries.append(match.groupdict())
        return pd.DataFrame(entries)