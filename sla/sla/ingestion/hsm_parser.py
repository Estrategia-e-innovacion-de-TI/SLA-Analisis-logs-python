import re
import pandas as pd
from .base_parser import BaseLogParser

class HSMParser(BaseLogParser):
    def parse(self) -> pd.DataFrame:
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