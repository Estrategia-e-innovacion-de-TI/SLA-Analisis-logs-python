import re
import pandas as pd
from .base_parser import BaseLogParser

class IBMMQParser(BaseLogParser):
    """
    IBMMQParser

    A parser for IBM MQ log files that extracts structured information from log entries
    and returns it as a pandas DataFrame.
            
    Raises:

        None: Any exceptions during file reading are caught and result in an empty DataFrame.
    """
    def parse(self) -> pd.DataFrame:
        """
        Parses the IBM MQ log file specified by `self.file_path` and extracts
            relevant information from each log entry. The extracted data is returned
            as a pandas DataFrame with the following columns:

                - Process: The process identifier.

                - Program: The program name.

                - Host: The host name.

                - Installation: The installation identifier.

                - VRMF: The version, release, modification, and fix level.

                - QMgr: The queue manager name.

                - Time: The timestamp of the log entry.

                - RemoteHost: The remote host name.

                - ArithInsert: The first arithmetic insert value.

                - CommentInsert1: The first comment insert value.

                - CommentInsert2: The second comment insert value.

                - CommentInsert3: The third comment insert value.

                - CodigoAMQ_Error: The AMQ error code.

                - DescripcionCodigoAMQ_Error: The description of the AMQ error code.

                - Explanation: The explanation of the error.
                
                - Action: The recommended action for the error.

        Returns:
        
                pd.DataFrame: A DataFrame containing the extracted log information.
                If the file cannot be read or parsed, an empty DataFrame is returned.
        """
        try:
            with open(self.file_path, "r") as file:
                content = file.read()
        except Exception:
            return pd.DataFrame()

        entries = [entry.strip() for entry in re.split(r'(?=----- amq)', content) if '-' in entry]

        patterns = {
            "Process": r'Process\((.*?)\)',
            "Program": r'Program\((.*?)\)',
            "Host": r'Host\((.*?)\)',
            "Installation": r'Installation\((.*?)\)',
            "VRMF": r'VRMF\((.*?)\)',
            "QMgr": r'QMgr\((.*?)\)',
            "Time": r'Time\((.*?)\)',
            "RemoteHost": r'RemoteHost\((.*?)\)',
            "ArithInsert": r'ArithInsert1\((.*?)\)',
            "CommentInsert1": r'CommentInsert1\((.*?)\)',
            "CommentInsert2": r'CommentInsert2\((.*?)\)',
            "CommentInsert3": r'CommentInsert3\((.*?)\)',
            "CodigoAMQ_Error": r'(AMQ[0-9]+[A-Z]?)',
            "DescripcionCodigoAMQ_Error": r'AMQ[0-9]+[A-Z]?:(.*?)\.',
            "Explanation": r'EXPLANATION:(.*?)ACTION:',
            "Action": r'ACTION:(.*)'
        }

        def extract_info(log):
            return {
                key: (re.search(pat, log, re.DOTALL).group(1).strip() if re.search(pat, log, re.DOTALL) else "")
                for key, pat in patterns.items()
            }

        return pd.DataFrame([extract_info(e) for e in entries])