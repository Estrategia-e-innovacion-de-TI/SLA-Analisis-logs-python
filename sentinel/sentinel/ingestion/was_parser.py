import re
import pandas as pd
from .base_parser import BaseLogParser

class WASParser(BaseLogParser):
    """
    WASParser is a log parser class that extends the BaseLogParser to process and extract structured data 
    from WebSphere Application Server (WAS) logs.        
    """
    def parse(self) -> pd.DataFrame:
        """
        Parses the WAS log file specified by `self.file_path` and extracts relevant information 
            into a pandas DataFrame. The method uses regular expressions to extract fields such as 
            timestamp, log level, service name, transaction ID, account details, transaction amount, 
            response code, and other metadata.

        Returns:

                pd.DataFrame: A DataFrame containing the parsed log data. If the log file cannot be 
                read or is empty, an empty DataFrame is returned.

        Attributes Extracted:

            - timestamp: The timestamp of the log entry.

            - log_level: The severity level of the log (e.g., INFO, ERROR).

            - service: The name of the service or component generating the log.

            - transaction_id: The unique identifier for the transaction.

            - account_from: The account ID of the sender.

            - account_to: The account ID of the receiver.

            - amount: The transaction amount.

            - response_code: The response code from the transaction.

            - trama_rq: The request payload (TramaRQ) sent to the iSeries system.

            - trama_uuid: The UUID associated with the TramaRQ.

            - trama_rs: The response payload (TramaRS) received from the iSeries system.

            - process_transaction: The ID of the transaction being processed.

            - connection_group: The connection group associated with the transaction.

        Raises:

            - Exception: If the log file cannot be opened or read, an empty DataFrame is returned instead 
            of raising an exception.

        Usage:

            parser = WASParser(file_path="path/to/log/file.log")
            
            parsed_data = parser.parse()
        """
        try:
            with open(self.file_path, "r", encoding="utf-8", errors="replace") as file:
                raw_log = file.read()
        except Exception:
            return pd.DataFrame()

        log_lines = raw_log.strip().split("\n")
        log_data = []
        current_log = {}

        regex_log = r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)\] \[(\w+)\] \[WebContainer : \d+\] (\S+)"
        regex_transaction_id = r'TRNUID="(\d+)"'
        regex_account_from = r"<ACCTFROM>.*?<ACCTID>([\d-]+)</ACCTID>"
        regex_account_to = r"<ACCTTO>.*?<ACCTID>([\d-]+)</ACCTID>"
        regex_amount = r"<TRNAMT>(\d+\.\d+)</TRNAMT>"
        regex_response_code = r'<STATUS CODE="(\d+)"'
        regex_trama_rq = r"TramaRQ iSeries:(.*)"
        regex_trama_uuid = r"TramaRQ UUID:([\w\d-]+)"
        regex_trama_rs = r"TramaRS iSeries:(.*)"
        regex_process_transaction = r"process transaction (\d+)"
        regex_connection_group = r"ConnectionGroup: (\S+)"

        for line in log_lines:
            match_log = re.search(regex_log, line)
            if match_log:
                if current_log:
                    log_data.append(current_log)
                current_log = {
                    "timestamp": match_log.group(1),
                    "log_level": match_log.group(2),
                    "service": match_log.group(3),
                    "transaction_id": None,
                    "account_from": None,
                    "account_to": None,
                    "amount": None,
                    "response_code": None,
                    "trama_rq": None,
                    "trama_uuid": None,
                    "trama_rs": None,
                    "process_transaction": None,
                    "connection_group": None,
                }

            for field, pattern in {
                "transaction_id": regex_transaction_id,
                "account_from": regex_account_from,
                "account_to": regex_account_to,
                "amount": regex_amount,
                "response_code": regex_response_code,
                "trama_rq": regex_trama_rq,
                "trama_uuid": regex_trama_uuid,
                "trama_rs": regex_trama_rs,
                "process_transaction": regex_process_transaction,
                "connection_group": regex_connection_group,
            }.items():
                match = re.search(pattern, line)
                if match:
                    current_log[field] = match.group(1)

        if current_log:
            log_data.append(current_log)

        return pd.DataFrame(log_data)