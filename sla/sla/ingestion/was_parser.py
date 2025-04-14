import re
import pandas as pd
from .base_parser import BaseLogParser

class WASParser(BaseLogParser):
    def parse(self) -> pd.DataFrame:
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