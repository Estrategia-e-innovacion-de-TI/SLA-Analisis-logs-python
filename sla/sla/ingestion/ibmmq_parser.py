import re
import pandas as pd
from .base_parser import BaseLogParser

class IBMMQParser(BaseLogParser):
    def parse(self) -> pd.DataFrame:
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