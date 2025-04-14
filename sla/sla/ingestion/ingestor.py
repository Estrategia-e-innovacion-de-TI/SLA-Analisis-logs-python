from .hsm_parser import HSMParser
from .hdc_parser import HDCParser
from .ibmmq_parser import IBMMQParser
from .was_parser import WASParser

class LogIngestor:
    parsers = {
        'HSM': HSMParser,
        'HDC': HDCParser,
        'IBM_MQ': IBMMQParser,
        'WAS': WASParser
    }

    @classmethod
    def ingest(cls, file_path: str, log_type: str):
        parser_class = cls.parsers.get(log_type.upper())
        if not parser_class:
            raise ValueError(f"Unsupported log type: {log_type}")
        parser = parser_class(file_path)
        return parser.parse()