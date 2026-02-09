from .hsm_parser import HSMParser
from .hdc_parser import HDCParser
from .ibmmq_parser import IBMMQParser
from .was_parser import WASParser

class LogIngestor:
    """
    Class LogIngestor

    A utility class for ingesting and parsing log files based on their type. 
    This class uses specific parsers to handle different log formats.

    Attributes:

        parsers (dict): A mapping of log type identifiers (str) to their corresponding parser classes.

    """
    parsers = {
        'HSM': HSMParser,
        'HDC': HDCParser,
        'IBM_MQ': IBMMQParser,
        'WAS': WASParser
    }

    @classmethod
    def ingest(cls, file_path: str, log_type: str):
        """
            Ingests and parses a log file based on the specified log type.

            Args:

                file_path (str): The path to the log file to be ingested.

                log_type (str): The type of the log file (e.g., 'HSM', 'HDC', 'IBM_MQ', 'WAS').

            Returns:

                Any: The parsed log data as returned by the corresponding parser.

            Raises:
            
                ValueError: If the specified log type is not supported.
        """
        parser_class = cls.parsers.get(log_type.upper())
        if not parser_class:
            raise ValueError(f"Unsupported log type: {log_type}")
        parser = parser_class(file_path)
        return parser.parse()