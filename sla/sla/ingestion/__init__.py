"""
This module is responsible for converting raw, unstructured log files into structured pandas DataFrames. 
It provides a base parser class and specific parsers for various log types, including:

- WAS (WebSphere Application Server)
- HSM (Hardware Security Module)
- HDC (High-Density Computing)
- IBMMQ (IBM Message Queue)

Each parser is designed to handle the unique structure and format of its respective log type, ensuring 
accurate and efficient transformation of log data into a structured format suitable for analysis and processing.

For those log types that have not yet been considered, custom implementations can be created using the base parser 
as a foundation.
"""

from .ingestor import LogIngestor
from .base_parser import BaseLogParser
from .hdc_parser import HDCParser
from .hsm_parser import HSMParser
from .ibmmq_parser import IBMMQParser
from .was_parser import WASParser

__all__ = [LogIngestor, BaseLogParser, HDCParser, HSMParser, IBMMQParser, WASParser]