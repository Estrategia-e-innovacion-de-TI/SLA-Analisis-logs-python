from abc import ABC, abstractmethod
import pandas as pd

class BaseLogParser(ABC):
    """
    BaseLogParser is an abstract base class that defines a blueprint for log parsers.

    It provides a common interface for parsing log files and extracting structured data.

    Subclasses must implement the parse method to define their specific parsing logic.

    This class is designed to be extended by specific log parsers for different log formats.
    
    Attributes:
        
        file_path (str): The path to the log file that will be parsed.

    Notes:
        This class follows the Abstract Factory design pattern, ensuring that subclasses implement the required
        parsing functionality.
    """
    def __init__(self, file_path: str):
        self.file_path = file_path

    @abstractmethod
    def parse(self) -> pd.DataFrame:
        """
        Abstract method to be implemented by subclasses.

        This method should define the logic for parsing data and return
        a pandas DataFrame containing the parsed information.

        Returns:
        
            pd.DataFrame: A DataFrame containing the parsed data.
        """
        pass