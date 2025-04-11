from abc import ABC, abstractmethod
import pandas as pd

class BaseLogParser(ABC):
    def __init__(self, file_path: str):
        self.file_path = file_path

    @abstractmethod
    def parse(self) -> pd.DataFrame:
        """
        Patrón de Diseño de tipo Abstract Factory
        Método abstracto que debe ser implementado por las subclases.
        """
        pass