import pandas as pd
import numpy as np
from typing import List, Dict, Union, Optional, Callable
from datetime import timedelta

class StringAggregator:
    """
    Clase para realizar agregaciones flexibles en DataFrames con soporte para ventanas temporales.

    Permite definir métricas personalizadas para cualquier tipo de columna.
    """

    def __init__(self, dataframe: pd.DataFrame, timestamp_column: str):
        """
        Inicializa el agregador con un DataFrame y columna de timestamp.

        Args:
            dataframe (pd.DataFrame): DataFrame de entrada
            timestamp_column (str): Nombre de la columna de timestamp
        """
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("El input debe ser un DataFrame de pandas")
        
        if timestamp_column not in dataframe.columns:
            raise ValueError(f"La columna {timestamp_column} no existe en el DataFrame")

        self.df = dataframe.copy()
        self.df[timestamp_column] = pd.to_datetime(self.df[timestamp_column])
        self.timestamp_column = timestamp_column

    def create_time_aggregation(
        self, 
        time_window: str = '5min', 
        column_metrics: Optional[Dict[str, List[Union[str, Callable]]]] = None,
        custom_metrics: Optional[Dict[str, Callable]] = None,
        category_count_columns: Optional[Dict[str, List[str]]] = None
    ) -> pd.DataFrame:
        """
        Realiza agregaciones flexibles por ventana de tiempo.

        Args:
            time_window (str, optional): Ventana temporal. Defaults to '5min'.
            column_metrics (Dict[str, List], optional): Métricas para columnas específicas.
            custom_metrics (Dict[str, Callable], optional): Métricas personalizadas.
            category_count_columns (Dict[str, List[str]], optional): Columnas para conteo por categoría.

        Returns:
            pd.DataFrame: DataFrame con agregaciones
        """
        # Agrupar por ventana de tiempo
        grouped = self.df.groupby(pd.Grouper(key=self.timestamp_column, freq=time_window))
        
        # Diccionario para almacenar resultados
        results = pd.DataFrame(index=grouped.groups.keys())
        
        # Métricas por defecto si no se proporcionan
        if column_metrics is None:
            column_metrics = {
                col: ['count', 'nunique'] 
                for col in self.df.columns 
                if col != self.timestamp_column
            }
        
        # Aplicar métricas por columna
        for column, metrics in column_metrics.items():
            if column not in self.df.columns:
                raise ValueError(f"La columna {column} no existe en el DataFrame")
            
            for metric in metrics:
                col_name = f"{column}_{metric}"
                
                # Manejar métricas de cadena
                if isinstance(metric, str):
                    if metric == 'count':
                        results[col_name] = grouped[column].count()
                    elif metric == 'nunique':
                        results[col_name] = grouped[column].nunique()
                    elif metric == 'mode':
                        results[col_name] = grouped[column].agg(lambda x: x.mode().iloc[0] if not x.empty else np.nan)
                
                # Manejar métricas personalizadas de cadena
                elif callable(metric):
                    results[col_name] = grouped[column].agg(metric)
        
        # Conteo por categorías específicas
        if category_count_columns:
            for column, categories in category_count_columns.items():
                if column not in self.df.columns:
                    raise ValueError(f"La columna {column} no existe en el DataFrame")
                
                for category in categories:
                    col_name = f"{column}_{category}_count"
                    results[col_name] = grouped.apply(lambda x: (x[column] == category).sum())
        
        # Métricas de tiempo entre eventos en segundos
        def avg_time_between_events(group):
            if len(group) <= 1:
                return 0
            times = group[self.timestamp_column].sort_values()
            diffs = times.diff().dropna()
            return diffs.dt.total_seconds().mean()
        
        # Métricas adicionales de tiempo entre eventos
        def min_time_between_events(group):
            if len(group) <= 1:
                return 0
            times = group[self.timestamp_column].sort_values()
            diffs = times.diff().dropna()
            return diffs.dt.total_seconds().min()
        
        def max_time_between_events(group):
            if len(group) <= 1:
                return 0
            times = group[self.timestamp_column].sort_values()
            diffs = times.diff().dropna()
            return diffs.dt.total_seconds().max()
        
        results['avg_time_between_events_seconds'] = grouped.apply(avg_time_between_events)
        results['min_time_between_events_seconds'] = grouped.apply(min_time_between_events)
        results['max_time_between_events_seconds'] = grouped.apply(max_time_between_events)
        
        # Aplicar métricas personalizadas globales
        if custom_metrics:
            for metric_name, metric_func in custom_metrics.items():
                results[metric_name] = grouped.apply(metric_func)
        
        return results

# Ejemplo de uso
def example_usage():
    """Ejemplo de cómo usar la clase FlexibleAggregator"""
    # Crear un DataFrame de ejemplo con más variedad de datos
    np.random.seed(42)  # Para reproducibilidad
    
    # Generar timestamps con más densidad
    timestamps = pd.date_range(start='2024-01-01', end='2024-01-02', freq='2min')
    
    data = {
        'timestamp': timestamps,
        'category': np.random.choice(['web', 'mobile', 'desktop'], len(timestamps)),
        'level': np.random.choice(['info', 'warning', 'error'], len(timestamps)),
        'ip': [f'192.168.1.{i%20}' for i in range(len(timestamps))],
        'response_time': np.random.uniform(10, 500, len(timestamps))
    }
    df = pd.DataFrame(data)
    
    # Crear instancia del agregador
    aggregator = StringAggregator(df, 'timestamp')
    
    # Definir métricas personalizadas
    column_metrics = {
        'category': ['count', 'nunique'],
        'level': ['count', 'mode'],
        'ip': ['nunique'],
        'response_time': ['mean', 'max', 'min']
    }
    
    # Definir conteo por categorías
    category_count_columns = {
        'level': ['info', 'warning', 'error'],
        'category': ['web', 'mobile', 'desktop']
    }
    
    # Métrica personalizada global
    def count_high_latency(group):
        return (group['response_time'] > 300).sum()
    
    # Realizar agregación
    result = aggregator.create_time_aggregation(
        time_window='5min', 
        column_metrics=column_metrics,
        category_count_columns=category_count_columns,
        custom_metrics={'high_latency_count': count_high_latency}
    )
    
    # Imprimir resultados
    print(result)

if __name__ == "__main__":
    example_usage()