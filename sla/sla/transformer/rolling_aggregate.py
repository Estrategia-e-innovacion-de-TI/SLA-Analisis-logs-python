"""Rolling aggregate transformer for anomaly detection pipelines."""
import numpy as np
import pandas as pd
from typing import Union, List, Optional, Dict, Any, Callable
from sklearn.base import BaseEstimator, TransformerMixin


class RollingAgregator(BaseEstimator, TransformerMixin):
    """Rolling aggregate transformer for anomaly detection pipelines.
    
    This transformer computes rolling window aggregations on time series data,
    suitable for use in scikit-learn pipelines for anomaly detection.
    
    Parameters
    ----------
    window_size : int
        Size of the rolling window.
    aggregation_functions : str, callable, or list, default='mean'
        Functions to apply to the rolling window. Can be:
        - String naming a pandas rolling method ('mean', 'std', etc.)
        - Callable function to apply to each window
        - List of strings or callables for multiple aggregations
    columns : str or list of str, optional
        Specific columns to apply rolling aggregations to.
        If None, applies to all numeric columns.
    window_type : str, optional
        Type of window ('fixed', 'expanding', or 'ewm').
        Fixed rolling window (traditional rolling window)
        Expanding window (growing window size)
        Exponentially weighted moving window (EWMA)
    min_periods : int, optional
        Minimum number of observations required to have a value.
        If None, defaults to window_size.
    center : bool, default=False
        If True, the window will be centered around the current point.
    suffix : str, default='_rolling'
        Suffix to append to column names for the new features.
    kwargs : dict
        Additional keyword arguments to pass to the rolling function.
    """
    
    def __init__(
        self,
        window_size: int,
        aggregation_functions: Union[str, Callable, List] = 'mean',
        columns: Optional[Union[str, List[str]]] = None,
        window_type: str = 'fixed',
        min_periods: Optional[int] = None,
        center: bool = False,
        suffix: str = '_rolling',
        **kwargs
    ):
        self.window_size = window_size
        self.aggregation_functions = aggregation_functions
        self.columns = columns
        self.window_type = window_type
        self.min_periods = min_periods if min_periods is not None else window_size
        self.center = center
        self.suffix = suffix
        self.kwargs = kwargs
        self._validate_params()
        
    def _validate_params(self) -> None:
        """Validate the parameters passed to the transformer."""
        valid_window_types = ['fixed', 'expanding', 'ewm']
        if self.window_type not in valid_window_types:
            raise ValueError(
                f"window_type must be one of {valid_window_types}, "
                f"got {self.window_type} instead."
            )
        
        if self.window_size < 1:
            raise ValueError(
                f"window_size must be at least 1, got {self.window_size} instead."
            )
            
    def _get_aggregation_list(self) -> List:
        """Convert aggregation_functions to a list of functions."""
        if isinstance(self.aggregation_functions, (str, Callable)):
            return [self.aggregation_functions]
        return self.aggregation_functions
    
    def _get_feature_names(self, X: pd.DataFrame) -> List[str]:
        """Get feature names for the transformed DataFrame."""
        if self.columns is None:
            # Use all numeric columns
            columns = X.select_dtypes(include=np.number).columns.tolist()
        elif isinstance(self.columns, str):
            columns = [self.columns]
        else:
            columns = self.columns
            
        feature_names = []
        agg_funcs = self._get_aggregation_list()
        
        for col in columns:
            for agg in agg_funcs:
                agg_name = agg if isinstance(agg, str) else agg.__name__
                feature_names.append(f"{col}{self.suffix}_{agg_name}")
                
        return feature_names
    
    def fit(self, X: pd.DataFrame, y: Optional[Any] = None) -> 'RollingAgregator':
        """Fit the transformer (no-op).
        
        Parameters
        ----------
        X : DataFrame
            Input data to fit.
        y : array-like or None, default=None
            Targets. Ignored in this transformer.
            
        Returns
        -------
        self : RollingAggregateTransformer
            Returns self.
        """
        # This transformer doesn't need to learn anything from the data
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply rolling aggregations to the input data.
        
        Parameters
        ----------
        X : DataFrame
            Input data to transform.
            
        Returns
        -------
        DataFrame
            Transformed data with rolling aggregations.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
            
        if self.columns is None:
            # Use all numeric columns
            columns = X.select_dtypes(include=np.number).columns.tolist()
        elif isinstance(self.columns, str):
            columns = [self.columns]
        else:
            columns = self.columns
            
        # Validate columns exist in DataFrame
        missing_cols = [col for col in columns if col not in X.columns]
        if missing_cols:
            raise ValueError(f"Columns {missing_cols} not found in input DataFrame")
            
        result_df = X.copy()
        agg_funcs = self._get_aggregation_list()
        
        for col in columns:
            # Choose the window type
            if self.window_type == 'fixed':
                window = X[col].rolling(
                    window=self.window_size,
                    min_periods=self.min_periods,
                    center=self.center,
                    **self.kwargs
                )
            elif self.window_type == 'expanding':
                window = X[col].expanding(
                    min_periods=self.min_periods,
                    **self.kwargs
                )
            elif self.window_type == 'ewm':
                window = X[col].ewm(
                    span=self.window_size,
                    min_periods=self.min_periods,
                    **self.kwargs
                )
            
            # Apply each aggregation function
            for agg in agg_funcs:
                if isinstance(agg, str):
                    if hasattr(window, agg):
                        method = getattr(window, agg)
                        agg_result = method()
                    else:
                        raise ValueError(f"Method {agg} not available for {self.window_type} window")
                else:
                    # agg is a callable
                    agg_result = window.apply(agg, raw=True)
                
                agg_name = agg if isinstance(agg, str) else agg.__name__
                result_df[f"{col}{self.suffix}_{agg_name}"] = agg_result
                
        return result_df
    
    def get_feature_names_out(self, input_features=None) -> List[str]:
        """Get output feature names for transformation.
        
        Parameters
        ----------
        input_features : array-like of str or None, default=None
            Input features. Ignored in this transformer.
            
        Returns
        -------
        list of str
            Output feature names.
        """
        if not hasattr(self, '_feature_names_out'):
            if isinstance(input_features, pd.DataFrame):
                self._feature_names_out = self._get_feature_names(input_features)
            else:
                raise ValueError(
                    "Cannot determine feature names without fitting or "
                    "providing a DataFrame as input_features."
                )
        return self._feature_names_out
    
    def fit_transform(self, X: pd.DataFrame, y: Optional[Any] = None) -> pd.DataFrame:
        """Fit and transform in one step.
        
        Parameters
        ----------
        X : DataFrame
            Input data to fit and transform.
        y : array-like or None, default=None
            Targets. Ignored in this transformer.
            
        Returns
        -------
        DataFrame
            Transformed data with rolling aggregations.
        """
        self._feature_names_out = self._get_feature_names(X)
        return self.transform(X)


# Example usage
if __name__ == "__main__":
    # Create sample time series data
    np.random.seed(42)
    date_range = pd.date_range(start='2023-01-01', periods=100, freq='D')
    normal_data = np.sin(np.linspace(0, 4*np.pi, 100)) + np.random.normal(0, 0.2, 100)
    
    # Insert some anomalies
    anomaly_indices = [25, 50, 75]
    normal_data[anomaly_indices] = [3, -3, 4]
    
    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': date_range,
        'value': normal_data
    })
    
    # Initialize transformer
    transformer = RollingAgregator(
        window_size=10,
        aggregation_functions=['mean', 'std', 'min', 'max'],
        columns='value'
    )
    
    # Apply transformation
    result_df = transformer.fit_transform(df)
    
    # Display results
    print(result_df.head(10))