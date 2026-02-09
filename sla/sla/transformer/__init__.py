"""Transformer module.

This module provides functionality for performing different types of aggregations 
over structured dataframes. It includes:

- StringAggregator: Allows aggregation over string values within a given period 
    of time, such as 30 seconds.
    
- RollingAgregator: Enables rolling window aggregations for structured data.

These tools are useful for processing and analyzing time-series or event-based 
data with flexible aggregation strategies.

"""
from .rolling_aggregate import RollingAgregator

from .string_aggregate import StringAggregator




__all__ = ['RollingAgregator', 'StringAggregator']