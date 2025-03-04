from .rolling_aggregate import RollingAgregator
from .string_aggregate import StringAggregator

"""Module of detectors.

A detector detects anomalous time points from time series.

"""


__all__ = ['RollingAgregator', 'StringAggregator']