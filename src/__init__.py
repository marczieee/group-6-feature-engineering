"""
Feature Engineering Pipeline - Group 6
"""

from .derive_computed_columns import derive_computed_columns, process_csv as derive_csv
from .encode_categorical_features import encode_categorical_features, process_csv as encode_csv
from .bin_numeric_ranges import bin_numeric_ranges, process_csv as bin_csv
from .time_based_feature_extraction import time_based_feature_extraction, process_csv as time_csv
from .flag_anomalies_column import flag_anomalies_column, process_csv as anomaly_csv

__version__ = "1.0.0"
__author__ = "Group 6"

__all__ = [
    'derive_computed_columns',
    'encode_categorical_features',
    'bin_numeric_ranges',
    'time_based_feature_extraction',
    'flag_anomalies_column',
    'derive_csv',
    'encode_csv',
    'bin_csv',
    'time_csv',
    'anomaly_csv',
]
