"""
E-Gen framework for hypothesis testing and data analysis.
"""

from .data_loader import DataManager, ExperimentalDataLoader
from .agents.falsification import SequentialFalsificationTest

__version__ = "0.1.0"

__all__ = [
    'DataManager',
    'ExperimentalDataLoader',
    'SequentialFalsificationTest'
] 