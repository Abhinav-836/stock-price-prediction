"""
Stock Price Prediction Package - FIXED

Fixes vs previous version:
  * Printed 3 banner lines on every import (spams Streamlit logs on every session).
  * Eagerly imported preprocess/train/predict/visualization (matplotlib, xgboost, lightgbm...)
    whenever ANY submodule was imported, e.g. `from src.data_loader import ...`, slowing cold
    starts, and the bare `except ImportError: pass` silently hid real import errors.
    Main classes are now loaded lazily on first access.
"""
import importlib
import logging

__version__ = '2.1.0'
__author__ = 'Stock Prediction Team'

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

PACKAGE_INFO = {
    'name': 'stock-prediction',
    'version': __version__,
    'author': __author__,
    'description': 'Advanced Stock Price Prediction with Machine Learning',
    'keywords': ['stock', 'prediction', 'machine-learning', 'finance', 'time-series'],
}

__all__ = [
    'data_loader', 'preprocess', 'train_model', 'evaluate', 'predict', 'visualization',
    'ensemble_model', 'advanced_features', 'auto_train', 'advanced_evaluate', 'lstm_model',
    'StockDataLoader', 'StockDataPreprocessor', 'ModelTrainer', 'AdvancedEvaluator',
    'PredictionEngine', 'StockVisualizer',
]

_LAZY = {
    'StockDataLoader': 'data_loader',
    'StockDataPreprocessor': 'preprocess',
    'ModelTrainer': 'train_model',
    'AdvancedEvaluator': 'evaluate',
    'PredictionEngine': 'predict',
    'StockVisualizer': 'visualization',
}


def __getattr__(name):
    if name in _LAZY:
        module = importlib.import_module(f'.{_LAZY[name]}', __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def get_version() -> str:
    return __version__


def get_info() -> dict:
    return PACKAGE_INFO.copy()


def setup_logging(level: str = 'INFO'):
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)