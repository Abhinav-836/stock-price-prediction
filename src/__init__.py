"""
Stock Price Prediction Package - Enhanced Version
"""
import logging
from typing import Optional

__version__ = '2.0.0'
__author__ = 'Stock Prediction Team'

# Configure package logger
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

# Package metadata
PACKAGE_INFO = {
    'name': 'stock-prediction',
    'version': __version__,
    'author': __author__,
    'description': 'Advanced Stock Price Prediction with Machine Learning',
    'keywords': ['stock', 'prediction', 'machine-learning', 'finance', 'time-series']
}

# Lazy imports to avoid circular dependencies
def get_version() -> str:
    """Return package version"""
    return __version__

def get_info() -> dict:
    """Return package information"""
    return PACKAGE_INFO.copy()

# Import submodules for easier access
__all__ = [
    'data_loader',
    'preprocess',
    'train_model',
    'evaluate',
    'predict',
    'visualization',
    'ensemble_model',
    'advanced_features',
    'auto_train',
    'advanced_evaluate'
]

# Optional: Import main classes for direct access
try:
    from .preprocess import StockDataPreprocessor
    from .train_model import ModelTrainer
    from .evaluate import AdvancedEvaluator
    from .predict import PredictionEngine
    from .visualization import StockVisualizer
except ImportError:
    pass

def setup_logging(level: str = 'INFO'):
    """Setup logging for the package"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    return logging.getLogger(__name__)

print(f"📈 Stock Prediction Package v{__version__} loaded successfully!")
print(f"   Author: {__author__}")
print(f"   Available modules: {', '.join(__all__)}")