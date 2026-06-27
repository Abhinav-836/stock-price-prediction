"""
Helper functions for the stock price prediction project
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
import pickle
import hashlib

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def save_model(model, model_name, models_dir, metadata=None):
    """
    Save a trained model to disk with metadata
    
    Args:
        model: Trained model object
        model_name: Name for the model file
        models_dir: Directory to save the model
        metadata: Additional metadata to save
    """
    os.makedirs(models_dir, exist_ok=True)
    
    # Save model
    filepath = os.path.join(models_dir, f"{model_name}.pkl")
    joblib.dump(model, filepath)
    
    # Save metadata
    if metadata:
        meta_path = os.path.join(models_dir, f"{model_name}_metadata.json")
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
    
    logger.info(f"✅ Model saved: {filepath}")
    return filepath


def load_model(model_name, models_dir, with_metadata=False):
    """
    Load a trained model from disk
    
    Args:
        model_name: Name of the model file
        models_dir: Directory where model is saved
        with_metadata: Whether to also load metadata
        
    Returns:
        Loaded model object or tuple (model, metadata)
    """
    filepath = os.path.join(models_dir, f"{model_name}.pkl")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model not found: {filepath}")
    
    model = joblib.load(filepath)
    
    if with_metadata:
        meta_path = os.path.join(models_dir, f"{model_name}_metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
            return model, metadata
    
    logger.info(f"✅ Model loaded: {filepath}")
    return model


def save_metrics(metrics, filename, output_dir):
    """Save evaluation metrics to JSON file"""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"{filename}.json")
    
    # Convert numpy types to native Python types
    metrics_clean = {}
    for key, value in metrics.items():
        if isinstance(value, (np.integer, np.floating)):
            metrics_clean[key] = float(value)
        elif isinstance(value, np.ndarray):
            metrics_clean[key] = value.tolist()
        else:
            metrics_clean[key] = value
    
    with open(filepath, 'w') as f:
        json.dump(metrics_clean, f, indent=4)
    
    logger.info(f"✅ Metrics saved: {filepath}")
    return filepath


def load_metrics(filename, output_dir):
    """Load evaluation metrics from JSON file"""
    filepath = os.path.join(output_dir, f"{filename}.json")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Metrics file not found: {filepath}")
    
    with open(filepath, 'r') as f:
        metrics = json.load(f)
    
    return metrics


def calculate_returns(prices):
    """Calculate percentage returns from price series"""
    return np.diff(prices) / prices[:-1] * 100


def calculate_log_returns(prices):
    """Calculate log returns from price series"""
    return np.log(prices[1:] / prices[:-1])


def get_timestamp():
    """Get current timestamp as string"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def print_section(title, char='='):
    """Print a formatted section header"""
    print("\n" + char*70)
    print(f"  {title}")
    print(char*70)


def save_dataframe(df, filename, output_dir, format='csv'):
    """Save DataFrame in specified format"""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    
    if format == 'csv':
        df.to_csv(filepath, index=False)
    elif format == 'excel':
        df.to_excel(filepath, index=False)
    elif format == 'json':
        df.to_json(filepath, orient='records', date_format='iso')
    else:
        raise ValueError(f"Unsupported format: {format}")
    
    logger.info(f"✅ Data saved: {filepath}")
    return filepath


def create_feature_hash(df, feature_cols):
    """Create a hash of feature columns for caching"""
    df_subset = df[feature_cols].copy()
    df_string = df_subset.to_string()
    return hashlib.md5(df_string.encode()).hexdigest()


def handle_missing_values(df, strategy='ffill', max_gap=5):
    """Handle missing values intelligently"""
    df = df.copy()
    
    if strategy == 'ffill':
        df = df.fillna(method='ffill')
    elif strategy == 'bfill':
        df = df.fillna(method='bfill')
    elif strategy == 'interpolate':
        df = df.interpolate(method='time', limit=max_gap)
    elif strategy == 'drop':
        df = df.dropna()
    
    return df


def detect_outliers(df, columns, method='iqr', threshold=1.5):
    """Detect outliers in specified columns"""
    outliers = {}
    
    for col in columns:
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - threshold * IQR
            upper = Q3 + threshold * IQR
            outliers[col] = (df[col] < lower) | (df[col] > upper)
        elif method == 'zscore':
            zscore = np.abs((df[col] - df[col].mean()) / df[col].std())
            outliers[col] = zscore > threshold
    
    return pd.DataFrame(outliers)


def normalize_data(df, columns, method='minmax'):
    """Normalize data using different methods"""
    df = df.copy()
    
    if method == 'minmax':
        for col in columns:
            df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
    elif method == 'standard':
        for col in columns:
            df[col] = (df[col] - df[col].mean()) / df[col].std()
    elif method == 'robust':
        for col in columns:
            median = df[col].median()
            q75, q25 = df[col].quantile(0.75), df[col].quantile(0.25)
            iqr = q75 - q25
            df[col] = (df[col] - median) / iqr
    
    return df


def calculate_confidence_interval(predictions, model_type='ensemble', n_bootstrap=1000):
    """Calculate confidence intervals using bootstrap"""
    if model_type == 'ensemble':
        # For ensemble, use standard deviation of ensemble members
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)
        lower = mean_pred - 1.96 * std_pred
        upper = mean_pred + 1.96 * std_pred
        return mean_pred, lower, upper
    
    elif model_type == 'bootstrap':
        # For single model, use bootstrap sampling
        bootstrapped = []
        n_samples = len(predictions)
        
        for _ in range(n_bootstrap):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            bootstrapped.append(predictions[indices])
        
        bootstrapped = np.array(bootstrapped)
        mean_pred = np.mean(bootstrapped, axis=0)
        lower = np.percentile(bootstrapped, 2.5, axis=0)
        upper = np.percentile(bootstrapped, 97.5, axis=0)
        return mean_pred, lower, upper
    
    return predictions, predictions, predictions


def validate_data(df, required_cols):
    """Validate that DataFrame has required columns"""
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return True


def get_data_stats(df, columns=None):
    """Get comprehensive statistics for data"""
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    
    stats = {}
    for col in columns:
        stats[col] = {
            'mean': df[col].mean(),
            'std': df[col].std(),
            'min': df[col].min(),
            'max': df[col].max(),
            'skew': df[col].skew(),
            'kurtosis': df[col].kurtosis(),
            'missing': df[col].isnull().sum()
        }
    
    return pd.DataFrame(stats).T


def memory_usage(df):
    """Get memory usage of DataFrame"""
    return df.memory_usage(deep=True).sum() / 1024**2  # MB


def chunk_data(df, chunk_size=1000):
    """Split DataFrame into chunks"""
    for i in range(0, len(df), chunk_size):
        yield df.iloc[i:i+chunk_size]


def progress_bar(iterable, desc="Progress", total=None):
    """Simple progress bar"""
    from tqdm import tqdm
    return tqdm(iterable, desc=desc, total=total)


def create_backup(filepath):
    """Create backup of a file"""
    if os.path.exists(filepath):
        backup_path = f"{filepath}.bak_{get_timestamp()}"
        import shutil
        shutil.copy2(filepath, backup_path)
        logger.info(f"✅ Backup created: {backup_path}")
        return backup_path
    return None


def safe_divide(a, b, default=0):
    """Safe division that handles division by zero"""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.divide(a, b)
        result[~np.isfinite(result)] = default
    return result


class Timer:
    """Context manager for timing code execution"""
    def __init__(self, name="Operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        return self
    
    def __exit__(self, *args):
        self.end_time = datetime.now()
        elapsed = (self.end_time - self.start_time).total_seconds()
        logger.info(f"⏱️  {self.name} completed in {elapsed:.2f} seconds")


if __name__ == "__main__":
    # Test helpers
    print("Testing helper functions...")
    
    # Test timer
    with Timer("Test"):
        import time
        time.sleep(0.5)
    
    # Test data validation
    df = pd.DataFrame({
        'col1': [1, 2, 3],
        'col2': [4, 5, 6]
    })
    validate_data(df, ['col1', 'col2'])
    print("✅ All tests passed!")