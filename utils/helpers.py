"""
Helper functions for the stock price prediction project - FIXED

Fixes vs previous version:
  * handle_missing_values: fillna(method=...) is removed in pandas 3 -> use .ffill()/.bfill();
    interpolate(method='time') crashed without a DatetimeIndex -> falls back to linear.
  * load_model(with_metadata=True) returned a bare model when no metadata file existed
    (inconsistent return type) -> always returns (model, metadata_or_None).
  * safe_divide crashed on scalars (item assignment on a numpy scalar).
  * calculate_returns / calculate_log_returns now accept pandas Series safely.
  * normalize_data guards against zero range / zero std / zero IQR.
  * logging.basicConfig no longer hijacks the root logger of the importing app.
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
import logging
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def save_model(model, model_name, models_dir, metadata=None):
    """Save a trained model to disk (joblib) with optional JSON metadata"""
    os.makedirs(models_dir, exist_ok=True)
    filepath = os.path.join(models_dir, f"{model_name}.pkl")
    joblib.dump(model, filepath)

    if metadata:
        meta_path = os.path.join(models_dir, f"{model_name}_metadata.json")
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)

    logger.info(f"✅ Model saved: {filepath}")
    return filepath


def load_model(model_name, models_dir, with_metadata=False):
    """Load a model. With with_metadata=True always returns (model, metadata_or_None)."""
    filepath = os.path.join(models_dir, f"{model_name}.pkl")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model not found: {filepath}")

    model = joblib.load(filepath)
    logger.info(f"✅ Model loaded: {filepath}")

    if with_metadata:
        metadata = None
        meta_path = os.path.join(models_dir, f"{model_name}_metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
        return model, metadata

    return model


def _to_jsonable(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        value = float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)  # JSON has no NaN/inf
    return value


def save_metrics(metrics, filename, output_dir):
    """Save evaluation metrics to JSON file"""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, f"{filename}.json")
    metrics_clean = {k: _to_jsonable(v) for k, v in metrics.items()}
    with open(filepath, 'w') as f:
        json.dump(metrics_clean, f, indent=4, default=str)
    logger.info(f"✅ Metrics saved: {filepath}")
    return filepath


def load_metrics(filename, output_dir):
    filepath = os.path.join(output_dir, f"{filename}.json")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Metrics file not found: {filepath}")
    with open(filepath, 'r') as f:
        return json.load(f)


def calculate_returns(prices):
    """Percentage returns from a price series/array"""
    prices = np.asarray(prices, dtype=float)
    return np.diff(prices) / prices[:-1] * 100


def calculate_log_returns(prices):
    """Log returns from a price series/array"""
    prices = np.asarray(prices, dtype=float)
    return np.log(prices[1:] / prices[:-1])


def get_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def print_section(title, char='='):
    print("\n" + char * 70)
    print(f"  {title}")
    print(char * 70)


def save_dataframe(df, filename, output_dir, format='csv'):
    """Save DataFrame in specified format"""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)

    if format == 'csv':
        df.to_csv(filepath, index=False)
    elif format == 'excel':
        df.to_excel(filepath, index=False)  # requires openpyxl
    elif format == 'json':
        df.to_json(filepath, orient='records', date_format='iso')
    else:
        raise ValueError(f"Unsupported format: {format}")

    logger.info(f"✅ Data saved: {filepath}")
    return filepath


def create_feature_hash(df, feature_cols):
    """Stable hash of feature values for caching"""
    hashed = pd.util.hash_pandas_object(df[feature_cols], index=True).values
    return hashlib.md5(hashed.tobytes()).hexdigest()


def handle_missing_values(df, strategy='ffill', max_gap=5):
    """Handle missing values (pandas 2 and 3 compatible)"""
    df = df.copy()

    if strategy == 'ffill':
        df = df.ffill(limit=max_gap)
    elif strategy == 'bfill':
        df = df.bfill(limit=max_gap)
    elif strategy == 'interpolate':
        method = 'time' if isinstance(df.index, pd.DatetimeIndex) else 'linear'
        df = df.interpolate(method=method, limit=max_gap)
    elif strategy == 'drop':
        df = df.dropna()
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return df


def detect_outliers(df, columns, method='iqr', threshold=1.5):
    """Boolean frame marking outliers in the given columns"""
    outliers = {}
    for col in columns:
        if method == 'iqr':
            q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
            iqr = q3 - q1
            outliers[col] = (df[col] < q1 - threshold * iqr) | (df[col] > q3 + threshold * iqr)
        elif method == 'zscore':
            std = df[col].std()
            if std:
                outliers[col] = np.abs((df[col] - df[col].mean()) / std) > threshold
            else:
                outliers[col] = pd.Series(False, index=df.index)
        else:
            raise ValueError(f"Unknown method: {method}")
    return pd.DataFrame(outliers)


def normalize_data(df, columns, method='minmax'):
    """Normalize columns; constant columns become 0 instead of NaN/inf"""
    df = df.copy()
    for col in columns:
        if method == 'minmax':
            rng = df[col].max() - df[col].min()
            df[col] = (df[col] - df[col].min()) / rng if rng else 0.0
        elif method == 'standard':
            std = df[col].std()
            df[col] = (df[col] - df[col].mean()) / std if std else 0.0
        elif method == 'robust':
            iqr = df[col].quantile(0.75) - df[col].quantile(0.25)
            df[col] = (df[col] - df[col].median()) / iqr if iqr else 0.0
        else:
            raise ValueError(f"Unknown method: {method}")
    return df


def calculate_confidence_interval(predictions, model_type='ensemble', n_bootstrap=1000):
    """
    'ensemble'  : predictions shape (n_models, n_points) -> mean ± 1.96 std across models
    'bootstrap' : predictions shape (n_points,) -> bootstrap CI of the MEAN prediction
    """
    predictions = np.asarray(predictions, dtype=float)

    if model_type == 'ensemble':
        mean_pred = predictions.mean(axis=0)
        std_pred = predictions.std(axis=0)
        return mean_pred, mean_pred - 1.96 * std_pred, mean_pred + 1.96 * std_pred

    if model_type == 'bootstrap':
        rng = np.random.default_rng(42)
        n = len(predictions)
        means = np.array([predictions[rng.integers(0, n, n)].mean() for _ in range(n_bootstrap)])
        return predictions.mean(), np.percentile(means, 2.5), np.percentile(means, 97.5)

    return predictions, predictions, predictions


def validate_data(df, required_cols):
    missing = set(required_cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return True


def get_data_stats(df, columns=None):
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns
    stats = {}
    for col in columns:
        stats[col] = {
            'mean': df[col].mean(), 'std': df[col].std(),
            'min': df[col].min(), 'max': df[col].max(),
            'skew': df[col].skew(), 'kurtosis': df[col].kurtosis(),
            'missing': df[col].isnull().sum()
        }
    return pd.DataFrame(stats).T


def memory_usage(df):
    """Memory usage in MB"""
    return df.memory_usage(deep=True).sum() / 1024 ** 2


def chunk_data(df, chunk_size=1000):
    for i in range(0, len(df), chunk_size):
        yield df.iloc[i:i + chunk_size]


def progress_bar(iterable, desc="Progress", total=None):
    try:
        from tqdm import tqdm
        return tqdm(iterable, desc=desc, total=total)
    except ImportError:
        return iterable


def create_backup(filepath):
    if os.path.exists(filepath):
        import shutil
        backup_path = f"{filepath}.bak_{get_timestamp()}"
        shutil.copy2(filepath, backup_path)
        logger.info(f"✅ Backup created: {backup_path}")
        return backup_path
    return None


def safe_divide(a, b, default=0):
    """Division that returns `default` where the result is not finite (works for scalars too)"""
    with np.errstate(divide='ignore', invalid='ignore'):
        result = np.divide(np.asarray(a, dtype=float), np.asarray(b, dtype=float))
    if np.ndim(result) == 0:
        return float(result) if np.isfinite(result) else default
    result = np.array(result)
    result[~np.isfinite(result)] = default
    return result


class Timer:
    """Context manager for timing code execution"""

    def __init__(self, name="Operation"):
        self.name = name
        self.start_time = None
        self.end_time = None
        self.elapsed = None

    def __enter__(self):
        self.start_time = datetime.now()
        return self

    def __exit__(self, *args):
        self.end_time = datetime.now()
        self.elapsed = (self.end_time - self.start_time).total_seconds()
        logger.info(f"⏱️  {self.name} completed in {self.elapsed:.2f} seconds")
        return False


if __name__ == "__main__":
    print("Testing helper functions...")
    import time
    with Timer("Test"):
        time.sleep(0.2)
    validate_data(pd.DataFrame({'col1': [1, 2, 3], 'col2': [4, 5, 6]}), ['col1', 'col2'])
    assert safe_divide(1, 0) == 0
    print("✅ All tests passed!")