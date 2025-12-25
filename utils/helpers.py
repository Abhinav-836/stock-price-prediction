"""
Helper functions for the stock price prediction project
"""
import os
import json
import joblib
import numpy as np
from datetime import datetime


def save_model(model, model_name, models_dir):
    """
    Save a trained model to disk
    
    Args:
        model: Trained model object
        model_name: Name for the model file
        models_dir: Directory to save the model
    """
    filepath = os.path.join(models_dir, f"{model_name}.pkl")
    joblib.dump(model, filepath)
    print(f"✅ Model saved: {filepath}")


def load_model(model_name, models_dir):
    """
    Load a trained model from disk
    
    Args:
        model_name: Name of the model file
        models_dir: Directory where model is saved
        
    Returns:
        Loaded model object
    """
    filepath = os.path.join(models_dir, f"{model_name}.pkl")
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model not found: {filepath}")
    
    model = joblib.load(filepath)
    print(f"✅ Model loaded: {filepath}")
    return model


def save_metrics(metrics, filename, output_dir):
    """
    Save evaluation metrics to JSON file
    
    Args:
        metrics: Dictionary of metrics
        filename: Name for the metrics file
        output_dir: Directory to save metrics
    """
    filepath = os.path.join(output_dir, f"{filename}.json")
    
    # Convert numpy types to native Python types
    metrics_clean = {}
    for key, value in metrics.items():
        if isinstance(value, (np.integer, np.floating)):
            metrics_clean[key] = float(value)
        else:
            metrics_clean[key] = value
    
    with open(filepath, 'w') as f:
        json.dump(metrics_clean, f, indent=4)
    
    print(f"✅ Metrics saved: {filepath}")


def load_metrics(filename, output_dir):
    """
    Load evaluation metrics from JSON file
    
    Args:
        filename: Name of the metrics file
        output_dir: Directory where metrics are saved
        
    Returns:
        Dictionary of metrics
    """
    filepath = os.path.join(output_dir, f"{filename}.json")
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Metrics file not found: {filepath}")
    
    with open(filepath, 'r') as f:
        metrics = json.load(f)
    
    return metrics


def calculate_returns(prices):
    """
    Calculate percentage returns from price series
    
    Args:
        prices: Array or Series of prices
        
    Returns:
        Array of percentage returns
    """
    return np.diff(prices) / prices[:-1] * 100


def get_timestamp():
    """
    Get current timestamp as string
    
    Returns:
        Formatted timestamp string
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def print_section(title):
    """
    Print a formatted section header
    
    Args:
        title: Section title
    """
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)