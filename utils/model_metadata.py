"""
Model metadata generation and management
"""
import json
import os
from datetime import datetime
import numpy as np


def create_model_metadata(model, model_name, train_metrics, test_metrics, 
                          X_train, X_test, feature_cols, params=None):
    """
    Create comprehensive metadata for a trained model
    
    Args:
        model: Trained model object
        model_name: Name of the model
        train_metrics: Dictionary of training metrics
        test_metrics: Dictionary of test metrics
        X_train: Training features
        X_test: Test features
        feature_cols: List of feature names
        params: Model parameters (optional)
    
    Returns:
        Dictionary with model metadata
    """
    
    # Get feature importances if available
    top_features = []
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:10]
        
        for i, idx in enumerate(indices, 1):
            top_features.append({
                'rank': i,
                'name': feature_cols[idx],
                'importance': float(importances[idx])
            })
    
    # Determine model type
    model_type = type(model).__name__
    
    # Calculate overfitting metrics
    r2_gap = train_metrics.get('R2', 0) - test_metrics.get('R2', 0)
    overfitting_risk = 'high' if r2_gap > 0.3 else 'medium' if r2_gap > 0.15 else 'low'
    
    metadata = {
        'model_info': {
            'model_name': model_name,
            'model_type': model_type,
            'version': '1.0.0',
            'created_date': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat()
        },
        'training_data': {
            'total_train_samples': len(X_train),
            'total_test_samples': len(X_test),
            'features_count': len(feature_cols),
            'target_variable': 'target'
        },
        'model_parameters': params if params else {},
        'performance_metrics': {
            'training': {k: float(v) for k, v in train_metrics.items()},
            'test': {k: float(v) for k, v in test_metrics.items()}
        },
        'top_features': top_features,
        'model_characteristics': {
            'overfitting_risk': overfitting_risk,
            'train_test_r2_gap': float(r2_gap),
            'prediction_horizon': '1_day',
            'suitable_for_production': test_metrics.get('R2', 0) > 0.5
        },
        'contact': {
            'last_validated': datetime.now().strftime('%Y-%m-%d')
        }
    }
    
    return metadata


def save_model_metadata(metadata, model_name, models_dir):
    """
    Save model metadata to JSON file
    
    Args:
        metadata: Dictionary with model metadata
        model_name: Name of the model
        models_dir: Directory to save metadata
    """
    filepath = os.path.join(models_dir, f'{model_name}_metadata.json')
    
    with open(filepath, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Metadata saved: {filepath}")


def load_model_metadata(model_name, models_dir):
    """
    Load model metadata from JSON file
    
    Args:
        model_name: Name of the model
        models_dir: Directory where metadata is saved
    
    Returns:
        Dictionary with model metadata
    """
    filepath = os.path.join(models_dir, f'{model_name}_metadata.json')
    
    if not os.path.exists(filepath):
        print(f"⚠️  Metadata file not found: {filepath}")
        return None
    
    with open(filepath, 'r') as f:
        metadata = json.load(f)
    
    return metadata


def print_model_info(metadata):
    """
    Print formatted model information
    
    Args:
        metadata: Dictionary with model metadata
    """
    print("\n" + "="*60)
    print("  MODEL INFORMATION")
    print("="*60)
    
    info = metadata.get('model_info', {})
    print(f"Model Name: {info.get('model_name')}")
    print(f"Model Type: {info.get('model_type')}")
    print(f"Version: {info.get('version')}")
    print(f"Created: {info.get('created_date')}")
    
    print("\n" + "-"*60)
    print("  PERFORMANCE METRICS")
    print("-"*60)
    
    train = metadata.get('performance_metrics', {}).get('training', {})
    test = metadata.get('performance_metrics', {}).get('test', {})
    
    print(f"\nTraining:")
    for metric, value in train.items():
        print(f"  {metric:8s}: {value:.4f}")
    
    print(f"\nTest:")
    for metric, value in test.items():
        print(f"  {metric:8s}: {value:.4f}")
    
    chars = metadata.get('model_characteristics', {})
    print("\n" + "-"*60)
    print("  MODEL CHARACTERISTICS")
    print("-"*60)
    print(f"Overfitting Risk: {chars.get('overfitting_risk')}")
    print(f"R² Gap: {chars.get('train_test_r2_gap', 0):.4f}")
    print(f"Production Ready: {chars.get('suitable_for_production')}")
    
    top_features = metadata.get('top_features', [])
    if top_features:
        print("\n" + "-"*60)
        print("  TOP FEATURES")
        print("-"*60)
        for feat in top_features[:5]:
            print(f"  {feat['rank']:2d}. {feat['name']:20s}: {feat['importance']:.4f}")
    
    print("="*60)


if __name__ == "__main__":
    # Example usage
    print("Model Metadata Module")
    print("Import this module to create and manage model metadata")
    
    # Example
    example_metadata = {
        'model_info': {
            'model_name': 'example_model',
            'model_type': 'XGBRegressor',
            'version': '1.0.0'
        },
        'performance_metrics': {
            'test': {
                'R2': 0.57,
                'RMSE': 45.12,
                'MAE': 32.45
            }
        }
    }
    
    print("\nExample metadata structure:")
    print(json.dumps(example_metadata, indent=2))