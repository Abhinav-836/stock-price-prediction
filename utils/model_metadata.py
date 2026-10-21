"""
Model metadata generation and management - FIXED

Fixes vs previous version:
  * Overfitting gap read 'R2' only, but ModelTrainer produces 'train_r2'/'test_r2'
    -> gap was always 0 and every model looked "low risk". Both naming styles now work.
  * float(v) crashed on non-numeric metric values; NaN/inf broke json.dump.
  * save_model_metadata crashed if models_dir didn't exist; params with numpy types
    or objects were not JSON serialisable.
  * Pipelines (e.g. StandardScaler+SVR) now report the final estimator type.
  * target variable is a parameter instead of hard-coded 'target'.
"""
import json
import os
from datetime import datetime
import numpy as np


def _clean_number(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    return f if np.isfinite(f) else None


def _get_r2(metrics: dict, prefix: str):
    for key in ('R2', 'r2', f'{prefix}_r2'):
        if key in metrics:
            return _clean_number(metrics[key]) or 0.0
    return 0.0


def create_model_metadata(model, model_name, train_metrics, test_metrics,
                          X_train, X_test, feature_cols, params=None,
                          target_variable='target_return_1d', prediction_horizon='1_day'):
    """Create comprehensive metadata for a trained model"""
    final_estimator = model.steps[-1][1] if hasattr(model, 'steps') else model

    top_features = []
    if hasattr(final_estimator, 'feature_importances_'):
        importances = np.asarray(final_estimator.feature_importances_)
        n = min(len(importances), len(feature_cols))
        for rank, idx in enumerate(np.argsort(importances[:n])[::-1][:10], 1):
            top_features.append({'rank': rank, 'name': feature_cols[idx], 'importance': float(importances[idx])})

    train_r2 = _get_r2(train_metrics, 'train')
    test_r2 = _get_r2(test_metrics, 'test')
    r2_gap = train_r2 - test_r2
    overfitting_risk = 'high' if r2_gap > 0.3 else 'medium' if r2_gap > 0.15 else 'low'

    return {
        'model_info': {
            'model_name': model_name,
            'model_type': type(final_estimator).__name__,
            'version': '1.0.0',
            'created_date': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat()
        },
        'training_data': {
            'total_train_samples': len(X_train),
            'total_test_samples': len(X_test),
            'features_count': len(feature_cols),
            'target_variable': target_variable
        },
        'model_parameters': params or {},
        'performance_metrics': {
            'training': {k: _clean_number(v) for k, v in train_metrics.items()},
            'test': {k: _clean_number(v) for k, v in test_metrics.items()}
        },
        'top_features': top_features,
        'model_characteristics': {
            'overfitting_risk': overfitting_risk,
            'train_test_r2_gap': float(r2_gap),
            'prediction_horizon': prediction_horizon,
            'suitable_for_production': test_r2 > 0.5
        },
        'contact': {'last_validated': datetime.now().strftime('%Y-%m-%d')}
    }


def save_model_metadata(metadata, model_name, models_dir):
    os.makedirs(models_dir, exist_ok=True)
    filepath = os.path.join(models_dir, f'{model_name}_metadata.json')
    with open(filepath, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    print(f"✅ Metadata saved: {filepath}")
    return filepath


def load_model_metadata(model_name, models_dir):
    filepath = os.path.join(models_dir, f'{model_name}_metadata.json')
    if not os.path.exists(filepath):
        print(f"⚠️  Metadata file not found: {filepath}")
        return None
    with open(filepath, 'r') as f:
        return json.load(f)


def _fmt(value):
    return f"{value:.4f}" if isinstance(value, (int, float)) else str(value)


def print_model_info(metadata):
    print("\n" + "=" * 60 + "\n  MODEL INFORMATION\n" + "=" * 60)
    info = metadata.get('model_info', {})
    print(f"Model Name: {info.get('model_name')}")
    print(f"Model Type: {info.get('model_type')}")
    print(f"Version: {info.get('version')}")
    print(f"Created: {info.get('created_date')}")

    print("\n" + "-" * 60 + "\n  PERFORMANCE METRICS\n" + "-" * 60)
    perf = metadata.get('performance_metrics', {})
    for split in ('training', 'test'):
        print(f"\n{split.title()}:")
        for metric, value in perf.get(split, {}).items():
            print(f"  {metric:12s}: {_fmt(value)}")

    chars = metadata.get('model_characteristics', {})
    print("\n" + "-" * 60 + "\n  MODEL CHARACTERISTICS\n" + "-" * 60)
    print(f"Overfitting Risk: {chars.get('overfitting_risk')}")
    print(f"R² Gap: {chars.get('train_test_r2_gap', 0):.4f}")
    print(f"Production Ready: {chars.get('suitable_for_production')}")

    top_features = metadata.get('top_features', [])
    if top_features:
        print("\n" + "-" * 60 + "\n  TOP FEATURES\n" + "-" * 60)
        for feat in top_features[:5]:
            print(f"  {feat['rank']:2d}. {feat['name']:25s}: {feat['importance']:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    print("Model Metadata Module")
    example = {
        'model_info': {'model_name': 'example_model', 'model_type': 'XGBRegressor', 'version': '1.0.0'},
        'performance_metrics': {'test': {'test_r2': 0.02, 'test_rmse': 1.6, 'test_dir_acc': 52.3}}
    }
    print(json.dumps(example, indent=2))