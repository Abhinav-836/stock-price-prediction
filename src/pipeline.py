"""
End-to-end pipeline shared by dashboard.py and main.py (NEW FILE)

load (with indicator warm-up history) -> features -> train -> pick best -> forecast
Keeping this in one place means the dashboard and the CLI can't drift apart again.
"""
import os
import sys
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.data_loader import StockDataLoader
from src.preprocess import StockDataPreprocessor
from src.train_model import ModelTrainer, prepare_features_target, split_data, DEFAULT_TARGET
from src.predict import PredictionEngine

# Extra calendar days fetched BEFORE the user's start date so 200-day SMAs,
# 52+26-day Ichimoku etc. are already warmed up on the first displayed day.
WARMUP_CALENDAR_DAYS = 450
MIN_TRAINING_ROWS = 150


def run_analysis(symbol: str, start, end, use_synthetic: bool = False, run_cv: bool = False,
                 verbose: bool = False, save_models: bool = False,
                 target_col: str = DEFAULT_TARGET) -> Dict[str, Any]:
    """Load data, engineer features, train all models. Raises ValueError with a readable message."""
    symbol = symbol.upper().strip()
    start_ts, end_ts = pd.Timestamp(start), pd.Timestamp(end)
    if start_ts >= end_ts:
        raise ValueError("Start date must be before end date.")
    fetch_start = (start_ts - pd.Timedelta(days=WARMUP_CALENDAR_DAYS)).strftime('%Y-%m-%d')
    end_str = end_ts.strftime('%Y-%m-%d')

    loader = StockDataLoader(symbol=symbol)
    if use_synthetic:
        raw, source = loader.generate_synthetic_data(symbol, fetch_start, end_str), 'synthetic'
    else:
        raw, source = loader.fetch(symbol, fetch_start, end_str, allow_synthetic=True, save=False)

    if raw is None or len(raw) == 0:
        raise ValueError(f"No data available for {symbol}.")

    processed_all = StockDataPreprocessor(raw, verbose=verbose).run_pipeline()
    processed = processed_all[processed_all['Date'] >= start_ts].reset_index(drop=True)

    X, y, feature_cols = prepare_features_target(processed, target_col=target_col, verbose=verbose)
    if len(X) < MIN_TRAINING_ROWS:
        raise ValueError(f"Only {len(X)} usable trading days for {symbol} in this range "
                         f"(need {MIN_TRAINING_ROWS}+). Pick an earlier start date.")

    X_train, X_test, y_train, y_test = split_data(X, y)
    trainer = ModelTrainer(X_train, y_train, X_test, y_test, feature_cols,
                           target_col=target_col, run_cv=run_cv, verbose=verbose)
    results = trainer.train_and_evaluate_all(save_models=save_models)
    _, best_name = trainer.get_best_model()

    # Honest yardstick: a model that always predicts "0% change"
    baseline_rmse = float(np.sqrt(np.mean(np.square(np.asarray(y_test))))) if 'return' in target_col else None

    raw_view = raw[raw['Date'] >= start_ts].reset_index(drop=True)
    return {
        'symbol': symbol,
        'source': source,
        'start': start_ts,
        'end': end_ts,
        'raw_full': raw,
        'raw_view': raw_view,
        'processed': processed,
        'feature_cols': feature_cols,
        'target_col': target_col,
        'models': trainer.models,
        'results': results,
        'comparison': trainer.create_comparison_table(),
        'best_name': best_name,
        'baseline_rmse': baseline_rmse,
        'test_dates': processed.loc[X_test.index, 'Date'].reset_index(drop=True),
        'y_test': y_test.reset_index(drop=True),
    }


def forecast(analysis: Dict[str, Any], days: int = 30, model_name: Optional[str] = None) -> pd.DataFrame:
    """Recursive multi-day forecast with the chosen (default: best) model"""
    name = model_name or analysis['best_name']
    metrics = analysis['results'][name]['metrics']
    engine = PredictionEngine(model_name=name, model=analysis['models'][name],
                              target_col=analysis['target_col'], symbol=analysis['symbol'])
    return engine.predict_future(
        analysis['processed'], analysis['feature_cols'], days=days,
        raw_df=analysis['raw_full'], daily_vol_pct=metrics.get('test_residual_std')
    )


def feature_importance(analysis: Dict[str, Any], top_n: int = 20) -> Optional[pd.DataFrame]:
    """Importances from the best model if it has them, else from the best tree model."""
    results, models = analysis['results'], analysis['models']
    ranked = sorted(results, key=lambda n: results[n]['metrics']['test_r2'], reverse=True)
    for name in ranked:
        if hasattr(models[name], 'feature_importances_'):
            imp = np.asarray(models[name].feature_importances_, dtype=float)
            if imp.sum() > 0:
                imp = imp / imp.sum()
            return (pd.DataFrame({'feature': analysis['feature_cols'], 'importance': imp, 'model': name})
                    .sort_values('importance', ascending=False).head(top_n).reset_index(drop=True))
    return None