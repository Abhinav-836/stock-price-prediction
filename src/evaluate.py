"""
Model evaluation module - FIXED

Fixes vs previous version:
  * Passing pandas Series (what ModelTrainer produces) broke the trading metrics:
    `np.diff(series) / series[:-1]` aligns on the index -> wrong values. Everything is
    converted to numpy arrays first.
  * Trading simulation had look-ahead: it used the predicted move for day t->t+1 and
    applied it to the same interval's actual return computed from y_true[t]. Positions
    are now decided from the prediction vs the LAST KNOWN actual value.
  * Works for both price targets and return targets (log of negative returns gave NaN).
  * Profit factor was printed with a '%' sign; MAPE divided by zero on return targets.
  * calculate_directional_accuracy() added (the tests import it).
"""
import pandas as pd
import numpy as np
import os
import sys
import json
from typing import Dict
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    explained_variance_score, max_error, median_absolute_error
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import OUTPUTS_DIR
from utils.helpers import print_section, Timer


def _arr(x) -> np.ndarray:
    return np.asarray(x, dtype=float).ravel()


def _safe_mape(y_true, y_pred) -> float:
    mask = y_true != 0
    if not mask.any():
        return float('nan')
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)


def calculate_directional_accuracy(y_true, y_pred, is_return_target: bool = False) -> float:
    """
    % of correct up/down calls.
    Price target : direction of (pred[t] - true[t-1]) vs (true[t] - true[t-1])
    Return target: sign(pred) vs sign(true)
    """
    y_true, y_pred = _arr(y_true), _arr(y_pred)
    if is_return_target:
        return float(np.mean(np.sign(y_pred) == np.sign(y_true)) * 100)
    if len(y_true) < 2:
        return float('nan')
    actual_dir = np.sign(y_true[1:] - y_true[:-1])
    pred_dir = np.sign(y_pred[1:] - y_true[:-1])
    return float(np.mean(actual_dir == pred_dir) * 100)


def _looks_like_returns(y_true: np.ndarray) -> bool:
    return bool((y_true <= 0).any())


class AdvancedEvaluator:
    """
    Model evaluation with regression, directional, trading and risk metrics
    """

    def __init__(self, y_true, y_pred, model_name='Model', is_return_target=None, verbose=True):
        self.y_true = _arr(y_true)
        self.y_pred = _arr(y_pred)
        self.model_name = model_name
        self.is_return_target = _looks_like_returns(self.y_true) if is_return_target is None else is_return_target
        self.verbose = verbose
        self.metrics = {}

    def calculate_basic_metrics(self) -> Dict:
        mse = mean_squared_error(self.y_true, self.y_pred)
        return {
            'MSE': mse,
            'RMSE': float(np.sqrt(mse)),
            'MAE': mean_absolute_error(self.y_true, self.y_pred),
            'MAPE': _safe_mape(self.y_true, self.y_pred),
            'R2': r2_score(self.y_true, self.y_pred),
            'Explained_Variance': explained_variance_score(self.y_true, self.y_pred),
            'Max_Error': max_error(self.y_true, self.y_pred),
            'Median_AE': median_absolute_error(self.y_true, self.y_pred),
        }

    def calculate_percentage_metrics(self) -> Dict:
        y, p = self.y_true, self.y_pred
        errors = y - p
        denom = np.abs(y) + np.abs(p)
        naive = np.mean(np.abs(np.diff(y))) if len(y) > 1 else np.nan
        nonzero = y != 0
        return {
            'SMAPE': float(np.mean(np.where(denom > 0, 2 * np.abs(errors) / np.where(denom > 0, denom, 1), 0)) * 100),
            'MASE': float(np.mean(np.abs(errors)) / naive) if naive and np.isfinite(naive) else float('nan'),
            'RMSPE': float(np.sqrt(np.mean((errors[nonzero] / y[nonzero]) ** 2)) * 100) if nonzero.any() else float('nan'),
        }

    def _strategy_returns(self) -> np.ndarray:
        """Long when the model predicts up, flat otherwise. No look-ahead."""
        y, p = self.y_true, self.y_pred
        if self.is_return_target:
            # y = realised % return of the next day, p = predicted % return
            return np.where(p > 0, 1.0, 0.0) * (y / 100)
        if len(y) < 2:
            return np.array([])
        actual_ret = y[1:] / y[:-1] - 1
        position = np.where(p[1:] > y[:-1], 1.0, 0.0)  # decided with info known at t-1
        return position * actual_ret

    def calculate_directional_metrics(self) -> Dict:
        strat = self._strategy_returns()
        wins, losses = strat[strat > 0], strat[strat < 0]
        traded = strat[strat != 0]
        return {
            'Directional_Accuracy': calculate_directional_accuracy(self.y_true, self.y_pred, self.is_return_target),
            'Win_Rate': float(np.mean(traded > 0) * 100) if len(traded) else 0.0,
            'Total_Return': float((np.prod(1 + strat) - 1) * 100) if len(strat) else 0.0,
            'Avg_Gain': float(wins.mean() * 100) if len(wins) else 0.0,
            'Avg_Loss': float(losses.mean() * 100) if len(losses) else 0.0,
            'Profit_Factor': float(wins.sum() / abs(losses.sum())) if len(losses) else float('inf'),
        }

    def calculate_risk_metrics(self) -> Dict:
        errors = self.y_true - self.y_pred
        mean_true = np.mean(self.y_true)
        std_error = np.std(errors)
        lower, upper = self.y_pred - 1.96 * std_error, self.y_pred + 1.96 * std_error

        metrics = {
            'Bias': float(np.mean(errors)),
            'MBD': float(np.mean(errors) / mean_true * 100) if mean_true else float('nan'),
            'MAD': float(np.mean(np.abs(errors - np.mean(errors)))),
            'Residual_Std': float(std_error),
            'Residual_Skew': float(pd.Series(errors).skew()) if len(errors) > 2 else 0.0,
            'Residual_Kurtosis': float(pd.Series(errors).kurtosis()) if len(errors) > 3 else 0.0,
            'Coverage_95': float(np.mean((self.y_true >= lower) & (self.y_true <= upper)) * 100),
        }

        # Theil's U on changes (only meaningful for strictly positive price series)
        if not self.is_return_target and (self.y_true > 0).all() and (self.y_pred > 0).all() and len(self.y_true) > 1:
            dt, dp = np.diff(np.log(self.y_true)), np.diff(np.log(self.y_pred))
            den = np.sqrt(np.mean(dt ** 2))
            metrics['Theil_U'] = float(np.sqrt(np.mean((dt - dp) ** 2)) / den) if den else 0.0
        else:
            den = np.sqrt(np.mean(self.y_true ** 2))
            metrics['Theil_U'] = float(np.sqrt(np.mean(errors ** 2)) / den) if den else 0.0

        strat = self._strategy_returns()
        if len(strat) > 1 and np.std(strat) > 0:
            metrics['Sharpe_Ratio'] = float(np.mean(strat) / np.std(strat) * np.sqrt(252))
            equity = np.cumprod(1 + strat)
            metrics['Max_Drawdown'] = float(np.min(equity / np.maximum.accumulate(equity) - 1) * 100)
        else:
            metrics['Sharpe_Ratio'] = 0.0
            metrics['Max_Drawdown'] = 0.0
        return metrics

    def calculate_all_metrics(self) -> Dict:
        if self.verbose:
            print_section(f"Evaluating {self.model_name}")
        self.metrics = {
            **self.calculate_basic_metrics(),
            **self.calculate_percentage_metrics(),
            **self.calculate_directional_metrics(),
            **self.calculate_risk_metrics(),
        }
        return self.metrics

    def print_metrics(self):
        if not self.metrics:
            self.calculate_all_metrics()
        m = self.metrics
        print("\n" + "=" * 70 + f"\n  {self.model_name} - PERFORMANCE METRICS\n" + "=" * 70)
        print("\n📊 Basic Metrics:")
        for key in ['R2', 'RMSE', 'MAE', 'MAPE', 'MSE', 'Explained_Variance']:
            print(f"  {key:20s}: {m[key]:.4f}")
        print("\n🎯 Directional / Trading Metrics:")
        for key in ['Directional_Accuracy', 'Win_Rate', 'Total_Return']:
            print(f"  {key:20s}: {m[key]:.2f}%")
        print(f"  {'Profit_Factor':20s}: {m['Profit_Factor']:.2f}")
        print(f"  {'Sharpe_Ratio':20s}: {m['Sharpe_Ratio']:.2f}")
        print("\n⚠️  Risk Metrics:")
        for key in ['Bias', 'MBD', 'Residual_Std', 'Coverage_95', 'Theil_U', 'Max_Drawdown']:
            print(f"  {key:20s}: {m[key]:.4f}")

    def save_metrics(self, output_dir: str = OUTPUTS_DIR):
        if not self.metrics:
            self.calculate_all_metrics()
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(output_dir, f"{self.model_name}_metrics_{timestamp}.json")
        clean = {k: (float(v) if np.isfinite(v) else str(v)) for k, v in self.metrics.items()}
        with open(filepath, 'w') as f:
            json.dump(clean, f, indent=4)
        print(f"✅ Metrics saved: {filepath}")
        return filepath


def compare_models(y_true, y_pred_dict: Dict[str, np.ndarray], is_return_target=None) -> pd.DataFrame:
    rows = []
    for name, y_pred in y_pred_dict.items():
        m = AdvancedEvaluator(y_true, y_pred, name, is_return_target, verbose=False).calculate_all_metrics()
        rows.append({'Model': name, 'R2': m['R2'], 'RMSE': m['RMSE'], 'MAE': m['MAE'], 'MAPE': m['MAPE'],
                     'Dir_Acc': m['Directional_Accuracy'], 'Profit_Factor': m['Profit_Factor'],
                     'Sharpe': m['Sharpe_Ratio']})
    return pd.DataFrame(rows).sort_values('R2', ascending=False)


# ============================================================================
# LEGACY SUPPORT FUNCTIONS
# ============================================================================

def calculate_metrics(y_true, y_pred, is_return_target=None):
    return AdvancedEvaluator(y_true, y_pred, is_return_target=is_return_target, verbose=False).calculate_all_metrics()


def evaluate_all_models(models_dict, X_train, y_train, X_test, y_test, save=True):
    results, y_pred_dict = {}, {}
    for name, model in models_dict.items():
        y_pred = model.predict(X_test)
        y_pred_dict[name] = y_pred
        evaluator = AdvancedEvaluator(y_test, y_pred, name)
        metrics = evaluator.calculate_all_metrics()
        evaluator.print_metrics()
        if save:
            evaluator.save_metrics(OUTPUTS_DIR)
        results[name] = {'metrics': metrics, 'predictions': y_pred}

    comparison = compare_models(y_test, y_pred_dict)
    print("\n" + "=" * 70 + "\n  MODEL COMPARISON\n" + "=" * 70)
    print(comparison.to_string(index=False))
    if save:
        os.makedirs(OUTPUTS_DIR, exist_ok=True)
        comparison.to_csv(os.path.join(OUTPUTS_DIR, 'model_comparison.csv'), index=False)
    return results


if __name__ == "__main__":
    rng = np.random.default_rng(42)
    y_true = 150 + np.cumsum(rng.normal(0, 2, 200))
    y_pred = y_true + rng.normal(0, 2, 200)
    evaluator = AdvancedEvaluator(y_true, y_pred, "Test Model")
    evaluator.calculate_all_metrics()
    evaluator.print_metrics()
    print("\n✅ Evaluation test complete!")