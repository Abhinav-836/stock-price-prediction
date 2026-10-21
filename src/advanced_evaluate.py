# src/advanced_evaluate.py - Advanced evaluation metrics - FIXED
"""
Fixes vs previous version:
  * Duplicated (and diverged from) evaluate.py. It crashed / returned NaN for:
    - pandas Series inputs (index-aligned division in returns),
    - zero std of strategy returns (division by zero in Sharpe),
    - return targets (log of negative numbers in Theil's U),
    - y_true containing 0 (MAPE division by zero).
  * Now a thin wrapper over the single, tested implementation in evaluate.py,
    keeping the original method name and output keys.
"""
try:
    from src.evaluate import AdvancedEvaluator
except ImportError:  # running from inside src/
    from evaluate import AdvancedEvaluator

_KEYS = ['MSE', 'RMSE', 'MAE', 'R2', 'MAPE', 'SMAPE', 'Directional_Accuracy', 'Profit_Factor',
         'Sharpe_Ratio', 'Max_Drawdown', 'Bias', 'MBD', 'Coverage_95', 'Theil_U']


class AdvancedMetrics:
    @staticmethod
    def calculate_all_metrics(y_true, y_pred, is_return_target=None):
        """Calculate comprehensive evaluation metrics (same keys as before)"""
        metrics = AdvancedEvaluator(y_true, y_pred, is_return_target=is_return_target,
                                    verbose=False).calculate_all_metrics()
        return {k: metrics[k] for k in _KEYS}