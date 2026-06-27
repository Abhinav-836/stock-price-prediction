# src/advanced_evaluate.py - Advanced evaluation metrics

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class AdvancedMetrics:
    @staticmethod
    def calculate_all_metrics(y_true, y_pred):
        """Calculate comprehensive evaluation metrics"""
        
        # Basic metrics
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Percentage errors
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        smape = np.mean(2 * np.abs(y_true - y_pred) / (np.abs(y_true) + np.abs(y_pred))) * 100
        
        # Directional accuracy
        actual_direction = np.sign(np.diff(y_true))
        pred_direction = np.sign(np.diff(y_pred))
        dir_acc = np.mean(actual_direction == pred_direction) * 100
        
        # Profit factor (trading simulation)
        returns = np.diff(y_true) / y_true[:-1]
        pred_returns = np.diff(y_pred) / y_pred[:-1]
        
        positions = np.where(pred_returns > 0, 1, 0)
        strategy_returns = positions[:-1] * returns[1:]  # Align lengths
        
        total_win = np.sum(strategy_returns[strategy_returns > 0])
        total_loss = np.abs(np.sum(strategy_returns[strategy_returns < 0]))
        profit_factor = total_win / total_loss if total_loss != 0 else np.inf
        
        # Sharpe ratio
        sharpe = np.mean(strategy_returns) / np.std(strategy_returns) * np.sqrt(252)
        
        # Maximum drawdown
        cumulative = np.cumprod(1 + strategy_returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = np.min(drawdown) * 100
        
        # Calibration metrics
        errors = y_true - y_pred
        bias = np.mean(errors)
        mbd = bias / np.mean(y_true) * 100  # Mean bias deviation
        
        # Prediction interval coverage (95% CI)
        # Assuming normal distribution of errors
        std_error = np.std(errors)
        lower_bound = y_pred - 1.96 * std_error
        upper_bound = y_pred + 1.96 * std_error
        coverage = np.mean((y_true >= lower_bound) & (y_true <= upper_bound)) * 100
        
        # Theil's U statistic
        y_true_log = np.log(y_true)
        y_pred_log = np.log(y_pred)
        y_true_diff = np.diff(y_true_log)
        y_pred_diff = np.diff(y_pred_log)
        
        numerator = np.sqrt(np.mean((y_true_diff - y_pred_diff) ** 2))
        denominator = np.sqrt(np.mean(y_true_diff ** 2))
        theil_u = numerator / denominator if denominator != 0 else 0
        
        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape,
            'SMAPE': smape,
            'Directional_Accuracy': dir_acc,
            'Profit_Factor': profit_factor,
            'Sharpe_Ratio': sharpe,
            'Max_Drawdown': max_drawdown,
            'Bias': bias,
            'MBD': mbd,
            'Coverage_95': coverage,
            'Theil_U': theil_u
        }