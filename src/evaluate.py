"""
Model evaluation module - Enhanced version with advanced metrics
"""
import pandas as pd
import numpy as np
import os
import sys
from typing import Dict, Any, Optional, List
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error, 
    r2_score, 
    mean_absolute_percentage_error,
    explained_variance_score,
    max_error,
    median_absolute_error
)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import DATA_PROCESSED_DIR, MODELS_DIR, STOCK_SYMBOL, OUTPUTS_DIR
from utils.helpers import load_model, print_section, Timer


class AdvancedEvaluator:
    """
    Advanced model evaluation with comprehensive metrics and visualizations
    """
    
    def __init__(self, y_true, y_pred, model_name='Model'):
        self.y_true = y_true
        self.y_pred = y_pred
        self.model_name = model_name
        self.metrics = {}
        
    def calculate_basic_metrics(self) -> Dict:
        """Calculate basic regression metrics"""
        metrics = {
            'MSE': mean_squared_error(self.y_true, self.y_pred),
            'RMSE': np.sqrt(mean_squared_error(self.y_true, self.y_pred)),
            'MAE': mean_absolute_error(self.y_true, self.y_pred),
            'MAPE': mean_absolute_percentage_error(self.y_true, self.y_pred) * 100,
            'R2': r2_score(self.y_true, self.y_pred),
            'Explained_Variance': explained_variance_score(self.y_true, self.y_pred),
            'Max_Error': max_error(self.y_true, self.y_pred),
            'Median_AE': median_absolute_error(self.y_true, self.y_pred)
        }
        
        return metrics
    
    def calculate_percentage_metrics(self) -> Dict:
        """Calculate percentage-based metrics"""
        errors = self.y_true - self.y_pred
        rel_errors = errors / self.y_true
        
        metrics = {
            'MAPE': np.mean(np.abs(rel_errors)) * 100,
            'SMAPE': np.mean(2 * np.abs(self.y_true - self.y_pred) / (np.abs(self.y_true) + np.abs(self.y_pred))) * 100,
            'MASE': np.mean(np.abs(errors)) / np.mean(np.abs(np.diff(self.y_true))),
            'RMSPE': np.sqrt(np.mean((errors / self.y_true) ** 2)) * 100
        }
        
        return metrics
    
    def calculate_directional_metrics(self) -> Dict:
        """Calculate directional accuracy metrics"""
        # Direction predictions
        actual_direction = np.diff(self.y_true) > 0
        pred_direction = np.diff(self.y_pred) > 0
        
        # Directional accuracy
        dir_acc = np.mean(actual_direction == pred_direction) * 100
        
        # Win rate
        actual_returns = np.diff(self.y_true) / self.y_true[:-1]
        pred_returns = np.diff(self.y_pred) / self.y_pred[:-1]
        
        # Buy/Sell signals
        positions = np.where(pred_returns > 0, 1, 0)
        strategy_returns = positions * actual_returns
        
        # Metrics
        win_rate = np.mean(strategy_returns > 0) * 100
        total_return = np.sum(strategy_returns) * 100
        
        metrics = {
            'Directional_Accuracy': dir_acc,
            'Win_Rate': win_rate,
            'Total_Return': total_return,
            'Avg_Gain': np.mean(strategy_returns[strategy_returns > 0]) * 100 if any(strategy_returns > 0) else 0,
            'Avg_Loss': np.mean(strategy_returns[strategy_returns < 0]) * 100 if any(strategy_returns < 0) else 0,
            'Profit_Factor': np.sum(strategy_returns[strategy_returns > 0]) / abs(np.sum(strategy_returns[strategy_returns < 0])) if any(strategy_returns < 0) else np.inf
        }
        
        return metrics
    
    def calculate_risk_metrics(self) -> Dict:
        """Calculate risk-related metrics"""
        errors = self.y_true - self.y_pred
        
        # Errors
        metrics = {
            'Bias': np.mean(errors),
            'MBD': np.mean(errors) / np.mean(self.y_true) * 100,  # Mean bias deviation
            'MAD': np.mean(np.abs(errors - np.mean(errors)))  # Mean absolute deviation
        }
        
        # Residual analysis
        metrics.update({
            'Residual_Std': np.std(errors),
            'Residual_Skew': pd.Series(errors).skew(),
            'Residual_Kurtosis': pd.Series(errors).kurtosis()
        })
        
        # Prediction interval coverage (95% CI)
        std_error = np.std(errors)
        lower_bound = self.y_pred - 1.96 * std_error
        upper_bound = self.y_pred + 1.96 * std_error
        metrics['Coverage_95'] = np.mean((self.y_true >= lower_bound) & (self.y_true <= upper_bound)) * 100
        
        # Theil's U statistic
        y_true_log = np.log(self.y_true)
        y_pred_log = np.log(self.y_pred)
        y_true_diff = np.diff(y_true_log)
        y_pred_diff = np.diff(y_pred_log)
        
        numerator = np.sqrt(np.mean((y_true_diff - y_pred_diff) ** 2))
        denominator = np.sqrt(np.mean(y_true_diff ** 2))
        metrics['Theil_U'] = numerator / denominator if denominator != 0 else 0
        
        return metrics
    
    def calculate_all_metrics(self) -> Dict:
        """Calculate all metrics"""
        print_section(f"Evaluating {self.model_name}")
        
        with Timer(f"{self.model_name} Evaluation"):
            # Basic metrics
            basic_metrics = self.calculate_basic_metrics()
            
            # Percentage metrics
            percentage_metrics = self.calculate_percentage_metrics()
            
            # Directional metrics
            directional_metrics = self.calculate_directional_metrics()
            
            # Risk metrics
            risk_metrics = self.calculate_risk_metrics()
            
            # Combine all metrics
            self.metrics = {
                **basic_metrics,
                **percentage_metrics,
                **directional_metrics,
                **risk_metrics
            }
        
        return self.metrics
    
    def print_metrics(self):
        """Print metrics in a formatted way"""
        if not self.metrics:
            self.calculate_all_metrics()
        
        print("\n" + "="*70)
        print(f"  {self.model_name} - PERFORMANCE METRICS")
        print("="*70)
        
        # Basic metrics
        print("\n📊 Basic Metrics:")
        for key in ['R2', 'RMSE', 'MAE', 'MAPE', 'MSE', 'Explained_Variance']:
            if key in self.metrics:
                print(f"  {key:20s}: {self.metrics[key]:.4f}")
        
        # Directional metrics
        print("\n🎯 Directional Metrics:")
        for key in ['Directional_Accuracy', 'Win_Rate', 'Total_Return', 'Profit_Factor']:
            if key in self.metrics:
                print(f"  {key:20s}: {self.metrics[key]:.2f}%")
        
        # Risk metrics
        print("\n⚠️  Risk Metrics:")
        for key in ['Bias', 'MBD', 'Residual_Std', 'Coverage_95', 'Theil_U']:
            if key in self.metrics:
                print(f"  {key:20s}: {self.metrics[key]:.4f}")
    
    def save_metrics(self, output_dir: str):
        """Save metrics to file"""
        if not self.metrics:
            self.calculate_all_metrics()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.model_name}_metrics_{timestamp}"
        filepath = os.path.join(output_dir, filename)
        
        # Convert numpy types to Python types
        metrics_clean = {}
        for key, value in self.metrics.items():
            if isinstance(value, (np.integer, np.floating)):
                metrics_clean[key] = float(value)
            else:
                metrics_clean[key] = value
        
        # Save as JSON
        import json
        with open(f"{filepath}.json", 'w') as f:
            json.dump(metrics_clean, f, indent=4)
        
        print(f"✅ Metrics saved: {filepath}.json")
        
        return filepath


def compare_models(y_true, y_pred_dict: Dict[str, np.ndarray]) -> pd.DataFrame:
    """Compare multiple models"""
    comparison = []
    
    for name, y_pred in y_pred_dict.items():
        evaluator = AdvancedEvaluator(y_true, y_pred, name)
        metrics = evaluator.calculate_all_metrics()
        
        row = {
            'Model': name,
            'R2': metrics.get('R2', np.nan),
            'RMSE': metrics.get('RMSE', np.nan),
            'MAE': metrics.get('MAE', np.nan),
            'MAPE': metrics.get('MAPE', np.nan),
            'Dir_Acc': metrics.get('Directional_Accuracy', np.nan),
            'Profit_Factor': metrics.get('Profit_Factor', np.nan)
        }
        comparison.append(row)
    
    df = pd.DataFrame(comparison)
    df = df.sort_values('R2', ascending=False)
    
    return df


# ============================================================================
# LEGACY SUPPORT FUNCTIONS
# ============================================================================

def calculate_metrics(y_true, y_pred):
    """Legacy function for backward compatibility"""
    evaluator = AdvancedEvaluator(y_true, y_pred)
    metrics = evaluator.calculate_all_metrics()
    return metrics


def evaluate_all_models(models_dict, X_train, y_train, X_test, y_test, save=True):
    """Legacy function for backward compatibility"""
    results = {}
    y_pred_dict = {}
    
    for name, model in models_dict.items():
        print(f"\n📊 Evaluating {name}...")
        
        y_pred = model.predict(X_test)
        y_pred_dict[name] = y_pred
        
        evaluator = AdvancedEvaluator(y_test, y_pred, name)
        metrics = evaluator.calculate_all_metrics()
        evaluator.print_metrics()
        
        if save:
            evaluator.save_metrics(OUTPUTS_DIR)
        
        results[name] = {
            'metrics': metrics,
            'predictions': y_pred
        }
    
    # Create comparison table
    comparison = compare_models(y_test, y_pred_dict)
    print("\n" + "="*70)
    print("  MODEL COMPARISON")
    print("="*70)
    print(comparison.to_string(index=False))
    
    # Save comparison
    if save:
        comparison_path = os.path.join(OUTPUTS_DIR, 'model_comparison.csv')
        comparison.to_csv(comparison_path, index=False)
        print(f"\n✅ Comparison saved: {comparison_path}")
    
    return results


if __name__ == "__main__":
    print("Testing Enhanced Evaluation...")
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 100
    y_true = np.random.uniform(100, 200, n_samples)
    y_pred = y_true + np.random.normal(0, 5, n_samples)  # Some error
    
    # Test evaluator
    evaluator = AdvancedEvaluator(y_true, y_pred, "Test Model")
    metrics = evaluator.calculate_all_metrics()
    evaluator.print_metrics()
    evaluator.save_metrics(OUTPUTS_DIR)
    
    print("\n✅ Evaluation test complete!")