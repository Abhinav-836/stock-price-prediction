"""
Model evaluation module
"""
import pandas as pd
import numpy as np
import os
import sys
from sklearn.metrics import (mean_squared_error, mean_absolute_error, 
                             r2_score, mean_absolute_percentage_error)

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import DATA_PROCESSED_DIR, MODELS_DIR, STOCK_SYMBOL, OUTPUTS_DIR
from utils.helpers import load_model, save_metrics, print_section
from src.train_model import prepare_features_target, split_data


def calculate_metrics(y_true, y_pred):
    """
    Calculate regression metrics
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary of metrics
    """
    metrics = {
        'MAE': mean_absolute_error(y_true, y_pred),
        'MSE': mean_squared_error(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'R2': r2_score(y_true, y_pred),
        'MAPE': mean_absolute_percentage_error(y_true, y_pred) * 100
    }
    
    return metrics


def evaluate_model(model, X_train, y_train, X_test, y_test, model_name='Model'):
    """
    Evaluate a single model on train and test sets
    
    Args:
        model: Trained model
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        model_name: Name of the model
        
    Returns:
        Dictionary with train and test metrics
    """
    print_section(f"Evaluating {model_name}")
    
    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # Calculate metrics
    train_metrics = calculate_metrics(y_train, y_train_pred)
    test_metrics = calculate_metrics(y_test, y_test_pred)
    
    # Display results
    print("\n📊 Training Metrics:")
    for metric, value in train_metrics.items():
        print(f"  {metric:8s}: {value:12.4f}")
    
    print("\n📊 Test Metrics:")
    for metric, value in test_metrics.items():
        print(f"  {metric:8s}: {value:12.4f}")
    
    # Calculate overfitting indicator
    r2_diff = train_metrics['R2'] - test_metrics['R2']
    if r2_diff > 0.1:
        print(f"\n⚠️  Warning: Possible overfitting (R² difference: {r2_diff:.4f})")
    
    results = {
        'model_name': model_name,
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'predictions': {
            'train': y_train_pred.tolist(),
            'test': y_test_pred.tolist()
        }
    }
    
    return results


def evaluate_all_models(models_dict, X_train, y_train, X_test, y_test, save=True):
    """
    Evaluate all models
    
    Args:
        models_dict: Dictionary of trained models
        X_train: Training features
        y_train: Training target
        X_test: Test features
        y_test: Test target
        save: Whether to save evaluation results
        
    Returns:
        Dictionary with all evaluation results
    """
    print_section("Evaluating All Models")
    
    all_results = {}
    
    for model_name, model in models_dict.items():
        results = evaluate_model(
            model, X_train, y_train, X_test, y_test, 
            model_name=model_name.replace('_', ' ').title()
        )
        all_results[model_name] = results
        
        # Save individual model metrics
        if save:
            save_metrics(results, f"{model_name}_metrics", OUTPUTS_DIR)
    
    # Create comparison DataFrame
    comparison = create_comparison_table(all_results)
    print_section("Model Comparison")
    print("\n", comparison.to_string(index=False))
    
    # Save comparison
    if save:
        filepath = os.path.join(OUTPUTS_DIR, 'model_comparison.csv')
        comparison.to_csv(filepath, index=False)
        print(f"\n✅ Model comparison saved to: {filepath}")
    
    return all_results


def create_comparison_table(results_dict):
    """
    Create a comparison table for all models
    
    Args:
        results_dict: Dictionary with evaluation results
        
    Returns:
        DataFrame with comparison
    """
    comparison_data = []
    
    for model_name, results in results_dict.items():
        row = {
            'Model': results['model_name'],
            'Train_R2': results['train_metrics']['R2'],
            'Test_R2': results['test_metrics']['R2'],
            'Train_RMSE': results['train_metrics']['RMSE'],
            'Test_RMSE': results['test_metrics']['RMSE'],
            'Train_MAE': results['train_metrics']['MAE'],
            'Test_MAE': results['test_metrics']['MAE'],
            'Test_MAPE': results['test_metrics']['MAPE']
        }
        comparison_data.append(row)
    
    df = pd.DataFrame(comparison_data)
    
    # Sort by Test R2 score
    df = df.sort_values('Test_R2', ascending=False)
    
    return df


def calculate_directional_accuracy(y_true, y_pred):
    """
    Calculate directional accuracy (did we predict the direction correctly?)
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Directional accuracy percentage
    """
    # Calculate actual and predicted directions
    actual_direction = np.diff(y_true) > 0
    predicted_direction = np.diff(y_pred) > 0
    
    # Calculate accuracy
    correct_predictions = np.sum(actual_direction == predicted_direction)
    total_predictions = len(actual_direction)
    
    accuracy = (correct_predictions / total_predictions) * 100
    
    return accuracy


def evaluate_trading_strategy(y_true, y_pred, initial_investment=10000):
    """
    Simulate a simple trading strategy and calculate returns
    
    Args:
        y_true: True prices
        y_pred: Predicted prices
        initial_investment: Starting capital
        
    Returns:
        Dictionary with strategy results
    """
    # Calculate predicted and actual returns
    predicted_returns = np.diff(y_pred) / y_pred[:-1]
    actual_returns = np.diff(y_true) / y_true[:-1]
    
    # Simple strategy: buy if we predict price will go up
    positions = np.where(predicted_returns > 0, 1, 0)
    
    # Calculate strategy returns
    strategy_returns = positions * actual_returns
    cumulative_returns = np.cumprod(1 + strategy_returns)
    
    final_value = initial_investment * cumulative_returns[-1]
    total_return = ((final_value - initial_investment) / initial_investment) * 100
    
    # Buy and hold strategy
    buy_hold_return = ((y_true[-1] - y_true[0]) / y_true[0]) * 100
    
    results = {
        'initial_investment': initial_investment,
        'final_value': final_value,
        'total_return': total_return,
        'buy_hold_return': buy_hold_return,
        'excess_return': total_return - buy_hold_return,
        'num_trades': np.sum(np.diff(positions) != 0)
    }
    
    return results


if __name__ == "__main__":
    print("Testing Model Evaluation Pipeline...")
    
    # Load processed data
    filepath = os.path.join(DATA_PROCESSED_DIR, f"{STOCK_SYMBOL}_processed.csv")
    
    if os.path.exists(filepath):
        print(f"📂 Loading processed data...")
        df = pd.read_csv(filepath)
        
        # Prepare features and target
        X, y, feature_cols = prepare_features_target(df)
        
        # Split data
        X_train, X_test, y_train, y_test = split_data(X, y)
        
        # Load trained models
        model_names = ['linear_regression', 'random_forest', 'xgboost']
        models = {}
        
        for model_name in model_names:
            try:
                models[model_name] = load_model(model_name, MODELS_DIR)
            except FileNotFoundError:
                print(f"⚠️  Model '{model_name}' not found. Skipping...")
        
        if models:
            # Evaluate all models
            results = evaluate_all_models(
                models, X_train, y_train, X_test, y_test, save=True
            )
            
            # Additional analysis for best model
            best_model_name = 'xgboost'  # or determine from results
            if best_model_name in models:
                print_section("Additional Analysis - XGBoost")
                
                y_test_pred = models[best_model_name].predict(X_test)
                
                # Directional accuracy
                dir_acc = calculate_directional_accuracy(
                    y_test.values, y_test_pred
                )
                print(f"\n📈 Directional Accuracy: {dir_acc:.2f}%")
                
                # Trading strategy simulation
                strategy_results = evaluate_trading_strategy(
                    y_test.values, y_test_pred
                )
                print("\n💰 Trading Strategy Results:")
                for key, value in strategy_results.items():
                    print(f"  {key}: {value:.2f}")
        else:
            print("❌ No trained models found. Please run train_model.py first.")
    else:
        print(f"❌ Processed data not found at: {filepath}")
        print("Please run the preprocessing pipeline first.")