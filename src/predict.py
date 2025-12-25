"""
Prediction module for future stock prices
"""
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import (DATA_PROCESSED_DIR, MODELS_DIR, STOCK_SYMBOL, 
                          PREDICTIONS_DIR)
from utils.helpers import load_model, print_section
from src.data_loader import load_stock_data
from src.preprocess import preprocess_data


def predict_next_day(model, last_features):
    """
    Predict the next day's closing price
    
    Args:
        model: Trained model
        last_features: Features from the last available day
        
    Returns:
        Predicted price
    """
    prediction = model.predict(last_features.reshape(1, -1))
    return prediction[0]


def predict_future(model, df, feature_cols, days=30):
    """
    Predict future prices for multiple days
    
    Args:
        model: Trained model
        df: DataFrame with historical data
        feature_cols: List of feature column names
        days: Number of days to predict
        
    Returns:
        DataFrame with predictions
    """
    print(f"\n🔮 Predicting next {days} days...")
    
    predictions = []
    dates = []
    
    # Get the last available date
    last_date = pd.to_datetime(df['Date'].iloc[-1])
    
    # Use the last row as starting point
    current_features = df[feature_cols].iloc[-1:].copy()
    
    for i in range(days):
        # Predict next day
        pred_price = predict_next_day(model, current_features.values[0])
        
        # Calculate next date (skip weekends)
        next_date = last_date + timedelta(days=i+1)
        while next_date.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
            next_date += timedelta(days=1)
        
        predictions.append(pred_price)
        dates.append(next_date)
        
        # Note: This is a simplified approach
        # In reality, we'd need to update features based on prediction
        # For now, we'll use the same features (not ideal)
    
    # Create predictions DataFrame
    pred_df = pd.DataFrame({
        'Date': dates,
        'Predicted_Close': predictions
    })
    
    return pred_df


def make_predictions(model_name='xgboost', prediction_days=30, save=True):
    """
    Make predictions using a trained model
    
    Args:
        model_name: Name of the model to use
        prediction_days: Number of days to predict
        save: Whether to save predictions
        
    Returns:
        DataFrame with predictions
    """
    print_section(f"Making Predictions - {model_name.upper()}")
    
    # Load model
    print(f"📦 Loading model...")
    model = load_model(model_name, MODELS_DIR)
    
    # Load processed data
    filepath = os.path.join(DATA_PROCESSED_DIR, f"{STOCK_SYMBOL}_processed.csv")
    print(f"📂 Loading processed data...")
    df = pd.read_csv(filepath)
    
    # Get feature columns (exclude target and date)
    exclude_cols = ['Date', 'target', 'target_class']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    # Make predictions
    predictions_df = predict_future(model, df, feature_cols, days=prediction_days)
    
    # Add actual last price for reference
    last_actual_price = df['Close'].iloc[-1]
    last_actual_date = df['Date'].iloc[-1]
    
    print(f"\n📊 Last Actual Price ({last_actual_date}): ${last_actual_price:.2f}")
    print(f"📊 Predicted Price (Next Day): ${predictions_df['Predicted_Close'].iloc[0]:.2f}")
    print(f"📊 Predicted Price (Day {prediction_days}): ${predictions_df['Predicted_Close'].iloc[-1]:.2f}")
    
    # Calculate predicted change
    price_change = predictions_df['Predicted_Close'].iloc[-1] - last_actual_price
    price_change_pct = (price_change / last_actual_price) * 100
    
    print(f"\n💹 Predicted Change over {prediction_days} days:")
    print(f"   ${price_change:.2f} ({price_change_pct:+.2f}%)")
    
    # Save predictions
    if save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{STOCK_SYMBOL}_predictions_{model_name}_{timestamp}.csv"
        filepath = os.path.join(PREDICTIONS_DIR, filename)
        predictions_df.to_csv(filepath, index=False)
        print(f"\n✅ Predictions saved to: {filepath}")
    
    return predictions_df


def predict_with_confidence(model, X, n_iterations=100):
    """
    Make predictions with confidence intervals (for ensemble models)
    
    Args:
        model: Trained ensemble model
        X: Features
        n_iterations: Number of iterations for confidence calculation
        
    Returns:
        Mean prediction, lower bound, upper bound
    """
    if hasattr(model, 'estimators_'):
        # For ensemble models like Random Forest
        predictions = np.array([estimator.predict(X) for estimator in model.estimators_])
        
        mean_pred = predictions.mean(axis=0)
        std_pred = predictions.std(axis=0)
        
        # 95% confidence interval (1.96 * std)
        lower_bound = mean_pred - 1.96 * std_pred
        upper_bound = mean_pred + 1.96 * std_pred
        
        return mean_pred, lower_bound, upper_bound
    else:
        # For models without ensembles, return simple prediction
        pred = model.predict(X)
        return pred, pred, pred


def create_prediction_report(predictions_df, stock_symbol=STOCK_SYMBOL):
    """
    Create a formatted prediction report
    
    Args:
        predictions_df: DataFrame with predictions
        stock_symbol: Stock ticker symbol
        
    Returns:
        Formatted report string
    """
    report = []
    report.append("="*60)
    report.append(f"  STOCK PRICE PREDICTION REPORT")
    report.append(f"  Symbol: {stock_symbol}")
    report.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("="*60)
    report.append("\nPredicted Prices:")
    report.append("-"*60)
    
    # Show first few predictions
    for i in range(min(5, len(predictions_df))):
        date = predictions_df['Date'].iloc[i]
        price = predictions_df['Predicted_Close'].iloc[i]
        report.append(f"  {date}: ${price:.2f}")
    
    if len(predictions_df) > 5:
        report.append("  ...")
        # Show last prediction
        date = predictions_df['Date'].iloc[-1]
        price = predictions_df['Predicted_Close'].iloc[-1]
        report.append(f"  {date}: ${price:.2f}")
    
    report.append("-"*60)
    report.append(f"\nTotal predictions: {len(predictions_df)} days")
    report.append("="*60)
    
    return "\n".join(report)


if __name__ == "__main__":
    print("Testing Prediction Pipeline...")
    
    # Check if we have trained models
    model_names = ['linear_regression', 'random_forest', 'xgboost']
    available_models = []
    
    for model_name in model_names:
        filepath = os.path.join(MODELS_DIR, f"{model_name}.pkl")
        if os.path.exists(filepath):
            available_models.append(model_name)
    
    if available_models:
        print(f"📦 Available models: {', '.join(available_models)}")
        
        # Use XGBoost by default, or first available model
        model_to_use = 'xgboost' if 'xgboost' in available_models else available_models[0]
        
        # Make predictions
        predictions_df = make_predictions(
            model_name=model_to_use,
            prediction_days=30,
            save=True
        )
        
        # Print report
        report = create_prediction_report(predictions_df)
        print("\n" + report)
        
    else:
        print("❌ No trained models found.")
        print("Please run train_model.py first to train models.")