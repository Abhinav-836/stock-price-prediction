"""
Prediction module for future stock prices - COMPLETE FIXED
"""
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import (
    DATA_PROCESSED_DIR, 
    MODELS_DIR, 
    STOCK_SYMBOL,
    PREDICTIONS_DIR,
    PREDICTION_DAYS
)
from utils.helpers import (
    load_model, 
    print_section, 
    save_dataframe,
    Timer
)


class PredictionEngine:
    """
    Advanced prediction engine with memory optimization
    """
    
    def __init__(self, model_name='xgboost', model=None):
        self.model_name = model_name
        self.model = model
        self.predictions = None
        self.confidence_intervals = None
        
    def load_model(self, models_dir=MODELS_DIR):
        """Load model from disk"""
        if self.model is None:
            self.model = load_model(self.model_name, models_dir)
        return self.model
    
    def predict_next_day(self, last_features: np.ndarray) -> float:
        """Predict next day's closing price"""
        if self.model is None:
            self.load_model()
        
        prediction = self.model.predict(last_features.reshape(1, -1))
        return prediction[0]
    
    def predict_future(self, df: pd.DataFrame, feature_cols: List[str], 
                       days: int = 30, return_uncertainty: bool = True) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """Predict future prices for multiple days - FIXED"""
        if self.model is None:
            self.load_model()
        
        print(f"\n🔮 Predicting next {days} days...")
        
        # Initialize lists BEFORE the loop
        predictions = []
        lower_bounds = []
        upper_bounds = []
        dates = []
        
        # Get the last available date
        last_date = pd.to_datetime(df['Date'].iloc[-1])
        
        # Use the last row as starting point
        current_features = df[feature_cols].iloc[-1:].copy()
        
        print(f"📊 Using {len(feature_cols)} features for prediction")
        
        for i in range(days):
            try:
                # Predict next day
                pred_price = self.predict_next_day(current_features.values[0])
            except ValueError as e:
                print(f"⚠️  Prediction error: {e}")
                # Use the last price as fallback
                pred_price = df['Close'].iloc[-1]
            
            # Get confidence interval
            if return_uncertainty and hasattr(self.model, 'estimators_'):
                predictions_ensemble = np.array([
                    estimator.predict(current_features.values.reshape(1, -1))
                    for estimator in self.model.estimators_
                ])
                mean_pred = np.mean(predictions_ensemble)
                std_pred = np.std(predictions_ensemble)
                lower = mean_pred - 1.96 * std_pred
                upper = mean_pred + 1.96 * std_pred
            else:
                # Simple 5% confidence interval
                lower = pred_price * 0.95
                upper = pred_price * 1.05
            
            # Calculate next date (skip weekends)
            next_date = last_date + timedelta(days=i+1)
            while next_date.weekday() >= 5:
                next_date += timedelta(days=1)
            
            # Append to lists
            predictions.append(pred_price)
            lower_bounds.append(lower)
            upper_bounds.append(upper)
            dates.append(next_date)
            
            # Update features for next prediction
            current_features = current_features.copy()
            if 'Close' in current_features.columns:
                current_features['Close'] = pred_price
        
        # Create predictions DataFrame
        pred_df = pd.DataFrame({
            'Date': dates,
            'Predicted_Close': predictions,
            'Lower_Bound': lower_bounds,
            'Upper_Bound': upper_bounds
        })
        
        self.predictions = pred_df
        
        return pred_df
    
    def generate_report(self, pred_df: pd.DataFrame, stock_symbol: str = STOCK_SYMBOL) -> str:
        """Generate a formatted prediction report"""
        report = []
        report.append("="*70)
        report.append(f"  STOCK PRICE PREDICTION REPORT")
        report.append(f"  Symbol: {stock_symbol}")
        report.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"  Model: {self.model_name}")
        report.append("="*70)
        
        # Prediction summary
        report.append("\nPrediction Summary:")
        report.append("-"*70)
        report.append(f"  Initial Price (Last Known): ${pred_df['Predicted_Close'].iloc[0]:.2f}")
        report.append(f"  Final Price (Day {len(pred_df)}): ${pred_df['Predicted_Close'].iloc[-1]:.2f}")
        
        # Change statistics
        price_change = pred_df['Predicted_Close'].iloc[-1] - pred_df['Predicted_Close'].iloc[0]
        price_change_pct = (price_change / pred_df['Predicted_Close'].iloc[0]) * 100
        report.append(f"  Total Change: ${price_change:.2f} ({price_change_pct:+.2f}%)")
        
        # Daily changes
        daily_changes = pred_df['Predicted_Close'].pct_change() * 100
        report.append(f"  Avg Daily Change: {daily_changes.mean():+.2f}%")
        report.append(f"  Max Daily Change: {daily_changes.max():+.2f}%")
        report.append(f"  Min Daily Change: {daily_changes.min():+.2f}%")
        
        # Confidence intervals
        if 'Lower_Bound' in pred_df.columns and 'Upper_Bound' in pred_df.columns:
            avg_ci_width = ((pred_df['Upper_Bound'] - pred_df['Lower_Bound']).mean())
            avg_ci_width_pct = (avg_ci_width / pred_df['Predicted_Close'].mean()) * 100
            report.append(f"\nConfidence Intervals (95%):")
            report.append(f"  Avg Width: ${avg_ci_width:.2f} ({avg_ci_width_pct:.1f}%)")
        
        # Prediction details
        report.append("\nDetailed Predictions:")
        report.append("-"*70)
        report.append(f"{'Date':<12} {'Predicted':<12} {'Lower':<12} {'Upper':<12} {'Change %':<10}")
        report.append("-"*70)
        
        for i in range(min(10, len(pred_df))):
            row = pred_df.iloc[i]
            date_str = row['Date'].strftime('%Y-%m-%d')
            change_pct = ((row['Predicted_Close'] - pred_df['Predicted_Close'].iloc[0]) / pred_df['Predicted_Close'].iloc[0]) * 100
            report.append(f"{date_str:<12} ${row['Predicted_Close']:>10.2f} "
                         f"${row.get('Lower_Bound', row['Predicted_Close']):>10.2f} "
                         f"${row.get('Upper_Bound', row['Predicted_Close']):>10.2f} "
                         f"{change_pct:>8.2f}%")
        
        if len(pred_df) > 10:
            report.append(f"... and {len(pred_df) - 10} more days")
        
        # Risk assessment
        final_change_pct = ((pred_df['Predicted_Close'].iloc[-1] - pred_df['Predicted_Close'].iloc[0]) / 
                           pred_df['Predicted_Close'].iloc[0]) * 100
        
        report.append("\nRisk Assessment:")
        report.append("-"*70)
        
        if final_change_pct > 5:
            risk = "LOW RISK - Strong Bullish Signal"
        elif final_change_pct > 2:
            risk = "MODERATE RISK - Mild Bullish Signal"
        elif final_change_pct > -2:
            risk = "MODERATE RISK - Neutral Outlook"
        elif final_change_pct > -5:
            risk = "MODERATE RISK - Mild Bearish Signal"
        else:
            risk = "HIGH RISK - Strong Bearish Signal"
        
        report.append(f"  {risk}")
        report.append(f"  Expected Return: {final_change_pct:+.2f}%")
        
        # Volatility assessment
        volatility = daily_changes.std()
        if volatility > 3:
            vol_risk = "HIGH VOLATILITY"
        elif volatility > 1.5:
            vol_risk = "MODERATE VOLATILITY"
        else:
            vol_risk = "LOW VOLATILITY"
        report.append(f"  Volatility: {vol_risk} ({volatility:.2f}% daily avg)")
        
        report.append("\n" + "="*70)
        report.append("  DISCLAIMER: This is for informational purposes only.")
        report.append("  Not financial advice. Always do your own research.")
        report.append("="*70)
        
        return "\n".join(report)
    
    def save_predictions(self, pred_df: pd.DataFrame, output_dir: str = PREDICTIONS_DIR):
        """Save predictions to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{STOCK_SYMBOL}_predictions_{self.model_name}_{timestamp}.csv"
        save_dataframe(pred_df, filename, output_dir, format='csv')
        
        # Save report with UTF-8 encoding
        report = self.generate_report(pred_df)
        report_filename = f"{STOCK_SYMBOL}_report_{self.model_name}_{timestamp}.txt"
        report_path = os.path.join(output_dir, report_filename)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"✅ Report saved: {report_path}")
        
        return filename


# ============================================================================
# LEGACY SUPPORT FUNCTIONS
# ============================================================================

def make_predictions(model_name='xgboost', prediction_days=30, save=True):
    """Legacy function - DON'T limit features"""
    print_section(f"Making Predictions - {model_name.upper()}")
    
    # Load model
    model = load_model(model_name, MODELS_DIR)
    
    # Load processed data
    filepath = os.path.join(DATA_PROCESSED_DIR, f"{STOCK_SYMBOL}_processed.csv")
    print(f"📂 Loading processed data...")
    
    # Load the full dataset
    df = pd.read_csv(filepath)
    print(f"✅ Loaded {len(df)} rows")
    
    # Get feature columns - use ALL features
    exclude_cols = ['Date', 'target', 'target_class'] + [col for col in df.columns if 'target_' in col]
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    feature_cols = [col for col in feature_cols if not col.endswith('d')]
    feature_cols = [col for col in feature_cols if 'target' not in col]
    
    print(f"📊 Using {len(feature_cols)} features for prediction")
    
    # Create prediction engine
    engine = PredictionEngine(model_name=model_name, model=model)
    
    # Make predictions with ALL features
    pred_df = engine.predict_future(df, feature_cols, days=prediction_days)
    
    # Generate report
    report = engine.generate_report(pred_df)
    print("\n" + report)
    
    # Save predictions
    if save:
        engine.save_predictions(pred_df)
    
    return pred_df


if __name__ == "__main__":
    print("Testing Enhanced Prediction Engine...")
    
    filepath = os.path.join(DATA_PROCESSED_DIR, f"{STOCK_SYMBOL}_processed.csv")
    
    if os.path.exists(filepath):
        predictions = make_predictions('xgboost', prediction_days=30, save=True)
        print(f"\n✅ Predictions shape: {predictions.shape}")
        print(predictions.head())
    else:
        print(f"❌ Processed data not found at: {filepath}")