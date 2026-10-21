"""
Prediction module for future stock prices - FIXED

Key fixes vs previous version:
  * Recursive forecasting actually updates the inputs. The old loop only overwrote a
    'Close' column and fed the SAME feature row to the model every day, so all 30 days got
    the same prediction (a flat line). Now each predicted day is appended to the price
    history and ALL features (lags, RSI, MAs, ...) are recomputed before the next step.
  * Duplicate dates fixed (old weekend-skip logic produced Mon, Mon, Mon for Sat/Sun/Mon).
  * GradientBoosting crash fixed: its `estimators_` is a 2-D numpy array, so calling
    `.predict` on its items raised AttributeError (only ValueError was caught).
  * Confidence band now widens with the horizon (sqrt-time) instead of a fixed ±5%.
  * Report "Initial price" is the real last close, not the first prediction.
"""
import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
from typing import Optional, List
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import DATA_PROCESSED_DIR, DATA_RAW_DIR, MODELS_DIR, STOCK_SYMBOL, PREDICTIONS_DIR
from utils.helpers import load_model, print_section, save_dataframe

try:
    from src.preprocess import StockDataPreprocessor, get_feature_columns
    from src.train_model import DEFAULT_TARGET, is_return_target
except ImportError:
    from preprocess import StockDataPreprocessor, get_feature_columns
    from train_model import DEFAULT_TARGET, is_return_target

RAW_COLS = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']


class PredictionEngine:
    """
    Recursive multi-day forecaster
    """

    def __init__(self, model_name='xgboost', model=None, target_col: str = DEFAULT_TARGET,
                 symbol: str = STOCK_SYMBOL):
        self.model_name = model_name
        self.model = model
        self.target_col = target_col
        self.symbol = symbol
        self.predictions = None
        self.last_close = None
        self.last_date = None

    def load_model(self, models_dir=MODELS_DIR):
        if self.model is None:
            self.model = load_model(self.model_name, models_dir)
        return self.model

    def predict_next_day(self, last_features) -> float:
        """Raw model output for one feature row (return % or price, depending on target)"""
        if self.model is None:
            self.load_model()
        if isinstance(last_features, pd.DataFrame):
            return float(self.model.predict(last_features)[0])
        return float(self.model.predict(np.asarray(last_features).reshape(1, -1))[0])

    @staticmethod
    def _synthesize_next_bar(hist: pd.DataFrame, next_close: float, next_date) -> dict:
        """Build a plausible OHLCV bar for a forecasted day, using recent typical candle shape."""
        recent = hist.tail(60)
        body_top = recent[['Open', 'Close']].max(axis=1)
        body_bot = recent[['Open', 'Close']].min(axis=1)
        up_wick = float(np.nanmedian(recent['High'] / body_top - 1)) if len(recent) else 0.005
        dn_wick = float(np.nanmedian(1 - recent['Low'] / body_bot)) if len(recent) else 0.005

        prev_close = float(hist['Close'].iloc[-1])
        open_ = prev_close
        high = max(open_, next_close) * (1 + max(up_wick, 0))
        low = min(open_, next_close) * (1 - max(dn_wick, 0))
        volume = float(hist['Volume'].tail(20).mean())
        return {'Date': next_date, 'Open': open_, 'High': high, 'Low': low,
                'Close': next_close, 'Volume': volume}

    def predict_future(self, df: pd.DataFrame, feature_cols: List[str], days: int = 30,
                       return_uncertainty: bool = True, raw_df: Optional[pd.DataFrame] = None,
                       daily_vol_pct: Optional[float] = None) -> pd.DataFrame:
        """
        Forecast `days` business days ahead.

        df          : processed dataframe (used for the raw columns if raw_df is not given)
        raw_df      : raw OHLCV history (preferred - must include the warm-up period)
        daily_vol_pct: daily std of returns in % for the confidence band
                       (defaults to the last 60 days of realised volatility)
        """
        if self.model is None:
            self.load_model()

        source = raw_df if raw_df is not None else df
        hist = source[RAW_COLS].copy()
        hist['Date'] = pd.to_datetime(hist['Date'], utc=True).dt.tz_localize(None)
        hist = hist.sort_values('Date').reset_index(drop=True)

        self.last_close = float(hist['Close'].iloc[-1])
        self.last_date = hist['Date'].iloc[-1]

        if daily_vol_pct is None:
            daily_vol_pct = float(hist['Close'].pct_change().tail(60).std() * 100)
        if not np.isfinite(daily_vol_pct) or daily_vol_pct <= 0:
            daily_vol_pct = 2.0

        future_dates = pd.bdate_range(self.last_date + pd.Timedelta(days=1), periods=days)
        return_target = is_return_target(self.target_col)

        predictions, dates = [], []
        for step, next_date in enumerate(future_dates):
            processed = StockDataPreprocessor(hist, verbose=False).run_pipeline()
            X_last = processed.reindex(columns=feature_cols).iloc[[-1]].fillna(0.0)
            current_close = float(hist['Close'].iloc[-1])

            try:
                raw_pred = self.predict_next_day(X_last)
            except Exception as e:
                print(f"⚠️  Prediction error on step {step + 1}: {e}")
                raw_pred = 0.0 if return_target else current_close

            if return_target:
                # Clamp to +/- 4 sigma so one bad step cannot explode the compounding path
                ret = float(np.clip(raw_pred, -4 * daily_vol_pct, 4 * daily_vol_pct))
                next_close = current_close * (1 + ret / 100)
            else:
                next_close = raw_pred

            predictions.append(next_close)
            dates.append(next_date)
            hist = pd.concat([hist, pd.DataFrame([self._synthesize_next_bar(hist, next_close, next_date)])],
                             ignore_index=True)

        pred_df = pd.DataFrame({'Date': dates, 'Predicted_Close': predictions})

        if return_uncertainty:
            horizon = np.arange(1, len(pred_df) + 1)
            band = 1.96 * (daily_vol_pct / 100) * np.sqrt(horizon)
            pred_df['Lower_Bound'] = pred_df['Predicted_Close'] * np.exp(-band)
            pred_df['Upper_Bound'] = pred_df['Predicted_Close'] * np.exp(band)

        self.predictions = pred_df
        return pred_df

    def generate_report(self, pred_df: pd.DataFrame, stock_symbol: Optional[str] = None) -> str:
        stock_symbol = stock_symbol or self.symbol
        base = self.last_close if self.last_close else float(pred_df['Predicted_Close'].iloc[0])

        report = ["=" * 70,
                  "  STOCK PRICE PREDICTION REPORT",
                  f"  Symbol: {stock_symbol}",
                  f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                  f"  Model: {self.model_name}  (target: {self.target_col})",
                  "=" * 70,
                  "\nPrediction Summary:", "-" * 70,
                  f"  Last Known Close: ${base:.2f}",
                  f"  Final Price (Day {len(pred_df)}): ${pred_df['Predicted_Close'].iloc[-1]:.2f}"]

        price_change = pred_df['Predicted_Close'].iloc[-1] - base
        price_change_pct = price_change / base * 100
        report.append(f"  Total Change: ${price_change:.2f} ({price_change_pct:+.2f}%)")

        path = pd.concat([pd.Series([base]), pred_df['Predicted_Close']], ignore_index=True)
        daily_changes = path.pct_change().dropna() * 100
        report.append(f"  Avg Daily Change: {daily_changes.mean():+.2f}%")
        report.append(f"  Max Daily Change: {daily_changes.max():+.2f}%")
        report.append(f"  Min Daily Change: {daily_changes.min():+.2f}%")

        if {'Lower_Bound', 'Upper_Bound'}.issubset(pred_df.columns):
            avg_ci_width = (pred_df['Upper_Bound'] - pred_df['Lower_Bound']).mean()
            report.append("\nConfidence Band (95%, widens with horizon):")
            report.append(f"  Avg Width: ${avg_ci_width:.2f} "
                          f"({avg_ci_width / pred_df['Predicted_Close'].mean() * 100:.1f}%)")

        report += ["\nDetailed Predictions:", "-" * 70,
                   f"{'Date':<12} {'Predicted':<12} {'Lower':<12} {'Upper':<12} {'Change %':<10}", "-" * 70]

        for i in range(min(10, len(pred_df))):
            row = pred_df.iloc[i]
            change_pct = (row['Predicted_Close'] - base) / base * 100
            report.append(f"{pd.Timestamp(row['Date']).strftime('%Y-%m-%d'):<12} ${row['Predicted_Close']:>10.2f} "
                          f"${row.get('Lower_Bound', row['Predicted_Close']):>10.2f} "
                          f"${row.get('Upper_Bound', row['Predicted_Close']):>10.2f} "
                          f"{change_pct:>8.2f}%")
        if len(pred_df) > 10:
            report.append(f"... and {len(pred_df) - 10} more days")

        report += ["\nOutlook:", "-" * 70]
        if price_change_pct > 5:
            outlook = "Strong Bullish Signal"
        elif price_change_pct > 2:
            outlook = "Mild Bullish Signal"
        elif price_change_pct > -2:
            outlook = "Neutral Outlook"
        elif price_change_pct > -5:
            outlook = "Mild Bearish Signal"
        else:
            outlook = "Strong Bearish Signal"
        report.append(f"  {outlook}")
        report.append(f"  Expected Return: {price_change_pct:+.2f}%")

        report += ["\n" + "=" * 70,
                   "  DISCLAIMER: This is for informational purposes only.",
                   "  Not financial advice. Always do your own research.",
                   "=" * 70]
        return "\n".join(report)

    def save_predictions(self, pred_df: pd.DataFrame, output_dir: str = PREDICTIONS_DIR):
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.symbol}_predictions_{self.model_name}_{timestamp}.csv"
        save_dataframe(pred_df, filename, output_dir, format='csv')

        report_path = os.path.join(output_dir, f"{self.symbol}_report_{self.model_name}_{timestamp}.txt")
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(self.generate_report(pred_df))
        print(f"✅ Report saved: {report_path}")
        return filename


# ============================================================================
# LEGACY SUPPORT FUNCTIONS
# ============================================================================

def make_predictions(model_name='xgboost', prediction_days=30, save=True, symbol=STOCK_SYMBOL,
                     target_col=DEFAULT_TARGET):
    print_section(f"Making Predictions - {model_name.upper()}")

    model = load_model(model_name, MODELS_DIR)

    processed_path = os.path.join(DATA_PROCESSED_DIR, f"{symbol}_processed.csv")
    df = pd.read_csv(processed_path)
    print(f"✅ Loaded {len(df)} processed rows")

    # Prefer the raw file (includes the warm-up history needed to recompute features)
    raw_path = os.path.join(DATA_RAW_DIR, f"{symbol}_raw.csv")
    raw_df = pd.read_csv(raw_path) if os.path.exists(raw_path) else None

    feature_cols = get_feature_columns(df)
    print(f"📊 Using {len(feature_cols)} features for prediction")

    engine = PredictionEngine(model_name=model_name, model=model, target_col=target_col, symbol=symbol)
    pred_df = engine.predict_future(df, feature_cols, days=prediction_days, raw_df=raw_df)

    print("\n" + engine.generate_report(pred_df))
    if save:
        engine.save_predictions(pred_df)
    return pred_df


if __name__ == "__main__":
    print("Testing Prediction Engine...")
    filepath = os.path.join(DATA_PROCESSED_DIR, f"{STOCK_SYMBOL}_processed.csv")
    if os.path.exists(filepath):
        predictions = make_predictions('xgboost', prediction_days=30, save=True)
        print(f"\n✅ Predictions shape: {predictions.shape}")
        print(predictions.head())
    else:
        print(f"❌ Processed data not found at: {filepath}")