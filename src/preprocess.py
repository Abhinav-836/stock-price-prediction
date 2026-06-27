"""
Data preprocessing and feature engineering module - COMPLETE FIXED VERSION
"""
import pandas as pd
import numpy as np
import os
import sys
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import (
    DATA_PROCESSED_DIR,
    DATA_RAW_DIR,
    STOCK_SYMBOL,
    MOVING_AVERAGES,
    PREDICTION_DAYS,
    CREATE_DATE_FEATURES,
    CREATE_PRICE_FEATURES,
    CREATE_TECHNICAL_INDICATORS,
    CREATE_VOLUME_FEATURES,
    CREATE_STATISTICAL_FEATURES,
    CREATE_PATTERN_FEATURES,
    CREATE_INTERACTION_FEATURES
)
from utils.helpers import print_section


class StockDataPreprocessor:
    """
    Advanced data preprocessing and feature engineering for stock data
    """
    
    def __init__(self, df: pd.DataFrame, config: Optional[Dict] = None):
        self.df = df.copy()
        self.config = config or {}
        self.feature_columns = []
        self.target_columns = []
        
    def clean_numeric_columns(self) -> 'StockDataPreprocessor':
        """Clean numeric columns with special handling"""
        df = self.df.copy()
        numeric_cols = ["Price", "Open", "Close", "High", "Low", "Volume", "Change %"]
        
        for col in numeric_cols:
            if col not in df.columns:
                continue
            
            if col == "Volume":
                df[col] = (df[col].astype(str)
                          .str.replace(",", "", regex=False)
                          .str.replace("-", "0")
                          .str.replace("M", "*1e6")
                          .str.replace("K", "*1e3")
                          .str.replace("B", "*1e9"))
                
                def convert_volume(x):
                    try:
                        if "*" in str(x):
                            return eval(str(x))
                        return float(x)
                    except Exception:
                        return np.nan
                
                df[col] = df[col].apply(convert_volume)
                continue
            
            if col == "Change %":
                df[col] = df[col].astype(str).str.replace("%", "", regex=False)
            
            df[col] = (df[col]
                      .astype(str)
                      .str.replace(",", "", regex=False)
                      .str.strip())
            df[col] = pd.to_numeric(df[col], errors="coerce")
        
        self.df = df
        return self
    
    def create_date_features(self) -> 'StockDataPreprocessor':
        """Create comprehensive date-based features - ULTRA FIXED"""
        if not CREATE_DATE_FEATURES:
            return self
        
        df = self.df.copy()
        
        # ====================================================================
        # ULTRA FIX: Handle ANY date format, timezone, or type
        # ====================================================================
        # Check if Date column exists
        if 'Date' not in df.columns:
            print("⚠️  No 'Date' column found. Skipping date features.")
            self.df = df
            return self
        
        # Try multiple approaches to convert to datetime
        try:
            # Approach 1: Try simple conversion
            df['Date'] = pd.to_datetime(df['Date'])
        except:
            try:
                # Approach 2: Try with utc=True
                df['Date'] = pd.to_datetime(df['Date'], utc=True)
            except:
                try:
                    # Approach 3: Try to convert string to datetime
                    df['Date'] = pd.to_datetime(df['Date'].astype(str))
                except:
                    print("⚠️  Could not convert 'Date' column to datetime")
                    self.df = df
                    return self
        
        # Remove timezone if present
        if hasattr(df['Date'].dt, 'tz') and df['Date'].dt.tz is not None:
            df['Date'] = df['Date'].dt.tz_localize(None)
        
        df = df.sort_values("Date")
        
        # Basic date features
        df["year"] = df["Date"].dt.year
        df["month"] = df["Date"].dt.month
        df["day"] = df["Date"].dt.day
        df["day_of_week"] = df["Date"].dt.dayofweek
        df["quarter"] = df["Date"].dt.quarter
        df["day_of_year"] = df["Date"].dt.dayofyear
        df["week_of_year"] = df["Date"].dt.isocalendar().week.astype(int)
        
        # Cyclical encoding for seasonal patterns
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
        
        # Special dates
        df["is_month_start"] = df["Date"].dt.is_month_start.astype(int)
        df["is_month_end"] = df["Date"].dt.is_month_end.astype(int)
        df["is_quarter_end"] = (df["month"] % 3 == 0).astype(int)
        df["is_year_end"] = (df["month"] == 12).astype(int)
        
        # Safe subtraction - both are now naive
        try:
            df["days_since_2010"] = (df["Date"] - pd.Timestamp("2010-01-01")).dt.days
        except:
            df["days_since_2010"] = 0
        
        try:
            df["days_since_min"] = (df["Date"] - df["Date"].min()).dt.days
        except:
            df["days_since_min"] = 0
        
        self.df = df
        return self
    
    def create_price_features(self) -> 'StockDataPreprocessor':
        """Create comprehensive price-based features"""
        if not CREATE_PRICE_FEATURES:
            return self
        
        df = self.df.copy()
        
        # Basic OHLC features
        df["open_close"] = df["Open"] - df["Close"]
        df["high_low"] = df["High"] - df["Low"]
        df["high_close"] = df["High"] - df["Close"]
        df["low_close"] = df["Low"] - df["Close"]
        
        # Price changes at different horizons
        for period in [1, 2, 3, 5, 10, 20, 50]:
            df[f"price_change_{period}"] = df["Close"].diff(period)
            df[f"price_change_pct_{period}"] = df["Close"].pct_change(period) * 100
        
        # Volume features
        for period in [1, 2, 5, 10, 20]:
            df[f"volume_change_{period}"] = df["Volume"].diff(period)
            df[f"volume_change_pct_{period}"] = df["Volume"].pct_change(period) * 100
        
        # Price ratios
        df["high_low_ratio"] = df["High"] / df["Low"]
        df["close_open_ratio"] = df["Close"] / df["Open"]
        df["high_open_ratio"] = df["High"] / df["Open"]
        df["low_open_ratio"] = df["Low"] / df["Open"]
        
        # Price acceleration
        df["price_acceleration"] = df["Close"].pct_change().pct_change()
        df["price_acceleration_5"] = df["Close"].pct_change(5).pct_change(5)
        
        # Range features
        df["daily_range"] = df["High"] - df["Low"]
        df["daily_range_pct"] = (df["High"] - df["Low"]) / df["Close"] * 100
        df["daily_range_ma_20"] = df["daily_range"].rolling(20).mean()
        df["daily_range_ratio"] = df["daily_range"] / df["daily_range_ma_20"]
        
        self.df = df
        return self
    
    def create_moving_averages(self, windows: Optional[List[int]] = None) -> 'StockDataPreprocessor':
        """Create moving average features"""
        if windows is None:
            windows = MOVING_AVERAGES
        
        df = self.df.copy()
        
        for window in windows:
            # Simple moving averages
            df[f"SMA_{window}"] = df["Close"].rolling(window=window).mean()
            df[f"SMA_{window}_Open"] = df["Open"].rolling(window=window).mean()
            df[f"SMA_{window}_High"] = df["High"].rolling(window=window).mean()
            df[f"SMA_{window}_Low"] = df["Low"].rolling(window=window).mean()
            
            # Exponential moving averages
            df[f"EMA_{window}"] = df["Close"].ewm(span=window, adjust=False).mean()
            
            # Price relative to moving average
            df[f"Close_SMA_{window}_ratio"] = df["Close"] / df[f"SMA_{window}"]
            df[f"Close_EMA_{window}_ratio"] = df["Close"] / df[f"EMA_{window}"]
            
            # Distance from moving average
            df[f"Close_SMA_{window}_diff"] = df["Close"] - df[f"SMA_{window}"]
            df[f"Close_SMA_{window}_diff_pct"] = (df["Close"] - df[f"SMA_{window}"]) / df[f"SMA_{window}"] * 100
        
        # Moving average crossovers
        for w1, w2 in [(5, 20), (20, 50), (50, 200)]:
            if f"SMA_{w1}" in df.columns and f"SMA_{w2}" in df.columns:
                df[f"MA_Crossover_{w1}_{w2}"] = np.where(
                    df[f"SMA_{w1}"] > df[f"SMA_{w2}"], 1, 0
                )
                df[f"MA_Crossover_{w1}_{w2}_diff"] = df[f"SMA_{w1}"] - df[f"SMA_{w2}"]
                df[f"MA_Crossover_{w1}_{w2}_diff_pct"] = (df[f"SMA_{w1}"] - df[f"SMA_{w2}"]) / df[f"SMA_{w2}"] * 100
        
        self.df = df
        return self
    
    def create_technical_indicators(self) -> 'StockDataPreprocessor':
        """Create comprehensive technical indicators"""
        if not CREATE_TECHNICAL_INDICATORS:
            return self
        
        df = self.df.copy()
        
        # RSI with multiple periods
        for period in [7, 14, 21, 28]:
            delta = df["Close"].diff()
            gain = delta.where(delta > 0, 0.0).rolling(period).mean()
            loss = -delta.where(delta < 0, 0.0).rolling(period).mean()
            rs = gain / loss
            df[f"RSI_{period}"] = 100 - (100 / (1 + rs))
        
        # MACD
        for fast, slow, signal in [(12, 26, 9), (8, 17, 9)]:
            ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
            ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()
            df[f"MACD_{fast}_{slow}"] = ema_fast - ema_slow
            df[f"MACD_{fast}_{slow}_signal"] = df[f"MACD_{fast}_{slow}"].ewm(span=signal, adjust=False).mean()
            df[f"MACD_{fast}_{slow}_diff"] = df[f"MACD_{fast}_{slow}"] - df[f"MACD_{fast}_{slow}_signal"]
        
        # Bollinger Bands
        for period in [20, 50]:
            rolling_mean = df["Close"].rolling(period).mean()
            rolling_std = df["Close"].rolling(period).std()
            
            df[f"BB_middle_{period}"] = rolling_mean
            df[f"BB_upper_{period}"] = rolling_mean + (rolling_std * 2)
            df[f"BB_lower_{period}"] = rolling_mean - (rolling_std * 2)
            df[f"BB_width_{period}"] = df[f"BB_upper_{period}"] - df[f"BB_lower_{period}"]
            df[f"BB_position_{period}"] = (df["Close"] - df[f"BB_lower_{period}"]) / df[f"BB_width_{period}"]
            df[f"BB_percent_{period}"] = (df["Close"] - df[f"BB_lower_{period}"]) / (df[f"BB_upper_{period}"] - df[f"BB_lower_{period}"]) * 100
        
        # Average True Range
        for period in [14, 21]:
            high_low = df["High"] - df["Low"]
            high_close = (df["High"] - df["Close"].shift()).abs()
            low_close = (df["Low"] - df["Close"].shift()).abs()
            ranges = pd.concat([high_low, high_close, low_close], axis=1)
            true_range = ranges.max(axis=1)
            df[f"ATR_{period}"] = true_range.rolling(period).mean()
            df[f"ATR_{period}_pct"] = df[f"ATR_{period}"] / df["Close"] * 100
        
        # Stochastic Oscillator
        for period in [14, 21]:
            low_min = df["Low"].rolling(period).min()
            high_max = df["High"].rolling(period).max()
            df[f"Stochastic_K_{period}"] = 100 * ((df["Close"] - low_min) / (high_max - low_min))
            df[f"Stochastic_D_{period}"] = df[f"Stochastic_K_{period}"].rolling(3).mean()
        
        # Ichimoku Cloud
        high_9 = df["High"].rolling(9).max()
        low_9 = df["Low"].rolling(9).min()
        df["Ichimoku_Conversion"] = (high_9 + low_9) / 2
        
        high_26 = df["High"].rolling(26).max()
        low_26 = df["Low"].rolling(26).min()
        df["Ichimoku_Base"] = (high_26 + low_26) / 2
        
        df["Ichimoku_SpanA"] = ((df["Ichimoku_Conversion"] + df["Ichimoku_Base"]) / 2).shift(26)
        
        high_52 = df["High"].rolling(52).max()
        low_52 = df["Low"].rolling(52).min()
        df["Ichimoku_SpanB"] = ((high_52 + low_52) / 2).shift(26)
        
        df["Ichimoku_Chikou"] = df["Close"].shift(-26)
        
        self.df = df
        return self
    
    def create_volume_features(self) -> 'StockDataPreprocessor':
        """Create volume-based features"""
        if not CREATE_VOLUME_FEATURES:
            return self
        
        df = self.df.copy()
        
        # Volume moving averages
        for period in [5, 10, 20, 50]:
            df[f"Volume_MA_{period}"] = df["Volume"].rolling(period).mean()
            df[f"Volume_MA_{period}_ratio"] = df["Volume"] / df[f"Volume_MA_{period}"]
        
        # Volume statistics
        df["Volume_std_20"] = df["Volume"].rolling(20).std()
        df["Volume_skew_20"] = df["Volume"].rolling(20).skew()
        df["Volume_kurt_20"] = df["Volume"].rolling(20).kurt()
        
        # On-Balance Volume
        obv = np.where(df["Close"] > df["Close"].shift(1), df["Volume"], 
                      np.where(df["Close"] < df["Close"].shift(1), -df["Volume"], 0))
        df["OBV"] = obv.cumsum()
        df["OBV_MA_10"] = df["OBV"].rolling(10).mean()
        df["OBV_ratio"] = df["OBV"] / df["OBV_MA_10"]
        
        # Volume-Weighted Average Price
        df["VWAP"] = (df["Volume"] * (df["High"] + df["Low"] + df["Close"]) / 3).cumsum() / df["Volume"].cumsum()
        df["VWAP_ratio"] = df["Close"] / df["VWAP"]
        
        # Money Flow Index
        typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
        money_flow = typical_price * df["Volume"]
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)
        
        for period in [14, 21]:
            mf_ratio = positive_flow.rolling(period).sum() / negative_flow.rolling(period).sum()
            df[f"MFI_{period}"] = 100 - (100 / (1 + mf_ratio))
        
        # Chaikin Money Flow
        for period in [20, 50]:
            mfm = ((df["Close"] - df["Low"]) - (df["High"] - df["Close"])) / (df["High"] - df["Low"])
            mf_volume = mfm * df["Volume"]
            df[f"CMF_{period}"] = mf_volume.rolling(period).sum() / df["Volume"].rolling(period).sum()
        
        self.df = df
        return self
    
    def create_statistical_features(self) -> 'StockDataPreprocessor':
        """Create statistical features"""
        if not CREATE_STATISTICAL_FEATURES:
            return self
        
        df = self.df.copy()
        windows = [5, 10, 20, 30, 50, 100]
        
        for w in windows:
            # Rolling statistics
            df[f"Close_mean_{w}"] = df["Close"].rolling(w).mean()
            df[f"Close_std_{w}"] = df["Close"].rolling(w).std()
            df[f"Close_skew_{w}"] = df["Close"].rolling(w).skew()
            df[f"Close_kurt_{w}"] = df["Close"].rolling(w).kurt()
            
            # Quantiles
            df[f"Close_q25_{w}"] = df["Close"].rolling(w).quantile(0.25)
            df[f"Close_q50_{w}"] = df["Close"].rolling(w).quantile(0.50)
            df[f"Close_q75_{w}"] = df["Close"].rolling(w).quantile(0.75)
            df[f"Close_iqr_{w}"] = df[f"Close_q75_{w}"] - df[f"Close_q25_{w}"]
            
            # Z-scores
            df[f"Close_zscore_{w}"] = (df["Close"] - df[f"Close_mean_{w}"]) / df[f"Close_std_{w}"]
            
            # Percentile rank
            df[f"Close_percentile_{w}"] = df["Close"].rolling(w).apply(
                lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min()) * 100 if x.max() != x.min() else 50
            )
            
            # Momentum indicators
            df[f"Momentum_{w}"] = df["Close"] / df["Close"].shift(w) - 1
            
            # Rate of Change
            df[f"ROC_{w}"] = (df["Close"] / df["Close"].shift(w) - 1) * 100
            
            # Volatility
            df[f"Volatility_{w}"] = df["Close"].pct_change().rolling(w).std() * np.sqrt(252)
        
        self.df = df
        return self
    
    def create_pattern_features(self) -> 'StockDataPreprocessor':
        """Create candlestick pattern features"""
        if not CREATE_PATTERN_FEATURES:
            return self
        
        df = self.df.copy()
        
        # Calculate body and shadows
        df["body"] = abs(df["Close"] - df["Open"])
        df["upper_shadow"] = df["High"] - df[["Open", "Close"]].max(axis=1)
        df["lower_shadow"] = df[["Open", "Close"]].min(axis=1) - df["Low"]
        df["body_pct"] = df["body"] / (df["High"] - df["Low"]) * 100
        
        # Doji
        df["is_doji"] = (df["body"] <= (df["High"] - df["Low"]) * 0.1).astype(int)
        
        # Hammer and Shooting Star
        df["is_hammer"] = ((df["lower_shadow"] > 2 * df["body"]) & 
                          (df["upper_shadow"] < df["body"]) & 
                          (df["body"] > 0)).astype(int)
        df["is_shooting_star"] = ((df["upper_shadow"] > 2 * df["body"]) & 
                                 (df["lower_shadow"] < df["body"]) & 
                                 (df["body"] > 0)).astype(int)
        
        # Engulfing patterns
        df["bullish_engulf"] = ((df["Close"] > df["Open"]) & 
                               (df["Close"].shift(1) < df["Open"].shift(1)) &
                               (df["Close"] > df["Open"].shift(1)) & 
                               (df["Open"] < df["Close"].shift(1))).astype(int)
        df["bearish_engulf"] = ((df["Close"] < df["Open"]) & 
                               (df["Close"].shift(1) > df["Open"].shift(1)) &
                               (df["Close"] < df["Open"].shift(1)) & 
                               (df["Open"] > df["Close"].shift(1))).astype(int)
        
        # Continuation patterns
        df["three_white_soldiers"] = (
            (df["Close"] > df["Open"]) &
            (df["Close"].shift(1) > df["Open"].shift(1)) &
            (df["Close"].shift(2) > df["Open"].shift(2)) &
            (df["Close"] > df["Close"].shift(1)) &
            (df["Close"].shift(1) > df["Close"].shift(2))
        ).astype(int)
        
        df["three_black_crows"] = (
            (df["Close"] < df["Open"]) &
            (df["Close"].shift(1) < df["Open"].shift(1)) &
            (df["Close"].shift(2) < df["Open"].shift(2)) &
            (df["Close"] < df["Close"].shift(1)) &
            (df["Close"].shift(1) < df["Close"].shift(2))
        ).astype(int)
        
        self.df = df
        return self
    
    def create_interaction_features(self) -> 'StockDataPreprocessor':
        """Create feature interaction terms"""
        if not CREATE_INTERACTION_FEATURES:
            return self
        
        df = self.df.copy()
        
        # Price-Volume interactions
        df["price_volume"] = df["Close"] * df["Volume"] / 1e6
        df["price_volume_ma_20"] = df["price_volume"].rolling(20).mean()
        df["price_volume_ratio"] = df["price_volume"] / df["price_volume_ma_20"]
        
        # Volatility-Volume interactions
        if "Volatility_20" in df.columns:
            df["vol_volume"] = df["Volatility_20"] * df["Volume"] / df["Volume"].rolling(20).mean()
        
        # RSI-Volume interactions
        for period in [14, 21]:
            if f"RSI_{period}" in df.columns:
                df[f"rsi_volume_{period}"] = df[f"RSI_{period}"] * df["Volume"] / df["Volume"].rolling(20).mean()
        
        # Price-RSI interactions
        for period in [14, 21]:
            if f"RSI_{period}" in df.columns:
                df[f"price_rsi_{period}"] = df["Close"] * df[f"RSI_{period}"] / 100
        
        # MACD-Volume interactions
        for macd_col in df.columns:
            if macd_col.startswith("MACD_"):
                df[f"{macd_col}_volume"] = df[macd_col] * df["Volume"] / df["Volume"].rolling(20).mean()
        
        # Bollinger Band positions with RSI
        for period in [20, 50]:
            if f"BB_position_{period}" in df.columns and f"RSI_14" in df.columns:
                df[f"bb_rsi_{period}"] = df[f"BB_position_{period}"] * df["RSI_14"] / 100
        
        self.df = df
        return self
    
    def create_lag_features(self, n_lags: int = 10) -> 'StockDataPreprocessor':
        """Create lagged features for time series"""
        df = self.df.copy()
        
        for i in range(1, n_lags + 1):
            df[f"close_lag_{i}"] = df["Close"].shift(i)
            df[f"volume_lag_{i}"] = df["Volume"].shift(i)
            df[f"high_lag_{i}"] = df["High"].shift(i)
            df[f"low_lag_{i}"] = df["Low"].shift(i)
            df[f"open_lag_{i}"] = df["Open"].shift(i)
        
        self.df = df
        return self
    
    def create_targets(self, prediction_days: Optional[List[int]] = None) -> 'StockDataPreprocessor':
        """Create multiple target variables for different prediction horizons"""
        if prediction_days is None:
            prediction_days = PREDICTION_DAYS
        
        df = self.df.copy()
        
        for days in prediction_days:
            # Future price
            df[f"target_{days}d"] = df["Close"].shift(-days)
            
            # Future return
            df[f"target_return_{days}d"] = (df["Close"].shift(-days) / df["Close"] - 1) * 100
            
            # Binary classification target (up/down)
            df[f"target_class_{days}d"] = (df[f"target_return_{days}d"] > 0).astype(int)
        
        self.target_columns = [f"target_{d}d" for d in prediction_days]
        self.df = df
        return self
    
    def handle_missing_values(self) -> 'StockDataPreprocessor':
        """Handle missing values intelligently - FIXED"""
        df = self.df.copy()
        
        missing_before = df.isnull().sum().sum()
        
        # Forward fill with limit
        df = df.fillna(method='ffill', limit=5)
        
        # Backward fill for remaining
        df = df.fillna(method='bfill', limit=5)
        
        # Interpolate any remaining - using linear method (time method not available)
        df = df.interpolate(method='linear', limit=10)
        
        # Drop any rows with remaining NaN
        df = df.dropna()
        
        missing_after = df.isnull().sum().sum()
        print(f"    Missing values: {missing_before} -> {missing_after} (removed)")
        
        self.df = df
        return self
    
    def handle_infinite_values(self) -> 'StockDataPreprocessor':
        """Handle infinite values in the dataset"""
        df = self.df.copy()
        
        # Replace inf with NaN
        df = df.replace([np.inf, -np.inf], np.nan)
        
        # Count infinities before
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        inf_before = np.isinf(df[numeric_cols]).sum().sum()
        
        # Fill NaN with forward fill
        df = df.fillna(method='ffill', limit=5)
        df = df.fillna(method='bfill', limit=5)
        
        # Drop any remaining NaN
        df = df.dropna()
        
        inf_after = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
        print(f"    Inf values: {inf_before} -> {inf_after} (removed)")
        
        self.df = df
        return self
    
    def detect_and_handle_outliers(self) -> 'StockDataPreprocessor':
        """Detect and handle outliers"""
        df = self.df.copy()
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if 'target' not in col]
        
        for col in numeric_cols:
            if col in df.columns:
                mean = df[col].mean()
                std = df[col].std()
                if std > 0:
                    df[col] = df[col].clip(mean - 3*std, mean + 3*std)
        
        self.df = df
        return self
    
    def scale_features(self, method: str = 'minmax') -> 'StockDataPreprocessor':
        """Scale features using different methods"""
        df = self.df.copy()
        
        feature_cols = [col for col in df.columns if 'target' not in col]
        feature_cols = [col for col in feature_cols if col not in ['Date']]
        
        if method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'standard':
            scaler = StandardScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown scaling method: {method}")
        
        df[feature_cols] = scaler.fit_transform(df[feature_cols])
        self.scaler = scaler
        self.df = df
        return self
    
    def run_pipeline(self, scale: bool = False, scale_method: str = 'minmax') -> pd.DataFrame:
        """Run the complete preprocessing pipeline"""
        print_section("Starting Advanced Preprocessing Pipeline")
        
        print("  🔧 Cleaning numeric columns...")
        self.clean_numeric_columns()
        
        print("  📅 Creating date features...")
        self.create_date_features()
        
        print("  💰 Creating price features...")
        self.create_price_features()
        
        print("  📊 Creating moving averages...")
        self.create_moving_averages()
        
        print("  📈 Creating technical indicators...")
        self.create_technical_indicators()
        
        print("  📊 Creating volume features...")
        self.create_volume_features()
        
        print("  📉 Creating statistical features...")
        self.create_statistical_features()
        
        print("  🎨 Creating pattern features...")
        self.create_pattern_features()
        
        print("  🔗 Creating interaction features...")
        self.create_interaction_features()
        
        print("  ⏰ Creating lag features...")
        self.create_lag_features(n_lags=10)
        
        print("  🎯 Creating target variables...")
        self.create_targets()
        
        print("  🧹 Handling missing values...")
        self.handle_missing_values()
        
        print("  ♾️  Handling infinite values...")
        self.handle_infinite_values()
        
        print("  🚨 Handling outliers...")
        self.detect_and_handle_outliers()
        
        if scale:
            print("  📏 Scaling features...")
            self.scale_features(method=scale_method)
        
        print(f"✅ Preprocessing complete! Final shape: {self.df.shape}")
        print(f"   Features: {len(self.df.columns)} columns")
        
        return self.df


# ============================================================================
# LEGACY SUPPORT FUNCTIONS
# ============================================================================

def preprocess_data(df, save=True, scale=False):
    """Legacy function for backward compatibility"""
    preprocessor = StockDataPreprocessor(df)
    processed_df = preprocessor.run_pipeline(scale=scale)
    
    if save:
        os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
        filepath = os.path.join(DATA_PROCESSED_DIR, f"{STOCK_SYMBOL}_processed.csv")
        processed_df.to_csv(filepath, index=False)
        print(f"\n✅ Processed data saved to: {filepath}")
    
    return processed_df


if __name__ == "__main__":
    print("Testing Advanced Preprocessing...")
    
    # Load raw data
    filepath = os.path.join(DATA_RAW_DIR, f"{STOCK_SYMBOL}_raw.csv")
    
    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        
        # Run preprocessing
        processed_df = preprocess_data(df, save=True, scale=False)
        
        print("\n" + "="*60)
        print("  PROCESSED DATA SAMPLE")
        print("="*60)
        print(processed_df.head())
        print(f"\nTotal columns: {len(processed_df.columns)}")
        print(f"Target columns: {[col for col in processed_df.columns if 'target' in col]}")
        
    else:
        print(f"❌ Raw data not found at: {filepath}")
        print("Please run create_enough_data.py first.")