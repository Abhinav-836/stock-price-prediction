"""
Data preprocessing and feature engineering module - FIXED VERSION

Key fixes vs previous version:
  * Removed look-ahead leakage (Ichimoku Chikou used Close.shift(-26) = future prices).
  * Targets are NEVER forward/back-filled (old code ffilled the last NaN targets with fake values).
  * Rows whose only NaNs are targets (i.e. the most recent days) are KEPT, so predictions
    start from the real last trading day instead of ~2 weeks earlier.
  * No backward-fill of features (bfill copies future values into the past).
  * Outlier clipping is opt-in and never touches price-level columns.
  * Removed eval() in volume parsing; vectorised the slow rolling.apply percentile.
"""
import pandas as pd
import numpy as np
import os
import sys
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from typing import Dict, List, Optional
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

# Raw columns that come from the data source and are never targets
RAW_COLUMNS = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume',
               'Dividends', 'Stock Splits', 'Symbol']

# Columns that are a pure function of calendar position / dataset start.
# They make tree models memorise time and break on a time-based split.
NON_STATIONARY_FEATURES = ['year', 'days_since_2010', 'days_since_min']


def is_target_column(col: str) -> bool:
    return col.startswith('target')


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Single source of truth for which columns are model features."""
    exclude = {'Date', 'Symbol'} | set(NON_STATIONARY_FEATURES)
    return [
        c for c in df.columns
        if c not in exclude
        and not is_target_column(c)
        and pd.api.types.is_numeric_dtype(df[c])
    ]


class StockDataPreprocessor:
    """
    Advanced data preprocessing and feature engineering for stock data
    """

    def __init__(self, df: pd.DataFrame, config: Optional[Dict] = None, verbose: bool = True):
        self.df = df.copy()
        self.config = config or {}
        self.verbose = verbose
        self.feature_columns = []
        self.target_columns = []
        self.scaler = None

    def _log(self, msg: str):
        if self.verbose:
            print(msg)

    # ------------------------------------------------------------------
    # Cleaning
    # ------------------------------------------------------------------
    def clean_numeric_columns(self) -> 'StockDataPreprocessor':
        """Clean numeric columns (handles strings like '1.2M', '3,400', '5.1%')"""
        df = self.df.copy()
        numeric_cols = ["Price", "Open", "Close", "High", "Low", "Volume", "Change %"]
        multipliers = {'K': 1e3, 'M': 1e6, 'B': 1e9}

        for col in numeric_cols:
            if col not in df.columns:
                continue
            if pd.api.types.is_numeric_dtype(df[col]):
                continue  # already numeric - nothing to do

            s = df[col].astype(str).str.replace(",", "", regex=False).str.replace("%", "", regex=False).str.strip()

            if col == "Volume":
                suffix = s.str[-1].str.upper()
                mult = suffix.map(multipliers).fillna(1.0)
                base = s.where(~suffix.isin(multipliers.keys()), s.str[:-1])
                base = base.replace({'-': '0', '': np.nan})
                df[col] = pd.to_numeric(base, errors='coerce') * mult
            else:
                df[col] = pd.to_numeric(s, errors="coerce")

        self.df = df
        return self

    # ------------------------------------------------------------------
    # Feature creation
    # ------------------------------------------------------------------
    def create_date_features(self) -> 'StockDataPreprocessor':
        """Create date-based features (timezone safe)"""
        if not CREATE_DATE_FEATURES:
            return self

        df = self.df.copy()

        if 'Date' not in df.columns:
            self._log("⚠️  No 'Date' column found. Skipping date features.")
            self.df = df
            return self

        try:
            df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.tz_localize(None)
        except Exception:
            try:
                df['Date'] = pd.to_datetime(df['Date'].astype(str), utc=True).dt.tz_localize(None)
            except Exception:
                self._log("⚠️  Could not convert 'Date' column to datetime")
                self.df = df
                return self

        df = df.sort_values("Date").reset_index(drop=True)

        df["year"] = df["Date"].dt.year
        df["month"] = df["Date"].dt.month
        df["day"] = df["Date"].dt.day
        df["day_of_week"] = df["Date"].dt.dayofweek
        df["quarter"] = df["Date"].dt.quarter
        df["day_of_year"] = df["Date"].dt.dayofyear
        df["week_of_year"] = df["Date"].dt.isocalendar().week.astype(int)

        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
        df["day_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
        df["day_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)

        df["is_month_start"] = df["Date"].dt.is_month_start.astype(int)
        df["is_month_end"] = df["Date"].dt.is_month_end.astype(int)
        df["is_quarter_end"] = df["Date"].dt.is_quarter_end.astype(int)
        df["is_year_end"] = df["Date"].dt.is_year_end.astype(int)

        df["days_since_2010"] = (df["Date"] - pd.Timestamp("2010-01-01")).dt.days
        df["days_since_min"] = (df["Date"] - df["Date"].min()).dt.days

        self.df = df
        return self

    def create_price_features(self) -> 'StockDataPreprocessor':
        """Create price-based features"""
        if not CREATE_PRICE_FEATURES:
            return self

        df = self.df.copy()

        df["open_close"] = df["Open"] - df["Close"]
        df["high_low"] = df["High"] - df["Low"]
        df["high_close"] = df["High"] - df["Close"]
        df["low_close"] = df["Low"] - df["Close"]

        for period in [1, 2, 3, 5, 10, 20, 50]:
            df[f"price_change_{period}"] = df["Close"].diff(period)
            df[f"price_change_pct_{period}"] = df["Close"].pct_change(period) * 100

        for period in [1, 2, 5, 10, 20]:
            df[f"volume_change_{period}"] = df["Volume"].diff(period)
            df[f"volume_change_pct_{period}"] = df["Volume"].pct_change(period) * 100

        df["high_low_ratio"] = df["High"] / df["Low"]
        df["close_open_ratio"] = df["Close"] / df["Open"]
        df["high_open_ratio"] = df["High"] / df["Open"]
        df["low_open_ratio"] = df["Low"] / df["Open"]

        df["price_acceleration"] = df["Close"].pct_change().pct_change()
        df["price_acceleration_5"] = df["Close"].pct_change(5).pct_change(5)

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
            df[f"SMA_{window}"] = df["Close"].rolling(window=window).mean()
            df[f"SMA_{window}_Open"] = df["Open"].rolling(window=window).mean()
            df[f"SMA_{window}_High"] = df["High"].rolling(window=window).mean()
            df[f"SMA_{window}_Low"] = df["Low"].rolling(window=window).mean()
            df[f"EMA_{window}"] = df["Close"].ewm(span=window, adjust=False).mean()
            df[f"Close_SMA_{window}_ratio"] = df["Close"] / df[f"SMA_{window}"]
            df[f"Close_EMA_{window}_ratio"] = df["Close"] / df[f"EMA_{window}"]
            df[f"Close_SMA_{window}_diff"] = df["Close"] - df[f"SMA_{window}"]
            df[f"Close_SMA_{window}_diff_pct"] = (df["Close"] - df[f"SMA_{window}"]) / df[f"SMA_{window}"] * 100

        for w1, w2 in [(5, 20), (20, 50), (50, 200)]:
            if f"SMA_{w1}" in df.columns and f"SMA_{w2}" in df.columns:
                df[f"MA_Crossover_{w1}_{w2}"] = np.where(df[f"SMA_{w1}"] > df[f"SMA_{w2}"], 1, 0)
                df[f"MA_Crossover_{w1}_{w2}_diff"] = df[f"SMA_{w1}"] - df[f"SMA_{w2}"]
                df[f"MA_Crossover_{w1}_{w2}_diff_pct"] = (df[f"SMA_{w1}"] - df[f"SMA_{w2}"]) / df[f"SMA_{w2}"] * 100

        self.df = df
        return self

    def create_technical_indicators(self) -> 'StockDataPreprocessor':
        """Create technical indicators (no future data used)"""
        if not CREATE_TECHNICAL_INDICATORS:
            return self

        df = self.df.copy()

        for period in [7, 14, 21, 28]:
            delta = df["Close"].diff()
            gain = delta.where(delta > 0, 0.0).rolling(period).mean()
            loss = -delta.where(delta < 0, 0.0).rolling(period).mean()
            rs = gain / loss
            df[f"RSI_{period}"] = 100 - (100 / (1 + rs))

        for fast, slow, signal in [(12, 26, 9), (8, 17, 9)]:
            ema_fast = df["Close"].ewm(span=fast, adjust=False).mean()
            ema_slow = df["Close"].ewm(span=slow, adjust=False).mean()
            df[f"MACD_{fast}_{slow}"] = ema_fast - ema_slow
            df[f"MACD_{fast}_{slow}_signal"] = df[f"MACD_{fast}_{slow}"].ewm(span=signal, adjust=False).mean()
            df[f"MACD_{fast}_{slow}_diff"] = df[f"MACD_{fast}_{slow}"] - df[f"MACD_{fast}_{slow}_signal"]

        for period in [20, 50]:
            rolling_mean = df["Close"].rolling(period).mean()
            rolling_std = df["Close"].rolling(period).std()
            df[f"BB_middle_{period}"] = rolling_mean
            df[f"BB_upper_{period}"] = rolling_mean + (rolling_std * 2)
            df[f"BB_lower_{period}"] = rolling_mean - (rolling_std * 2)
            df[f"BB_width_{period}"] = df[f"BB_upper_{period}"] - df[f"BB_lower_{period}"]
            df[f"BB_position_{period}"] = (df["Close"] - df[f"BB_lower_{period}"]) / df[f"BB_width_{period}"]
            df[f"BB_percent_{period}"] = df[f"BB_position_{period}"] * 100

        for period in [14, 21]:
            high_low = df["High"] - df["Low"]
            high_close = (df["High"] - df["Close"].shift()).abs()
            low_close = (df["Low"] - df["Close"].shift()).abs()
            true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df[f"ATR_{period}"] = true_range.rolling(period).mean()
            df[f"ATR_{period}_pct"] = df[f"ATR_{period}"] / df["Close"] * 100

        for period in [14, 21]:
            low_min = df["Low"].rolling(period).min()
            high_max = df["High"].rolling(period).max()
            df[f"Stochastic_K_{period}"] = 100 * ((df["Close"] - low_min) / (high_max - low_min))
            df[f"Stochastic_D_{period}"] = df[f"Stochastic_K_{period}"].rolling(3).mean()

        # Ichimoku Cloud (only past data)
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

        # FIX: the old "Ichimoku_Chikou = Close.shift(-26)" was the price 26 days in the FUTURE
        # (target leakage). Use the backward-looking equivalent instead.
        df["Ichimoku_Chikou_diff"] = df["Close"] - df["Close"].shift(26)

        self.df = df
        return self

    def create_volume_features(self) -> 'StockDataPreprocessor':
        """Create volume-based features"""
        if not CREATE_VOLUME_FEATURES:
            return self

        df = self.df.copy()

        for period in [5, 10, 20, 50]:
            df[f"Volume_MA_{period}"] = df["Volume"].rolling(period).mean()
            df[f"Volume_MA_{period}_ratio"] = df["Volume"] / df[f"Volume_MA_{period}"]

        df["Volume_std_20"] = df["Volume"].rolling(20).std()
        df["Volume_skew_20"] = df["Volume"].rolling(20).skew()
        df["Volume_kurt_20"] = df["Volume"].rolling(20).kurt()

        obv = np.where(df["Close"] > df["Close"].shift(1), df["Volume"],
                       np.where(df["Close"] < df["Close"].shift(1), -df["Volume"], 0))
        df["OBV"] = obv.cumsum()
        df["OBV_MA_10"] = df["OBV"].rolling(10).mean()
        df["OBV_ratio"] = df["OBV"] / df["OBV_MA_10"]

        df["VWAP"] = (df["Volume"] * (df["High"] + df["Low"] + df["Close"]) / 3).cumsum() / df["Volume"].cumsum()
        df["VWAP_ratio"] = df["Close"] / df["VWAP"]

        typical_price = (df["High"] + df["Low"] + df["Close"]) / 3
        money_flow = typical_price * df["Volume"]
        positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
        negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)

        for period in [14, 21]:
            mf_ratio = positive_flow.rolling(period).sum() / negative_flow.rolling(period).sum()
            df[f"MFI_{period}"] = 100 - (100 / (1 + mf_ratio))

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
        returns = df["Close"].pct_change()

        for w in windows:
            roll = df["Close"].rolling(w)
            df[f"Close_mean_{w}"] = roll.mean()
            df[f"Close_std_{w}"] = roll.std()
            df[f"Close_skew_{w}"] = roll.skew()
            df[f"Close_kurt_{w}"] = roll.kurt()

            df[f"Close_q25_{w}"] = roll.quantile(0.25)
            df[f"Close_q50_{w}"] = roll.quantile(0.50)
            df[f"Close_q75_{w}"] = roll.quantile(0.75)
            df[f"Close_iqr_{w}"] = df[f"Close_q75_{w}"] - df[f"Close_q25_{w}"]

            df[f"Close_zscore_{w}"] = (df["Close"] - df[f"Close_mean_{w}"]) / df[f"Close_std_{w}"]

            # Vectorised (identical result to the old slow rolling.apply lambda)
            rmin, rmax = roll.min(), roll.max()
            rng = rmax - rmin
            df[f"Close_percentile_{w}"] = np.where(rng != 0, (df["Close"] - rmin) / rng.replace(0, np.nan) * 100, 50.0)
            df.loc[rmin.isna(), f"Close_percentile_{w}"] = np.nan

            df[f"Momentum_{w}"] = df["Close"] / df["Close"].shift(w) - 1
            df[f"ROC_{w}"] = df[f"Momentum_{w}"] * 100
            df[f"Volatility_{w}"] = returns.rolling(w).std() * np.sqrt(252)

        self.df = df
        return self

    def create_pattern_features(self) -> 'StockDataPreprocessor':
        """Create candlestick pattern features"""
        if not CREATE_PATTERN_FEATURES:
            return self

        df = self.df.copy()

        df["body"] = (df["Close"] - df["Open"]).abs()
        df["upper_shadow"] = df["High"] - df[["Open", "Close"]].max(axis=1)
        df["lower_shadow"] = df[["Open", "Close"]].min(axis=1) - df["Low"]
        df["body_pct"] = df["body"] / (df["High"] - df["Low"]) * 100

        df["is_doji"] = (df["body"] <= (df["High"] - df["Low"]) * 0.1).astype(int)

        df["is_hammer"] = ((df["lower_shadow"] > 2 * df["body"]) &
                           (df["upper_shadow"] < df["body"]) &
                           (df["body"] > 0)).astype(int)
        df["is_shooting_star"] = ((df["upper_shadow"] > 2 * df["body"]) &
                                  (df["lower_shadow"] < df["body"]) &
                                  (df["body"] > 0)).astype(int)

        df["bullish_engulf"] = ((df["Close"] > df["Open"]) &
                                (df["Close"].shift(1) < df["Open"].shift(1)) &
                                (df["Close"] > df["Open"].shift(1)) &
                                (df["Open"] < df["Close"].shift(1))).astype(int)
        df["bearish_engulf"] = ((df["Close"] < df["Open"]) &
                                (df["Close"].shift(1) > df["Open"].shift(1)) &
                                (df["Close"] < df["Open"].shift(1)) &
                                (df["Open"] > df["Close"].shift(1))).astype(int)

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
        vol_ratio_20 = df["Volume"] / df["Volume"].rolling(20).mean()

        df["price_volume"] = df["Close"] * df["Volume"] / 1e6
        df["price_volume_ma_20"] = df["price_volume"].rolling(20).mean()
        df["price_volume_ratio"] = df["price_volume"] / df["price_volume_ma_20"]

        if "Volatility_20" in df.columns:
            df["vol_volume"] = df["Volatility_20"] * vol_ratio_20

        for period in [14, 21]:
            if f"RSI_{period}" in df.columns:
                df[f"rsi_volume_{period}"] = df[f"RSI_{period}"] * vol_ratio_20
                df[f"price_rsi_{period}"] = df["Close"] * df[f"RSI_{period}"] / 100

        # Snapshot the list first - the old code iterated df.columns while adding to it
        macd_cols = [c for c in df.columns if c.startswith("MACD_") and not c.endswith("_volume")]
        for macd_col in macd_cols:
            df[f"{macd_col}_volume"] = df[macd_col] * vol_ratio_20

        for period in [20, 50]:
            if f"BB_position_{period}" in df.columns and "RSI_14" in df.columns:
                df[f"bb_rsi_{period}"] = df[f"BB_position_{period}"] * df["RSI_14"] / 100

        self.df = df
        return self

    def create_lag_features(self, n_lags: int = 10) -> 'StockDataPreprocessor':
        """Create lagged features"""
        df = self.df.copy()
        lags = {}
        for i in range(1, n_lags + 1):
            lags[f"close_lag_{i}"] = df["Close"].shift(i)
            lags[f"volume_lag_{i}"] = df["Volume"].shift(i)
            lags[f"high_lag_{i}"] = df["High"].shift(i)
            lags[f"low_lag_{i}"] = df["Low"].shift(i)
            lags[f"open_lag_{i}"] = df["Open"].shift(i)
            lags[f"return_lag_{i}"] = df["Close"].pct_change().shift(i - 1) * 100
        self.df = pd.concat([df, pd.DataFrame(lags, index=df.index)], axis=1)
        return self

    def create_targets(self, prediction_days: Optional[List[int]] = None) -> 'StockDataPreprocessor':
        """Create target variables. The last `days` rows will legitimately be NaN."""
        if prediction_days is None:
            prediction_days = PREDICTION_DAYS
        # The 1-day horizon is required by the training / forecasting code
        prediction_days = sorted(set(list(prediction_days) + [1]))

        df = self.df.copy()

        for days in prediction_days:
            future_close = df["Close"].shift(-days)
            df[f"target_{days}d"] = future_close
            df[f"target_return_{days}d"] = (future_close / df["Close"] - 1) * 100
            # Keep NaN (not 0) where the future is unknown
            df[f"target_class_{days}d"] = np.where(
                future_close.isna(), np.nan, (future_close > df["Close"]).astype(float)
            )

        self.target_columns = [f"target_{d}d" for d in prediction_days]
        self.df = df
        return self

    # ------------------------------------------------------------------
    # NaN / inf / outliers
    # ------------------------------------------------------------------
    def handle_infinite_values(self) -> 'StockDataPreprocessor':
        """Replace +/-inf with NaN (they are then handled by handle_missing_values)"""
        df = self.df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        inf_count = int(np.isinf(df[numeric_cols]).sum().sum())
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        self._log(f"    Inf values replaced with NaN: {inf_count}")
        self.df = df
        return self

    def handle_missing_values(self) -> 'StockDataPreprocessor':
        """
        FIX: only forward-fill FEATURES (short gaps), never targets, never bfill.
        Rolling warm-up rows are dropped. Recent rows with NaN targets are kept.
        """
        df = self.df.copy()
        feature_cols = get_feature_columns(df)
        missing_before = int(df[feature_cols].isnull().sum().sum())

        df[feature_cols] = df[feature_cols].ffill(limit=5)
        df = df.dropna(subset=feature_cols).reset_index(drop=True)

        missing_after = int(df[feature_cols].isnull().sum().sum())
        self._log(f"    Feature missing values: {missing_before} -> {missing_after} "
                  f"(warm-up rows dropped, {len(df)} rows left)")

        self.df = df
        return self

    def detect_and_handle_outliers(self) -> 'StockDataPreprocessor':
        """
        Optional winsorising of ratio-type features. Price-level columns are NOT clipped
        (clipping Close/SMA at mean +/- 3std distorts any trending stock).
        """
        df = self.df.copy()
        price_like = ('Open', 'High', 'Low', 'Close', 'Adj Close', 'SMA_', 'EMA_', 'BB_upper', 'BB_lower',
                      'BB_middle', 'Ichimoku_', 'VWAP', 'close_lag', 'open_lag', 'high_lag', 'low_lag',
                      'Close_mean', 'Close_q', 'OBV', 'Volume')
        for col in get_feature_columns(df):
            if col.startswith(price_like) or df[col].nunique() <= 2:
                continue
            mean, std = df[col].mean(), df[col].std()
            if std > 0:
                df[col] = df[col].clip(mean - 4 * std, mean + 4 * std)
        self.df = df
        return self

    def scale_features(self, method: str = 'minmax') -> 'StockDataPreprocessor':
        """Scale features (fit on full data - use only for exploration, not for training)"""
        df = self.df.copy()
        feature_cols = get_feature_columns(df)

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

    def run_pipeline(self, scale: bool = False, scale_method: str = 'minmax',
                     clip_outliers: bool = False) -> pd.DataFrame:
        """Run the complete preprocessing pipeline"""
        if self.verbose:
            print_section("Starting Advanced Preprocessing Pipeline")

        steps = [
            ("  🔧 Cleaning numeric columns...", self.clean_numeric_columns),
            ("  📅 Creating date features...", self.create_date_features),
            ("  💰 Creating price features...", self.create_price_features),
            ("  📊 Creating moving averages...", self.create_moving_averages),
            ("  📈 Creating technical indicators...", self.create_technical_indicators),
            ("  📊 Creating volume features...", self.create_volume_features),
            ("  📉 Creating statistical features...", self.create_statistical_features),
            ("  🎨 Creating pattern features...", self.create_pattern_features),
            ("  🔗 Creating interaction features...", self.create_interaction_features),
            ("  ⏰ Creating lag features...", lambda: self.create_lag_features(n_lags=10)),
            ("  🎯 Creating target variables...", self.create_targets),
            ("  ♾️  Handling infinite values...", self.handle_infinite_values),
            ("  🧹 Handling missing values...", self.handle_missing_values),
        ]
        for msg, fn in steps:
            self._log(msg)
            fn()

        if clip_outliers:
            self._log("  🚨 Clipping outliers...")
            self.detect_and_handle_outliers()

        if scale:
            self._log("  📏 Scaling features...")
            self.scale_features(method=scale_method)

        self.feature_columns = get_feature_columns(self.df)
        self.df = self.df.copy()  # de-fragment
        self._log(f"✅ Preprocessing complete! Final shape: {self.df.shape}")
        self._log(f"   Features: {len(self.feature_columns)} columns")

        return self.df


# ============================================================================
# LEGACY SUPPORT FUNCTIONS
# ============================================================================

def preprocess_data(df, save=True, scale=False, verbose=True, symbol=STOCK_SYMBOL):
    """Legacy function for backward compatibility"""
    preprocessor = StockDataPreprocessor(df, verbose=verbose)
    processed_df = preprocessor.run_pipeline(scale=scale)

    if save:
        os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
        filepath = os.path.join(DATA_PROCESSED_DIR, f"{symbol}_processed.csv")
        processed_df.to_csv(filepath, index=False)
        print(f"\n✅ Processed data saved to: {filepath}")

    return processed_df


if __name__ == "__main__":
    print("Testing Advanced Preprocessing...")

    filepath = os.path.join(DATA_RAW_DIR, f"{STOCK_SYMBOL}_raw.csv")

    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        processed_df = preprocess_data(df, save=True, scale=False)

        print("\n" + "=" * 60)
        print("  PROCESSED DATA SAMPLE")
        print("=" * 60)
        print(processed_df.head())
        print(f"\nTotal columns: {len(processed_df.columns)}")
        print(f"Target columns: {[c for c in processed_df.columns if is_target_column(c)]}")
    else:
        print(f"❌ Raw data not found at: {filepath}")
        print("Please run create_enough_data.py first.")