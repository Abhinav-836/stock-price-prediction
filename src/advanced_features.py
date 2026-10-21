# src/advanced_features.py - Extended feature set using the `ta` library - FIXED
"""
Fixes vs previous version:
  * _interaction_features() read df['RSI_14'], df['volatility_20'], df['return_1'],
    df['bb_low'], df['rsi_14'] from the RAW dataframe, where none of them exist
    -> KeyError on every call. Interactions are now built from the features computed above.
  * `ta` is not in requirements.txt, so importing this module crashed the whole app.
    The import is now guarded with a clear message (`pip install ta`).
  * The output keeps the input index and the Date column so it can be joined back.
  * Removed unused imports (scipy.stats, add_all_ta_features, VWAP).
"""
import pandas as pd
import numpy as np

try:
    from ta.momentum import RSIIndicator, StochasticOscillator
    from ta.trend import MACD, IchimokuIndicator, AroonIndicator
    from ta.volatility import BollingerBands, AverageTrueRange
    from ta.volume import OnBalanceVolume
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False

EPS = 1e-6


class AdvancedFeatureEngineer:
    def __init__(self, df: pd.DataFrame):
        if not TA_AVAILABLE:
            raise ImportError("AdvancedFeatureEngineer needs the 'ta' package: pip install ta")
        required = {'Open', 'High', 'Low', 'Close', 'Volume'}
        missing = required - set(df.columns)
        if missing:
            raise KeyError(f"Missing columns: {missing}")
        self.df = df.copy()

    def create_all_features(self) -> pd.DataFrame:
        features = {}
        features.update(self._price_features())
        features.update(self._technical_indicators())
        features.update(self._statistical_features())
        features.update(self._pattern_features())
        features.update(self._volume_features())
        features.update(self._sentiment_features())
        features.update(self._interaction_features(features))

        out = pd.DataFrame(features, index=self.df.index)
        if 'Date' in self.df.columns:
            out.insert(0, 'Date', self.df['Date'])
        return out.replace([np.inf, -np.inf], np.nan)

    def _price_features(self):
        df = self.df
        close = df['Close']
        return {
            'return_1': close.pct_change(),
            'return_5': close.pct_change(5),
            'return_10': close.pct_change(10),
            'return_20': close.pct_change(20),
            'return_50': close.pct_change(50),
            'log_return_1': np.log(close / close.shift(1)),
            'log_return_5': np.log(close / close.shift(5)),
            'high_low_ratio': df['High'] / df['Low'],
            'close_open_ratio': close / df['Open'],
            'acceleration': close.pct_change().diff(),
            'daily_range': df['High'] - df['Low'],
            'daily_range_pct': (df['High'] - df['Low']) / close,
        }

    def _technical_indicators(self):
        df = self.df
        macd = MACD(df['Close'])
        bb = BollingerBands(df['Close'], window=20, window_dev=2)
        stoch = StochasticOscillator(df['High'], df['Low'], df['Close'], window=14)
        ichi = IchimokuIndicator(df['High'], df['Low'])
        aroon = AroonIndicator(df['High'], df['Low'], window=25)
        return {
            'rsi_14': RSIIndicator(df['Close'], window=14).rsi(),
            'rsi_21': RSIIndicator(df['Close'], window=21).rsi(),
            'macd': macd.macd(),
            'macd_signal': macd.macd_signal(),
            'macd_diff': macd.macd_diff(),
            'bb_high': bb.bollinger_hband(),
            'bb_low': bb.bollinger_lband(),
            'bb_mid': bb.bollinger_mavg(),
            'bb_width': bb.bollinger_wband(),
            'bb_percent': bb.bollinger_pband(),
            'atr': AverageTrueRange(df['High'], df['Low'], df['Close'], window=14).average_true_range(),
            'stoch_k': stoch.stoch(),
            'stoch_d': stoch.stoch_signal(),
            'ichimoku_a': ichi.ichimoku_a(),
            'ichimoku_b': ichi.ichimoku_b(),
            'aroon_up': aroon.aroon_up(),
            'aroon_down': aroon.aroon_down(),
            'obv': OnBalanceVolume(df['Close'], df['Volume']).on_balance_volume(),
        }

    def _statistical_features(self):
        close = self.df['Close']
        returns = close.pct_change()
        features = {}
        for w in [5, 10, 20, 30, 50]:
            roll = close.rolling(w)
            mean, std = roll.mean(), roll.std()
            features[f'close_mean_{w}'] = mean
            features[f'close_std_{w}'] = std
            features[f'close_skew_{w}'] = roll.skew()
            features[f'close_kurt_{w}'] = roll.kurt()
            features[f'close_quantile_25_{w}'] = roll.quantile(0.25)
            features[f'close_quantile_75_{w}'] = roll.quantile(0.75)
            features[f'close_zscore_{w}'] = (close - mean) / (std + EPS)
            features[f'momentum_{w}'] = close / close.shift(w) - 1
            features[f'volatility_{w}'] = returns.rolling(w).std() * np.sqrt(252)
        return features

    def _pattern_features(self):
        df = self.df
        upper_shadow = df['High'] - df[['Open', 'Close']].max(axis=1)
        lower_shadow = df[['Open', 'Close']].min(axis=1) - df['Low']
        body = (df['Close'] - df['Open']).abs()
        prev_open, prev_close = df['Open'].shift(1), df['Close'].shift(1)
        return {
            'is_doji': (body <= (df['High'] - df['Low']) * 0.1).astype(int),
            'hammer': ((lower_shadow > 2 * body) & (upper_shadow < body)).astype(int),
            'shooting_star': ((upper_shadow > 2 * body) & (lower_shadow < body)).astype(int),
            'bullish_engulf': ((df['Close'] > df['Open']) & (prev_close < prev_open) &
                               (df['Close'] > prev_open) & (df['Open'] < prev_close)).astype(int),
            'bearish_engulf': ((df['Close'] < df['Open']) & (prev_close > prev_open) &
                               (df['Close'] < prev_open) & (df['Open'] > prev_close)).astype(int),
        }

    def _volume_features(self):
        vol = self.df['Volume']
        vol_ma_20 = vol.rolling(20).mean()
        return {
            'volume_ma_5': vol.rolling(5).mean(),
            'volume_ma_20': vol_ma_20,
            'volume_ratio': vol / (vol_ma_20 + EPS),
            'volume_std_20': vol.rolling(20).std(),
            'volume_skew_20': vol.rolling(20).skew(),
            'volume_to_price': vol / self.df['Close'],
            'volume_change_1': vol.pct_change(),
            'volume_change_5': vol.pct_change(5),
        }

    def _sentiment_features(self):
        df = self.df
        up_volume = df['Volume'].where(df['Close'] > df['Open'], 0)
        down_volume = df['Volume'].where(df['Close'] < df['Open'], 0)
        money_flow = df['Volume'] * (df['Close'] - df['Open']) / (df['High'] - df['Low'] + EPS)
        return {
            'up_volume_ratio': up_volume / (up_volume + down_volume + EPS),
            'down_volume_ratio': down_volume / (up_volume + down_volume + EPS),
            'money_flow': money_flow,
            'money_flow_ma_10': money_flow.rolling(10).mean(),
        }

    def _interaction_features(self, f: dict):
        """Built from already-computed features (the old version read non-existent raw columns)"""
        df = self.df
        vol_ratio = f['volume_ratio']
        bb_pos = (df['Close'] - f['bb_low']) / (f['bb_high'] - f['bb_low'] + EPS)
        return {
            'rsi_volume_14': f['rsi_14'] * vol_ratio,
            'price_volume': df['Close'] * df['Volume'] / 1e6,
            'vol_volume': f['volatility_20'] * vol_ratio,
            'return_rsi_14': f['return_1'] * f['rsi_14'],
            'bb_rsi': bb_pos * f['rsi_14'],
        }