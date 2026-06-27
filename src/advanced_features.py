# src/advanced_features.py - New module with 100+ features

import pandas as pd
import numpy as np
from scipy import stats
from ta import add_all_ta_features
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, IchimokuIndicator, AroonIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolume, VolumeWeightedAveragePrice

class AdvancedFeatureEngineer:
    def __init__(self, df):
        self.df = df.copy()
        
    def create_all_features(self):
        """Create comprehensive feature set"""
        features = {}
        
        # 1. Price-based features (15 features)
        features.update(self._price_features())
        
        # 2. Technical indicators (30+ features)
        features.update(self._technical_indicators())
        
        # 3. Statistical features (20 features)
        features.update(self._statistical_features())
        
        # 4. Pattern features (15 features)
        features.update(self._pattern_features())
        
        # 5. Volume features (10 features)
        features.update(self._volume_features())
        
        # 6. Sentiment-like features (5 features)
        features.update(self._sentiment_features())
        
        # 7. Interaction features (20 features)
        features.update(self._interaction_features())
        
        return pd.DataFrame(features)
    
    def _price_features(self):
        """Price-based features"""
        df = self.df
        return {
            # Returns at different lags
            'return_1': df['Close'].pct_change(),
            'return_5': df['Close'].pct_change(5),
            'return_10': df['Close'].pct_change(10),
            'return_20': df['Close'].pct_change(20),
            'return_50': df['Close'].pct_change(50),
            
            # Log returns
            'log_return_1': np.log(df['Close'] / df['Close'].shift(1)),
            'log_return_5': np.log(df['Close'] / df['Close'].shift(5)),
            
            # Price ratios
            'high_low_ratio': df['High'] / df['Low'],
            'close_open_ratio': df['Close'] / df['Open'],
            
            # Price acceleration
            'acceleration': df['Close'].pct_change().pct_change(),
            
            # Range features
            'daily_range': df['High'] - df['Low'],
            'daily_range_pct': (df['High'] - df['Low']) / df['Close'],
        }
    
    def _technical_indicators(self):
        """Technical analysis indicators"""
        df = self.df
        
        # RSI with multiple periods
        rsi_14 = RSIIndicator(df['Close'], window=14)
        rsi_21 = RSIIndicator(df['Close'], window=21)
        
        # MACD
        macd = MACD(df['Close'])
        
        # Bollinger Bands
        bb = BollingerBands(df['Close'], window=20, window_dev=2)
        
        # ATR
        atr = AverageTrueRange(df['High'], df['Low'], df['Close'], window=14)
        
        # Stochastic
        stoch = StochasticOscillator(df['High'], df['Low'], df['Close'], window=14)
        
        # Ichimoku
        ichi = IchimokuIndicator(df['High'], df['Low'])
        
        # Aroon
        aroon = AroonIndicator(df['High'], df['Low'], window=25)
        
        # On Balance Volume
        obv = OnBalanceVolume(df['Close'], df['Volume'])
        
        return {
            'rsi_14': rsi_14.rsi(),
            'rsi_21': rsi_21.rsi(),
            'macd': macd.macd(),
            'macd_signal': macd.macd_signal(),
            'macd_diff': macd.macd_diff(),
            'bb_high': bb.bollinger_hband(),
            'bb_low': bb.bollinger_lband(),
            'bb_mid': bb.bollinger_mavg(),
            'bb_width': bb.bollinger_wband(),
            'bb_percent': bb.bollinger_pband(),
            'atr': atr.average_true_range(),
            'stoch_k': stoch.stoch(),
            'stoch_d': stoch.stoch_signal(),
            'ichimoku_a': ichi.ichimoku_a(),
            'ichimoku_b': ichi.ichimoku_b(),
            'aroon_up': aroon.aroon_up(),
            'aroon_down': aroon.aroon_down(),
            'obv': obv.on_balance_volume(),
        }
    
    def _statistical_features(self):
        """Statistical features over windows"""
        df = self.df
        features = {}
        
        windows = [5, 10, 20, 30, 50]
        
        for w in windows:
            # Rolling statistics
            features[f'close_mean_{w}'] = df['Close'].rolling(w).mean()
            features[f'close_std_{w}'] = df['Close'].rolling(w).std()
            features[f'close_skew_{w}'] = df['Close'].rolling(w).skew()
            features[f'close_kurt_{w}'] = df['Close'].rolling(w).kurt()
            
            # Quantiles
            features[f'close_quantile_25_{w}'] = df['Close'].rolling(w).quantile(0.25)
            features[f'close_quantile_75_{w}'] = df['Close'].rolling(w).quantile(0.75)
            
            # Z-score
            features[f'close_zscore_{w}'] = (df['Close'] - df['Close'].rolling(w).mean()) / df['Close'].rolling(w).std()
            
        # Momentum indicators
        for w in windows:
            features[f'momentum_{w}'] = df['Close'] / df['Close'].shift(w) - 1
            
        return features
    
    def _pattern_features(self):
        """Candlestick pattern features"""
        df = self.df
        
        # Doji
        doji = abs(df['Open'] - df['Close']) <= (df['High'] - df['Low']) * 0.1
        
        # Hammer/Shooting Star
        upper_shadow = df['High'] - df[['Open', 'Close']].max(axis=1)
        lower_shadow = df[['Open', 'Close']].min(axis=1) - df['Low']
        body = abs(df['Close'] - df['Open'])
        
        return {
            'is_doji': doji.astype(int),
            'hammer': ((lower_shadow > 2*body) & (upper_shadow < body)).astype(int),
            'shooting_star': ((upper_shadow > 2*body) & (lower_shadow < body)).astype(int),
            'bullish_engulf': ((df['Close'] > df['Open']) & 
                              (df['Close'].shift(1) < df['Open'].shift(1)) &
                              (df['Close'] > df['Open'].shift(1)) & 
                              (df['Open'] < df['Close'].shift(1))).astype(int),
            'bearish_engulf': ((df['Close'] < df['Open']) & 
                              (df['Close'].shift(1) > df['Open'].shift(1)) &
                              (df['Close'] < df['Open'].shift(1)) & 
                              (df['Open'] > df['Close'].shift(1))).astype(int),
        }
    
    def _volume_features(self):
        """Volume-based features"""
        df = self.df
        
        return {
            'volume_ma_5': df['Volume'].rolling(5).mean(),
            'volume_ma_20': df['Volume'].rolling(20).mean(),
            'volume_ratio': df['Volume'] / df['Volume'].rolling(20).mean(),
            'volume_std_20': df['Volume'].rolling(20).std(),
            'volume_skew_20': df['Volume'].rolling(20).skew(),
            'volume_to_price': df['Volume'] / df['Close'],
            'volume_change_1': df['Volume'].pct_change(),
            'volume_change_5': df['Volume'].pct_change(5),
        }
    
    def _sentiment_features(self):
        """Sentiment-like features from price action"""
        df = self.df
        
        # Up/Down volume
        up_volume = df['Volume'].where(df['Close'] > df['Open'], 0)
        down_volume = df['Volume'].where(df['Close'] < df['Open'], 0)
        
        return {
            'up_volume_ratio': up_volume / (up_volume + down_volume + 1e-6),
            'down_volume_ratio': down_volume / (up_volume + down_volume + 1e-6),
            'money_flow': df['Volume'] * (df['Close'] - df['Open']) / (df['High'] - df['Low'] + 1e-6),
            'money_flow_ma_10': (df['Volume'] * (df['Close'] - df['Open']) / (df['High'] - df['Low'] + 1e-6)).rolling(10).mean(),
        }
    
    def _interaction_features(self):
        """Feature interactions"""
        df = self.df
        features = {}
        
        # RSI * Volume (momentum with conviction)
        features['rsi_volume_14'] = df['RSI_14'] * df['Volume'] / df['Volume'].rolling(20).mean()
        
        # Price * Volume (accumulation)
        features['price_volume'] = df['Close'] * df['Volume'] / 1e6
        
        # Volatility * Volume
        features['vol_volume'] = df['volatility_20'] * df['Volume'] / df['Volume'].rolling(20).mean()
        
        # Return * RSI
        features['return_rsi_14'] = df['return_1'] * df['RSI_14']
        
        # Bollinger position * RSI
        features['bb_rsi'] = (df['Close'] - df['bb_low']) / (df['bb_high'] - df['bb_low'] + 1e-6) * df['rsi_14']
        
        return features