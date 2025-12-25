"""
Unit tests for preprocessing module
"""
import unittest
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess import (create_date_features, create_price_features, 
                            create_moving_averages, create_technical_indicators,
                            create_lag_features, create_target)


class TestPreprocessing(unittest.TestCase):
    """Test cases for preprocessing functions"""
    
    def setUp(self):
        """Set up test data"""
        dates = pd.date_range('2020-01-01', periods=100, freq='D')
        self.df = pd.DataFrame({
            'Date': dates,
            'Open': np.random.uniform(100, 200, 100),
            'High': np.random.uniform(100, 200, 100),
            'Low': np.random.uniform(100, 200, 100),
            'Close': np.random.uniform(100, 200, 100),
            'Volume': np.random.uniform(1000000, 5000000, 100)
        })
    
    def test_create_date_features(self):
        """Test date feature creation"""
        df = create_date_features(self.df)
        
        # Check if new columns are created
        self.assertIn('year', df.columns)
        self.assertIn('month', df.columns)
        self.assertIn('day', df.columns)
        self.assertIn('day_of_week', df.columns)
        self.assertIn('quarter', df.columns)
        self.assertIn('is_quarter_end', df.columns)
        
        # Check data types
        self.assertTrue(df['year'].dtype in [np.int32, np.int64])
        self.assertTrue(df['month'].dtype in [np.int32, np.int64])
        
        # Check value ranges
        self.assertTrue(all(df['month'] >= 1) and all(df['month'] <= 12))
        self.assertTrue(all(df['day_of_week'] >= 0) and all(df['day_of_week'] <= 6))
        self.assertTrue(all(df['quarter'] >= 1) and all(df['quarter'] <= 4))
    
    def test_create_price_features(self):
        """Test price feature creation"""
        df = create_price_features(self.df)
        
        # Check if new columns are created
        self.assertIn('open_close', df.columns)
        self.assertIn('high_low', df.columns)
        self.assertIn('price_change', df.columns)
        self.assertIn('price_change_pct', df.columns)
        
        # Check calculations
        self.assertTrue(all(df['open_close'] == df['Open'] - df['Close']))
        self.assertTrue(all(df['high_low'] == df['High'] - df['Low']))
    
    def test_create_moving_averages(self):
        """Test moving average creation"""
        df = create_moving_averages(self.df, windows=[5, 10])
        
        # Check if MA columns are created
        self.assertIn('MA_5', df.columns)
        self.assertIn('MA_10', df.columns)
        self.assertIn('EMA_5', df.columns)
        self.assertIn('EMA_10', df.columns)
        
        # Check that first values are NaN
        self.assertTrue(pd.isna(df['MA_5'].iloc[0]))
        self.assertTrue(pd.isna(df['MA_10'].iloc[0]))
    
    def test_create_technical_indicators(self):
        """Test technical indicator creation"""
        df = create_technical_indicators(self.df)
        
        # Check if indicator columns are created
        self.assertIn('RSI', df.columns)
        self.assertIn('MACD', df.columns)
        self.assertIn('MACD_signal', df.columns)
        self.assertIn('BB_middle', df.columns)
        self.assertIn('BB_upper', df.columns)
        self.assertIn('BB_lower', df.columns)
        self.assertIn('volatility', df.columns)
        
        # Check RSI bounds (should be between 0 and 100)
        valid_rsi = df['RSI'].dropna()
        if len(valid_rsi) > 0:
            self.assertTrue(all(valid_rsi >= 0) and all(valid_rsi <= 100))
    
    def test_create_lag_features(self):
        """Test lag feature creation"""
        df = create_lag_features(self.df, n_lags=3)
        
        # Check if lag columns are created
        self.assertIn('close_lag_1', df.columns)
        self.assertIn('close_lag_2', df.columns)
        self.assertIn('close_lag_3', df.columns)
        
        # Check that lag values are shifted correctly
        self.assertEqual(df['close_lag_1'].iloc[1], df['Close'].iloc[0])
        self.assertEqual(df['close_lag_2'].iloc[2], df['Close'].iloc[0])
    
    def test_create_target(self):
        """Test target variable creation"""
        df = create_target(self.df, prediction_days=1)
        
        # Check if target columns are created
        self.assertIn('target', df.columns)
        self.assertIn('target_class', df.columns)
        
        # Check target is shifted correctly
        self.assertEqual(df['target'].iloc[0], df['Close'].iloc[1])
        
        # Check target_class is binary
        valid_target_class = df['target_class'].dropna()
        self.assertTrue(all(valid_target_class.isin([0, 1])))
    
    def test_empty_dataframe(self):
        """Test handling of empty DataFrame"""
        empty_df = pd.DataFrame()
        
        with self.assertRaises(Exception):
            create_date_features(empty_df)
    
    def test_missing_columns(self):
        """Test handling of missing required columns"""
        incomplete_df = pd.DataFrame({
            'Date': pd.date_range('2020-01-01', periods=10),
            'Close': np.random.uniform(100, 200, 10)
        })
        
        # Should work for functions that only need Close
        df = create_lag_features(incomplete_df, n_lags=2)
        self.assertIn('close_lag_1', df.columns)
        
        # Should fail for functions that need OHLC
        with self.assertRaises(KeyError):
            create_price_features(incomplete_df)


class TestDataIntegrity(unittest.TestCase):
    """Test data integrity checks"""
    
    def test_no_data_leakage(self):
        """Ensure target doesn't leak into features"""
        dates = pd.date_range('2020-01-01', periods=50, freq='D')
        df = pd.DataFrame({
            'Date': dates,
            'Close': np.random.uniform(100, 200, 50)
        })
        
        df = create_target(df, prediction_days=1)
        
        # Last row should have NaN target (no future data)
        self.assertTrue(pd.isna(df['target'].iloc[-1]))


def run_tests():
    """Run all tests"""
    unittest.main(argv=[''], verbosity=2, exit=False)


if __name__ == '__main__':
    run_tests()