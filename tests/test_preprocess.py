"""
Unit tests for the preprocessing module - FIXED

The old tests imported module-level functions (create_date_features, create_target, ...)
and column names (MA_5, RSI, target, volatility) that don't exist in src/preprocess.py,
so the whole suite failed at import. Rewritten against the real StockDataPreprocessor API,
plus real leakage tests.
"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess import StockDataPreprocessor, get_feature_columns


def make_ohlcv(n=400, seed=0):
    rng = np.random.default_rng(seed)
    close = 100 * np.exp(np.cumsum(rng.normal(0, 0.015, n)))
    open_ = close * (1 + rng.normal(0, 0.003, n))
    return pd.DataFrame({
        'Date': pd.bdate_range('2020-01-01', periods=n),
        'Open': open_,
        'High': np.maximum(open_, close) * 1.01,
        'Low': np.minimum(open_, close) * 0.99,
        'Close': close,
        'Volume': rng.integers(1_000_000, 5_000_000, n).astype(float),
    })


class TestPreprocessing(unittest.TestCase):

    def setUp(self):
        self.df = make_ohlcv()

    def test_create_date_features(self):
        df = StockDataPreprocessor(self.df, verbose=False).create_date_features().df
        for col in ['year', 'month', 'day', 'day_of_week', 'quarter', 'is_quarter_end']:
            self.assertIn(col, df.columns)
        self.assertTrue(df['month'].between(1, 12).all())
        self.assertTrue(df['day_of_week'].between(0, 4).all())   # business days only
        self.assertTrue(df['quarter'].between(1, 4).all())

    def test_timezone_aware_dates(self):
        df = self.df.copy()
        df['Date'] = df['Date'].dt.tz_localize('America/New_York')
        out = StockDataPreprocessor(df, verbose=False).create_date_features().df
        self.assertIsNone(out['Date'].dt.tz)

    def test_create_price_features(self):
        df = StockDataPreprocessor(self.df, verbose=False).create_price_features().df
        for col in ['open_close', 'high_low', 'price_change_1', 'price_change_pct_1']:
            self.assertIn(col, df.columns)
        np.testing.assert_allclose(df['open_close'], df['Open'] - df['Close'])
        np.testing.assert_allclose(df['high_low'], df['High'] - df['Low'])

    def test_create_moving_averages(self):
        df = StockDataPreprocessor(self.df, verbose=False).create_moving_averages(windows=[5, 10]).df
        for col in ['SMA_5', 'SMA_10', 'EMA_5', 'EMA_10']:
            self.assertIn(col, df.columns)
        self.assertTrue(pd.isna(df['SMA_5'].iloc[3]))
        self.assertAlmostEqual(df['SMA_5'].iloc[4], df['Close'].iloc[:5].mean())

    def test_create_technical_indicators(self):
        df = StockDataPreprocessor(self.df, verbose=False).create_technical_indicators().df
        for col in ['RSI_14', 'MACD_12_26', 'MACD_12_26_signal', 'BB_middle_20', 'BB_upper_20', 'BB_lower_20']:
            self.assertIn(col, df.columns)
        rsi = df['RSI_14'].dropna()
        self.assertTrue(((rsi >= 0) & (rsi <= 100)).all())

    def test_no_future_ichimoku_chikou(self):
        df = StockDataPreprocessor(self.df, verbose=False).create_technical_indicators().df
        self.assertNotIn('Ichimoku_Chikou', df.columns)  # was Close.shift(-26) = future leak

    def test_create_lag_features(self):
        df = StockDataPreprocessor(self.df, verbose=False).create_lag_features(n_lags=3).df
        for i in (1, 2, 3):
            self.assertIn(f'close_lag_{i}', df.columns)
        self.assertEqual(df['close_lag_1'].iloc[1], df['Close'].iloc[0])
        self.assertEqual(df['close_lag_2'].iloc[2], df['Close'].iloc[0])

    def test_create_targets(self):
        df = StockDataPreprocessor(self.df, verbose=False).create_targets(prediction_days=[1, 5]).df
        for col in ['target_1d', 'target_return_1d', 'target_class_1d', 'target_5d']:
            self.assertIn(col, df.columns)
        self.assertEqual(df['target_1d'].iloc[0], df['Close'].iloc[1])
        self.assertTrue(df['target_class_1d'].dropna().isin([0, 1]).all())
        self.assertTrue(pd.isna(df['target_1d'].iloc[-1]))
        self.assertTrue(pd.isna(df['target_class_1d'].iloc[-1]))  # unknown, not "down"

    def test_volume_string_parsing(self):
        df = self.df.head(3).copy()
        df['Volume'] = ['1.5M', '200K', '1,234']
        out = StockDataPreprocessor(df, verbose=False).clean_numeric_columns().df
        np.testing.assert_allclose(out['Volume'], [1.5e6, 2e5, 1234])

    def test_empty_dataframe(self):
        with self.assertRaises(Exception):
            StockDataPreprocessor(pd.DataFrame(), verbose=False).run_pipeline()

    def test_missing_columns(self):
        incomplete = pd.DataFrame({'Date': pd.bdate_range('2020-01-01', periods=10),
                                   'Close': np.linspace(100, 110, 10)})
        with self.assertRaises(KeyError):
            StockDataPreprocessor(incomplete, verbose=False).create_price_features()


class TestDataIntegrity(unittest.TestCase):

    def test_full_pipeline_keeps_latest_rows(self):
        raw = make_ohlcv(400)
        out = StockDataPreprocessor(raw, verbose=False).run_pipeline()
        # Last real trading day must survive so forecasting starts from "today"
        self.assertEqual(out['Date'].iloc[-1], raw['Date'].iloc[-1])
        self.assertTrue(pd.isna(out['target_1d'].iloc[-1]))
        self.assertEqual(int(out[get_feature_columns(out)].isna().sum().sum()), 0)

    def test_targets_not_in_features(self):
        out = StockDataPreprocessor(make_ohlcv(400), verbose=False).run_pipeline()
        self.assertFalse(any(c.startswith('target') for c in get_feature_columns(out)))

    def test_no_lookahead_in_features(self):
        """
        Changing FUTURE prices (x1.5) must not change features of earlier rows.

        Tolerance note: pandas 2.x computes rolling skew/kurtosis with running sums over the
        whole column, so later values perturb earlier results at the ~1e-7 level (pure
        floating-point noise). A real leak would differ by tens of percent here, so
        rtol=1e-5 / atol=1e-6 still catches any genuine look-ahead.
        """
        raw = make_ohlcv(400)
        tampered = raw.copy()
        tampered.loc[350:, ['Open', 'High', 'Low', 'Close']] *= 1.5

        a = StockDataPreprocessor(raw, verbose=False).run_pipeline().set_index('Date')
        b = StockDataPreprocessor(tampered, verbose=False).run_pipeline().set_index('Date')
        cols = get_feature_columns(a.reset_index())
        cutoff = raw['Date'].iloc[349]
        common = a.index.intersection(b.index)
        common = common[common <= cutoff]
        self.assertGreater(len(common), 50)
        pd.testing.assert_frame_equal(a.loc[common, cols], b.loc[common, cols],
                                      check_exact=False, rtol=1e-5, atol=1e-6)


if __name__ == '__main__':
    unittest.main(verbosity=2)