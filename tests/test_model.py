"""
Unit tests for model training, evaluation and forecasting - FIXED

The old file imported train_linear_regression / train_random_forest / train_xgboost /
calculate_directional_accuracy, none of which existed, so the module failed at import.
Those helpers now exist; tests also cover the forecasting bugs that were fixed.
"""
import os
import sys
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.train_model import (train_linear_regression, train_random_forest, train_xgboost,
                             prepare_features_target, split_data, ModelTrainer)
from src.evaluate import calculate_metrics, calculate_directional_accuracy
from src.preprocess import StockDataPreprocessor
from src.predict import PredictionEngine
from tests.test_preprocess import make_ohlcv


class TestModelTraining(unittest.TestCase):

    def setUp(self):
        rng = np.random.default_rng(42)
        cols = [f'feature_{i}' for i in range(5)]
        self.X_train = pd.DataFrame(rng.normal(size=(100, 5)), columns=cols)
        self.y_train = pd.Series(150 + self.X_train['feature_0'] * 10 + rng.normal(0, 1, 100))
        self.X_test = pd.DataFrame(rng.normal(size=(20, 5)), columns=cols)

    def test_linear_regression_training(self):
        model = train_linear_regression(self.X_train, self.y_train)
        pred = model.predict(self.X_test)
        self.assertEqual(len(pred), len(self.X_test))
        self.assertTrue(np.isfinite(pred).all())

    def test_random_forest_training(self):
        model = train_random_forest(self.X_train, self.y_train, n_estimators=10)
        self.assertEqual(len(model.predict(self.X_test)), 20)
        self.assertEqual(len(model.feature_importances_), self.X_train.shape[1])

    def test_xgboost_training(self):
        model = train_xgboost(self.X_train, self.y_train,
                              params={'n_estimators': 10, 'max_depth': 3, 'learning_rate': 0.1, 'n_jobs': 1})
        self.assertEqual(len(model.predict(self.X_test)), 20)
        self.assertTrue(hasattr(model, 'feature_importances_'))

    def test_model_prediction_range(self):
        pred = train_linear_regression(self.X_train, self.y_train).predict(self.X_test)
        lo, hi = self.y_train.min(), self.y_train.max()
        margin = (hi - lo) * 0.5
        self.assertGreater(np.mean((pred >= lo - margin) & (pred <= hi + margin)), 0.5)

    def test_svr_pipeline_scales_test_data(self):
        """Old bug: SVR trained on scaled X but evaluated on unscaled X."""
        trainer = ModelTrainer(self.X_train * 1000, self.y_train, self.X_test * 1000, self.y_train[:20],
                               verbose=False, run_cv=False, target_col='target_1d')
        trainer.train_svm()
        pred = trainer.models['svr'].predict(self.X_test * 1000)
        self.assertLess(np.abs(pred - self.y_train.mean()).max(), 100)


class TestModelEvaluation(unittest.TestCase):

    def setUp(self):
        self.y_true = np.array([100, 110, 120, 115, 125, 130, 140, 135], dtype=float)
        self.y_pred = np.array([105, 108, 122, 118, 123, 128, 142, 138], dtype=float)

    def test_calculate_metrics(self):
        m = calculate_metrics(self.y_true, self.y_pred)
        for key in ['MAE', 'MSE', 'RMSE', 'R2', 'MAPE']:
            self.assertIn(key, m)
        for value in m.values():
            self.assertIsInstance(value, (int, float, np.number))
        self.assertGreater(m['MAE'], 0)
        self.assertGreaterEqual(m['RMSE'], m['MAE'])
        self.assertLessEqual(m['R2'], 1.0)

    def test_pandas_series_input(self):
        """Old bug: Series inputs were index-aligned in return calculations."""
        idx = pd.RangeIndex(500, 508)
        m1 = calculate_metrics(pd.Series(self.y_true, index=idx), pd.Series(self.y_pred, index=idx))
        m2 = calculate_metrics(self.y_true, self.y_pred)
        self.assertAlmostEqual(m1['Total_Return'], m2['Total_Return'])

    def test_perfect_predictions(self):
        y = np.array([100, 110, 120, 130], dtype=float)
        m = calculate_metrics(y, y)
        for key in ['MAE', 'MSE', 'RMSE', 'MAPE']:
            self.assertAlmostEqual(m[key], 0, places=10)
        self.assertAlmostEqual(m['R2'], 1.0, places=10)

    def test_directional_accuracy(self):
        acc = calculate_directional_accuracy([100, 110, 105, 120, 115], [105, 115, 100, 125, 110])
        self.assertTrue(0 <= acc <= 100)
        self.assertAlmostEqual(calculate_directional_accuracy([1, -1, 2], [0.5, -0.2, -1], is_return_target=True),
                               200 / 3)

    def test_metrics_with_single_value(self):
        m = calculate_metrics(np.array([100, 110, 120, 130.]), np.array([110, 110, 110, 110.]))
        self.assertLess(m['R2'], 0)

    def test_return_target_metrics_are_finite(self):
        rng = np.random.default_rng(1)
        y = rng.normal(0, 1.5, 200)
        m = calculate_metrics(y, y * 0.3 + rng.normal(0, 1, 200), is_return_target=True)
        for key in ['R2', 'RMSE', 'Directional_Accuracy', 'Sharpe_Ratio', 'Theil_U']:
            self.assertTrue(np.isfinite(m[key]), key)


class TestFeaturePreparation(unittest.TestCase):

    def test_prepare_features_target(self):
        rng = np.random.default_rng(0)
        df = pd.DataFrame({
            'Date': pd.bdate_range('2020-01-01', periods=50),
            'feature_1': rng.normal(size=50),
            'feature_2': rng.normal(size=50),
            'target_return_1d': rng.normal(size=50),
            'target_class_1d': rng.integers(0, 2, 50),
        })
        df.loc[49, 'target_return_1d'] = np.nan          # latest day: future unknown
        X, y, feature_cols = prepare_features_target(df, verbose=False)
        self.assertEqual(len(X), 49)                      # NaN-target row excluded from training
        self.assertEqual(feature_cols, ['feature_1', 'feature_2'])

    def test_split_is_chronological(self):
        X = pd.DataFrame({'a': range(100)})
        X_train, X_test, _, _ = split_data(X, pd.Series(range(100)))
        self.assertLess(X_train['a'].max(), X_test['a'].min())


class TestForecasting(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.raw = make_ohlcv(450, seed=3)
        cls.processed = StockDataPreprocessor(cls.raw, verbose=False).run_pipeline()
        X, y, cls.feature_cols = prepare_features_target(cls.processed, verbose=False)
        cls.model = train_random_forest(X, y, n_estimators=20)

    def test_forecast_not_flat_and_dates_unique(self):
        engine = PredictionEngine('random_forest', self.model, target_col='target_return_1d')
        pred = engine.predict_future(self.processed, self.feature_cols, days=10, raw_df=self.raw)
        self.assertEqual(len(pred), 10)
        self.assertTrue(pred['Date'].is_unique)
        self.assertTrue((pred['Date'].dt.dayofweek < 5).all())
        self.assertGreater(pred['Date'].iloc[0], self.raw['Date'].iloc[-1])
        self.assertGreater(pred['Predicted_Close'].round(6).nunique(), 1)

    def test_band_widens_with_horizon(self):
        engine = PredictionEngine('random_forest', self.model, target_col='target_return_1d')
        pred = engine.predict_future(self.processed, self.feature_cols, days=5, raw_df=self.raw)
        width = (pred['Upper_Bound'] - pred['Lower_Bound']) / pred['Predicted_Close']
        self.assertTrue(width.is_monotonic_increasing)


if __name__ == '__main__':
    unittest.main(verbosity=2)