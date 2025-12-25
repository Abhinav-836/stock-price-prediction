"""
Unit tests for model training and evaluation
"""
import unittest
import pandas as pd
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.train_model import (train_linear_regression, train_random_forest, 
                             train_xgboost, prepare_features_target)
from src.evaluate import calculate_metrics, calculate_directional_accuracy


class TestModelTraining(unittest.TestCase):
    """Test cases for model training"""
    
    def setUp(self):
        """Set up test data"""
        np.random.seed(42)
        n_samples = 100
        n_features = 5
        
        self.X_train = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        self.y_train = pd.Series(np.random.uniform(100, 200, n_samples))
        
        self.X_test = pd.DataFrame(
            np.random.randn(20, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        self.y_test = pd.Series(np.random.uniform(100, 200, 20))
    
    def test_linear_regression_training(self):
        """Test Linear Regression training"""
        model = train_linear_regression(self.X_train, self.y_train)
        
        # Check model is trained
        self.assertIsNotNone(model)
        
        # Check predictions can be made
        predictions = model.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.y_test))
        
        # Check predictions are numeric
        self.assertTrue(all(isinstance(p, (int, float, np.number)) for p in predictions))
    
    def test_random_forest_training(self):
        """Test Random Forest training"""
        model = train_random_forest(self.X_train, self.y_train, n_estimators=10)
        
        # Check model is trained
        self.assertIsNotNone(model)
        
        # Check predictions can be made
        predictions = model.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.y_test))
        
        # Check feature importances exist
        self.assertTrue(hasattr(model, 'feature_importances_'))
        self.assertEqual(len(model.feature_importances_), self.X_train.shape[1])
    
    def test_xgboost_training(self):
        """Test XGBoost training"""
        params = {
            'n_estimators': 10,
            'max_depth': 3,
            'learning_rate': 0.1,
            'random_state': 42
        }
        model = train_xgboost(self.X_train, self.y_train, params=params)
        
        # Check model is trained
        self.assertIsNotNone(model)
        
        # Check predictions can be made
        predictions = model.predict(self.X_test)
        self.assertEqual(len(predictions), len(self.y_test))
        
        # Check feature importances exist
        self.assertTrue(hasattr(model, 'feature_importances_'))
    
    def test_model_prediction_range(self):
        """Test that predictions are in reasonable range"""
        model = train_linear_regression(self.X_train, self.y_train)
        predictions = model.predict(self.X_test)
        
        # Predictions should be somewhat close to training data range
        train_min, train_max = self.y_train.min(), self.y_train.max()
        margin = (train_max - train_min) * 0.5  # Allow 50% margin
        
        # Most predictions should be in reasonable range
        in_range = sum((predictions >= train_min - margin) & 
                      (predictions <= train_max + margin))
        self.assertGreater(in_range / len(predictions), 0.5)


class TestModelEvaluation(unittest.TestCase):
    """Test cases for model evaluation"""
    
    def setUp(self):
        """Set up test data"""
        self.y_true = np.array([100, 110, 120, 115, 125, 130, 140, 135])
        self.y_pred = np.array([105, 108, 122, 118, 123, 128, 142, 138])
    
    def test_calculate_metrics(self):
        """Test metrics calculation"""
        metrics = calculate_metrics(self.y_true, self.y_pred)
        
        # Check all metrics are present
        required_metrics = ['MAE', 'MSE', 'RMSE', 'R2', 'MAPE']
        for metric in required_metrics:
            self.assertIn(metric, metrics)
        
        # Check metrics are numeric
        for value in metrics.values():
            self.assertTrue(isinstance(value, (int, float, np.number)))
        
        # Check MAE is positive
        self.assertGreater(metrics['MAE'], 0)
        
        # Check RMSE >= MAE (mathematical property)
        self.assertGreaterEqual(metrics['RMSE'], metrics['MAE'])
        
        # Check R2 is between -inf and 1
        self.assertLessEqual(metrics['R2'], 1.0)
    
    def test_perfect_predictions(self):
        """Test metrics with perfect predictions"""
        y_true = np.array([100, 110, 120, 130])
        y_pred = np.array([100, 110, 120, 130])
        
        metrics = calculate_metrics(y_true, y_pred)
        
        # Perfect predictions should have:
        self.assertAlmostEqual(metrics['MAE'], 0, places=10)
        self.assertAlmostEqual(metrics['MSE'], 0, places=10)
        self.assertAlmostEqual(metrics['RMSE'], 0, places=10)
        self.assertAlmostEqual(metrics['R2'], 1.0, places=10)
        self.assertAlmostEqual(metrics['MAPE'], 0, places=10)
    
    def test_directional_accuracy(self):
        """Test directional accuracy calculation"""
        y_true = np.array([100, 110, 105, 120, 115])
        y_pred = np.array([105, 115, 100, 125, 110])
        
        accuracy = calculate_directional_accuracy(y_true, y_pred)
        
        # Should be between 0 and 100
        self.assertGreaterEqual(accuracy, 0)
        self.assertLessEqual(accuracy, 100)
        
        # Should be a number
        self.assertTrue(isinstance(accuracy, (int, float, np.number)))
    
    def test_metrics_with_single_value(self):
        """Test metrics with constant predictions"""
        y_true = np.array([100, 110, 120, 130])
        y_pred = np.array([110, 110, 110, 110])
        
        metrics = calculate_metrics(y_true, y_pred)
        
        # R2 should be negative (worse than mean baseline)
        self.assertLess(metrics['R2'], 0)


class TestFeaturePreparation(unittest.TestCase):
    """Test cases for feature preparation"""
    
    def test_prepare_features_target(self):
        """Test feature and target preparation"""
        df = pd.DataFrame({
            'Date': pd.date_range('2020-01-01', periods=50),
            'feature_1': np.random.randn(50),
            'feature_2': np.random.randn(50),
            'target': np.random.uniform(100, 200, 50),
            'target_class': np.random.randint(0, 2, 50)
        })
        
        X, y, feature_cols = prepare_features_target(df)
        
        # Check shapes
        self.assertEqual(X.shape[0], df.shape[0])
        self.assertEqual(len(y), df.shape[0])
        
        # Check that Date and target are not in features
        self.assertNotIn('Date', feature_cols)
        self.assertNotIn('target', feature_cols)
        self.assertNotIn('target_class', feature_cols)
        
        # Check feature columns match
        self.assertEqual(len(feature_cols), X.shape[1])


def run_tests():
    """Run all tests"""
    unittest.main(argv=[''], verbosity=2, exit=False)


if __name__ == '__main__':
    run_tests()