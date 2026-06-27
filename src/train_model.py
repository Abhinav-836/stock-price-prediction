"""
Model training module for stock price prediction - FIXED for memory issues
"""
import pandas as pd
import numpy as np
import os
import sys
import warnings
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime
warnings.filterwarnings('ignore')

from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import (
    MODELS_DIR, 
    DATA_PROCESSED_DIR, 
    STOCK_SYMBOL,
    TEST_SIZE,
    RANDOM_STATE,
    XGB_PARAMS,
    LGBM_PARAMS,
    RF_PARAMS,
    TIME_SERIES_SPLIT
)
from utils.helpers import print_section, save_model, save_metrics, Timer


class ModelTrainer:
    """
    Advanced model training with cross-validation and hyperparameter tuning
    """
    
    def __init__(self, X_train, y_train, X_test, y_test, feature_cols=None):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.feature_cols = feature_cols
        self.models = {}
        self.results = {}
        
    def prepare_features_target(self, df, feature_cols=None, target_col='target_1d'):
        """
        Prepare features and target for training - FIXED to handle multiple target columns
        """
        if feature_cols is None:
            exclude_cols = ['Date', 'target_class']
            target_cols = [col for col in df.columns if col.startswith('target_')]
            exclude_cols.extend(target_cols)
            
            feature_cols = [col for col in df.columns if col not in exclude_cols]
            feature_cols = [col for col in feature_cols if not col.endswith('d')]
        
        if target_col not in df.columns:
            target_cols = [col for col in df.columns if col.startswith('target_') and col.endswith('d')]
            if target_cols:
                target_col = target_cols[0]
                print(f"📌 Using target column: {target_col}")
            else:
                raise KeyError(f"No target column found. Available columns: {list(df.columns)}")
        
        X = df[feature_cols]
        y = df[target_col]
        
        print(f"📊 Features shape: {X.shape}")
        print(f"🎯 Target shape: {y.shape}")
        print(f"📝 Number of features: {len(feature_cols)}")
        
        return X, y, feature_cols
    
    def train_linear_models(self) -> 'ModelTrainer':
        """Train linear models"""
        print_section("Training Linear Models")
        
        print("\n📊 Training Linear Regression...")
        with Timer("Linear Regression"):
            lr = LinearRegression()
            lr.fit(self.X_train, self.y_train)
            self.models['linear_regression'] = lr
        
        print("\n📊 Training Ridge Regression...")
        with Timer("Ridge Regression"):
            ridge = Ridge(alpha=1.0)
            ridge.fit(self.X_train, self.y_train)
            self.models['ridge'] = ridge
        
        print("\n📊 Training Lasso Regression...")
        with Timer("Lasso Regression"):
            lasso = Lasso(alpha=0.1)
            lasso.fit(self.X_train, self.y_train)
            self.models['lasso'] = lasso
        
        print("\n📊 Training ElasticNet...")
        with Timer("ElasticNet"):
            elastic = ElasticNet(alpha=0.1, l1_ratio=0.5)
            elastic.fit(self.X_train, self.y_train)
            self.models['elasticnet'] = elastic
        
        return self
    
    def train_tree_models(self) -> 'ModelTrainer':
        """Train tree-based models with memory optimization"""
        print_section("Training Tree-Based Models")
        
        # Random Forest - Memory optimized
        print("\n📊 Training Random Forest...")
        with Timer("Random Forest"):
            rf = RandomForestRegressor(
                n_estimators=50,  # Reduced for memory
                max_depth=8,
                min_samples_split=10,
                min_samples_leaf=5,
                random_state=RANDOM_STATE,
                n_jobs=1  # Single core to avoid memory issues
            )
            rf.fit(self.X_train, self.y_train)
            self.models['random_forest'] = rf
        
        # Gradient Boosting
        print("\n📊 Training Gradient Boosting...")
        with Timer("Gradient Boosting"):
            gb = GradientBoostingRegressor(
                n_estimators=100,  # Reduced
                max_depth=4,
                learning_rate=0.05,
                random_state=RANDOM_STATE
            )
            gb.fit(self.X_train, self.y_train)
            self.models['gradient_boosting'] = gb
        
        # XGBoost - Memory optimized
        print("\n📊 Training XGBoost...")
        with Timer("XGBoost"):
            xgb_params = XGB_PARAMS.copy()
            xgb_params['n_estimators'] = 100  # Reduced
            xgb_params['max_depth'] = 5
            xgb_params['n_jobs'] = 1
            xgb = XGBRegressor(**xgb_params)
            xgb.fit(self.X_train, self.y_train)
            self.models['xgboost'] = xgb
        
        # LightGBM
        print("\n📊 Training LightGBM...")
        with Timer("LightGBM"):
            lgbm_params = LGBM_PARAMS.copy()
            lgbm_params['n_estimators'] = 100  # Reduced
            lgbm_params['max_depth'] = 5
            lgbm_params['n_jobs'] = 1
            lgbm = LGBMRegressor(**lgbm_params)
            lgbm.fit(self.X_train, self.y_train)
            self.models['lightgbm'] = lgbm
        
        return self
    
    def train_svm(self) -> 'ModelTrainer':
        """Train SVM"""
        print_section("Training SVM")
        
        # Scale data for SVM
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(self.X_train)
        
        print("\n📊 Training SVR...")
        with Timer("SVR"):
            svr = SVR(kernel='rbf', C=10, gamma=0.1, epsilon=0.1)  # Reduced C
            svr.fit(X_train_scaled, self.y_train)
            self.models['svr'] = svr
            self.scaler = scaler
        
        return self
    
    def evaluate_model(self, model, model_name):
        """Evaluate a single model - FIXED for memory issues"""
        # Predictions
        y_train_pred = model.predict(self.X_train)
        y_test_pred = model.predict(self.X_test)
        
        # Calculate metrics
        metrics = {
            'train_r2': r2_score(self.y_train, y_train_pred),
            'test_r2': r2_score(self.y_test, y_test_pred),
            'train_rmse': np.sqrt(mean_squared_error(self.y_train, y_train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(self.y_test, y_test_pred)),
            'train_mae': mean_absolute_error(self.y_train, y_train_pred),
            'test_mae': mean_absolute_error(self.y_test, y_test_pred)
        }
        
        # Cross-validation - FIXED: use n_jobs=1 to avoid memory issues
        if TIME_SERIES_SPLIT:
            try:
                tscv = TimeSeriesSplit(n_splits=3)  # Reduced from 5 to 3
                cv_scores = cross_val_score(
                    model, self.X_train, self.y_train,
                    cv=tscv, scoring='r2', 
                    n_jobs=1,  # Use single core to avoid memory issues
                    error_score='raise'
                )
                metrics['cv_r2_mean'] = cv_scores.mean()
                metrics['cv_r2_std'] = cv_scores.std()
            except Exception as e:
                print(f"   ⚠️  Cross-validation skipped: {str(e)[:100]}")
                metrics['cv_r2_mean'] = np.nan
                metrics['cv_r2_std'] = np.nan
        
        self.results[model_name] = {
            'metrics': metrics,
            'predictions': {
                'train': y_train_pred,
                'test': y_test_pred
            },
            'model': model
        }
        
        return metrics
    
    def train_and_evaluate_all(self, save_models=True) -> Dict:
        """Train and evaluate all models"""
        print_section("Training All Models")
        
        # Train linear models
        self.train_linear_models()
        
        # Train tree models
        self.train_tree_models()
        
        # Train SVM
        self.train_svm()
        
        # Evaluate all models
        print("\n" + "="*70)
        print("  MODEL EVALUATION RESULTS")
        print("="*70)
        
        for name, model in self.models.items():
            print(f"\n📊 {name.replace('_', ' ').title()}:")
            metrics = self.evaluate_model(model, name)
            
            print(f"   Train R²: {metrics['train_r2']:.4f}")
            print(f"   Test R²:  {metrics['test_r2']:.4f}")
            print(f"   Train RMSE: ${metrics['train_rmse']:.2f}")
            print(f"   Test RMSE:  ${metrics['test_rmse']:.2f}")
            if 'cv_r2_mean' in metrics and not np.isnan(metrics['cv_r2_mean']):
                print(f"   CV R²:    {metrics['cv_r2_mean']:.4f} (±{metrics['cv_r2_std']:.4f})")
            
            # Save model
            if save_models:
                save_model(model, name, MODELS_DIR)
                save_metrics(metrics, f"{name}_metrics", MODELS_DIR)
        
        # Create comparison table
        comparison = self.create_comparison_table()
        print("\n" + "="*70)
        print("  MODEL COMPARISON (Sorted by Test R²)")
        print("="*70)
        print(comparison.to_string(index=False))
        
        return self.results
    
    def create_comparison_table(self) -> pd.DataFrame:
        """Create comparison table of all models"""
        comparison_data = []
        
        for name, result in self.results.items():
            metrics = result['metrics']
            row = {
                'Model': name.replace('_', ' ').title(),
                'Train R²': metrics['train_r2'],
                'Test R²': metrics['test_r2'],
                'Train RMSE': metrics['train_rmse'],
                'Test RMSE': metrics['test_rmse'],
                'Train MAE': metrics['train_mae'],
                'Test MAE': metrics['test_mae']
            }
            if 'cv_r2_mean' in metrics and not np.isnan(metrics['cv_r2_mean']):
                row['CV R²'] = metrics['cv_r2_mean']
            comparison_data.append(row)
        
        df = pd.DataFrame(comparison_data)
        df = df.sort_values('Test R²', ascending=False)
        
        return df
    
    def get_feature_importance(self, model_name='xgboost', top_n=20) -> pd.DataFrame:
        """Get feature importance from a tree-based model"""
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
        
        model = self.models[model_name]
        
        if not hasattr(model, 'feature_importances_'):
            print(f"⚠️  Model '{model_name}' does not have feature_importances_")
            return None
        
        importance_df = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n🔝 Top {top_n} Important Features ({model_name}):")
        print(importance_df.head(top_n).to_string(index=False))
        
        return importance_df
    
    def get_best_model(self) -> Tuple[Any, str]:
        """Get the best performing model"""
        best_name = None
        best_score = -np.inf
        
        for name, result in self.results.items():
            score = result['metrics']['test_r2']
            if score > best_score:
                best_score = score
                best_name = name
        
        return self.models[best_name], best_name


# ============================================================================
# LEGACY SUPPORT FUNCTIONS - FIXED
# ============================================================================

def prepare_features_target(df, feature_cols=None, target_col='target_1d'):
    """Legacy function for backward compatibility - FIXED"""
    trainer = ModelTrainer(None, None, None, None)
    return trainer.prepare_features_target(df, feature_cols, target_col)


def split_data(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """Legacy split function with time series support"""
    if TIME_SERIES_SPLIT:
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
    else:
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=False
        )
    
    return X_train, X_test, y_train, y_test


def train_all_models(X_train, y_train, save_models=True):
    """Legacy function for backward compatibility"""
    trainer = ModelTrainer(X_train, y_train, None, None)
    
    trainer.train_linear_models()
    trainer.train_tree_models()
    trainer.train_svm()
    
    if save_models:
        for name, model in trainer.models.items():
            save_model(model, name, MODELS_DIR)
    
    return trainer.models


def get_feature_importance(model, feature_cols, top_n=10):
    """Legacy function for backward compatibility"""
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n🔝 Top {top_n} Important Features:")
        print(importance_df.head(top_n).to_string(index=False))
        
        return importance_df
    else:
        print("⚠️  Model does not have feature_importances_ attribute")
        return None


if __name__ == "__main__":
    print("Testing Enhanced Model Training...")
    
    filepath = os.path.join(DATA_PROCESSED_DIR, f"{STOCK_SYMBOL}_processed.csv")
    
    if os.path.exists(filepath):
        print(f"📂 Loading data from: {filepath}")
        df = pd.read_csv(filepath)
        
        X, y, feature_cols = prepare_features_target(df, target_col='target_1d')
        X_train, X_test, y_train, y_test = split_data(X, y)
        
        print(f"Training set: {X_train.shape}")
        print(f"Test set: {X_test.shape}")
        
        trainer = ModelTrainer(X_train, y_train, X_test, y_test, feature_cols)
        results = trainer.train_and_evaluate_all(save_models=True)
        
        trainer.get_feature_importance('xgboost', top_n=20)
        
        best_model, best_name = trainer.get_best_model()
        print(f"\n🏆 Best Model: {best_name}")
        
    else:
        print(f"❌ Processed data not found at: {filepath}")