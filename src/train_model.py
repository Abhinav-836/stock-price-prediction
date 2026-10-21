"""
Model training module for stock price prediction - FIXED

Key fixes vs previous version:
  * Default target is now the next-day RETURN (%) instead of the raw price level.
    Tree models (XGBoost/LightGBM/RF/GB) cannot extrapolate beyond prices seen in
    training, so on a time-based split they produced the negative test R² seen in the README.
  * SVR was trained on scaled data but evaluated/predicted on UNSCALED data -> fixed with
    a Pipeline(StandardScaler, SVR). Linear models also get a scaler.
  * Rows whose target is NaN (the most recent days) are excluded from training only.
  * Feature list comes from one shared function (old `not col.endswith('d')` filter also
    silently dropped real features like 'is_month_end').
  * Directional accuracy added to the metrics.
"""
import pandas as pd
import numpy as np
import os
import sys
import warnings
from typing import Dict, Any, Optional, List, Tuple
warnings.filterwarnings('ignore')

from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import (
    MODELS_DIR,
    DATA_PROCESSED_DIR,
    STOCK_SYMBOL,
    TEST_SIZE,
    RANDOM_STATE,
    XGB_PARAMS,
    LGBM_PARAMS,
    TIME_SERIES_SPLIT
)
from utils.helpers import print_section, save_model, save_metrics, Timer

try:
    from src.preprocess import get_feature_columns
except ImportError:  # when run from inside src/
    from preprocess import get_feature_columns

DEFAULT_TARGET = 'target_return_1d'


def is_return_target(target_col: str) -> bool:
    return 'return' in target_col


class ModelTrainer:
    """
    Model training with time-series cross-validation
    """

    def __init__(self, X_train, y_train, X_test, y_test, feature_cols=None,
                 target_col: str = DEFAULT_TARGET, run_cv: bool = True, verbose: bool = True):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.feature_cols = feature_cols
        self.target_col = target_col
        self.run_cv = run_cv
        self.verbose = verbose
        self.models = {}
        self.results = {}

    def _log(self, msg=""):
        if self.verbose:
            print(msg)

    def prepare_features_target(self, df, feature_cols=None, target_col=DEFAULT_TARGET):
        """Prepare features and target; drops rows where the target is unknown."""
        if target_col not in df.columns:
            # Legacy callers passed 'target' -> map to the 1-day return target
            fallback = [DEFAULT_TARGET, 'target_1d', 'target']
            target_col = next((c for c in fallback if c in df.columns), None)
            if target_col is None:
                raise KeyError(f"No target column found. Available columns: {list(df.columns)}")
            self._log(f"📌 Using target column: {target_col}")

        if feature_cols is None:
            feature_cols = get_feature_columns(df)

        train_df = df.dropna(subset=[target_col])
        X = train_df[feature_cols]
        y = train_df[target_col]

        self._log(f"📊 Features shape: {X.shape}")
        self._log(f"🎯 Target: {target_col}  shape: {y.shape}")

        self.target_col = target_col
        return X, y, feature_cols

    def _fit(self, name, model):
        with Timer(name) if self.verbose else _NullCtx():
            model.fit(self.X_train, self.y_train)
        self.models[name] = model

    def train_linear_models(self) -> 'ModelTrainer':
        if self.verbose:
            print_section("Training Linear Models")
        # Plain LinearRegression is NOT trained here: with ~340 highly collinear features it is
        # numerically unstable (test R² in the -1000s). Ridge/Lasso/ElasticNet cover linear models.
        self._fit('ridge', make_pipeline(StandardScaler(), Ridge(alpha=10.0)))
        self._fit('lasso', make_pipeline(StandardScaler(), Lasso(alpha=0.01, max_iter=5000)))
        self._fit('elasticnet', make_pipeline(StandardScaler(), ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000)))
        return self

    def train_tree_models(self) -> 'ModelTrainer':
        if self.verbose:
            print_section("Training Tree-Based Models")

        self._fit('random_forest', RandomForestRegressor(
            n_estimators=50, max_depth=8, min_samples_split=10, min_samples_leaf=5,
            random_state=RANDOM_STATE, n_jobs=1
        ))

        self._fit('gradient_boosting', GradientBoostingRegressor(
            n_estimators=100, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=RANDOM_STATE
        ))

        xgb_params = dict(XGB_PARAMS)
        xgb_params.update({'n_estimators': 100, 'max_depth': 4, 'n_jobs': 1})
        xgb_params.setdefault('random_state', RANDOM_STATE)
        self._fit('xgboost', XGBRegressor(**xgb_params))

        lgbm_params = dict(LGBM_PARAMS)
        lgbm_params.update({'n_estimators': 100, 'max_depth': 4, 'n_jobs': 1, 'verbose': -1})
        lgbm_params.setdefault('random_state', RANDOM_STATE)
        self._fit('lightgbm', LGBMRegressor(**lgbm_params))
        return self

    def train_svm(self) -> 'ModelTrainer':
        if self.verbose:
            print_section("Training SVM")
        # FIX: scaler lives inside the pipeline, so predict() on raw features works
        self._fit('svr', make_pipeline(StandardScaler(), SVR(kernel='rbf', C=1.0, gamma='scale', epsilon=0.1)))
        return self

    def evaluate_model(self, model, model_name):
        """Evaluate a single model"""
        y_train_pred = model.predict(self.X_train)
        y_test_pred = model.predict(self.X_test)
        y_test = np.asarray(self.y_test)

        metrics = {
            'train_r2': r2_score(self.y_train, y_train_pred),
            'test_r2': r2_score(y_test, y_test_pred),
            'train_rmse': np.sqrt(mean_squared_error(self.y_train, y_train_pred)),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'train_mae': mean_absolute_error(self.y_train, y_train_pred),
            'test_mae': mean_absolute_error(y_test, y_test_pred),
        }

        if is_return_target(self.target_col):
            metrics['test_dir_acc'] = float(np.mean(np.sign(y_test_pred) == np.sign(y_test)) * 100)
            metrics['test_residual_std'] = float(np.std(y_test - y_test_pred))

        if TIME_SERIES_SPLIT and self.run_cv:
            try:
                cv_scores = cross_val_score(
                    model, self.X_train, self.y_train,
                    cv=TimeSeriesSplit(n_splits=3), scoring='r2', n_jobs=1, error_score='raise'
                )
                metrics['cv_r2_mean'] = cv_scores.mean()
                metrics['cv_r2_std'] = cv_scores.std()
            except Exception as e:
                self._log(f"   ⚠️  Cross-validation skipped: {str(e)[:100]}")
                metrics['cv_r2_mean'] = np.nan
                metrics['cv_r2_std'] = np.nan

        self.results[model_name] = {
            'metrics': metrics,
            'predictions': {'train': y_train_pred, 'test': y_test_pred},
            'model': model
        }
        return metrics

    def train_and_evaluate_all(self, save_models=True) -> Dict:
        """Train and evaluate all models"""
        if self.verbose:
            print_section("Training All Models")

        self.train_linear_models()
        self.train_tree_models()
        self.train_svm()

        unit = '%' if is_return_target(self.target_col) else '$'
        self._log("\n" + "=" * 70 + "\n  MODEL EVALUATION RESULTS\n" + "=" * 70)

        for name, model in self.models.items():
            metrics = self.evaluate_model(model, name)
            self._log(f"\n📊 {name.replace('_', ' ').title()}:")
            self._log(f"   Train R²: {metrics['train_r2']:.4f}")
            self._log(f"   Test R²:  {metrics['test_r2']:.4f}")
            self._log(f"   Test RMSE: {metrics['test_rmse']:.4f} {unit}")
            if 'test_dir_acc' in metrics:
                self._log(f"   Test directional accuracy: {metrics['test_dir_acc']:.1f}%")
            if not np.isnan(metrics.get('cv_r2_mean', np.nan)):
                self._log(f"   CV R²:    {metrics['cv_r2_mean']:.4f} (±{metrics['cv_r2_std']:.4f})")

            if save_models:
                save_model(model, name, MODELS_DIR)
                save_metrics(metrics, f"{name}_metrics", MODELS_DIR)

        if self.verbose:
            print("\n" + "=" * 70 + "\n  MODEL COMPARISON (Sorted by Test R²)\n" + "=" * 70)
            print(self.create_comparison_table().to_string(index=False))

        return self.results

    def create_comparison_table(self) -> pd.DataFrame:
        rows = []
        for name, result in self.results.items():
            m = result['metrics']
            row = {
                'Model': name.replace('_', ' ').title(),
                'Train R²': m['train_r2'],
                'Test R²': m['test_r2'],
                'Train RMSE': m['train_rmse'],
                'Test RMSE': m['test_rmse'],
                'Test MAE': m['test_mae'],
            }
            if 'test_dir_acc' in m:
                row['Dir. Acc %'] = m['test_dir_acc']
            if not np.isnan(m.get('cv_r2_mean', np.nan)):
                row['CV R²'] = m['cv_r2_mean']
            rows.append(row)
        return pd.DataFrame(rows).sort_values('Test R²', ascending=False)

    def get_feature_importance(self, model_name='xgboost', top_n=20) -> Optional[pd.DataFrame]:
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
        model = self.models[model_name]
        if not hasattr(model, 'feature_importances_'):
            self._log(f"⚠️  Model '{model_name}' does not have feature_importances_")
            return None
        importance_df = pd.DataFrame({
            'feature': self.feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        self._log(f"\n🔝 Top {top_n} Important Features ({model_name}):")
        self._log(importance_df.head(top_n).to_string(index=False))
        return importance_df

    def get_best_model(self) -> Tuple[Any, str]:
        best_name = max(self.results, key=lambda n: self.results[n]['metrics']['test_r2'])
        return self.models[best_name], best_name


class _NullCtx:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


# ============================================================================
# LEGACY SUPPORT FUNCTIONS
# ============================================================================

def prepare_features_target(df, feature_cols=None, target_col=DEFAULT_TARGET, verbose=True):
    trainer = ModelTrainer(None, None, None, None, verbose=verbose)
    return trainer.prepare_features_target(df, feature_cols, target_col)


def split_data(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """Chronological split (never shuffle time series)"""
    split_idx = int(len(X) * (1 - test_size))
    return X.iloc[:split_idx], X.iloc[split_idx:], y.iloc[:split_idx], y.iloc[split_idx:]


def train_all_models(X_train, y_train, save_models=True):
    trainer = ModelTrainer(X_train, y_train, None, None)
    trainer.train_linear_models()
    trainer.train_tree_models()
    trainer.train_svm()
    if save_models:
        for name, model in trainer.models.items():
            save_model(model, name, MODELS_DIR)
    return trainer.models


def train_linear_regression(X_train, y_train):
    """Scaled linear regression (used by tests / notebooks)"""
    return make_pipeline(StandardScaler(), LinearRegression()).fit(X_train, y_train)


def train_random_forest(X_train, y_train, n_estimators=50, **kwargs):
    params = dict(n_estimators=n_estimators, max_depth=8, min_samples_leaf=5,
                  random_state=RANDOM_STATE, n_jobs=1)
    params.update(kwargs)
    return RandomForestRegressor(**params).fit(X_train, y_train)


def train_xgboost(X_train, y_train, params=None):
    xgb_params = dict(XGB_PARAMS)
    xgb_params.update(params or {})
    return XGBRegressor(**xgb_params).fit(X_train, y_train)


def get_feature_importance(model, feature_cols, top_n=10):
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': feature_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        print(f"\n🔝 Top {top_n} Important Features:")
        print(importance_df.head(top_n).to_string(index=False))
        return importance_df
    print("⚠️  Model does not have feature_importances_ attribute")
    return None


if __name__ == "__main__":
    print("Testing Model Training...")
    filepath = os.path.join(DATA_PROCESSED_DIR, f"{STOCK_SYMBOL}_processed.csv")

    if os.path.exists(filepath):
        df = pd.read_csv(filepath)
        X, y, feature_cols = prepare_features_target(df)
        X_train, X_test, y_train, y_test = split_data(X, y)
        print(f"Training set: {X_train.shape}  Test set: {X_test.shape}")

        trainer = ModelTrainer(X_train, y_train, X_test, y_test, feature_cols)
        trainer.train_and_evaluate_all(save_models=True)
        trainer.get_feature_importance('xgboost', top_n=20)
        _, best_name = trainer.get_best_model()
        print(f"\n🏆 Best Model: {best_name}")
    else:
        print(f"❌ Processed data not found at: {filepath}")