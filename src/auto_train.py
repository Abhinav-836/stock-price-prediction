# src/auto_train.py - Automated hyperparameter search with Optuna - FIXED
"""
Fixes vs previous version:
  * The final models were rebuilt from `study.best_params` only, dropping random_state
    (non-reproducible) and LightGBM's verbose=-1 (log spam).
  * `.iloc` crashed when numpy arrays were passed -> inputs are converted to DataFrame/Series.
  * optuna is optional; a clear ImportError is raised instead of crashing on import.
  * Single-threaded models + optional timeout so it can't freeze a small cloud instance.
  * README referenced `python src/auto_train.py`, but the file had no entry point.
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import RANDOM_STATE

try:
    import optuna
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False


class AutoTrainer:
    def __init__(self, X_train, y_train, X_test, y_test, n_splits: int = 5):
        if not OPTUNA_AVAILABLE:
            raise ImportError("AutoTrainer needs optuna: pip install optuna")
        self.X_train = pd.DataFrame(X_train).reset_index(drop=True)
        self.y_train = pd.Series(np.asarray(y_train).ravel())
        self.X_test = pd.DataFrame(X_test).reset_index(drop=True)
        self.y_test = pd.Series(np.asarray(y_test).ravel())
        self.n_splits = n_splits
        self.best_params = {}
        self.best_models = {}

    def _cv_score(self, make_model) -> float:
        scores = []
        for train_idx, val_idx in TimeSeriesSplit(n_splits=self.n_splits).split(self.X_train):
            model = make_model()
            model.fit(self.X_train.iloc[train_idx], self.y_train.iloc[train_idx])
            pred = model.predict(self.X_train.iloc[val_idx])
            scores.append(r2_score(self.y_train.iloc[val_idx], pred))
        return float(np.mean(scores))

    def _run_study(self, objective, n_trials, timeout):
        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=RANDOM_STATE))
        study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
        return study

    def optimize_xgboost(self, n_trials=50, timeout=None):
        fixed = {'random_state': RANDOM_STATE, 'n_jobs': 1, 'verbosity': 0}

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 2, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 20),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10, log=True),
            }
            return self._cv_score(lambda: XGBRegressor(**params, **fixed))

        study = self._run_study(objective, n_trials, timeout)
        self.best_params['xgboost'] = study.best_params
        self.best_models['xgboost'] = XGBRegressor(**study.best_params, **fixed)
        return study.best_params

    def optimize_lightgbm(self, n_trials=50, timeout=None):
        fixed = {'random_state': RANDOM_STATE, 'n_jobs': 1, 'verbose': -1}

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 2, 8),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'num_leaves': trial.suggest_int('num_leaves', 8, 64),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'subsample_freq': 1,
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.3, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10, log=True),
                'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10, log=True),
            }
            return self._cv_score(lambda: LGBMRegressor(**params, **fixed))

        study = self._run_study(objective, n_trials, timeout)
        params = {**study.best_params, 'subsample_freq': 1}
        self.best_params['lightgbm'] = params
        self.best_models['lightgbm'] = LGBMRegressor(**params, **fixed)
        return params

    def train_best_models(self):
        results = {}
        for name, model in self.best_models.items():
            print(f"Training {name} with best parameters...")
            model.fit(self.X_train, self.y_train)
            train_pred, test_pred = model.predict(self.X_train), model.predict(self.X_test)
            results[name] = {
                'model': model,
                'params': self.best_params[name],
                'train_r2': r2_score(self.y_train, train_pred),
                'test_r2': r2_score(self.y_test, test_pred),
                'train_rmse': float(np.sqrt(mean_squared_error(self.y_train, train_pred))),
                'test_rmse': float(np.sqrt(mean_squared_error(self.y_test, test_pred))),
                'test_dir_acc': float(np.mean(np.sign(test_pred) == np.sign(self.y_test)) * 100),
            }
            print(f"  {name} - Test R²: {results[name]['test_r2']:.4f}  "
                  f"Dir.Acc: {results[name]['test_dir_acc']:.1f}%")
        return results


if __name__ == "__main__":
    from utils.config import DATA_PROCESSED_DIR, STOCK_SYMBOL
    from src.train_model import prepare_features_target, split_data

    path = os.path.join(DATA_PROCESSED_DIR, f"{STOCK_SYMBOL}_processed.csv")
    if not os.path.exists(path):
        print(f"❌ Processed data not found at {path}. Run: python main.py")
        sys.exit(1)
    X, y, _ = prepare_features_target(pd.read_csv(path))
    X_train, X_test, y_train, y_test = split_data(X, y)
    trainer = AutoTrainer(X_train, y_train, X_test, y_test)
    trainer.optimize_xgboost(n_trials=20, timeout=600)
    trainer.optimize_lightgbm(n_trials=20, timeout=600)
    trainer.train_best_models()