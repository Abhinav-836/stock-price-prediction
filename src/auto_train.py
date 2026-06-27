# src/auto_train.py - Automated training with Optuna

import optuna
from optuna.samplers import TPESampler
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error, r2_score
import joblib

class AutoTrainer:
    def __init__(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.best_params = {}
        self.best_models = {}
        
    def optimize_xgboost(self, n_trials=50):
        """Optimize XGBoost with Optuna"""
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'random_state': 42
            }
            
            # Time series cross validation
            tscv = TimeSeriesSplit(n_splits=5)
            scores = []
            
            for train_idx, val_idx in tscv.split(self.X_train):
                X_t, X_v = self.X_train.iloc[train_idx], self.X_train.iloc[val_idx]
                y_t, y_v = self.y_train.iloc[train_idx], self.y_train.iloc[val_idx]
                
                model = XGBRegressor(**params)
                model.fit(X_t, y_t)
                pred = model.predict(X_v)
                scores.append(r2_score(y_v, pred))
            
            return np.mean(scores)
        
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42)
        )
        
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        self.best_params['xgboost'] = study.best_params
        self.best_models['xgboost'] = XGBRegressor(**study.best_params)
        
        return study.best_params
    
    def optimize_lightgbm(self, n_trials=50):
        """Optimize LightGBM with Optuna"""
        
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'num_leaves': trial.suggest_int('num_leaves', 10, 100),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
                'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
                'random_state': 42
            }
            
            tscv = TimeSeriesSplit(n_splits=5)
            scores = []
            
            for train_idx, val_idx in tscv.split(self.X_train):
                X_t, X_v = self.X_train.iloc[train_idx], self.X_train.iloc[val_idx]
                y_t, y_v = self.y_train.iloc[train_idx], self.y_train.iloc[val_idx]
                
                model = LGBMRegressor(**params, verbose=-1)
                model.fit(X_t, y_t)
                pred = model.predict(X_v)
                scores.append(r2_score(y_v, pred))
            
            return np.mean(scores)
        
        study = optuna.create_study(
            direction='maximize',
            sampler=TPESampler(seed=42)
        )
        
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        self.best_params['lightgbm'] = study.best_params
        self.best_models['lightgbm'] = LGBMRegressor(**study.best_params)
        
        return study.best_params
    
    def train_best_models(self):
        """Train all models with best parameters"""
        results = {}
        
        for name, params in self.best_params.items():
            print(f"Training {name} with best parameters...")
            model = self.best_models[name]
            model.fit(self.X_train, self.y_train)
            
            # Evaluate
            train_pred = model.predict(self.X_train)
            test_pred = model.predict(self.X_test)
            
            results[name] = {
                'model': model,
                'train_r2': r2_score(self.y_train, train_pred),
                'test_r2': r2_score(self.y_test, test_pred),
                'train_rmse': np.sqrt(mean_squared_error(self.y_train, train_pred)),
                'test_rmse': np.sqrt(mean_squared_error(self.y_test, test_pred))
            }
            
            print(f"  {name} - Test R²: {results[name]['test_r2']:.4f}")
        
        return results