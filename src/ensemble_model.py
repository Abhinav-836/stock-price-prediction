# src/ensemble_model.py - Ensemble with LSTM + XGBoost + LightGBM

import numpy as np
import pandas as pd
from sklearn.ensemble import VotingRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional, Attention
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

class SuperEnsemble:
    """Advanced ensemble with LSTM + Tree models"""
    
    def __init__(self, lookback=60):
        self.lookback = lookback
        self.lstm_model = None
        self.tree_models = []
        self.scaler = StandardScaler()
        
    def create_lstm_model(self, input_shape):
        """Create enhanced LSTM with attention"""
        model = Sequential([
            Bidirectional(LSTM(128, return_sequences=True), input_shape=input_shape),
            Dropout(0.2),
            Bidirectional(LSTM(64, return_sequences=True)),
            Dropout(0.2),
            tf.keras.layers.Attention(),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation='relu'),
            Dense(1)
        ])
        
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='mse',
            metrics=['mae']
        )
        
        return model
    
    def train_ensemble(self, X_train, y_train, X_test, y_test):
        """Train all models in ensemble"""
        
        # 1. LSTM
        print("📊 Training LSTM...")
        lstm_input = self._prepare_lstm_data(X_train)
        self.lstm_model = self.create_lstm_model(lstm_input.shape[1:])
        
        callbacks = [
            EarlyStopping(patience=10, restore_best_weights=True),
            ReduceLROnPlateau(factor=0.5, patience=5)
        ]
        
        self.lstm_model.fit(
            lstm_input, y_train,
            epochs=100,
            batch_size=32,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=0
        )
        
        # 2. XGBoost
        print("📊 Training XGBoost...")
        xgb = XGBRegressor(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        xgb.fit(X_train, y_train)
        
        # 3. LightGBM
        print("📊 Training LightGBM...")
        lgbm = LGBMRegressor(
            n_estimators=300,
            max_depth=8,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbose=-1
        )
        lgbm.fit(X_train, y_train)
        
        # 4. Stacking Ensemble
        print("📊 Training Stacking Ensemble...")
        self.stacking = StackingRegressor(
            estimators=[
                ('xgb', xgb),
                ('lgbm', lgbm)
            ],
            final_estimator=Ridge(alpha=0.1),
            cv=5
        )
        self.stacking.fit(X_train, y_train)
        
        # Store individual models
        self.tree_models = [xgb, lgbm, self.stacking]
        
    def predict(self, X, return_uncertainty=False):
        """Make predictions with uncertainty estimation"""
        
        # LSTM prediction
        lstm_input = self._prepare_lstm_data(X)
        lstm_pred = self.lstm_model.predict(lstm_input, verbose=0)
        
        # Tree model predictions
        tree_preds = []
        for model in self.tree_models:
            pred = model.predict(X)
            tree_preds.append(pred)
        
        # Combine predictions (weighted average)
        # LSTM usually better for time series, so give it more weight
        weights = [0.4, 0.2, 0.2, 0.2]  # LSTM, XGB, LGBM, Stacking
        
        all_preds = [lstm_pred.flatten()] + tree_preds
        weighted_pred = sum(w * p for w, p in zip(weights, all_preds))
        
        if return_uncertainty:
            # Calculate confidence intervals using bootstrap
            std_pred = np.std(all_preds, axis=0)
            return weighted_pred, weighted_pred - 1.96*std_pred, weighted_pred + 1.96*std_pred
        
        return weighted_pred
    
    def _prepare_lstm_data(self, X):
        """Prepare data for LSTM"""
        # X should be 2D, convert to 3D for LSTM
        if len(X.shape) == 2:
            # Use sliding window
            n_samples = len(X)
            n_features = X.shape[1]
            X_lstm = []
            
            for i in range(self.lookback, n_samples):
                X_lstm.append(X[i-self.lookback:i])
            
            return np.array(X_lstm)
        return X