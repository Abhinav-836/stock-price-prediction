# src/ensemble_model.py - Ensemble with LSTM + XGBoost + LightGBM + Stacking - FIXED
"""
Fixes vs previous version:
  * `import tensorflow` at module top crashed the import when TF is not installed
    (it is not on Streamlit Cloud). TF is now optional; without it the ensemble is tree-only.
  * `tf.keras.layers.Attention()` inside Sequential is invalid (needs [query, value]).
  * LSTM inputs were windowed (n - lookback rows) but trained against the FULL y_train
    (n rows) -> shape mismatch in fit().
  * predict() added an (n - lookback) LSTM vector to n-length tree vectors -> broadcast error.
    All model outputs are now aligned to the same rows.
  * LSTM inputs were never scaled (the StandardScaler was created but unused).
  * StackingRegressor(cv=5) fits its meta-model on KFold predictions where earlier rows are
    predicted by models trained on LATER rows (time-series leakage). StackingRegressor does
    not accept TimeSeriesSplit (it needs every row predicted), so the stack is built manually
    from chronological out-of-fold predictions.
"""
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor

try:
    from src.lstm_model import TF_AVAILABLE, LSTMModel
except ImportError:
    from lstm_model import TF_AVAILABLE, LSTMModel


class SuperEnsemble:
    """Weighted ensemble of LSTM (optional) + XGBoost + LightGBM + time-series stacked blend"""

    def __init__(self, lookback: int = 60, use_lstm: bool = True, random_state: int = 42):
        self.lookback = lookback
        self.use_lstm = use_lstm and TF_AVAILABLE
        self.random_state = random_state
        self.lstm = None
        self.xgb = None
        self.lgbm = None
        self.meta = None
        self.weights = None

    def _tree_models(self):
        xgb = XGBRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
                           colsample_bytree=0.8, random_state=self.random_state, n_jobs=1, verbosity=0)
        lgbm = LGBMRegressor(n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.8, random_state=self.random_state, n_jobs=1, verbose=-1)
        return xgb, lgbm

    def train_ensemble(self, X_train, y_train, X_test=None, y_test=None, lstm_epochs: int = 50):
        X_train = np.asarray(X_train, dtype=float)
        y_train = np.asarray(y_train, dtype=float).ravel()

        # Out-of-fold predictions on chronological folds for the stacking meta-model
        oof = np.full((len(X_train), 2), np.nan)
        for tr, va in TimeSeriesSplit(n_splits=5).split(X_train):
            xgb, lgbm = self._tree_models()
            xgb.fit(X_train[tr], y_train[tr])
            lgbm.fit(X_train[tr], y_train[tr])
            oof[va, 0] = xgb.predict(X_train[va])
            oof[va, 1] = lgbm.predict(X_train[va])
        mask = ~np.isnan(oof).any(axis=1)
        self.meta = Ridge(alpha=1.0).fit(oof[mask], y_train[mask])

        self.xgb, self.lgbm = self._tree_models()
        self.xgb.fit(X_train, y_train)
        self.lgbm.fit(X_train, y_train)

        if self.use_lstm:
            self.lstm = LSTMModel(lookback=self.lookback, model_type='lstm', units=[64, 32], use_attention=False)
            self.lstm.train(X_train, y_train, X_test, y_test, epochs=lstm_epochs, verbose=False)
            self.weights = {'lstm': 0.4, 'xgb': 0.2, 'lgbm': 0.2, 'stack': 0.2}
        else:
            self.weights = {'xgb': 0.35, 'lgbm': 0.35, 'stack': 0.3}
        return self

    def _component_predictions(self, X) -> dict:
        X = np.asarray(X, dtype=float)
        p_xgb, p_lgbm = self.xgb.predict(X), self.lgbm.predict(X)
        preds = {'xgb': p_xgb, 'lgbm': p_lgbm, 'stack': self.meta.predict(np.column_stack([p_xgb, p_lgbm]))}
        if self.use_lstm and self.lstm is not None:
            start = self.lookback - 1          # LSTM only predicts once a full window exists
            preds = {k: v[start:] for k, v in preds.items()}
            preds['lstm'] = self.lstm.predict(X)
        return preds

    def predict(self, X, return_uncertainty: bool = False):
        """
        Returns predictions for rows lookback-1..n-1 when the LSTM is used,
        otherwise for all rows. Uncertainty = spread across ensemble members.
        """
        preds = self._component_predictions(X)
        weighted = sum(self.weights[k] * preds[k] for k in self.weights)
        if return_uncertainty:
            std = np.std(np.vstack(list(preds.values())), axis=0)
            return weighted, weighted - 1.96 * std, weighted + 1.96 * std
        return weighted

    def aligned_target(self, y):
        """Slice y the same way predict() slices its output"""
        y = np.asarray(y, dtype=float).ravel()
        return y[self.lookback - 1:] if (self.use_lstm and self.lstm is not None) else y