"""
LSTM Model for Stock Price Prediction - FIXED (TensorFlow optional)

Fixes vs previous version:
  * `_build_model` logged `{model_type}` (undefined name) -> NameError on EVERY model build.
  * Input layer was hard-coded to shape (lookback, 1) but the data has N features -> shape error.
    The model is now built at train time with the real feature count.
  * ONE MinMaxScaler was used for both X and y: fitting it on y overwrote the X scaling,
    then `scaler.transform(X_val)` raised (1 feature expected, N given). Separate scalers now.
  * `Attention()` inside a Sequential model is invalid (it needs [query, value] inputs).
    Rebuilt with the functional API (self-attention + pooling).
  * Windows excluded the current row, and targets were misaligned by one step. Fixed.
  * predict(return_scaled=True) returned SCALED values while evaluate() compared them with
    real prices. predict() now returns real units by default.
  * Keras models were pickled with joblib (fragile) and checkpointed to '.h5'
    (rejected by Keras 3). Saved as '.keras' now.
"""
import numpy as np
import os
import sys
import logging
from typing import Optional, Dict, Any, List
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import LSTM_PARAMS, MODELS_DIR, RANDOM_STATE
from utils.helpers import print_section

logger = logging.getLogger(__name__)

TF_AVAILABLE = False
try:
    import tensorflow as tf
    from tensorflow.keras import Model, Input
    from tensorflow.keras.layers import (LSTM, GRU, Dense, Dropout, Bidirectional, Conv1D,
                                         MaxPooling1D, Attention, GlobalAveragePooling1D)
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras.models import load_model as keras_load_model
    TF_AVAILABLE = True
except ImportError:
    logger.warning("⚠️  TensorFlow not available - LSTM models disabled (pip install tensorflow)")


def make_windows(X: np.ndarray, lookback: int) -> np.ndarray:
    """Window ending at (and including) row i -> shape (n - lookback + 1, lookback, n_features)"""
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        return X
    if len(X) < lookback:
        raise ValueError(f"Need at least {lookback} rows, got {len(X)}")
    return np.stack([X[i - lookback + 1:i + 1] for i in range(lookback - 1, len(X))])


class LSTMModel:
    """
    LSTM / BiLSTM / GRU regressor over sliding windows of features
    """

    def __init__(self, lookback: int = 60, model_type: str = 'bidirectional',
                 units: Optional[List[int]] = None, dropout_rate: float = 0.2,
                 learning_rate: float = 0.001, use_attention: bool = True, use_conv: bool = False):
        self.lookback = lookback
        self.model_type = model_type
        self.units = list(units) if units else [128, 64, 32]  # no mutable default arg
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.use_attention = use_attention
        self.use_conv = use_conv
        self.model = None
        self.x_scaler = MinMaxScaler()
        self.y_scaler = MinMaxScaler()
        self.trained = False
        self.n_features = None

    def _recurrent(self, units, return_sequences):
        if self.model_type == 'gru':
            return GRU(units, return_sequences=return_sequences, dropout=self.dropout_rate)
        layer = LSTM(units, return_sequences=return_sequences, dropout=self.dropout_rate)
        return Bidirectional(layer) if self.model_type == 'bidirectional' else layer

    def _build_model(self, n_features: int) -> None:
        if not TF_AVAILABLE:
            raise ImportError("TensorFlow is required for LSTMModel")
        tf.random.set_seed(RANDOM_STATE)
        self.n_features = n_features

        inputs = Input(shape=(self.lookback, n_features))
        x = inputs
        if self.use_conv:
            x = Conv1D(64, 3, activation='relu', padding='same')(x)
            x = MaxPooling1D(pool_size=2)(x)
            x = Dropout(self.dropout_rate)(x)

        use_attention = self.use_attention
        for i, u in enumerate(self.units):
            is_last = i == len(self.units) - 1
            x = self._recurrent(u, return_sequences=(not is_last) or use_attention)(x)

        if use_attention:
            x = Attention()([x, x])          # self-attention over time steps
            x = GlobalAveragePooling1D()(x)

        x = Dropout(self.dropout_rate)(x)
        x = Dense(64, activation='relu', kernel_regularizer=l2(0.001))(x)
        x = Dropout(self.dropout_rate)(x)
        x = Dense(32, activation='relu', kernel_regularizer=l2(0.001))(x)
        outputs = Dense(1)(x)

        self.model = Model(inputs, outputs)
        self.model.compile(optimizer=Adam(learning_rate=self.learning_rate), loss='mse', metrics=['mae'])
        logger.info(f"📊 LSTM Model Architecture: {self.model_type}")

    def prepare_data(self, X: np.ndarray) -> np.ndarray:
        return make_windows(X, self.lookback)

    def _align_y(self, y: np.ndarray) -> np.ndarray:
        return np.asarray(y, dtype=np.float32).ravel()[self.lookback - 1:]

    def train(self, X_train, y_train, X_val=None, y_val=None, epochs: int = 100,
              batch_size: int = 32, verbose: bool = True) -> Dict[str, Any]:
        if not TF_AVAILABLE:
            return {'error': 'TensorFlow not available'}
        print_section("Training LSTM Model")

        X_train = np.asarray(X_train, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.float32).reshape(-1, 1)
        if self.model is None or self.n_features != X_train.shape[1]:
            self._build_model(X_train.shape[1])

        X_tr = self.prepare_data(self.x_scaler.fit_transform(X_train))
        y_tr = self._align_y(self.y_scaler.fit_transform(y_train))

        validation_data = None
        if X_val is not None and y_val is not None and len(X_val) >= self.lookback:
            X_v = self.prepare_data(self.x_scaler.transform(np.asarray(X_val, dtype=np.float32)))
            y_v = self._align_y(self.y_scaler.transform(np.asarray(y_val, dtype=np.float32).reshape(-1, 1)))
            validation_data = (X_v, y_v)

        monitor = 'val_loss' if validation_data else 'loss'
        os.makedirs(MODELS_DIR, exist_ok=True)
        callbacks = [
            EarlyStopping(monitor=monitor, patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor=monitor, factor=0.5, patience=5),
            ModelCheckpoint(os.path.join(MODELS_DIR, 'lstm_best.keras'), monitor=monitor, save_best_only=True),
        ]

        history = self.model.fit(X_tr, y_tr, epochs=epochs, batch_size=batch_size,
                                 validation_data=validation_data, callbacks=callbacks,
                                 verbose=1 if verbose else 0)
        self.trained = True
        self.save(os.path.join(MODELS_DIR, 'lstm_model.keras'))

        h = history.history
        return {'history': h, 'epochs_run': len(h['loss']), 'final_loss': h['loss'][-1],
                'final_val_loss': h['val_loss'][-1] if 'val_loss' in h else None}

    def predict(self, X, return_scaled: bool = False) -> np.ndarray:
        """Predictions for every row from index lookback-1 onwards (real units by default)"""
        if not TF_AVAILABLE or self.model is None:
            raise RuntimeError("Model not trained or TensorFlow not available")
        X_w = self.prepare_data(self.x_scaler.transform(np.asarray(X, dtype=np.float32)))
        pred = self.model.predict(X_w, verbose=0).reshape(-1, 1)
        if not return_scaled:
            pred = self.y_scaler.inverse_transform(pred)
        return pred.ravel()

    def evaluate(self, X_test, y_test) -> Dict[str, float]:
        if not self.trained:
            logger.warning("Model not trained. Call train() first.")
            return {}
        pred = self.predict(X_test)
        y = self._align_y(y_test)
        nonzero = y != 0
        return {
            'rmse': float(np.sqrt(mean_squared_error(y, pred))),
            'mae': float(mean_absolute_error(y, pred)),
            'r2': float(r2_score(y, pred)),
            'mape': float(np.mean(np.abs((y[nonzero] - pred[nonzero]) / y[nonzero])) * 100) if nonzero.any() else float('nan'),
        }

    def save(self, filepath: str) -> None:
        if TF_AVAILABLE and self.model is not None:
            if not filepath.endswith('.keras'):
                filepath += '.keras'
            self.model.save(filepath)
            logger.info(f"✅ LSTM model saved to: {filepath}")

    def load(self, filepath: str) -> None:
        """Note: scalers are not stored inside the .keras file - refit/keep them alongside."""
        if TF_AVAILABLE:
            self.model = keras_load_model(filepath)
            self.n_features = self.model.input_shape[-1]
            self.trained = True

    def get_model_summary(self) -> str:
        if TF_AVAILABLE and self.model is not None:
            lines = []
            self.model.summary(print_fn=lines.append)
            return "\n".join(lines)
        return "Model not built or TensorFlow not available"


class AdvancedLSTMEnsemble:
    """Ensemble of recurrent models, weighted by inverse validation loss"""

    def __init__(self, lookback: int = 60):
        self.lookback = lookback
        self.models: List[LSTMModel] = []
        self.weights: List[float] = []

    def build_ensemble(self):
        architectures = [
            {'type': 'lstm', 'units': [64, 32], 'use_attention': False},
            {'type': 'bidirectional', 'units': [64, 32], 'use_attention': False},
            {'type': 'bidirectional', 'units': [128, 64, 32], 'use_attention': True},
            {'type': 'gru', 'units': [64, 32], 'use_attention': False},
        ]
        self.models = [LSTMModel(lookback=self.lookback, model_type=a['type'], units=a['units'],
                                 use_attention=a['use_attention']) for a in architectures]
        self.weights = [1.0 / len(self.models)] * len(self.models)
        return self.models

    def train_all(self, X_train, y_train, X_val, y_val, epochs=50):
        histories = []
        for i, model in enumerate(self.models):
            h = model.train(X_train, y_train, X_val, y_val, epochs=epochs, verbose=False)
            histories.append(h)
            if h.get('final_val_loss') is not None:
                self.weights[i] = 1.0 / (h['final_val_loss'] + 1e-6)
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]
        return histories

    def predict(self, X):
        return np.sum([m.predict(X) * w for m, w in zip(self.models, self.weights)], axis=0)


# ============================================================================
# LEGACY SUPPORT FUNCTIONS
# ============================================================================

def train_lstm(X_train, y_train, X_test, y_test, params=None):
    params = params or LSTM_PARAMS
    model = LSTMModel(lookback=params.get('sequence_length', 60), units=params.get('units', [128, 64, 32]),
                      dropout_rate=params.get('dropout', 0.2), learning_rate=params.get('learning_rate', 0.001))
    history = model.train(X_train, y_train, X_test, y_test, epochs=params.get('epochs', 100),
                          batch_size=params.get('batch_size', 32))
    return model, history


if __name__ == "__main__":
    if not TF_AVAILABLE:
        print("⚠️  TensorFlow not available. pip install tensorflow")
        sys.exit(0)
    rng = np.random.default_rng(42)
    X = rng.normal(size=(500, 10))
    y = np.sin(np.linspace(0, 10, 500)) + rng.normal(0, 0.1, 500)
    model = LSTMModel(lookback=20, model_type='bidirectional', units=[32, 16])
    model.train(X[:400], y[:400], X[400:], y[400:], epochs=3, batch_size=16)
    print(model.evaluate(X[400:], y[400:]))