"""
LSTM Model for Stock Price Prediction - Enhanced Version
Supports both TensorFlow and PyTorch backends with fallback
"""
import numpy as np
import pandas as pd
import os
import sys
import logging
from typing import Optional, Dict, Any, Tuple, List
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import LSTM_PARAMS, MODELS_DIR, RANDOM_STATE
from utils.helpers import print_section, save_model, Timer

logger = logging.getLogger(__name__)

# Try importing TensorFlow
TF_AVAILABLE = False
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import (
        LSTM, Dense, Dropout, Bidirectional, 
        GRU, Conv1D, MaxPooling1D, Flatten,
        Attention, LayerNormalization, Input,
        Concatenate, Reshape, TimeDistributed
    )
    from tensorflow.keras.callbacks import (
        EarlyStopping, ModelCheckpoint, ReduceLROnPlateau,
        TensorBoard, LearningRateScheduler
    )
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.regularizers import l2
    TF_AVAILABLE = True
    logger.info("✅ TensorFlow available")
except ImportError:
    logger.warning("⚠️  TensorFlow not available. Using placeholder LSTM model.")
    logger.warning("   Install: pip install tensorflow")


class LSTMModel:
    """
    Advanced LSTM model with multiple architectures and fallback support
    """
    
    def __init__(
        self, 
        lookback: int = 60,
        model_type: str = 'bidirectional',
        units: List[int] = [128, 64, 32],
        dropout_rate: float = 0.2,
        learning_rate: float = 0.001,
        use_attention: bool = True,
        use_conv: bool = False
    ):
        """
        Initialize LSTM model
        
        Args:
            lookback: Number of time steps to look back
            model_type: 'lstm', 'bidirectional', 'stacked', 'gru'
            units: List of units for each layer
            dropout_rate: Dropout rate
            learning_rate: Learning rate
            use_attention: Whether to use attention mechanism
            use_conv: Whether to use convolutional layers
        """
        self.lookback = lookback
        self.model_type = model_type
        self.units = units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.use_attention = use_attention
        self.use_conv = use_conv
        self.model = None
        self.scaler = MinMaxScaler()
        self.trained = False
        
        if TF_AVAILABLE:
            self._build_model()
        else:
            logger.warning("⚠️  LSTM model requires TensorFlow")
    
    def _build_model(self) -> None:
        """Build the LSTM model architecture"""
        if not TF_AVAILABLE:
            return
        
        model = Sequential()
        
        # Input layer
        model.add(Input(shape=(self.lookback, 1)))
        
        # Convolutional layer (optional)
        if self.use_conv:
            model.add(Conv1D(filters=64, kernel_size=3, activation='relu', padding='same'))
            model.add(MaxPooling1D(pool_size=2))
            model.add(Dropout(self.dropout_rate))
        
        # LSTM layers based on type
        if self.model_type == 'lstm':
            # Simple LSTM
            for i, units in enumerate(self.units):
                return_sequences = i < len(self.units) - 1
                model.add(LSTM(units, return_sequences=return_sequences, 
                              dropout=self.dropout_rate))
        
        elif self.model_type == 'bidirectional':
            # Bidirectional LSTM
            for i, units in enumerate(self.units):
                return_sequences = i < len(self.units) - 1
                model.add(Bidirectional(
                    LSTM(units, return_sequences=return_sequences, 
                         dropout=self.dropout_rate)
                ))
        
        elif self.model_type == 'stacked':
            # Stacked LSTM with increasing complexity
            for i, units in enumerate(self.units):
                return_sequences = i < len(self.units) - 1
                model.add(LSTM(units, return_sequences=return_sequences,
                              dropout=self.dropout_rate))
        
        elif self.model_type == 'gru':
            # GRU (faster than LSTM)
            for i, units in enumerate(self.units):
                return_sequences = i < len(self.units) - 1
                model.add(GRU(units, return_sequences=return_sequences,
                             dropout=self.dropout_rate))
        
        # Attention layer (optional)
        if self.use_attention and len(self.units) > 1:
            model.add(Attention())
        
        # Dense layers
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.001)))
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(32, activation='relu', kernel_regularizer=l2(0.001)))
        model.add(Dense(1))
        
        # Compile model
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(
            optimizer=optimizer,
            loss='mse',
            metrics=['mae', 'mse']
        )
        
        self.model = model
        
        # Print model summary
        logger.info(f"📊 LSTM Model Architecture: {model_type}")
        model.summary()
    
    def prepare_data(self, X: np.ndarray) -> np.ndarray:
        """
        Prepare data for LSTM training
        
        Args:
            X: Input data (features)
            
        Returns:
            Reshaped data for LSTM
        """
        if len(X.shape) == 2:
            # Reshape for LSTM: (samples, timesteps, features)
            # Using sliding window approach
            n_samples = len(X)
            X_lstm = []
            
            for i in range(self.lookback, n_samples):
                X_lstm.append(X[i-self.lookback:i])
            
            X_lstm = np.array(X_lstm)
            return X_lstm
        else:
            return X
    
    def train(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 100,
        batch_size: int = 32,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        Train the LSTM model
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            epochs: Number of epochs
            batch_size: Batch size
            verbose: Whether to show progress
            
        Returns:
            Training history
        """
        if not TF_AVAILABLE:
            logger.error("TensorFlow not available. Cannot train LSTM.")
            return {'error': 'TensorFlow not available'}
        
        if self.model is None:
            self._build_model()
        
        print_section("Training LSTM Model")
        
        # Scale data
        with Timer("Scaling data"):
            X_train_scaled = self.scaler.fit_transform(X_train)
            y_train_scaled = self.scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
            
            if X_val is not None:
                X_val_scaled = self.scaler.transform(X_val)
                y_val_scaled = self.scaler.transform(y_val.reshape(-1, 1)).flatten()
        
        # Prepare data for LSTM
        with Timer("Preparing LSTM data"):
            X_train_lstm = self.prepare_data(X_train_scaled)
            y_train_lstm = y_train_scaled[self.lookback:]
            
            if X_val is not None:
                X_val_lstm = self.prepare_data(X_val_scaled)
                y_val_lstm = y_val_scaled[self.lookback:]
            else:
                X_val_lstm = None
                y_val_lstm = None
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=15, restore_best_weights=True, verbose=1),
            ReduceLROnPlateau(factor=0.5, patience=5, verbose=1),
            ModelCheckpoint(
                filepath=os.path.join(MODELS_DIR, 'lstm_best.h5'),
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train model
        print(f"\n🚀 Training LSTM for {epochs} epochs...")
        with Timer("LSTM Training"):
            history = self.model.fit(
                X_train_lstm, y_train_lstm,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_val_lstm, y_val_lstm) if X_val_lstm is not None else None,
                callbacks=callbacks,
                verbose=1 if verbose else 0
            )
        
        self.trained = True
        
        # Save model
        save_model(self.model, 'lstm_model', MODELS_DIR)
        
        # Return training history
        return {
            'history': history.history,
            'best_epoch': len(history.history['loss']),
            'final_loss': history.history['loss'][-1],
            'final_val_loss': history.history['val_loss'][-1] if 'val_loss' in history.history else None
        }
    
    def predict(self, X: np.ndarray, return_scaled: bool = True) -> np.ndarray:
        """
        Make predictions using the trained model
        
        Args:
            X: Input features
            return_scaled: Whether to return unscaled predictions
            
        Returns:
            Predictions
        """
        if not TF_AVAILABLE or self.model is None:
            logger.error("Model not trained or TensorFlow not available")
            return np.zeros(len(X))
        
        # Scale features
        X_scaled = self.scaler.transform(X)
        
        # Prepare data for LSTM
        X_lstm = self.prepare_data(X_scaled)
        
        # Make predictions
        predictions = self.model.predict(X_lstm, verbose=0)
        
        if not return_scaled:
            # Inverse transform predictions
            predictions = self.scaler.inverse_transform(predictions)
        
        return predictions.flatten()
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate model performance
        
        Args:
            X_test: Test features
            y_test: Test target
            
        Returns:
            Dictionary of metrics
        """
        if not self.trained:
            logger.warning("Model not trained. Call train() first.")
            return {}
        
        predictions = self.predict(X_test, return_scaled=True)
        
        # Calculate metrics
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_test, predictions)),
            'mae': mean_absolute_error(y_test, predictions),
            'r2': r2_score(y_test, predictions),
            'mape': np.mean(np.abs((y_test - predictions) / y_test)) * 100
        }
        
        return metrics
    
    def save(self, filepath: str) -> None:
        """
        Save the trained model
        
        Args:
            filepath: Path to save the model
        """
        if TF_AVAILABLE and self.model is not None:
            self.model.save(filepath)
            logger.info(f"✅ LSTM model saved to: {filepath}")
    
    def load(self, filepath: str) -> None:
        """
        Load a trained model
        
        Args:
            filepath: Path to the saved model
        """
        if TF_AVAILABLE:
            self.model = load_model(filepath)
            self.trained = True
            logger.info(f"✅ LSTM model loaded from: {filepath}")
    
    def get_model_summary(self) -> str:
        """Get model summary as string"""
        if TF_AVAILABLE and self.model is not None:
            from io import StringIO
            import sys
            old_stdout = sys.stdout
            sys.stdout = StringIO()
            self.model.summary()
            summary = sys.stdout.getvalue()
            sys.stdout = old_stdout
            return summary
        return "Model not built or TensorFlow not available"


class AdvancedLSTMEnsemble:
    """
    Ensemble of multiple LSTM models with different architectures
    """
    
    def __init__(self, lookback: int = 60):
        self.lookback = lookback
        self.models = []
        self.weights = []
        
    def build_ensemble(self):
        """Build multiple LSTM models"""
        architectures = [
            {'type': 'lstm', 'units': [64, 32], 'use_attention': False},
            {'type': 'bidirectional', 'units': [64, 32], 'use_attention': False},
            {'type': 'bidirectional', 'units': [128, 64, 32], 'use_attention': True},
            {'type': 'stacked', 'units': [64, 64, 32], 'use_attention': False},
            {'type': 'gru', 'units': [64, 32], 'use_attention': False}
        ]
        
        for arch in architectures:
            model = LSTMModel(
                lookback=self.lookback,
                model_type=arch['type'],
                units=arch['units'],
                use_attention=arch['use_attention']
            )
            self.models.append(model)
            self.weights.append(1.0 / len(architectures))
        
        logger.info(f"✅ Built ensemble with {len(self.models)} models")
        return self.models
    
    def train_all(self, X_train, y_train, X_val, y_val, epochs=50):
        """Train all models in ensemble"""
        histories = []
        
        for i, model in enumerate(self.models):
            logger.info(f"\n📊 Training Model {i+1}/{len(self.models)}")
            history = model.train(
                X_train, y_train,
                X_val, y_val,
                epochs=epochs,
                verbose=True
            )
            histories.append(history)
            
            # Adjust weights based on validation performance
            if 'history' in history:
                val_loss = history['history']['val_loss'][-1] if 'val_loss' in history['history'] else 0
                self.weights[i] = 1.0 / (val_loss + 1e-6)
        
        # Normalize weights
        total_weight = sum(self.weights)
        self.weights = [w / total_weight for w in self.weights]
        
        return histories
    
    def predict(self, X):
        """Make ensemble predictions"""
        predictions = []
        
        for model, weight in zip(self.models, self.weights):
            pred = model.predict(X)
            predictions.append(pred * weight)
        
        return np.sum(predictions, axis=0)


# ============================================================================
# LEGACY SUPPORT FUNCTIONS
# ============================================================================

def train_lstm(X_train, y_train, X_test, y_test, params=None):
    """Legacy function for backward compatibility"""
    if params is None:
        params = LSTM_PARAMS
    
    model = LSTMModel(
        lookback=params.get('sequence_length', 60),
        units=params.get('units', [128, 64, 32]),
        dropout_rate=params.get('dropout', 0.2),
        learning_rate=params.get('learning_rate', 0.001)
    )
    
    history = model.train(
        X_train, y_train,
        X_test, y_test,
        epochs=params.get('epochs', 100),
        batch_size=params.get('batch_size', 32)
    )
    
    return model, history


if __name__ == "__main__":
    print_section("Testing Enhanced LSTM Model")
    
    if TF_AVAILABLE:
        print("✅ TensorFlow available")
        
        # Generate sample data
        np.random.seed(42)
        n_samples = 500
        X = np.random.randn(n_samples, 10)
        y = np.sin(np.linspace(0, 10, n_samples)) + np.random.randn(n_samples) * 0.1
        
        # Split data
        split = int(n_samples * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        # Test LSTM model
        print("\n📊 Testing Basic LSTM...")
        model = LSTMModel(lookback=20, model_type='bidirectional', units=[64, 32])
        
        # Train
        history = model.train(
            X_train, y_train,
            X_test, y_test,
            epochs=10,
            batch_size=16,
            verbose=True
        )
        
        # Evaluate
        metrics = model.evaluate(X_test, y_test)
        print(f"\n📊 Evaluation Metrics:")
        for key, value in metrics.items():
            print(f"  {key:8s}: {value:.4f}")
        
        # Test ensemble
        print("\n📊 Testing LSTM Ensemble...")
        ensemble = AdvancedLSTMEnsemble(lookback=20)
        ensemble.build_ensemble()
        ensemble.train_all(X_train, y_train, X_test, y_test, epochs=5)
        
        preds = ensemble.predict(X_test)
        print(f"Ensemble predictions shape: {preds.shape}")
        
    else:
        print("⚠️  TensorFlow not available. Install it for LSTM support:")
        print("   pip install tensorflow")