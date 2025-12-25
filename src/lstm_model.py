"""
Simple LSTM model placeholder - use when TensorFlow has issues
"""

class LSTMModel:
    """Placeholder LSTM model when TensorFlow not available"""
    
    def __init__(self, lookback=60):
        self.lookback = lookback
        print("⚠️  TensorFlow not available. Using placeholder LSTM model.")
        print("   Install: pip install tensorflow")
    
    def train(self, *args, **kwargs):
        print("LSTM training skipped (TensorFlow not installed)")
        return None
    
    def predict(self, X):
        print("LSTM predictions not available")
        return None

def train_lstm_simple():
    """Placeholder training function"""
    print("⚠️  Cannot train LSTM without TensorFlow")
    print("   To use LSTM, install: pip install tensorflow")
    return None, None