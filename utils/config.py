"""
Configuration file for stock price prediction project
"""
import os
from datetime import datetime

# ============================================================================
# PATHS
# ============================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')
FIGURES_DIR = os.path.join(OUTPUTS_DIR, 'figures')
PREDICTIONS_DIR = os.path.join(OUTPUTS_DIR, 'predictions')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')
CACHE_DIR = os.path.join(BASE_DIR, 'cache')
DASHBOARD_DIR = os.path.join(BASE_DIR, 'dashboard_data')

# Create directories if they don't exist
for directory in [DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR, 
                  OUTPUTS_DIR, FIGURES_DIR, PREDICTIONS_DIR,
                  LOGS_DIR, CACHE_DIR, DASHBOARD_DIR]:
    os.makedirs(directory, exist_ok=True)

# ============================================================================
# STOCK CONFIGURATION
# ============================================================================
STOCK_SYMBOL = 'AAPL'  # Default symbol
START_DATE = '2010-01-01'
END_DATE = datetime.now().strftime('%Y-%m-%d')
DAYS = 365  # Days of data to generate if needed

# ============================================================================
# DATA CONFIGURATION
# ============================================================================
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1
RANDOM_STATE = 42
TIME_SERIES_SPLIT = True  # Use time series split instead of random split

# ============================================================================
# FEATURE ENGINEERING
# ============================================================================
MOVING_AVERAGES = [5, 10, 20, 50, 100, 200]
FEATURE_COLS = ['Open', 'High', 'Low', 'Close', 'Volume']
TARGET_COL = 'Close'
PREDICTION_DAYS = [1, 3, 5, 7, 14, 30]  # Multiple prediction horizons

# Advanced feature flags
CREATE_DATE_FEATURES = True
CREATE_PRICE_FEATURES = True
CREATE_TECHNICAL_INDICATORS = True
CREATE_VOLUME_FEATURES = True
CREATE_STATISTICAL_FEATURES = True
CREATE_PATTERN_FEATURES = True
CREATE_INTERACTION_FEATURES = True

# Feature selection
SELECT_BEST_FEATURES = True
MAX_FEATURES = 50
MIN_FEATURE_IMPORTANCE = 0.01

# ============================================================================
# MODEL PARAMETERS
# ============================================================================

# XGBoost - Base parameters
XGB_PARAMS = {
    'n_estimators': 200,
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'verbosity': 0
}

# LightGBM - Base parameters
LGBM_PARAMS = {
    'n_estimators': 200,
    'max_depth': 6,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'verbose': -1
}

# Random Forest
RF_PARAMS = {
    'n_estimators': 200,
    'max_depth': 10,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'max_features': 'sqrt',
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}

# LSTM
LSTM_PARAMS = {
    'sequence_length': 60,
    'epochs': 100,
    'batch_size': 32,
    'units': [128, 64, 32],
    'dropout': 0.2,
    'learning_rate': 0.001,
    'patience': 10,
    'validation_split': 0.2
}

# Ensemble
ENSEMBLE_PARAMS = {
    'weights': {'lstm': 0.4, 'xgboost': 0.25, 'lightgbm': 0.2, 'random_forest': 0.15},
    'stacking_cv': 5,
    'use_voting': True,
    'use_stacking': True
}

# ============================================================================
# HYPERPARAMETER OPTIMIZATION
# ============================================================================
OPTIMIZE_MODELS = True
OPTIMIZATION_TRIALS = 50
OPTIMIZATION_TIMEOUT = 3600  # seconds
OPTIMIZATION_METRIC = 'neg_mean_squared_error'  # or 'r2'

# ============================================================================
# DASHBOARD CONFIGURATION
# ============================================================================
DASHBOARD_CONFIG = {
    'theme': 'dark',  # 'dark' or 'light'
    'refresh_interval': 60,  # seconds
    'enable_alerts': True,
    'enable_email': False,
    'enable_telegram': False,
    'alert_thresholds': {
        'price_change': 0.05,  # 5% change triggers alert
        'rsi_overbought': 70,
        'rsi_oversold': 30,
        'volume_spike': 2.0,  # 2x average
        'volatility': 0.03  # 3% daily volatility
    }
}

# ============================================================================
# PERFORMANCE TRACKING
# ============================================================================
TRACK_PERFORMANCE = True
PERFORMANCE_METRICS = ['r2', 'rmse', 'mae', 'mape', 'directional_accuracy']
SAVE_PREDICTIONS_HISTORY = True
MAX_HISTORY_DAYS = 365

# ============================================================================
# LOGGING
# ============================================================================
LOG_LEVEL = 'INFO'
LOG_TO_FILE = True
LOG_FILE_NAME = f'stock_predictor_{datetime.now().strftime("%Y%m%d")}.log'

# ============================================================================
# API CONFIGURATION
# ============================================================================
USE_API_CACHE = True
CACHE_EXPIRY = 3600  # 1 hour
MAX_RETRIES = 3
RETRY_DELAY = 5  # seconds

# ============================================================================
# ADVANCED FEATURES
# ============================================================================
ENABLE_SENTIMENT_ANALYSIS = False  # Requires external API
ENABLE_NEWS_FEATURES = False
ENABLE_FUNDAMENTAL_ANALYSIS = False
ENABLE_ALTERNATIVE_DATA = False

# ============================================================================
# EXPORT CONFIGURATION
# ============================================================================
EXPORT_FORMATS = ['csv', 'json', 'excel']
EXPORT_INTERVAL = 'daily'  # 'daily', 'weekly', 'monthly'

# ============================================================================
# NOTIFICATION SETTINGS
# ============================================================================
NOTIFICATIONS = {
    'email': {
        'enabled': False,
        'smtp_server': 'smtp.gmail.com',
        'smtp_port': 587,
        'sender': 'your_email@gmail.com',
        'recipients': ['recipient@gmail.com']
    },
    'telegram': {
        'enabled': False,
        'bot_token': 'YOUR_BOT_TOKEN',
        'chat_id': 'YOUR_CHAT_ID'
    }
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_model_config(model_name):
    """Get configuration for specific model"""
    config_map = {
        'xgboost': XGB_PARAMS,
        'lightgbm': LGBM_PARAMS,
        'random_forest': RF_PARAMS,
        'lstm': LSTM_PARAMS
    }
    return config_map.get(model_name, {})

def get_feature_config():
    """Get feature engineering configuration"""
    return {
        'create_date_features': CREATE_DATE_FEATURES,
        'create_price_features': CREATE_PRICE_FEATURES,
        'create_technical_indicators': CREATE_TECHNICAL_INDICATORS,
        'create_volume_features': CREATE_VOLUME_FEATURES,
        'create_statistical_features': CREATE_STATISTICAL_FEATURES,
        'create_pattern_features': CREATE_PATTERN_FEATURES,
        'create_interaction_features': CREATE_INTERACTION_FEATURES
    }

if __name__ == "__main__":
    print("="*60)
    print("CONFIGURATION SUMMARY")
    print("="*60)
    print(f"Stock Symbol: {STOCK_SYMBOL}")
    print(f"Data Range: {START_DATE} to {END_DATE}")
    print(f"Test Size: {TEST_SIZE}")
    print(f"Features: {len(FEATURE_COLS)} base + engineered")
    print(f"Models: XGBoost, LightGBM, Random Forest, LSTM")
    print(f"Ensemble: {ENSEMBLE_PARAMS['use_voting'] and ENSEMBLE_PARAMS['use_stacking']}")
    print(f"Optimization: {OPTIMIZE_MODELS} ({OPTIMIZATION_TRIALS} trials)")
    print("="*60)