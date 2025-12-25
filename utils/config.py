"""
Configuration file for stock price prediction project
"""
import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
DATA_PROCESSED_DIR = os.path.join(BASE_DIR, 'data', 'processed')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')
FIGURES_DIR = os.path.join(OUTPUTS_DIR, 'figures')
PREDICTIONS_DIR = os.path.join(OUTPUTS_DIR, 'predictions')

# Create directories if they don't exist
for directory in [DATA_RAW_DIR, DATA_PROCESSED_DIR, MODELS_DIR, 
                  OUTPUTS_DIR, FIGURES_DIR, PREDICTIONS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Stock Configuration
STOCK_SYMBOL = 'NVDA'
START_DATE = '2010-01-01'
END_DATE = '2023-12-31'
DAYS = 200 
# Data Configuration
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1
RANDOM_STATE = 42

# Feature Engineering
MOVING_AVERAGES = [5, 10, 20, 50]
FEATURE_COLS = ['Open', 'High', 'Low', 'Close', 'Volume']
TARGET_COL = 'Close'

# Model Parameters
# Linear Regression
LR_PARAMS = {}

# XGBoost
XGB_PARAMS = {
    'n_estimators': 100,
    'max_depth': 5,
    'learning_rate': 0.1,
    'random_state': RANDOM_STATE
}

# LSTM
LSTM_PARAMS = {
    'sequence_length': 60,
    'epochs': 50,
    'batch_size': 32,
    'units': [50, 50],
    'dropout': 0.2
}

# Visualization
PLOT_STYLE = 'seaborn-v0_8'
FIGURE_SIZE = (15, 6)
DPI = 100