# Models Directory

This directory stores trained machine learning models for stock price prediction.

## 📊 Stored Models

Models are saved in pickle format (`.pkl`) for easy loading and reuse, along with JSON metadata files containing evaluation metrics.

### Available Models

| Model | Description | Best For |
|-------|-------------|----------|
| **xgboost.pkl** | Gradient boosting with regularization | Maximum accuracy (⭐ Best Overall) |
| **lightgbm.pkl** | Lightweight gradient boosting | Fast training with good accuracy |
| **random_forest.pkl** | Ensemble of decision trees | Robust predictions with feature importance |
| **ridge.pkl** | Linear regression with L2 regularization | Stable predictions with many features |
| **lasso.pkl** | Linear regression with L1 regularization | Feature selection |
| **elasticnet.pkl** | Linear regression with L1+L2 regularization | Balanced feature selection |
| **linear_regression.pkl** | Simple linear regression | Baseline comparison |
| **gradient_boosting.pkl** | Sequential ensemble | Complex patterns |
| **svr.pkl** | Support Vector Regression | Non-linear relationships |

## 📁 Model Files

When you run the training pipeline, these files are automatically created:
models/
├── xgboost.pkl (~1-10 MB)
├── xgboost_metrics.json (evaluation metrics)
├── lightgbm.pkl (~1-5 MB)
├── lightgbm_metrics.json (evaluation metrics)
├── random_forest.pkl (~50-500 MB)
├── random_forest_metrics.json (evaluation metrics)
├── ridge.pkl (~100 KB)
├── ridge_metrics.json (evaluation metrics)
├── lasso.pkl (~100 KB)
├── lasso_metrics.json (evaluation metrics)
├── elasticnet.pkl (~100 KB)
├── elasticnet_metrics.json (evaluation metrics)
├── linear_regression.pkl (~100 KB)
├── linear_regression_metrics.json (evaluation metrics)
├── gradient_boosting.pkl (~10-50 MB)
├── gradient_boosting_metrics.json (evaluation metrics)
├── svr.pkl (~10-50 MB)
├── svr_metrics.json (evaluation metrics)
└── README.md (this file)

text

## 🚀 Usage

### Training Models

```bash
# Train all models
python main.py --mode train

# Train with specific symbol
python main.py --symbol AAPL --mode train

# Run full pipeline (data + train + predict)
python main.py --symbol AAPL --days 30
Loading Models
python
from utils.helpers import load_model
from utils.config import MODELS_DIR

# Load a specific model
model = load_model('xgboost', MODELS_DIR)

# Load with metadata
model, metadata = load_model('xgboost', MODELS_DIR, with_metadata=True)

# Make predictions
predictions = model.predict(X_test)
Model Evaluation
python
from src.evaluate import AdvancedEvaluator

# Evaluate model
evaluator = AdvancedEvaluator(y_true, y_pred, model_name='xgboost')
metrics = evaluator.calculate_all_metrics()
evaluator.print_metrics()
📊 Model Performance
Typical performance metrics on AAPL stock data (2020-2023):

Model	Test R²	Test RMSE	Test MAE	Directional Accuracy
XGBoost	0.251	$5.85	$4.82	~55%
Ridge	0.175	$6.14	$5.12	~54%
Lasso	-0.715	$8.85	$7.58	~52%
Gradient Boosting	-0.865	$9.23	$7.90	~50%
LightGBM	-1.337	$10.33	$8.78	~51%
ElasticNet	-1.645	$11.00	$9.82	~49%
Random Forest	-1.745	$11.20	$9.71	~48%
SVR	-10.041	$22.46	$21.42	~45%
Linear Regression	-1401.58	$253.20	$215.26	~48%
Note: Performance varies by stock, date range, and market conditions

🔧 Model Configuration
XGBoost (Best Model)
python
from xgboost import XGBRegressor

model = XGBRegressor(
    n_estimators=100,      # Number of boosting rounds
    max_depth=5,           # Maximum tree depth
    learning_rate=0.05,    # Step size shrinkage
    subsample=0.8,         # Subsample ratio of training data
    colsample_bytree=0.8,  # Subsample ratio of columns
    reg_alpha=0.1,         # L1 regularization
    reg_lambda=1.0,        # L2 regularization
    random_state=42
)
LightGBM
python
from lightgbm import LGBMRegressor

model = LGBMRegressor(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.05,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42
)
Random Forest
python
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(
    n_estimators=50,       # Number of trees (reduced for memory)
    max_depth=8,           # Maximum depth
    min_samples_split=10,  # Minimum samples to split
    min_samples_leaf=5,    # Minimum samples per leaf
    random_state=42,
    n_jobs=1               # Single core to avoid memory issues
)
📈 Feature Importance
XGBoost and Random Forest provide feature importance:

python
from src.train_model import get_feature_importance

# Get importance
importance_df = get_feature_importance(model, feature_cols, top_n=20)

# Plot (in dashboard)
# Feature importance is automatically displayed in the dashboard
Top Features Typically Include:
Lagged prices (close_lag_1, close_lag_2, close_lag_6)

Lagged volumes (volume_lag_1, volume_lag_2)

Technical indicators (RSI, MACD, Bollinger Bands)

Price ratios (High/Low, Close/Open)

Volatility measures (ATR, rolling standard deviation)

Volume indicators (OBV, VWAP, Money Flow Index)

🔄 Model Versioning
Models are overwritten when retrained. To keep multiple versions:

python
from utils.helpers import save_model
from datetime import datetime

# Save with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_model(model, f'xgboost_{timestamp}', MODELS_DIR)
🧪 Hyperparameter Tuning
Manual Tuning (config.py)
Edit utils/config.py:

python
# XGBoost Parameters
XGB_PARAMS = {
    'n_estimators': 100,
    'max_depth': 5,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': RANDOM_STATE,
    'n_jobs': 1
}
Automated Tuning (Optuna)
bash
# Run auto-training with Optuna
python src/auto_train.py
📊 Evaluation Metrics
Models are evaluated on:

Metric	Description	Target
R² Score	Proportion of variance explained	Higher is better (max 1.0)
RMSE	Root Mean Squared Error	Lower is better
MAE	Mean Absolute Error	Lower is better
MAPE	Mean Absolute Percentage Error	Lower is better
Directional Accuracy	% of correct direction predictions	>50% is better than random
Sharpe Ratio	Risk-adjusted return	>1 is good
Max Drawdown	Maximum loss from peak	Lower is better
⚠️ Important Notes
Memory & Performance
Random Forest models can be very large (100+ MB)

XGBoost and LightGBM are more memory efficient

Models are ignored by Git (see .gitignore)

Use n_jobs=1 to avoid memory issues on Windows

Retraining
Models should be retrained periodically (monthly recommended)

Market conditions change over time

New data improves predictions

Limitations
Past performance doesn't guarantee future results

Stock prediction is inherently difficult

~50-60% directional accuracy is typical

Models perform better on liquid, stable stocks

🐛 Troubleshooting
Problem	Solution
Models not saving	Check write permissions for models/ directory
Model file too large	Reduce n_estimators for tree-based models
Loading fails	Ensure same library versions as when model was saved
Memory errors	Set n_jobs=1 and reduce n_estimators
Out-of-memory during training	Use n_jobs=1 and reduce feature count
📚 References
scikit-learn Model Persistence

XGBoost Documentation

LightGBM Documentation

Optuna Hyperparameter Optimization

joblib Documentation

Models are automatically saved when running the training pipeline.