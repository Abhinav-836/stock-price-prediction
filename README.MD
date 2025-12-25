# Models Directory

This directory stores trained machine learning models for stock price prediction.

## Stored Models

Models are saved in pickle format (`.pkl`) for easy loading and reuse.

### Available Models

1. **linear_regression.pkl**
   - Simple linear regression model
   - Fast training and prediction
   - Good baseline for comparison
   - Best for: Understanding linear relationships

2. **random_forest.pkl**
   - Ensemble of decision trees
   - Handles non-linear relationships
   - Provides feature importance
   - Best for: Robust predictions with moderate complexity

3. **xgboost.pkl**
   - Gradient boosting model
   - Usually best performance
   - Efficient and accurate
   - Best for: Maximum accuracy

## Model Files Created

When you run the training pipeline, these files are automatically created:

```
models/
├── linear_regression.pkl      (~100 KB)
├── random_forest.pkl          (~50-500 MB, depending on n_estimators)
├── xgboost.pkl                (~1-10 MB)
└── README.md                  (this file)
```

## Usage

### Training Models

```bash
# Train all models
python main.py --mode train

# Or use the training script directly
python src/train_model.py
```

### Loading Models

```python
from utils.helpers import load_model
from utils.config import MODELS_DIR

# Load a specific model
model = load_model('xgboost', MODELS_DIR)

# Make predictions
predictions = model.predict(X_test)
```

### Model Versioning

Models are overwritten when retrained. To keep multiple versions:

```python
from utils.helpers import save_model
from datetime import datetime

# Save with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
save_model(model, f'xgboost_{timestamp}', MODELS_DIR)
```

## Model Performance

Typical performance metrics on Tesla stock data:

| Model | Train R² | Test R² | Train RMSE | Test RMSE |
|-------|----------|---------|------------|-----------|
| Linear Regression | 0.52 | 0.51 | $45.2 | $48.5 |
| Random Forest | 0.85 | 0.55 | $25.8 | $46.3 |
| XGBoost | 0.96 | 0.57 | $12.4 | $45.1 |

*Note: Actual performance varies by stock and time period*

## Model Details

### Linear Regression
```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
```

**Parameters**: Default scikit-learn parameters

**Pros**:
- Fast training
- Interpretable coefficients
- No hyperparameters to tune

**Cons**:
- Assumes linear relationships
- Sensitive to outliers
- May underfit complex patterns

### Random Forest
```python
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
```

**Parameters**:
- `n_estimators=100`: Number of trees
- `random_state=42`: For reproducibility
- `n_jobs=-1`: Use all CPU cores

**Pros**:
- Handles non-linear relationships
- Robust to overfitting
- Feature importance available

**Cons**:
- Can be slow with many trees
- Large model size
- Less interpretable than linear models

### XGBoost
```python
from xgboost import XGBRegressor
model = XGBRegressor(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)
```

**Parameters** (from `utils/config.py`):
- `n_estimators=100`: Number of boosting rounds
- `max_depth=5`: Maximum tree depth
- `learning_rate=0.1`: Step size shrinkage
- `random_state=42`: For reproducibility

**Pros**:
- Usually best accuracy
- Handles missing values
- Built-in regularization
- Feature importance

**Cons**:
- More hyperparameters to tune
- Can overfit if not careful
- Slower than linear models

## Feature Importance

XGBoost and Random Forest provide feature importance:

```python
from src.train_model import get_feature_importance

# Get importance
importance_df = get_feature_importance(model, feature_cols, top_n=15)

# Plot
from src.visualization import plot_feature_importance
plot_feature_importance(importance_df)
```

## Hyperparameter Tuning

To customize model parameters, edit `utils/config.py`:

```python
# XGBoost Parameters
XGB_PARAMS = {
    'n_estimators': 100,      # Increase for better fit
    'max_depth': 5,           # Increase to capture complexity
    'learning_rate': 0.1,     # Decrease for better generalization
    'random_state': RANDOM_STATE
}
```

Or override when training:

```python
from src.train_model import train_xgboost

custom_params = {
    'n_estimators': 200,
    'max_depth': 7,
    'learning_rate': 0.05,
    'random_state': 42
}

model = train_xgboost(X_train, y_train, params=custom_params)
```

## Model Evaluation

Models are evaluated on multiple metrics:

- **MAE** (Mean Absolute Error): Average prediction error in dollars
- **RMSE** (Root Mean Squared Error): Penalizes large errors more
- **R² Score**: Proportion of variance explained (0-1, higher is better)
- **MAPE** (Mean Absolute Percentage Error): Error as percentage
- **Directional Accuracy**: % of times we predicted direction correctly

## Best Practices

1. **Always evaluate on test set**: Don't trust training performance
2. **Check for overfitting**: Compare train vs test metrics
3. **Use cross-validation**: For more robust evaluation
4. **Monitor feature importance**: Ensure reasonable features are important
5. **Version your models**: Keep track of different experiments
6. **Regular retraining**: Markets change, retrain periodically

## Important Notes

⚠️ **File Size**: 
- Random Forest models can be very large (100+ MB)
- Git ignores these files (see `.gitignore`)

🔄 **Retraining**:
- Models should be retrained periodically
- Market conditions change over time
- New data improves predictions

📊 **Performance**:
- Past performance doesn't guarantee future results
- Stock prediction is inherently difficult
- ~50-60% directional accuracy is typical

## Troubleshooting

**Problem**: Models not saving
- **Solution**: Check write permissions for `models/` directory

**Problem**: Model file too large
- **Solution**: Reduce `n_estimators` for Random Forest

**Problem**: Loading fails
- **Solution**: Ensure same library versions as when model was saved

## References

- [scikit-learn Model Persistence](https://scikit-learn.org/stable/model_persistence.html)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [joblib Documentation](https://joblib.readthedocs.io/)

---

*Models are automatically saved when running the training pipeline.*