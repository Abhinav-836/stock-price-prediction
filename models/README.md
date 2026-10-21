# Models Directory

Trained models are written here by `python main.py` (`<name>.pkl`, joblib) together with
`<name>_metrics.json`. The Streamlit dashboard does **not** write here; it keeps models in memory.
Everything except this README is git-ignored.

## What the models predict

**Target:** `target_return_1d`, the next trading day's return in **%**.

Earlier versions predicted the raw price. Tree models (XGBoost, LightGBM, Random Forest,
Gradient Boosting) cannot output values outside the price range seen in training, so on a
chronological test split they produced the large negative R² values the old table showed.
Returns don't have that problem. Multi-day forecasts are built recursively: predict a return,
update the price history, recompute all features, repeat.

## Models trained automatically

| Key | Model | Notes |
|-----|-------|-------|
| `ridge` | StandardScaler + Ridge | linear baseline |
| `lasso` | StandardScaler + Lasso | sparse linear |
| `elasticnet` | StandardScaler + ElasticNet | |
| `random_forest` | RandomForestRegressor | 50 trees, depth 8 |
| `gradient_boosting` | GradientBoostingRegressor | |
| `xgboost` | XGBRegressor | params from `utils/config.py` |
| `lightgbm` | LGBMRegressor | params from `utils/config.py` |
| `svr` | StandardScaler + SVR | scaler inside the pipeline |

Plain `LinearRegression` is no longer trained automatically: with ~340 collinear features
it is numerically unstable. `train_linear_regression()` still exists in `src/train_model.py`.

## Usage

```bash
python main.py                                  # default symbol, config date range
python main.py --symbol MSFT --days 60
python main.py --symbol RELIANCE.NS --start-date 2019-01-01
python main.py --synthetic                      # offline demo data
python main.py --model ridge --no-cv
python src/auto_train.py                        # Optuna tuning (pip install optuna)
streamlit run dashboard.py
```

```python
from utils.helpers import load_model
from utils.config import MODELS_DIR

model = load_model('xgboost', MODELS_DIR)
model, metadata = load_model('xgboost', MODELS_DIR, with_metadata=True)  # metadata may be None
```

## How to read the metrics

Daily returns are close to random, so realistic results are:

| Metric | Realistic range | Meaning |
|--------|-----------------|---------|
| Test R² | about -0.1 to 0.05 | near 0 is normal for daily returns |
| Directional accuracy | 48-56 % | 50 % = coin flip |
| Test RMSE vs baseline | slightly below = useful | baseline = always predict 0 % change |

A model is only worth using if it beats the zero-change baseline consistently, across symbols
and time periods. The dashboard's **Model Performance** tab shows this comparison.
Very high R² on daily data usually means leakage, not skill.

## Retraining

- Retrain when the date range changes; markets drift.
- Models must be loaded with the same library versions they were saved with.
- Saved models expect exactly the feature columns produced by `src/preprocess.py`
  (`get_feature_columns`). Changing feature engineering requires retraining.

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `Model not found` | run `python main.py` first |
| Dashboard says "Synthetic data" | bad ticker, Yahoo rate limit, or old yfinance; check `pip show yfinance` (needs ≥ 0.2.54) |
| "Only N usable trading days" | choose an earlier start date (≈200 rows go to indicator warm-up, fetched automatically) |
| Memory errors | keep `n_jobs=1`, reduce `n_estimators` |

> Educational project. Not financial advice.