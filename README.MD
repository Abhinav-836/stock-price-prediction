# Stock Price Prediction

Download a stock's history from Yahoo Finance, build ~340 technical features, train eight ML models on a chronological split, and forecast the next 1–90 trading days — from the command line or an interactive Streamlit dashboard. Works for US tickers (`AAPL`, `MSFT`) and Indian ones (`RELIANCE.NS`).

The project is honest about what's possible: it predicts **next-day returns**, compares every model against a "no change" baseline, and shows when a model isn't beating it.

**Live:** [stocks-price-predictions.streamlit.app](https://stocks-price-predictions.streamlit.app)

## How it works

```
Yahoo Finance (with retries + local cache)  ─or─  synthetic data (offline)
  ─► extra ~200 days fetched for indicator warm-up
  ─► features: price, moving averages, RSI / MACD / Bollinger / ATR, volume (OBV, VWAP, MFI),
               rolling statistics, candlestick patterns, interactions, 10 lags of each
  ─► target: next trading day's return in %
  ─► chronological 80 / 20 split (never shuffled) + 3-fold time-series cross-validation
  ─► 8 models ─► best test R² wins
  ─► recursive forecast: predict a return → update the price → recompute features → repeat
```

**Why returns, not prices.** Earlier versions predicted the raw price. Tree models (XGBoost, LightGBM, Random Forest) can't output a price higher than anything they saw in training, so on a time-ordered test set they failed badly. Predicting returns removes that ceiling, and multi-day price paths are rebuilt from the predicted returns.

**Models:** Ridge, Lasso, ElasticNet, Random Forest, Gradient Boosting, XGBoost, LightGBM and SVR. Linear models and SVR are scaled inside their own pipeline, so the scaler never sees test data.

## What to expect

Daily stock returns are close to random. Realistic results on real data:

| Metric | Realistic range | Meaning |
|---|---|---|
| Test R² | about −0.1 to 0.05 | Near 0 is normal for daily returns |
| Directional accuracy | 48–56 % | 50 % is a coin flip |
| Test RMSE vs baseline | slightly below = useful | Baseline = always predict 0 % change |

A model is only worth using if it beats the zero-change baseline across several symbols and periods. A very high R² on daily data almost always means data leakage, not skill. The dashboard's **Model Performance** tab shows this comparison for every run.

## Dashboard

```bash
streamlit run dashboard.py
```

Pick a symbol and a forecast horizon (1–90 business days) in the sidebar, optionally switching to offline synthetic data or forcing a fresh download. Tabs:

- **Forecast** — the predicted price path; the best model is selected by default and you can switch to any of the others
- **Model Performance** — R², RMSE, MAE and directional accuracy for all eight models vs. the baseline
- **Feature Importance** — which features the tree models relied on
- **Risk & Returns** — buy-and-hold Sharpe ratio, drawdown, return distribution
- **Data** — the raw and engineered data

The dashboard shows where the data came from (Yahoo, cache or synthetic), so a failed download never silently turns into fake data.

## Command line

```bash
python main.py                                    # AAPL, 2010 → today, 30-day forecast
python main.py --symbol MSFT --days 60
python main.py --symbol RELIANCE.NS --start-date 2019-01-01
python main.py --model ridge --no-cv              # specific model, skip cross-validation
python main.py --synthetic                        # offline demo data
```

Trained models are saved to `models/` as `<name>.pkl` with a `<name>_metrics.json` next to each. See [models/README.md](models/README.md) for loading them in your own code.

## Run locally

Needs Python 3.10–3.13.

```bash
git clone https://github.com/Abhinav-836/stock-price-prediction.git
cd stock-price-prediction
python -m venv venv
venv\Scripts\activate                 # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
streamlit run dashboard.py
```

Optional extras, not needed for the dashboard:

```bash
pip install pytest coverage              # tests
pip install optuna                       # hyperparameter tuning: python src/auto_train.py
pip install ta                           # src/advanced_features.py
pip install statsmodels seaborn          # notebooks/
pip install tensorflow                   # experimental LSTM: src/lstm_model.py
```

## Tests

```bash
python tests/run_tests.py
```

29 tests covering preprocessing (including checks that no future data leaks into features) and model training. Reports are written to `reports/`.

## Project structure

```
main.py                 CLI pipeline
dashboard.py            Streamlit app
src/
  data_loader.py        Yahoo Finance download, retries, cache, synthetic fallback
  preprocess.py         feature engineering and targets
  train_model.py        the eight models, time-series CV, chronological split
  pipeline.py           load → features → train → pick best → forecast
  evaluate.py           metrics incl. directional accuracy and baseline
  auto_train.py         Optuna tuning (optional)
  lstm_model.py         experimental LSTM (optional, TensorFlow)
utils/config.py         default symbol, dates, model parameters
models/                 saved models (git-ignored)
notebooks/              exploration, feature analysis, experiments
tests/                  unit tests + runner
```

## Limitations

- Past prices alone carry very little signal about tomorrow's return; expect results close to the baseline.
- Recursive forecasts compound errors, so the further out the forecast, the less it means.
- Yahoo Finance rate-limits; if a download fails, retry or use the cached data.

> Educational project. Not financial advice.

---

Built by **Abhinav Ashutosh** · [GitHub](https://github.com/Abhinav-836)