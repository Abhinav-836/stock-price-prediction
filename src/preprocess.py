"""
Data preprocessing and feature engineering module
"""
import pandas as pd
import numpy as np
import os
import sys
from sklearn.preprocessing import MinMaxScaler

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import (
    DATA_PROCESSED_DIR,
    MOVING_AVERAGES,
    STOCK_SYMBOL,
    DATA_RAW_DIR
)


# ===================================================================
#  CLEANING RAW CSV COLUMNS (Investing.com, Yahoo, etc.)
# ===================================================================
def clean_numeric_columns(df):
    """
    Cleans numeric-like columns that contain symbols, letters, commas,
    %, M/K/B multipliers, dashes, or non-numeric characters.

    Args:
        df: DataFrame with raw columns

    Returns:
        Cleaned DataFrame with numeric columns fixed
    """
    df = df.copy()

    numeric_cols = ["Price", "Open", "Close", "High", "Low", "Volume", "Change %"]

    for col in numeric_cols:
        if col not in df.columns:
            continue

        # Convert Volume values like 1.2M, 450K, etc.
        if col == "Volume":
            df[col] = (
                df[col].astype(str)
                .str.replace(",", "")
                .str.replace("-", "0")
                .str.replace("M", "*1e6")
                .str.replace("K", "*1e3")
                .str.replace("B", "*1e9")
            )

            def convert(x):
                try:
                    if "*" in str(x):
                        return eval(str(x))
                    return float(x)
                except:
                    return np.nan

            df[col] = df[col].apply(convert)
            continue

        # Remove % sign from Change %
        if col == "Change %":
            df[col] = df[col].astype(str).str.replace("%", "")

        # Remove commas and whitespace from all numeric columns
        df[col] = (
            df[col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.strip()
        )

        # Convert into numeric
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


# ===================================================================
#  FEATURE ENGINEERING
# ===================================================================

def create_date_features(df):
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    df["year"] = df["Date"].dt.year
    df["month"] = df["Date"].dt.month
    df["day"] = df["Date"].dt.day
    df["day_of_week"] = df["Date"].dt.dayofweek
    df["quarter"] = df["Date"].dt.quarter
    df["is_quarter_end"] = np.where(df["month"] % 3 == 0, 1, 0)
    df["is_year_end"] = np.where(df["month"] == 12, 1, 0)

    return df


def create_price_features(df):
    df = df.copy()

    df["open_close"] = df["Open"] - df["Close"]
    df["high_low"] = df["High"] - df["Low"]
    df["price_change"] = df["Close"].diff()
    df["price_change_pct"] = df["Close"].pct_change() * 100
    df["volume_change"] = df["Volume"].diff()
    df["volume_change_pct"] = df["Volume"].pct_change() * 100

    return df


def create_moving_averages(df, windows=MOVING_AVERAGES):
    df = df.copy()
    for window in windows:
        df[f"MA_{window}"] = df["Close"].rolling(window=window).mean()
        df[f"EMA_{window}"] = df["Close"].ewm(span=window, adjust=False).mean()
    return df


def create_technical_indicators(df):
    df = df.copy()

    # RSI
    delta = df["Close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # MACD
    exp1 = df["Close"].ewm(span=12, adjust=False).mean()
    exp2 = df["Close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = exp1 - exp2
    df["MACD_signal"] = df["MACD"].ewm(span=9, adjust=False).mean()

    # Bollinger Bands
    df["BB_middle"] = df["Close"].rolling(20).mean()
    std = df["Close"].rolling(20).std()
    df["BB_upper"] = df["BB_middle"] + (std * 2)
    df["BB_lower"] = df["BB_middle"] - (std * 2)
    df["BB_width"] = df["BB_upper"] - df["BB_lower"]

    # Volatility
    df["volatility"] = df["Close"].rolling(20).std()

    return df


def create_lag_features(df, n_lags=5):
    df = df.copy()
    for i in range(1, n_lags + 1):
        df[f"close_lag_{i}"] = df["Close"].shift(i)
    return df


def create_target(df, prediction_days=1):
    df = df.copy()
    df["target"] = df["Close"].shift(-prediction_days)
    df["target_class"] = np.where(
        df["Close"].shift(-prediction_days) > df["Close"], 1, 0
    )
    return df


# ===================================================================
#  PREPROCESSING PIPELINE
# ===================================================================

def preprocess_data(df, save=True):
    print("\n🔧 Starting data preprocessing...")

    df = df.copy()

    # Step 1: Clean numeric columns first
    print("  ➤ Cleaning numeric columns...")
    df = clean_numeric_columns(df)

    # Step 2: Feature engineering
    print("  ➤ Creating date features...")
    df = create_date_features(df)

    print("  ➤ Creating price features...")
    df = create_price_features(df)

    print("  ➤ Creating moving averages...")
    df = create_moving_averages(df)

    print("  ➤ Creating technical indicators...")
    df = create_technical_indicators(df)

    print("  ➤ Creating lag features...")
    df = create_lag_features(df, n_lags=5)

    print("  ➤ Creating target variable...")
    df = create_target(df)

    # Remove NaN values caused by rolling windows
    print("  ➤ Handling missing values...")
    initial = len(df)
    df = df.dropna()
    print(f"    Removed {initial - len(df)} rows")

    # Save processed file
    if save:
        filepath = os.path.join(DATA_PROCESSED_DIR, f"{STOCK_SYMBOL}_processed.csv")
        df.to_csv(filepath, index=False)
        print(f"\n✅ Processed data saved to: {filepath}")

    print(f"✅ Preprocessing complete! Final shape: {df.shape}")

    return df


def scale_features(X_train, X_test, feature_cols):
    scaler = MinMaxScaler()

    X_train_scaled = X_train.copy()
    X_test_scaled = X_test.copy()

    X_train_scaled[feature_cols] = scaler.fit_transform(X_train[feature_cols])
    X_test_scaled[feature_cols] = scaler.transform(X_test[feature_cols])

    return X_train_scaled, X_test_scaled, scaler


# ===================================================================
#  DEBUG MODE
# ===================================================================
if __name__ == "__main__":
    print("Testing Preprocessing Pipeline...")

    filepath = os.path.join(DATA_RAW_DIR, f"{STOCK_SYMBOL}_raw.csv")

    if os.path.exists(filepath):
        df = pd.read_csv(filepath)

        df["Date"] = pd.to_datetime(df["Date"])

        df_processed = preprocess_data(df)

        print("\n" + "=" * 60)
        print("  PROCESSED DATA SAMPLE")
        print("=" * 60)
        print(df_processed.head())

    else:
        print(f"❌ Raw data not found at: {filepath}")
