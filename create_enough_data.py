"""
Create enough stock data for preprocessing
"""
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from utils.config import STOCK_SYMBOL, DAYS as DAYS_OF_DATA


def create_working_dataset(symbol, days):
    """Create stock data for preprocessing"""
    print(f"📊 Creating {days} days of {symbol} data...")

    # Create date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq='B')

    n = len(dates)
    np.random.seed(42)

    base_price = 150
    returns = np.random.normal(0.0005, 0.02, n)
    prices = base_price * np.exp(np.cumsum(returns))
    trend = np.linspace(0, 50, n)
    prices += trend
    prices = np.maximum(prices, 10)

    df = pd.DataFrame({
        'Date': dates,
        'Open': prices * (1 + np.random.uniform(-0.005, 0.005, n)),
        'High': prices * (1 + np.random.uniform(0.01, 0.025, n)),
        'Low': prices * (1 - np.random.uniform(0.01, 0.025, n)),
        'Close': prices,
        'Adj Close': prices,
        'Volume': np.random.randint(1000000, 50000000, n)
    })

    df['High'] = df[['Open', 'High', 'Close']].max(axis=1) * 1.001
    df['Low'] = df[['Open', 'Low', 'Close']].min(axis=1) * 0.999

    print(f"✅ Created {n} days of data")
    print(f"   Price range: ${df['Close'].min():.2f} to ${df['Close'].max():.2f}")

    return df


# -----------------------------------------
# 🚀 Use config values only — no hardcoding
# -----------------------------------------

df = create_working_dataset(STOCK_SYMBOL, DAYS_OF_DATA)

# Save using stock name from config
os.makedirs('data/raw', exist_ok=True)
file_path = f"data/raw/{STOCK_SYMBOL}_raw.csv"
df.to_csv(file_path, index=False)

print(f"\n💾 Saved to: {file_path}")
print("\n📊 Sample data:")
print(df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].head())

print("\n🚀 Now run: python src/preprocess.py")
