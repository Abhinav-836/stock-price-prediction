"""
Create synthetic stock data - FIXED

Fixes vs previous version:
  * Every symbol produced IDENTICAL data (global np.random.seed(42) for all symbols).
    The seed is now derived from the symbol, and a local Generator is used so the
    global numpy random state of the app is not reset.
  * Can generate data ending on a chosen date (used by the dashboard's date range).
  * Path setup works whether this file is run from the project root or imported.
"""
import pandas as pd
import numpy as np
import os
import sys
import zlib
import argparse
from datetime import datetime, timedelta
from typing import Optional, List, Dict
import warnings
warnings.filterwarnings('ignore')

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from utils.config import STOCK_SYMBOL, DAYS, DATA_RAW_DIR
from utils.helpers import print_section


def symbol_seed(symbol: str) -> int:
    return zlib.crc32(symbol.upper().encode()) % (2 ** 31)


class StockDataGenerator:
    """
    Synthetic stock data generator (geometric Brownian motion + OHLC candles)
    """

    def __init__(self, symbol: str = STOCK_SYMBOL, days: int = DAYS, seed: Optional[int] = None,
                 end_date: Optional[datetime] = None, verbose: bool = True):
        self.symbol = symbol
        self.days = days
        self.seed = symbol_seed(symbol) if seed is None else seed
        self.end_date = pd.Timestamp(end_date) if end_date is not None else pd.Timestamp(datetime.now()).normalize()
        self.verbose = verbose
        self.rng = np.random.default_rng(self.seed)

    def generate_price_path(self, n: int, base_price: Optional[float] = None, volatility: float = 0.018,
                            drift: float = 0.0004) -> np.ndarray:
        if base_price is None:
            base_price = float(self.rng.uniform(40, 400))
        returns = self.rng.normal(drift - 0.5 * volatility ** 2, volatility, n)
        prices = base_price * np.exp(np.cumsum(returns))
        return np.maximum(prices, 1.0)

    def generate_ohlc(self, prices: np.ndarray, spread: float = 0.005) -> pd.DataFrame:
        n = len(prices)
        prev_close = np.concatenate([[prices[0]], prices[:-1]])
        open_prices = prev_close * (1 + self.rng.normal(0, spread, n))

        body_top = np.maximum(open_prices, prices)
        body_bot = np.minimum(open_prices, prices)
        high_prices = body_top * (1 + self.rng.uniform(0, spread * 2, n))
        low_prices = body_bot * (1 - self.rng.uniform(0, spread * 2, n))

        volumes = (self.rng.integers(1_000_000, 50_000_000, n) * self.rng.uniform(0.8, 1.2, n)).astype(np.int64)

        return pd.DataFrame({'Open': open_prices, 'High': high_prices, 'Low': low_prices,
                             'Close': prices, 'Volume': volumes})

    def generate_weekday_data(self, start_date, end_date) -> pd.DataFrame:
        dates = pd.bdate_range(start=start_date, end=end_date)
        ohlc = self.generate_ohlc(self.generate_price_path(len(dates)))
        df = pd.DataFrame({'Date': dates})
        df[['Open', 'High', 'Low', 'Close', 'Volume']] = ohlc.values
        df['Volume'] = df['Volume'].astype(np.int64)
        df['Adj Close'] = df['Close']
        return df[['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']]

    def create_working_dataset(self) -> pd.DataFrame:
        if self.verbose:
            print_section(f"Creating {self.days} days of synthetic {self.symbol} data")

        start_date = self.end_date - timedelta(days=self.days)
        df = self.generate_weekday_data(start_date, self.end_date)

        if self.verbose:
            returns = df['Close'].pct_change().dropna()
            print(f"✅ Created {len(df)} trading days of data")
            print(f"   Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
            print(f"   Price range: ${df['Close'].min():.2f} to ${df['Close'].max():.2f}")
            print(f"   Mean daily return: {returns.mean() * 100:.3f}%  Volatility: {returns.std() * 100:.3f}%")
        return df

    def save_dataset(self, df: pd.DataFrame, suffix: str = "_synthetic") -> str:
        """Saved with a '_synthetic' suffix so it is never mistaken for real cached data"""
        os.makedirs(DATA_RAW_DIR, exist_ok=True)
        file_path = os.path.join(DATA_RAW_DIR, f"{self.symbol}{suffix}_raw.csv")
        df.to_csv(file_path, index=False)
        if self.verbose:
            print(f"\n💾 Saved to: {file_path}")
        return file_path

    def create_multiple_stocks(self, symbols: List[str], days: int = None) -> Dict[str, pd.DataFrame]:
        days = days or self.days
        result = {}
        for symbol in symbols:
            generator = StockDataGenerator(symbol=symbol, days=days, end_date=self.end_date, verbose=self.verbose)
            df = generator.create_working_dataset()
            generator.save_dataset(df)
            result[symbol] = df
        return result


# ============================================================================
# LEGACY SUPPORT FUNCTIONS
# ============================================================================

def create_working_dataset(symbol, days):
    return StockDataGenerator(symbol=symbol, days=days).create_working_dataset()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create synthetic stock data for testing')
    parser.add_argument('--symbol', type=str, default=STOCK_SYMBOL)
    parser.add_argument('--symbols', type=str, help='Comma-separated list of symbols')
    parser.add_argument('--days', type=int, default=DAYS)
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(',')] if args.symbols else [args.symbol]
    generator = StockDataGenerator(symbol=symbols[0], days=args.days, seed=args.seed)

    if len(symbols) == 1:
        generator.save_dataset(generator.create_working_dataset())
    else:
        generator.create_multiple_stocks(symbols, days=args.days)

    print("\n✅ DATA GENERATION COMPLETE\n🚀 Now you can run:  python main.py --synthetic")