"""
Data loading module for stock price prediction - FIXED for Streamlit Cloud

Fixes vs previous version:
  * Every Yahoo download FAILED and silently fell back to synthetic data:
    `history(..., repair=True)` adds an extra 'Repaired?' column, so the hard-coded
    8-name `df.columns = [...]` assignment raised ValueError on every attempt.
    Columns are now selected by name. (Removed repair/back_adjust; deprecated & need scipy.)
  * Synthetic fallback data was saved as `<SYMBOL>_raw.csv`, i.e. cached as if it were real,
    so every later run served fake data. Synthetic data is never written to the real cache.
  * The data source ('yahoo' / 'cache' / 'synthetic') is reported, so the UI can show it.
  * load_stock_data ignored the requested date range; yfinance `end` is exclusive (+1 day fix).
  * lru_cache on an instance method kept the loader alive and cached failures forever.
  * Timezone-aware dates are normalised to naive dates.
"""
import yfinance as yf
import pandas as pd
import numpy as np
import os
import sys
import logging
import time
from typing import Optional, List, Dict, Tuple

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from utils.config import STOCK_SYMBOL, START_DATE, END_DATE, DATA_RAW_DIR, CACHE_DIR
from utils.helpers import print_section

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']


def _normalise_dates(series: pd.Series) -> pd.Series:
    return pd.to_datetime(series, utc=True).dt.tz_localize(None).dt.normalize()


class StockDataLoader:
    """
    Stock data loader: Yahoo Finance with retries -> optional synthetic fallback
    """

    def __init__(self, symbol: str = STOCK_SYMBOL, cache_dir: str = CACHE_DIR):
        self.symbol = symbol
        self.cache_dir = cache_dir
        self._cache: Dict[str, pd.DataFrame] = {}
        self._info_cache: Dict[str, dict] = {}
        self.last_source: Optional[str] = None
        os.makedirs(cache_dir, exist_ok=True)
        os.makedirs(DATA_RAW_DIR, exist_ok=True)

    def get_stock_info(self, symbol: str) -> dict:
        if symbol in self._info_cache:
            return self._info_cache[symbol]
        try:
            info = yf.Ticker(symbol).info or {}
            result = {
                'symbol': symbol,
                'name': info.get('longName', symbol),
                'sector': info.get('sector', 'Unknown'),
                'industry': info.get('industry', 'Unknown'),
                'market_cap': info.get('marketCap', 0),
                'pe_ratio': info.get('trailingPE', 0),
                'dividend_yield': info.get('dividendYield', 0),
                'beta': info.get('beta', 0),
                'fifty_two_week_high': info.get('fiftyTwoWeekHigh', 0),
                'fifty_two_week_low': info.get('fiftyTwoWeekLow', 0),
                'volume': info.get('volume', 0),
                'avg_volume': info.get('averageVolume', 0),
                'currency': info.get('currency', 'USD'),
            }
            self._info_cache[symbol] = result
            return result
        except Exception as e:
            logger.warning(f"Could not get stock info for {symbol}: {e}")
            return {'symbol': symbol}

    # ------------------------------------------------------------------
    # Sources
    # ------------------------------------------------------------------
    def download_from_yahoo(self, symbol: str, start: str, end: str, max_retries: int = 3) -> Optional[pd.DataFrame]:
        """Download daily OHLCV from Yahoo Finance. Returns None on failure."""
        end_exclusive = (pd.to_datetime(end) + pd.Timedelta(days=1)).strftime('%Y-%m-%d')

        for attempt in range(max_retries):
            try:
                logger.info(f"📥 Downloading {symbol} {start} -> {end} (attempt {attempt + 1})")
                df = yf.Ticker(symbol).history(
                    start=start, end=end_exclusive, interval='1d',
                    auto_adjust=True, actions=False
                )
                if df is None or df.empty:
                    logger.warning(f"No data returned for {symbol} (attempt {attempt + 1})")
                    time.sleep(1 + attempt)
                    continue

                df = df.reset_index()
                if 'Date' not in df.columns:
                    df = df.rename(columns={'Datetime': 'Date', 'index': 'Date'})
                missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
                if missing:
                    logger.warning(f"Yahoo response missing columns {missing}: {list(df.columns)}")
                    return None

                df = df[REQUIRED_COLUMNS].copy()
                df['Date'] = _normalise_dates(df['Date'])
                df = (df.dropna(subset=['Close'])
                        .drop_duplicates(subset='Date', keep='last')
                        .sort_values('Date')
                        .reset_index(drop=True))
                logger.info(f"✅ Downloaded {len(df)} rows ({df['Date'].min().date()} -> {df['Date'].max().date()})")
                return df

            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed for {symbol}: {e}")
                time.sleep(2 ** attempt)

        return None

    def generate_synthetic_data(self, symbol: str, start: str = START_DATE, end: str = END_DATE) -> pd.DataFrame:
        """Synthetic fallback covering the requested range. NEVER written to the real cache."""
        from create_enough_data import StockDataGenerator
        days = max(365, (pd.to_datetime(end) - pd.to_datetime(start)).days)
        generator = StockDataGenerator(symbol=symbol, days=days, end_date=pd.to_datetime(end), verbose=False)
        df = generator.create_working_dataset()
        return df[REQUIRED_COLUMNS].copy()

    def fetch(self, symbol: str, start: str = START_DATE, end: str = END_DATE,
              allow_synthetic: bool = True, save: bool = True) -> Tuple[Optional[pd.DataFrame], str]:
        """
        Main entry point. Returns (dataframe, source) where source is 'yahoo' or 'synthetic'.
        """
        symbol = symbol.upper().strip()
        df = self.download_from_yahoo(symbol, str(start), str(end))
        if df is not None and len(df) > 0:
            if save:
                self._save_real(df, symbol)
            self.last_source = 'yahoo'
            return df, 'yahoo'

        if not allow_synthetic:
            self.last_source = 'none'
            return None, 'none'

        logger.warning(f"⚠️ Yahoo Finance failed for {symbol}. Using SYNTHETIC data.")
        self.last_source = 'synthetic'
        return self.generate_synthetic_data(symbol, str(start), str(end)), 'synthetic'

    def _save_real(self, df: pd.DataFrame, symbol: str):
        try:
            os.makedirs(DATA_RAW_DIR, exist_ok=True)
            filepath = os.path.join(DATA_RAW_DIR, f"{symbol}_raw.csv")
            df.to_csv(filepath, index=False)
            logger.info(f"💾 Data saved to: {filepath}")
        except OSError as e:  # read-only filesystems must not break the app
            logger.warning(f"Could not save data: {e}")

    # ------------------------------------------------------------------
    # Legacy-compatible API
    # ------------------------------------------------------------------
    def download_stock_data(self, symbol: str = STOCK_SYMBOL, start: str = START_DATE, end: str = END_DATE,
                            save: bool = True, max_retries: int = 3) -> Optional[pd.DataFrame]:
        df, _ = self.fetch(symbol, start, end, allow_synthetic=True, save=save)
        return df

    def load_stock_data(self, symbol: str = STOCK_SYMBOL, force_download: bool = False,
                        start: str = START_DATE, end: str = END_DATE) -> Optional[pd.DataFrame]:
        """Load REAL cached CSV if present (and not forced), else download (synthetic fallback)."""
        symbol = symbol.upper().strip()
        cache_key = f"{symbol}_{start}_{end}"
        if cache_key in self._cache and not force_download:
            return self._cache[cache_key]

        filepath = os.path.join(DATA_RAW_DIR, f"{symbol}_raw.csv")
        if os.path.exists(filepath) and not force_download:
            df = pd.read_csv(filepath)
            df['Date'] = _normalise_dates(df['Date'])
            df = df[[c for c in REQUIRED_COLUMNS if c in df.columns]]
            mask = (df['Date'] >= pd.to_datetime(start)) & (df['Date'] <= pd.to_datetime(end))
            if mask.sum() > 0:
                self.last_source = 'cache'
                df = df[mask].reset_index(drop=True)
                self._cache[cache_key] = df
                return df

        df, _ = self.fetch(symbol, start, end)
        if df is not None:
            self._cache[cache_key] = df
        return df

    def get_multiple_stocks(self, symbols: List[str], start: str = START_DATE,
                            end: str = END_DATE) -> Dict[str, pd.DataFrame]:
        stocks_data = {}
        for symbol in symbols:
            df = self.download_stock_data(symbol=symbol, start=start, end=end, save=True)
            if df is not None:
                stocks_data[symbol] = df
        logger.info(f"✅ Loaded {len(stocks_data)} stocks")
        return stocks_data

    def get_data_info(self, df: pd.DataFrame) -> None:
        print("\n" + "=" * 70 + "\n  DATASET INFORMATION\n" + "=" * 70)
        print(f"\n📊 Shape: {df.shape}")
        print(f"📅 Date Range: {df['Date'].min()} to {df['Date'].max()}")
        print(f"📈 Trading Days: {len(df)} days")
        print(f"\n🔍 Data Quality:")
        print(f"  - Missing Values: {df.isnull().sum().sum()}")
        print(f"  - Duplicate Rows: {df.duplicated().sum()}")
        print(f"  - Memory Usage: {df.memory_usage(deep=True).sum() / 1024 ** 2:.2f} MB")
        print(f"\n📋 Columns: {list(df.columns)}")
        print(f"\n📊 Basic Statistics:")
        print(df.describe().round(2))

        if 'Close' in df.columns and len(df) > 1:
            returns = df['Close'].pct_change().dropna()
            print(f"\n📈 Returns Statistics:")
            print(f"  Mean Daily Return: {returns.mean() * 100:.3f}%")
            print(f"  Std Daily Return: {returns.std() * 100:.3f}%")
            print(f"  Skewness: {returns.skew():.3f}")
            print(f"  Kurtosis: {returns.kurtosis():.3f}")
            print(f"  Total Return: {((df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1) * 100:.2f}%")

    def validate_data(self, df: pd.DataFrame) -> Dict[str, bool]:
        tol = 1e-6  # float rounding in adjusted prices
        results = {
            'has_data': len(df) > 0,
            'has_complete_dates': bool(df['Date'].notnull().all()),
            'has_positive_prices': bool((df['Close'] > 0).all()),
            'has_positive_volume': bool((df['Volume'] >= 0).all()),
            'has_ohlc_valid': bool(
                (df['High'] + tol >= df['Low']).all()
                and (df['High'] + tol >= df[['Open', 'Close']].max(axis=1)).all()
                and (df['Low'] - tol <= df[['Open', 'Close']].min(axis=1)).all()
            ),
            'no_missing_values': bool(df[REQUIRED_COLUMNS].isnull().sum().sum() == 0),
            'sufficient_data': len(df) >= 300,  # ~200 rows are consumed by indicator warm-up
        }
        results['is_valid'] = all(results.values())
        return results


# ============================================================================
# LEGACY SUPPORT FUNCTIONS
# ============================================================================

def download_stock_data(symbol=STOCK_SYMBOL, start=START_DATE, end=END_DATE, save=True):
    return StockDataLoader().download_stock_data(symbol, start, end, save)


def load_stock_data(symbol=STOCK_SYMBOL, force_download=False, start=START_DATE, end=END_DATE):
    return StockDataLoader().load_stock_data(symbol, force_download, start, end)


def get_multiple_stocks(symbols, start=START_DATE, end=END_DATE):
    return StockDataLoader().get_multiple_stocks(symbols, start, end)


def get_data_info(df):
    StockDataLoader().get_data_info(df)


if __name__ == "__main__":
    print_section("Testing Data Loader")
    loader = StockDataLoader()
    df, source = loader.fetch(STOCK_SYMBOL, START_DATE, END_DATE)
    print(f"Source: {source}")
    if df is not None:
        loader.get_data_info(df)
        for key, value in loader.validate_data(df).items():
            print(f"  {key:20s}: {value}")