"""
Data loading module for stock price prediction - Enhanced Version
"""
import yfinance as yf
import pandas as pd
import numpy as np
import os
import sys
import logging
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Union
from functools import lru_cache
import requests
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import STOCK_SYMBOL, START_DATE, END_DATE, DATA_RAW_DIR, CACHE_DIR
from utils.helpers import Timer, print_section

# Setup logging
logger = logging.getLogger(__name__)


class StockDataLoader:
    """
    Advanced stock data loader with caching, retries, and multiple data sources
    """
    
    def __init__(self, symbol: str = STOCK_SYMBOL, cache_dir: str = CACHE_DIR):
        self.symbol = symbol
        self.cache_dir = cache_dir
        self._cache = {}
        os.makedirs(cache_dir, exist_ok=True)
        
    @lru_cache(maxsize=128)
    def get_stock_info(self, symbol: str) -> dict:
        """Get stock information with caching"""
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            return {
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
                'avg_volume': info.get('averageVolume', 0)
            }
        except Exception as e:
            logger.warning(f"Could not get stock info for {symbol}: {e}")
            return {'symbol': symbol}
    
    def download_stock_data(
        self, 
        symbol: str = STOCK_SYMBOL, 
        start: str = START_DATE, 
        end: str = END_DATE,
        save: bool = True,
        max_retries: int = 3
    ) -> Optional[pd.DataFrame]:
        """
        Download stock data from Yahoo Finance with retries
        
        Args:
            symbol: Stock ticker symbol
            start: Start date (YYYY-MM-DD format)
            end: End date (YYYY-MM-DD format)
            save: Whether to save data to CSV
            max_retries: Maximum number of retry attempts
            
        Returns:
            DataFrame with stock data
        """
        logger.info(f"📥 Downloading {symbol} data from {start} to {end}...")
        
        for attempt in range(max_retries):
            try:
                # Download data
                stock = yf.Ticker(symbol)
                df = stock.history(start=start, end=end)
                
                if df.empty:
                    logger.warning(f"No data found for {symbol} (attempt {attempt + 1})")
                    time.sleep(2)
                    continue
                
                # Reset index to make Date a column
                df.reset_index(inplace=True)
                
                # Standardize column names
                df.columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume', 'Dividends', 'Stock Splits']
                
                # Add additional metrics
                df['Symbol'] = symbol
                df['Date'] = pd.to_datetime(df['Date'])
                
                logger.info(f"✅ Downloaded {len(df)} rows of data")
                logger.info(f"📊 Date range: {df['Date'].min()} to {df['Date'].max()}")
                
                # Save to CSV if requested
                if save:
                    filepath = os.path.join(DATA_RAW_DIR, f"{symbol}_raw.csv")
                    df.to_csv(filepath, index=False)
                    logger.info(f"💾 Data saved to: {filepath}")
                
                return df
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1} failed: {e}")
                time.sleep(2 ** attempt)  # Exponential backoff
        
        logger.error(f"❌ Failed to download data after {max_retries} attempts")
        return None
    
    def load_stock_data(self, symbol: str = STOCK_SYMBOL, force_download: bool = False) -> Optional[pd.DataFrame]:
        """Load stock data from CSV or download if not available"""
        cache_key = f"{symbol}_data"
        if cache_key in self._cache and not force_download:
            logger.info(f"📦 Returning cached data for {symbol}")
            return self._cache[cache_key]

        filepath = os.path.join(DATA_RAW_DIR, f"{symbol}_raw.csv")

        if os.path.exists(filepath) and not force_download:
            logger.info(f"📂 Loading existing data from: {filepath}")
            df = pd.read_csv(filepath)
            df['Date'] = pd.to_datetime(df['Date'])

            if 'Symbol' in df.columns:
                df = df.drop('Symbol', axis=1)

            logger.info(f"✅ Loaded {len(df)} rows")
            self._cache[cache_key] = df
            return df

        df = self.download_stock_data(symbol=symbol, save=True)
        if df is not None:
            if 'Symbol' in df.columns:
                df = df.drop('Symbol', axis=1)
            self._cache[cache_key] = df
        return df
    
    def get_multiple_stocks(
        self, 
        symbols: List[str], 
        start: str = START_DATE, 
        end: str = END_DATE,
        parallel: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Download data for multiple stock symbols
        
        Args:
            symbols: List of stock ticker symbols
            start: Start date
            end: End date
            parallel: Whether to download in parallel (currently sequential)
            
        Returns:
            Dictionary with symbol as key and DataFrame as value
        """
        stocks_data = {}
        
        if parallel:
            # Parallel download (simplified - use ThreadPoolExecutor for production)
            for symbol in symbols:
                logger.info(f"\n📥 Processing {symbol}...")
                df = self.download_stock_data(symbol=symbol, start=start, end=end, save=True)
                if df is not None:
                    stocks_data[symbol] = df
        else:
            for symbol in symbols:
                logger.info(f"\n📥 Processing {symbol}...")
                df = self.download_stock_data(symbol=symbol, start=start, end=end, save=True)
                if df is not None:
                    stocks_data[symbol] = df
        
        logger.info(f"\n✅ Successfully loaded {len(stocks_data)} stocks")
        return stocks_data
    
    def get_data_info(self, df: pd.DataFrame) -> None:
        """
        Display comprehensive information about the dataset
        """
        print("\n" + "="*70)
        print("  DATASET INFORMATION")
        print("="*70)
        
        # Basic info
        print(f"\n📊 Shape: {df.shape}")
        print(f"📅 Date Range: {df['Date'].min()} to {df['Date'].max()}")
        print(f"📈 Trading Days: {len(df)} days")
        
        # Data quality
        print(f"\n🔍 Data Quality:")
        print(f"  - Missing Values: {df.isnull().sum().sum()}")
        print(f"  - Duplicate Rows: {df.duplicated().sum()}")
        
        # Memory usage
        memory_mb = df.memory_usage(deep=True).sum() / 1024**2
        print(f"  - Memory Usage: {memory_mb:.2f} MB")
        
        # Column info
        print(f"\n📋 Columns: {list(df.columns)}")
        
        # Data types
        print(f"\n📊 Data Types:")
        for col, dtype in df.dtypes.items():
            print(f"  {col:15s}: {dtype}")
        
        # Missing values
        missing = df.isnull().sum()
        if missing.sum() > 0:
            print(f"\n⚠️  Missing Values:")
            for col, count in missing.items():
                if count > 0:
                    print(f"  {col:15s}: {count} ({count/len(df)*100:.2f}%)")
        
        # Statistical summary
        print(f"\n📊 Basic Statistics:")
        print(df.describe().round(2))
        
        # Additional metrics
        if 'Close' in df.columns:
            returns = df['Close'].pct_change().dropna()
            print(f"\n📈 Returns Statistics:")
            print(f"  Mean Daily Return: {returns.mean()*100:.3f}%")
            print(f"  Std Daily Return: {returns.std()*100:.3f}%")
            print(f"  Skewness: {returns.skew():.3f}")
            print(f"  Kurtosis: {returns.kurtosis():.3f}")
            print(f"  Total Return: {((df['Close'].iloc[-1] / df['Close'].iloc[0]) - 1) * 100:.2f}%")
    
    def validate_data(self, df: pd.DataFrame) -> Dict[str, bool]:
        """
        Validate data quality
        
        Returns:
            Dictionary of validation results
        """
        results = {
            'has_data': len(df) > 0,
            'has_complete_dates': df['Date'].isnull().sum() == 0,
            'has_positive_prices': (df['Close'] > 0).all(),
            'has_positive_volume': (df['Volume'] > 0).all(),
            'has_ohlc_valid': (
                (df['High'] >= df['Low']).all() &
                (df['High'] >= df['Close']).all() &
                (df['Low'] <= df['Close']).all()
            ),
            'no_missing_values': df.isnull().sum().sum() == 0,
            'sufficient_data': len(df) >= 100
        }
        
        results['is_valid'] = all(results.values())
        
        return results


# ============================================================================
# LEGACY SUPPORT FUNCTIONS
# ============================================================================

def download_stock_data(symbol=STOCK_SYMBOL, start=START_DATE, end=END_DATE, save=True):
    """Legacy function for backward compatibility"""
    loader = StockDataLoader()
    return loader.download_stock_data(symbol, start, end, save)


def load_stock_data(symbol=STOCK_SYMBOL, force_download=False):
    """Legacy function for backward compatibility"""
    loader = StockDataLoader()
    return loader.load_stock_data(symbol, force_download)


def get_multiple_stocks(symbols, start=START_DATE, end=END_DATE):
    """Legacy function for backward compatibility"""
    loader = StockDataLoader()
    return loader.get_multiple_stocks(symbols, start, end)


def get_data_info(df):
    """Legacy function for backward compatibility"""
    loader = StockDataLoader()
    loader.get_data_info(df)


if __name__ == "__main__":
    print_section("Testing Enhanced Data Loader")
    
    # Create loader
    loader = StockDataLoader()
    
    # Load data
    df = loader.load_stock_data(force_download=False)
    
    if df is not None:
        # Display info
        loader.get_data_info(df)
        
        # Validate
        validation = loader.validate_data(df)
        print(f"\n✅ Validation Results:")
        for key, value in validation.items():
            print(f"  {key:20s}: {value}")
        
        # Display sample
        print("\n" + "="*70)
        print("  SAMPLE DATA")
        print("="*70)
        print(df.head())
        print("\n" + "="*70)
        print(df.tail())
        
        # Get stock info
        info = loader.get_stock_info(STOCK_SYMBOL)
        print(f"\n📊 Stock Info:")
        for key, value in info.items():
            if key != 'symbol':
                print(f"  {key:20s}: {value}")