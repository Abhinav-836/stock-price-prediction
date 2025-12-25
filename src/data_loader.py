"""
Data loading module for stock price prediction
"""
import yfinance as yf
import pandas as pd
import os
import sys

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import STOCK_SYMBOL, START_DATE, END_DATE, DATA_RAW_DIR


def download_stock_data(symbol=STOCK_SYMBOL, start=START_DATE, end=END_DATE, save=True):
    """
    Download stock data from Yahoo Finance
    
    Args:
        symbol: Stock ticker symbol (default: from config)
        start: Start date (YYYY-MM-DD format)
        end: End date (YYYY-MM-DD format)
        save: Whether to save data to CSV
        
    Returns:
        DataFrame with stock data
    """
    print(f"📥 Downloading {symbol} data from {start} to {end}...")
    
    try:
        # Download data
        stock = yf.Ticker(symbol)
        df = stock.history(start=start, end=end)
        
        # Reset index to make Date a column
        df.reset_index(inplace=True)
        
        print(f"✅ Downloaded {len(df)} rows of data")
        print(f"📊 Date range: {df['Date'].min()} to {df['Date'].max()}")
        
        # Save to CSV if requested
        if save:
            filepath = os.path.join(DATA_RAW_DIR, f"{symbol}_raw.csv")
            df.to_csv(filepath, index=False)
            print(f"💾 Data saved to: {filepath}")
        
        return df
    
    except Exception as e:
        print(f"❌ Error downloading data: {e}")
        return None


def load_stock_data(symbol=STOCK_SYMBOL, force_download=False):
    """
    Load stock data from CSV or download if not available
    
    Args:
        symbol: Stock ticker symbol
        force_download: Force download even if CSV exists
        
    Returns:
        DataFrame with stock data
    """
    filepath = os.path.join(DATA_RAW_DIR, f"{symbol}_raw.csv")
    
    # Check if file exists and we're not forcing download
    if os.path.exists(filepath) and not force_download:
        print(f"📂 Loading existing data from: {filepath}")
        df = pd.read_csv(filepath)
        df['Date'] = pd.to_datetime(df['Date'])
        print(f"✅ Loaded {len(df)} rows")
        return df
    else:
        # Download data
        return download_stock_data(symbol=symbol, save=True)


def get_multiple_stocks(symbols, start=START_DATE, end=END_DATE):
    """
    Download data for multiple stock symbols
    
    Args:
        symbols: List of stock ticker symbols
        start: Start date
        end: End date
        
    Returns:
        Dictionary with symbol as key and DataFrame as value
    """
    stocks_data = {}
    
    for symbol in symbols:
        print(f"\n📥 Processing {symbol}...")
        df = download_stock_data(symbol=symbol, start=start, end=end, save=True)
        if df is not None:
            stocks_data[symbol] = df
    
    print(f"\n✅ Successfully loaded {len(stocks_data)} stocks")
    return stocks_data


def get_data_info(df):
    """
    Display information about the dataset
    
    Args:
        df: DataFrame with stock data
    """
    print("\n" + "="*60)
    print("  DATASET INFORMATION")
    print("="*60)
    print(f"\nShape: {df.shape}")
    print(f"Date Range: {df['Date'].min()} to {df['Date'].max()}")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nMissing Values:\n{df.isnull().sum()}")
    print(f"\nData Types:\n{df.dtypes}")
    print(f"\nBasic Statistics:\n{df.describe()}")


if __name__ == "__main__":
    # Test the data loader
    print("Testing Data Loader...")
    
    # Load data
    df = load_stock_data(force_download=True)
    
    if df is not None:
        # Display info
        get_data_info(df)
        
        # Display first few rows
        print("\n" + "="*60)
        print("  SAMPLE DATA")
        print("="*60)
        print(df.head())
        print("\n" + "="*60)
        print(df.tail())