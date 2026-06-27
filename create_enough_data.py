"""
Create enough stock data for preprocessing - Enhanced Version
"""
import pandas as pd
import numpy as np
import os
import sys
import argparse
from datetime import datetime, timedelta
from typing import Optional, List, Dict
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import STOCK_SYMBOL, DAYS, DATA_RAW_DIR, RANDOM_STATE
from utils.helpers import print_section


class StockDataGenerator:
    """
    Advanced synthetic stock data generator with realistic patterns
    """
    
    def __init__(self, symbol: str = STOCK_SYMBOL, days: int = DAYS, seed: int = RANDOM_STATE):
        self.symbol = symbol
        self.days = days
        self.seed = seed
        np.random.seed(seed)
        
    def generate_price_path(
        self, 
        n: int, 
        base_price: float = 150,
        volatility: float = 0.02,
        drift: float = 0.0005,
        trend: float = 0.01
    ) -> np.ndarray:
        """
        Generate realistic price path using geometric Brownian motion
        """
        returns = np.random.normal(drift - 0.5 * volatility**2, volatility, n)
        trend_component = np.linspace(0, trend * n / 252, n)
        prices = base_price * np.exp(np.cumsum(returns) + trend_component)
        prices = np.maximum(prices, 10)
        return prices
    
    def generate_ohlc(self, prices: np.ndarray, spread: float = 0.005) -> pd.DataFrame:
        """Generate OHLC data from closing prices"""
        n = len(prices)
        
        open_prices = prices * (1 + np.random.uniform(-spread, spread, n))
        open_prices[0] = prices[0] * (1 + np.random.uniform(-spread, spread))
        
        high_prices = np.maximum(
            prices * (1 + np.random.uniform(spread, spread * 2, n)),
            open_prices * (1 + np.random.uniform(0, spread, n))
        )
        low_prices = np.minimum(
            prices * (1 - np.random.uniform(spread, spread * 2, n)),
            open_prices * (1 - np.random.uniform(0, spread, n))
        )
        
        high_prices = np.maximum(high_prices, low_prices + 0.01)
        
        volumes = np.random.randint(
            1000000, 50000000, n
        ) * np.random.uniform(0.8, 1.2, n)
        volumes = volumes.astype(int)
        
        return pd.DataFrame({
            'Open': open_prices,
            'High': high_prices,
            'Low': low_prices,
            'Close': prices,
            'Volume': volumes
        })
    
    def generate_weekday_data(self, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """Generate data with only weekdays"""
        dates = pd.date_range(start=start_date, end=end_date, freq='B')
        n = len(dates)
        
        prices = self.generate_price_path(n)
        ohlc_data = self.generate_ohlc(prices)
        
        df = pd.DataFrame({
            'Date': dates,
            'Open': ohlc_data['Open'],
            'High': ohlc_data['High'],
            'Low': ohlc_data['Low'],
            'Close': ohlc_data['Close'],
            'Adj Close': ohlc_data['Close'],
            'Volume': ohlc_data['Volume']
        })
        
        return df
    
    def create_working_dataset(self) -> pd.DataFrame:
        """Create a complete working dataset - NO SYMBOL COLUMN"""
        print_section(f"Creating {self.days} days of {self.symbol} data")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.days)
        
        df = self.generate_weekday_data(start_date, end_date)
        # REMOVED: df['Symbol'] = self.symbol  <-- This was causing the error
        
        print(f"✅ Created {len(df)} trading days of data")
        print(f"   Date range: {df['Date'].min()} to {df['Date'].max()}")
        print(f"   Price range: ${df['Close'].min():.2f} to ${df['Close'].max():.2f}")
        print(f"   Avg Volume: {df['Volume'].mean():.0f}")
        
        returns = df['Close'].pct_change().dropna()
        print(f"   Mean daily return: {returns.mean()*100:.3f}%")
        print(f"   Volatility: {returns.std()*100:.3f}%")
        
        return df
    
    def save_dataset(self, df: pd.DataFrame) -> str:
        """Save dataset to CSV"""
        os.makedirs(DATA_RAW_DIR, exist_ok=True)
        file_path = os.path.join(DATA_RAW_DIR, f"{self.symbol}_raw.csv")
        df.to_csv(file_path, index=False)
        print(f"\n💾 Saved to: {file_path}")
        return file_path
    
    def create_multiple_stocks(self, symbols: List[str], days: int = None) -> Dict[str, pd.DataFrame]:
        """Create datasets for multiple stocks"""
        if days is None:
            days = self.days
        
        result = {}
        
        for symbol in symbols:
            print(f"\n📊 Generating data for {symbol}...")
            generator = StockDataGenerator(symbol=symbol, days=days, seed=self.seed)
            df = generator.create_working_dataset()
            generator.save_dataset(df)
            result[symbol] = df
        
        print(f"\n✅ Generated data for {len(symbols)} stocks")
        return result


# ============================================================================
# LEGACY SUPPORT FUNCTIONS
# ============================================================================

def create_working_dataset(symbol, days):
    """Legacy function for backward compatibility"""
    generator = StockDataGenerator(symbol=symbol, days=days)
    return generator.create_working_dataset()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Create synthetic stock data for testing'
    )
    
    parser.add_argument(
        '--symbol',
        type=str,
        default=STOCK_SYMBOL,
        help=f'Stock symbol (default: {STOCK_SYMBOL})'
    )
    
    parser.add_argument(
        '--symbols',
        type=str,
        help='Comma-separated list of symbols'
    )
    
    parser.add_argument(
        '--days',
        type=int,
        default=DAYS,
        help=f'Number of days (default: {DAYS})'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=RANDOM_STATE,
        help=f'Random seed (default: {RANDOM_STATE})'
    )
    
    args = parser.parse_args()
    
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(',')]
    else:
        symbols = [args.symbol]
    
    generator = StockDataGenerator(symbol=symbols[0], days=args.days, seed=args.seed)
    
    if len(symbols) == 1:
        df = generator.create_working_dataset()
        generator.save_dataset(df)
    else:
        generator.create_multiple_stocks(symbols, days=args.days)
    
    print("\n" + "="*70)
    print("  ✅ DATA GENERATION COMPLETE")
    print("="*70)
    print("\n🚀 Now you can run:")
    print("   python main.py --mode full")