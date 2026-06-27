"""
Main pipeline for Stock Price Prediction - Fully Automated
Run this file to execute the complete workflow with automatic data generation
"""
import argparse
import os
import sys
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from utils.config import STOCK_SYMBOL, START_DATE, END_DATE, DAYS, LOGS_DIR
from utils.helpers import print_section, Timer
from src.data_loader import StockDataLoader, load_stock_data
from src.preprocess import StockDataPreprocessor, preprocess_data
from src.train_model import ModelTrainer, prepare_features_target, split_data
from src.evaluate import AdvancedEvaluator
from src.predict import PredictionEngine, make_predictions
from src.visualization import StockVisualizer, create_dashboard

# Setup logging
os.makedirs(LOGS_DIR, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(LOGS_DIR, f'pipeline_{datetime.now().strftime("%Y%m%d")}.log')),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AutomatedStockPipeline:
    """
    Fully automated pipeline with automatic data generation
    """
    
    def __init__(self, symbol=STOCK_SYMBOL, start_date=START_DATE, end_date=END_DATE):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.df_raw = None
        self.df_processed = None
        self.models = None
        self.results = None
        self.predictions = None
        self.days_of_data = DAYS
        
    def ensure_data_exists(self, force_download=False):
        """
        Ensure data exists - if not, generate synthetic data automatically
        """
        print_section("📊 DATA AVAILABILITY CHECK")
        
        filepath = os.path.join('data', 'raw', f"{self.symbol}_raw.csv")
        
        # Check if data exists
        data_exists = os.path.exists(filepath)
        
        if data_exists and not force_download:
            print(f"✅ Data exists for {self.symbol}")
            return True
        
        print(f"⚠️  No data found for {self.symbol} or force_download=True")
        print(f"🔄 Generating synthetic data automatically...")
        
        # Import and run create_enough_data
        try:
            from create_enough_data import StockDataGenerator
            
            # Generate synthetic data
            generator = StockDataGenerator(symbol=self.symbol, days=self.days_of_data)
            df = generator.create_working_dataset()
            generator.save_dataset(df)
            
            print(f"✅ Synthetic data generated successfully!")
            return True
            
        except ImportError:
            print("⚠️  Could not import StockDataGenerator. Using fallback...")
            # Fallback: Simple data generation
            self._generate_fallback_data()
            return True
        except Exception as e:
            print(f"❌ Error generating data: {e}")
            return False
    
    def _generate_fallback_data(self):
        """Fallback data generation if module not available"""
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta
        
        print("📊 Generating fallback data...")
        
        # Create date range
        end_date = datetime.now()
        start_date = end_date - timedelta(days=self.days_of_data)
        dates = pd.date_range(start=start_date, end=end_date, freq='B')
        
        n = len(dates)
        np.random.seed(42)
        
        # Generate realistic price data
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
        
        # Ensure High >= Low
        df['High'] = df[['Open', 'High', 'Close']].max(axis=1) * 1.001
        df['Low'] = df[['Open', 'Low', 'Close']].min(axis=1) * 0.999
        
        # Save
        os.makedirs('data/raw', exist_ok=True)
        file_path = f"data/raw/{self.symbol}_raw.csv"
        df.to_csv(file_path, index=False)
        
        print(f"✅ Fallback data saved to: {file_path}")
        return df
    
    def run_data_pipeline(self, force_download=False):
        """Run data loading and preprocessing"""
        print_section("📊 DATA PIPELINE")
        
        # Ensure data exists
        if not self.ensure_data_exists(force_download):
            logger.error("Failed to ensure data exists")
            return False
        
        # Load data
        print("\n1️⃣ Loading Stock Data")
        with Timer("Data Loading"):
            self.df_raw = load_stock_data(self.symbol, force_download=force_download)
        
        if self.df_raw is None:
            logger.error("Failed to load data")
            return False
        
        # Show data info
        loader = StockDataLoader()
        loader.get_data_info(self.df_raw)
        
        # Check data sufficiency
        if len(self.df_raw) < 30:
            logger.warning("Insufficient data. Generating more...")
            self._generate_fallback_data()
            self.df_raw = load_stock_data(self.symbol, force_download=True)
        
        # Preprocess data
        print("\n2️⃣ Preprocessing Data")
        with Timer("Preprocessing"):
            self.df_processed = preprocess_data(self.df_raw, save=True, scale=False)
        
        print(f"   Final shape: {self.df_processed.shape}")
        print(f"   Features: {len(self.df_processed.columns)} columns")
        
        return True
    
    def run_training_pipeline(self, target_col='target'):
        """Run model training"""
        print_section("🤖 TRAINING PIPELINE")
        
        if self.df_processed is None:
            logger.error("No processed data available")
            return False
        
        # Prepare features and target
        print("\n1️⃣ Preparing Features and Target")
        X, y, feature_cols = prepare_features_target(self.df_processed, target_col=target_col)
        
        # Split data
        print("\n2️⃣ Splitting Data")
        X_train, X_test, y_train, y_test = split_data(X, y)
        
        print(f"   Training set: {X_train.shape}")
        print(f"   Test set: {X_test.shape}")
        print(f"   Features: {len(feature_cols)}")
        
        # Train models
        print("\n3️⃣ Training Models")
        trainer = ModelTrainer(X_train, y_train, X_test, y_test, feature_cols)
        self.results = trainer.train_and_evaluate_all(save_models=True)
        self.models = trainer.models
        
        # Get feature importance
        print("\n4️⃣ Feature Importance Analysis")
        importance_df = trainer.get_feature_importance('xgboost', top_n=20)
        
        # Get best model
        best_model, best_name = trainer.get_best_model()
        logger.info(f"🏆 Best Model: {best_name}")
        logger.info(f"   Test R²: {trainer.results[best_name]['metrics']['test_r2']:.4f}")
        
        return True
    
    def run_prediction_pipeline(self, days=30, model_name='xgboost'):
        """Run prediction pipeline"""
        print_section("🔮 PREDICTION PIPELINE")
        
        if self.df_processed is None:
            logger.error("No processed data available")
            return False
        
        # Make predictions
        print(f"\n🔮 Predicting next {days} days...")
        self.predictions = make_predictions(
            model_name=model_name,
            prediction_days=days,
            save=True
        )
        
        return True
    
    def run_full_pipeline(self, force_download=False, predict_days=30):
        """Run complete pipeline automatically"""
        print("="*80)
        print(" "*20 + "🚀 AUTOMATED STOCK PREDICTION PIPELINE")
        print(" "*30 + f"Symbol: {self.symbol}")
        print(" "*25 + f"Date Range: {self.start_date} to {self.end_date}")
        print("="*80)
        
        start_time = datetime.now()
        
        try:
            # Data pipeline (with automatic data generation)
            if not self.run_data_pipeline(force_download=force_download):
                return False
            
            # Training pipeline
            if not self.run_training_pipeline():
                return False
            
            # Prediction pipeline
            if not self.run_prediction_pipeline(days=predict_days):
                return False
            
            # Visualizations will be generated by dashboard
            print_section("📊 VISUALIZATIONS READY")
            print("\n✅ All data processing complete!")
            print(f"   - Raw data: {len(self.df_raw)} rows")
            print(f"   - Processed data: {self.df_processed.shape}")
            print(f"   - Models trained: {len(self.models)}")
            print(f"   - Predictions: {len(self.predictions)} days")
            
            # Final summary
            elapsed = (datetime.now() - start_time).total_seconds()
            print_section("✅ PIPELINE COMPLETED SUCCESSFULLY! ✨")
            print(f"\n⏱️  Total execution time: {elapsed:.2f} seconds")
            print("\n🚀 Launch dashboard with: streamlit run dashboard.py")
            
            return True
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            import traceback
            traceback.print_exc()
            return False


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Automated Stock Price Prediction Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline (automatic)
  python main.py
  
  # Run with specific symbol
  python main.py --symbol AAPL
  
  # Force download fresh data
  python main.py --force-download
  
  # Predict more days
  python main.py --days 60
        """
    )
    
    parser.add_argument(
        '--symbol',
        type=str,
        default=STOCK_SYMBOL,
        help=f'Stock ticker symbol (default: {STOCK_SYMBOL})'
    )
    
    parser.add_argument(
        '--force-download',
        action='store_true',
        help='Force download fresh data'
    )
    
    parser.add_argument(
        '--days',
        type=int,
        default=30,
        help='Number of days to predict (default: 30)'
    )
    
    parser.add_argument(
        '--start-date',
        type=str,
        default=START_DATE,
        help=f'Start date (default: {START_DATE})'
    )
    
    parser.add_argument(
        '--end-date',
        type=str,
        default=END_DATE,
        help=f'End date (default: {END_DATE})'
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = AutomatedStockPipeline(
        symbol=args.symbol,
        start_date=args.start_date,
        end_date=args.end_date
    )
    
    # Run full pipeline
    pipeline.run_full_pipeline(
        force_download=args.force_download,
        predict_days=args.days
    )


if __name__ == "__main__":
    main()