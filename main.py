"""
Main pipeline for Stock Price Prediction - FIXED

Fixes vs previous version:
  * ensure_data_exists() generated SYNTHETIC data whenever no CSV existed - before even
    trying Yahoo - and saved it as the real cache. First runs always trained on fake data.
  * --start-date / --end-date were accepted but ignored (load_stock_data never used them).
  * Training used target_col='target' (doesn't exist) and relied on a fallback; prediction
    then reloaded 'xgboost' from disk instead of using the best model just trained.
  * Hard-coded 'data/raw' paths ignored config.DATA_RAW_DIR (broke when run from another cwd).
  * Not enough data -> generated fallback data, then re-downloaded the SAME symbol with
    force_download=True (which overwrote the fallback). Now a clear error is raised.
  * Now shares src/pipeline.py with the dashboard, so both behave identically.
"""
import argparse
import os
import sys
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from utils.config import STOCK_SYMBOL, START_DATE, END_DATE, LOGS_DIR, DATA_PROCESSED_DIR, MODELS_DIR
from utils.helpers import print_section, save_model, save_metrics
from src.pipeline import run_analysis, forecast, feature_importance
from src.predict import PredictionEngine

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
    """CLI wrapper around src.pipeline (data -> train -> predict -> save)"""

    def __init__(self, symbol=STOCK_SYMBOL, start_date=START_DATE, end_date=END_DATE, use_synthetic=False):
        self.symbol = symbol.upper()
        self.start_date = start_date
        self.end_date = end_date
        self.use_synthetic = use_synthetic
        self.analysis = None
        self.predictions = None

    def run_training(self, run_cv=True):
        print_section("📊 DATA + 🤖 TRAINING PIPELINE")
        self.analysis = run_analysis(self.symbol, self.start_date, self.end_date,
                                     use_synthetic=self.use_synthetic, run_cv=run_cv, verbose=True)
        a = self.analysis
        print(f"\n📡 Data source: {a['source'].upper()}  |  rows: {len(a['raw_view'])}  "
              f"|  features: {len(a['feature_cols'])}")
        if a['source'] == 'synthetic':
            logger.warning("⚠️  Using SYNTHETIC data - results say nothing about the real stock.")

        os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
        a['processed'].to_csv(os.path.join(DATA_PROCESSED_DIR, f"{self.symbol}_processed.csv"), index=False)

        best = a['best_name']
        for name, model in a['models'].items():
            save_model(model, name, MODELS_DIR)
            save_metrics(a['results'][name]['metrics'], f"{name}_metrics", MODELS_DIR)

        print(f"\n🏆 Best model: {best}  (test R² {a['results'][best]['metrics']['test_r2']:.4f})")
        if a['baseline_rmse'] is not None:
            print(f"   Zero-return baseline RMSE: {a['baseline_rmse']:.4f}%  vs  "
                  f"best model: {a['results'][best]['metrics']['test_rmse']:.4f}%")

        imp = feature_importance(a, top_n=15)
        if imp is not None:
            print(f"\n🔝 Top features ({imp['model'].iloc[0]}):")
            print(imp[['feature', 'importance']].to_string(index=False))
        return True

    def run_prediction(self, days=30, model_name=None, save=True):
        print_section("🔮 PREDICTION PIPELINE")
        self.predictions = forecast(self.analysis, days=days, model_name=model_name)

        name = model_name or self.analysis['best_name']
        engine = PredictionEngine(model_name=name, model=self.analysis['models'][name],
                                  target_col=self.analysis['target_col'], symbol=self.symbol)
        engine.last_close = float(self.analysis['raw_full']['Close'].iloc[-1])
        print("\n" + engine.generate_report(self.predictions))
        if save:
            engine.save_predictions(self.predictions)
        return True

    def run_full_pipeline(self, predict_days=30, model_name=None, run_cv=True):
        print("=" * 80)
        print(f"🚀 AUTOMATED STOCK PREDICTION PIPELINE | {self.symbol} | {self.start_date} → {self.end_date}")
        print("=" * 80)
        started = datetime.now()
        try:
            self.run_training(run_cv=run_cv)
            self.run_prediction(days=predict_days, model_name=model_name)
        except ValueError as e:
            logger.error(f"❌ {e}")
            return False
        except Exception as e:
            logger.exception(f"Pipeline failed: {e}")
            return False

        print_section("✅ PIPELINE COMPLETED SUCCESSFULLY")
        print(f"⏱️  Total time: {(datetime.now() - started).total_seconds():.1f}s")
        print("🚀 Launch dashboard with: streamlit run dashboard.py")
        return True


def main():
    parser = argparse.ArgumentParser(
        description='Automated Stock Price Prediction Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                              # default symbol, config date range
  python main.py --symbol MSFT --days 60
  python main.py --symbol RELIANCE.NS --start-date 2019-01-01
  python main.py --synthetic                  # offline demo data
  python main.py --model ridge --no-cv        # force a model, skip cross-validation
        """
    )
    parser.add_argument('--symbol', type=str, default=STOCK_SYMBOL)
    parser.add_argument('--days', type=int, default=30, help='Business days to forecast')
    parser.add_argument('--start-date', type=str, default=START_DATE)
    parser.add_argument('--end-date', type=str, default=END_DATE)
    parser.add_argument('--synthetic', action='store_true', help='Use synthetic data (offline)')
    parser.add_argument('--model', type=str, default=None, help='Model for forecasting (default: best)')
    parser.add_argument('--no-cv', action='store_true', help='Skip time-series cross-validation')
    args = parser.parse_args()

    pipeline = AutomatedStockPipeline(args.symbol, args.start_date, args.end_date, args.synthetic)
    ok = pipeline.run_full_pipeline(predict_days=args.days, model_name=args.model, run_cv=not args.no_cv)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()