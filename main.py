"""
Main pipeline for Stock Price Prediction
Run this file to execute the complete workflow
"""
import argparse
import os
import sys

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from utils.config import STOCK_SYMBOL
from utils.helpers import print_section
from src.data_loader import load_stock_data, get_data_info
from src.preprocess import preprocess_data
from src.train_model import prepare_features_target, split_data, train_all_models, get_feature_importance
from src.evaluate import evaluate_all_models
from src.predict import make_predictions
from src.visualization import (plot_price_history, plot_volume, plot_moving_averages, 
                               plot_correlation_matrix, create_dashboard, 
                               plot_predictions_vs_actual, plot_residuals)


def create_enough_data():
    """
    Run create_enough_data.py first to generate synthetic data
    if there's not enough historical data
    """
    print_section("CHECKING DATA AVAILABILITY")
    
    try:
        # First check if we have enough data
        from utils.config import DATA_RAW_DIR
        import pandas as pd
        
        filepath = os.path.join(DATA_RAW_DIR, f"{STOCK_SYMBOL}.csv")
        
        if os.path.exists(filepath):
            # Check if we have enough data
            df_raw = pd.read_csv(filepath)
            if len(df_raw) < 100:  # Minimum data points needed
                print(f"⚠️  Insufficient data ({len(df_raw)} rows). Generating synthetic data...")
                # Import and run create_enough_data
                exec(open('create_enough_data.py').read())
            else:
                print(f"✅ Sufficient data available ({len(df_raw)} rows)")
        else:
            print("📊 No existing data found. Will download fresh data...")
            
    except Exception as e:
        print(f"⚠️  Error checking data: {e}")
        print("Proceeding with data download...")


def run_data_pipeline(force_download=False):
    """
    Run the data loading and preprocessing pipeline
    
    Args:
        force_download: Force download data even if it exists
        
    Returns:
        Processed DataFrame
    """
    print_section("DATA PIPELINE")
    
    # First ensure we have enough data
    create_enough_data()
    
    # Load data
    print("\n1️⃣ Loading Stock Data")
    df_raw = load_stock_data(STOCK_SYMBOL, force_download=force_download)
    
    if df_raw is None:
        print("❌ Failed to load data. Exiting...")
        return None
    
    # Show data info
    get_data_info(df_raw)
    
    # Check if we have enough data for processing
    if len(df_raw) < 30:
        print("❌ Insufficient data for processing. Need at least 30 data points.")
        return None
    
    # Preprocess data
    print("\n2️⃣ Preprocessing Data")
    df_processed = preprocess_data(df_raw, save=True)
    
    return df_processed


def run_training_pipeline(df):
    """
    Run the model training pipeline
    
    Args:
        df: Processed DataFrame
        
    Returns:
        Dictionary of trained models, test data
    """
    print_section("TRAINING PIPELINE")
    
    # Prepare features and target
    print("\n1️⃣ Preparing Features and Target")
    X, y, feature_cols = prepare_features_target(df)
    
    # Split data
    print("\n2️⃣ Splitting Data")
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    # Train models
    print("\n3️⃣ Training Models")
    models = train_all_models(X_train, y_train, save_models=True)
    
    # Show feature importance
    print("\n4️⃣ Feature Importance Analysis")
    if 'xgboost' in models:
        importance_df = get_feature_importance(models['xgboost'], feature_cols, top_n=15)
    
    return models, X_train, X_test, y_train, y_test


def run_evaluation_pipeline(models, X_train, y_train, X_test, y_test):
    """
    Run the model evaluation pipeline
    
    Args:
        models: Dictionary of trained models
        X_train, X_test, y_train, y_test: Train and test data
        
    Returns:
        Evaluation results
    """
    print_section("EVALUATION PIPELINE")
    
    # Evaluate all models
    results = evaluate_all_models(
        models, X_train, y_train, X_test, y_test, save=True
    )
    
    return results


def run_visualization_pipeline(df, models, X_test, y_test):
    """
    Run the visualization pipeline
    
    Args:
        df: Processed DataFrame
        models: Dictionary of trained models
        X_test: Test features
        y_test: Test target
    """
    print_section("VISUALIZATION PIPELINE")
    
    print("\n1️⃣ Generating Basic Plots")
    plot_price_history(df, save=True)
    plot_volume(df, save=True)
    plot_moving_averages(df, save=True)
    
    print("\n2️⃣ Generating Analysis Plots")
    plot_correlation_matrix(df, save=True)
    create_dashboard(df, save=True)
    
    print("\n3️⃣ Generating Prediction Plots")
    # Plot predictions vs actual for best model
    if 'xgboost' in models:
        y_pred = models['xgboost'].predict(X_test)
        plot_predictions_vs_actual(y_test, y_pred, 
                                   title='XGBoost Predictions', 
                                   filename='xgboost_predictions.png')
        plot_residuals(y_test, y_pred)


def run_prediction_pipeline(model_name='xgboost', days=30):
    """
    Run the prediction pipeline
    
    Args:
        model_name: Name of model to use
        days: Number of days to predict
        
    Returns:
        Predictions DataFrame
    """
    print_section("PREDICTION PIPELINE")
    
    predictions = make_predictions(
        model_name=model_name,
        prediction_days=days,
        save=True
    )
    
    return predictions


def main(mode='full', force_download=False, predict_days=30):
    """
    Main function to run the entire pipeline
    
    Args:
        mode: Execution mode ('data', 'train', 'evaluate', 'visualize', 'predict', 'full')
        force_download: Force download data
        predict_days: Number of days to predict
    """
    print("="*80)
    print(" "*20 + "STOCK PRICE PREDICTION PIPELINE")
    print(" "*30 + f"Symbol: {STOCK_SYMBOL}")
    print("="*80)
    
    try:
        if mode in ['data', 'full']:
            # Data pipeline (this now includes create_enough_data check)
            df = run_data_pipeline(force_download=force_download)
            if df is None:
                return
        else:
            # Load processed data
            from utils.config import DATA_PROCESSED_DIR
            import pandas as pd
            filepath = os.path.join(DATA_PROCESSED_DIR, f"{STOCK_SYMBOL}_processed.csv")
            if not os.path.exists(filepath):
                print("❌ Processed data not found. Please run with mode='data' first.")
                return
            df = pd.read_csv(filepath)
            print(f"✅ Loaded processed data: {df.shape}")
        
        if mode in ['train', 'full']:
            # Training pipeline
            models, X_train, X_test, y_train, y_test = run_training_pipeline(df)
        else:
            if mode in ['evaluate', 'visualize']:
                # Load models and prepare data
                from utils.helpers import load_model
                from utils.config import MODELS_DIR
                
                X, y, feature_cols = prepare_features_target(df)
                X_train, X_test, y_train, y_test = split_data(X, y)
                
                model_names = ['linear_regression', 'random_forest', 'xgboost']
                models = {}
                for model_name in model_names:
                    try:
                        models[model_name] = load_model(model_name, MODELS_DIR)
                    except FileNotFoundError:
                        print(f"⚠️  Model '{model_name}' not found.")
                
                if not models:
                    print("❌ No trained models found. Please run with mode='train' first.")
                    return
        
        if mode in ['evaluate', 'full']:
            # Evaluation pipeline
            results = run_evaluation_pipeline(models, X_train, y_train, X_test, y_test)
        
        if mode in ['visualize', 'full']:
            # Visualization pipeline
            run_visualization_pipeline(df, models, X_test, y_test)
        
        if mode in ['predict', 'full']:
            # Prediction pipeline
            predictions = run_prediction_pipeline(
                model_name='xgboost',
                days=predict_days
            )
        
        # Final summary
        print_section("PIPELINE COMPLETED SUCCESSFULLY! ✨")
        print("\n📊 Summary:")
        print(f"   ✅ Data processed: {df.shape}")
        if mode in ['train', 'full']:
            print(f"   ✅ Models trained: {len(models)}")
        if mode in ['visualize', 'full']:
            print(f"   ✅ Visualizations generated")
        if mode in ['predict', 'full']:
            print(f"   ✅ Predictions made for next {predict_days} days")
        
        print("\n🎉 All done! Check the 'outputs' and 'models' folders for results.")
        
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Stock Price Prediction Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run full pipeline
  python main.py --mode full
  
  # Only download and preprocess data
  python main.py --mode data
  
  # Only train models
  python main.py --mode train
  
  # Only evaluate models
  python main.py --mode evaluate
  
  # Only generate visualizations
  python main.py --mode visualize
  
  # Only make predictions
  python main.py --mode predict --days 60
  
  # Force download fresh data
  python main.py --mode full --force-download
        """
    )
    
    parser.add_argument(
        '--mode',
        type=str,
        default='full',
        choices=['data', 'train', 'evaluate', 'visualize', 'predict', 'full'],
        help='Execution mode (default: full)'
    )
    
    parser.add_argument(
        '--force-download',
        action='store_true',
        help='Force download data even if it exists'
    )
    
    parser.add_argument(
        '--days',
        type=int,
        default=30,
        help='Number of days to predict (default: 30)'
    )
    
    args = parser.parse_args()
    
    main(
        mode=args.mode,
        force_download=args.force_download,
        predict_days=args.days
    )