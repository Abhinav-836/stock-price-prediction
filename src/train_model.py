"""
Model training module for stock price prediction
"""
import pandas as pd
import numpy as np
import os
import sys
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import joblib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import (TEST_SIZE, RANDOM_STATE, XGB_PARAMS, 
                          MODELS_DIR, DATA_PROCESSED_DIR, STOCK_SYMBOL)
from utils.helpers import save_model, print_section


def prepare_features_target(df, feature_cols=None, target_col='target'):
    """
    Prepare features and target for training
    
    Args:
        df: Processed DataFrame
        feature_cols: List of feature columns (if None, use all except target and Date)
        target_col: Name of target column
        
    Returns:
        X (features), y (target), and list of feature names
    """
    # Exclude columns we don't want as features
    exclude_cols = ['Date', 'target', 'target_class']
    
    if feature_cols is None:
        feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols]
    y = df[target_col]
    
    print(f"📊 Features shape: {X.shape}")
    print(f"🎯 Target shape: {y.shape}")
    print(f"📝 Number of features: {len(feature_cols)}")
    
    return X, y, feature_cols


def split_data(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE):
    """
    Split data into training and testing sets
    
    Args:
        X: Features
        y: Target
        test_size: Proportion of test set
        random_state: Random seed
        
    Returns:
        X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=False
    )
    
    print(f"\n📦 Training set size: {len(X_train)}")
    print(f"📦 Test set size: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test


def train_linear_regression(X_train, y_train):
    """
    Train Linear Regression model
    
    Args:
        X_train: Training features
        y_train: Training target
        
    Returns:
        Trained model
    """
    print_section("Training Linear Regression")
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    print("✅ Linear Regression trained successfully!")
    return model


def train_random_forest(X_train, y_train, n_estimators=100):
    """
    Train Random Forest model
    
    Args:
        X_train: Training features
        y_train: Training target
        n_estimators: Number of trees
        
    Returns:
        Trained model
    """
    print_section("Training Random Forest")
    
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    print("✅ Random Forest trained successfully!")
    return model


def train_xgboost(X_train, y_train, params=None):
    """
    Train XGBoost model
    
    Args:
        X_train: Training features
        y_train: Training target
        params: XGBoost parameters
        
    Returns:
        Trained model
    """
    print_section("Training XGBoost")
    
    if params is None:
        params = XGB_PARAMS
    
    model = XGBRegressor(**params)
    model.fit(X_train, y_train)
    
    print("✅ XGBoost trained successfully!")
    return model


def train_all_models(X_train, y_train, save_models=True):
    """
    Train all models
    
    Args:
        X_train: Training features
        y_train: Training target
        save_models: Whether to save trained models
        
    Returns:
        Dictionary of trained models
    """
    models = {}
    
    # Train Linear Regression
    models['linear_regression'] = train_linear_regression(X_train, y_train)
    if save_models:
        save_model(models['linear_regression'], 'linear_regression', MODELS_DIR)
    
    # Train Random Forest
    models['random_forest'] = train_random_forest(X_train, y_train)
    if save_models:
        save_model(models['random_forest'], 'random_forest', MODELS_DIR)
    
    # Train XGBoost
    models['xgboost'] = train_xgboost(X_train, y_train)
    if save_models:
        save_model(models['xgboost'], 'xgboost', MODELS_DIR)
    
    print_section("All Models Trained Successfully!")
    print(f"📦 Trained {len(models)} models: {list(models.keys())}")
    
    return models


def get_feature_importance(model, feature_names, top_n=10):
    """
    Get feature importance from tree-based models
    
    Args:
        model: Trained model (must have feature_importances_)
        feature_names: List of feature names
        top_n: Number of top features to display
        
    Returns:
        DataFrame with feature importances
    """
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n🔝 Top {top_n} Important Features:")
        print(importance_df.head(top_n).to_string(index=False))
        
        return importance_df
    else:
        print("⚠️  Model does not have feature_importances_ attribute")
        return None


if __name__ == "__main__":
    print("Testing Model Training Pipeline...")
    
    # Load processed data
    filepath = os.path.join(DATA_PROCESSED_DIR, f"{STOCK_SYMBOL}_processed.csv")
    
    if os.path.exists(filepath):
        print(f"📂 Loading processed data from: {filepath}")
        df = pd.read_csv(filepath)
        
        # Prepare features and target
        X, y, feature_cols = prepare_features_target(df)
        
        # Split data
        X_train, X_test, y_train, y_test = split_data(X, y)
        
        # Train all models
        models = train_all_models(X_train, y_train, save_models=True)
        
        # Show feature importance for XGBoost
        print_section("Feature Importance (XGBoost)")
        get_feature_importance(models['xgboost'], feature_cols, top_n=15)
        
    else:
        print(f"❌ Processed data not found at: {filepath}")
        print("Please run preprocess.py first to create processed data.")