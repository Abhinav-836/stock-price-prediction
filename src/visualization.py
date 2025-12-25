"""
Visualization module for stock price prediction
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import (FIGURES_DIR, DATA_PROCESSED_DIR, STOCK_SYMBOL, 
                          FIGURE_SIZE, DPI)
from utils.helpers import print_section

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def plot_price_history(df, save=True):
    """
    Plot historical stock prices
    
    Args:
        df: DataFrame with Date and Close columns
        save: Whether to save the plot
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=DPI)
    
    ax.plot(pd.to_datetime(df['Date']), df['Close'], linewidth=2)
    ax.set_title(f'{STOCK_SYMBOL} Stock Price History', fontsize=16, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price ($)', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save:
        filepath = os.path.join(FIGURES_DIR, 'price_history.png')
        plt.savefig(filepath, dpi=DPI, bbox_inches='tight')
        print(f"✅ Plot saved: {filepath}")
    
    plt.show()


def plot_ohlc_data(df, last_n_days=100, save=True):
    """
    Plot OHLC (Open, High, Low, Close) data
    
    Args:
        df: DataFrame with OHLC data
        last_n_days: Number of recent days to plot
        save: Whether to save the plot
    """
    df_recent = df.tail(last_n_days).copy()
    df_recent['Date'] = pd.to_datetime(df_recent['Date'])
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), dpi=DPI)
    
    # Open
    axes[0, 0].plot(df_recent['Date'], df_recent['Open'], color='blue', linewidth=2)
    axes[0, 0].set_title('Opening Price', fontweight='bold')
    axes[0, 0].set_ylabel('Price ($)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # High
    axes[0, 1].plot(df_recent['Date'], df_recent['High'], color='green', linewidth=2)
    axes[0, 1].set_title('High Price', fontweight='bold')
    axes[0, 1].set_ylabel('Price ($)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Low
    axes[1, 0].plot(df_recent['Date'], df_recent['Low'], color='red', linewidth=2)
    axes[1, 0].set_title('Low Price', fontweight='bold')
    axes[1, 0].set_xlabel('Date')
    axes[1, 0].set_ylabel('Price ($)')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Close
    axes[1, 1].plot(df_recent['Date'], df_recent['Close'], color='purple', linewidth=2)
    axes[1, 1].set_title('Closing Price', fontweight='bold')
    axes[1, 1].set_xlabel('Date')
    axes[1, 1].set_ylabel('Price ($)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save:
        filepath = os.path.join(FIGURES_DIR, 'ohlc_data.png')
        plt.savefig(filepath, dpi=DPI, bbox_inches='tight')
        print(f"✅ Plot saved: {filepath}")
    
    plt.show()


def plot_volume(df, save=True):
    """
    Plot trading volume
    
    Args:
        df: DataFrame with Volume data
        save: Whether to save the plot
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=DPI)
    
    dates = pd.to_datetime(df['Date'])
    ax.bar(dates, df['Volume'], alpha=0.7, color='steelblue')
    ax.set_title(f'{STOCK_SYMBOL} Trading Volume', fontsize=16, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Volume', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save:
        filepath = os.path.join(FIGURES_DIR, 'trading_volume.png')
        plt.savefig(filepath, dpi=DPI, bbox_inches='tight')
        print(f"✅ Plot saved: {filepath}")
    
    plt.show()


def plot_moving_averages(df, save=True):
    """
    Plot price with moving averages
    
    Args:
        df: DataFrame with Close and MA columns
        save: Whether to save the plot
    """
    fig, ax = plt.subplots(figsize=FIGURE_SIZE, dpi=DPI)
    
    dates = pd.to_datetime(df['Date'])
    
    # Plot close price
    ax.plot(dates, df['Close'], label='Close Price', linewidth=2, alpha=0.8)
    
    # Plot moving averages if they exist
    ma_cols = [col for col in df.columns if col.startswith('MA_')]
    colors = ['orange', 'green', 'red', 'purple']
    
    for i, col in enumerate(ma_cols):
        if col in df.columns:
            ax.plot(dates, df[col], label=col.replace('_', ' '), 
                   linewidth=1.5, alpha=0.7, color=colors[i % len(colors)])
    
    ax.set_title(f'{STOCK_SYMBOL} Price with Moving Averages', 
                fontsize=16, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price ($)', fontsize=12)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save:
        filepath = os.path.join(FIGURES_DIR, 'moving_averages.png')
        plt.savefig(filepath, dpi=DPI, bbox_inches='tight')
        print(f"✅ Plot saved: {filepath}")
    
    plt.show()


def plot_correlation_matrix(df, save=True):
    """
    Plot correlation matrix heatmap
    
    Args:
        df: DataFrame with features
        save: Whether to save the plot
    """
    # Select only numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    exclude_cols = ['year', 'month', 'day', 'day_of_week', 'quarter']
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    # Calculate correlation
    corr = df[feature_cols].corr()
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 10), dpi=DPI)
    sns.heatmap(corr, annot=False, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5, ax=ax, cbar_kws={"shrink": 0.8})
    
    ax.set_title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    if save:
        filepath = os.path.join(FIGURES_DIR, 'correlation_matrix.png')
        plt.savefig(filepath, dpi=DPI, bbox_inches='tight')
        print(f"✅ Plot saved: {filepath}")
    
    plt.show()


def plot_predictions_vs_actual(y_true, y_pred, title='Predictions vs Actual', save=True, filename='predictions_vs_actual.png'):
    """
    Plot predicted vs actual values
    
    Args:
        y_true: True values
        y_pred: Predicted values
        title: Plot title
        save: Whether to save the plot
        filename: Filename for saving
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5), dpi=DPI)
    
    # Time series plot
    ax1.plot(y_true, label='Actual', linewidth=2, alpha=0.8)
    ax1.plot(y_pred, label='Predicted', linewidth=2, alpha=0.8)
    ax1.set_title(f'{title} - Time Series', fontweight='bold')
    ax1.set_xlabel('Sample')
    ax1.set_ylabel('Price ($)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Scatter plot
    ax2.scatter(y_true, y_pred, alpha=0.5, s=50)
    
    # Add perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax2.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    ax2.set_title(f'{title} - Scatter', fontweight='bold')
    ax2.set_xlabel('Actual Price ($)')
    ax2.set_ylabel('Predicted Price ($)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save:
        filepath = os.path.join(FIGURES_DIR, filename)
        plt.savefig(filepath, dpi=DPI, bbox_inches='tight')
        print(f"✅ Plot saved: {filepath}")
    
    plt.show()


def plot_residuals(y_true, y_pred, save=True):
    """
    Plot prediction residuals
    
    Args:
        y_true: True values
        y_pred: Predicted values
        save: Whether to save the plot
    """
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), dpi=DPI)
    
    # Residuals over time
    axes[0].plot(residuals, linewidth=1, alpha=0.8)
    axes[0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0].set_title('Residuals Over Time', fontweight='bold')
    axes[0].set_xlabel('Sample')
    axes[0].set_ylabel('Residual ($)')
    axes[0].grid(True, alpha=0.3)
    
    # Residuals distribution
    axes[1].hist(residuals, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    axes[1].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[1].set_title('Residuals Distribution', fontweight='bold')
    axes[1].set_xlabel('Residual ($)')
    axes[1].set_ylabel('Frequency')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save:
        filepath = os.path.join(FIGURES_DIR, 'residuals.png')
        plt.savefig(filepath, dpi=DPI, bbox_inches='tight')
        print(f"✅ Plot saved: {filepath}")
    
    plt.show()


def plot_feature_importance(importance_df, top_n=15, save=True):
    """
    Plot feature importance
    
    Args:
        importance_df: DataFrame with feature and importance columns
        top_n: Number of top features to plot
        save: Whether to save the plot
    """
    top_features = importance_df.head(top_n)
    
    fig, ax = plt.subplots(figsize=(10, 8), dpi=DPI)
    
    ax.barh(range(len(top_features)), top_features['importance'], color='steelblue')
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['feature'])
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title(f'Top {top_n} Feature Importances', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    
    if save:
        filepath = os.path.join(FIGURES_DIR, 'feature_importance.png')
        plt.savefig(filepath, dpi=DPI, bbox_inches='tight')
        print(f"✅ Plot saved: {filepath}")
    
    plt.show()


def create_dashboard(df, save=True):
    """
    Create a comprehensive dashboard
    
    Args:
        df: DataFrame with all data
        save: Whether to save the plot
    """
    fig = plt.figure(figsize=(18, 12), dpi=DPI)
    
    # Price history
    ax1 = plt.subplot(3, 2, 1)
    dates = pd.to_datetime(df['Date'])
    ax1.plot(dates, df['Close'], linewidth=2, color='steelblue')
    ax1.set_title('Price History', fontweight='bold')
    ax1.set_ylabel('Price ($)')
    ax1.grid(True, alpha=0.3)
    
    # Volume
    ax2 = plt.subplot(3, 2, 2)
    ax2.bar(dates, df['Volume'], alpha=0.7, color='coral')
    ax2.set_title('Trading Volume', fontweight='bold')
    ax2.set_ylabel('Volume')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Price distribution
    ax3 = plt.subplot(3, 2, 3)
    ax3.hist(df['Close'], bins=50, alpha=0.7, color='green', edgecolor='black')
    ax3.set_title('Price Distribution', fontweight='bold')
    ax3.set_xlabel('Price ($)')
    ax3.set_ylabel('Frequency')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Returns distribution
    ax4 = plt.subplot(3, 2, 4)
    if 'price_change_pct' in df.columns:
        returns = df['price_change_pct'].dropna()
        ax4.hist(returns, bins=50, alpha=0.7, color='purple', edgecolor='black')
        ax4.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax4.set_title('Daily Returns Distribution', fontweight='bold')
    ax4.set_xlabel('Return (%)')
    ax4.set_ylabel('Frequency')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Moving averages
    ax5 = plt.subplot(3, 2, 5)
    ax5.plot(dates, df['Close'], label='Close', linewidth=2, alpha=0.8)
    ma_cols = [col for col in df.columns if col.startswith('MA_')]
    for col in ma_cols[:3]:  # Plot first 3 MAs
        if col in df.columns:
            ax5.plot(dates, df[col], label=col, linewidth=1.5, alpha=0.7)
    ax5.set_title('Moving Averages', fontweight='bold')
    ax5.set_ylabel('Price ($)')
    ax5.legend(loc='best', fontsize=8)
    ax5.grid(True, alpha=0.3)
    
    # Technical indicators
    ax6 = plt.subplot(3, 2, 6)
    if 'RSI' in df.columns:
        ax6.plot(dates, df['RSI'], linewidth=1.5, color='orange')
        ax6.axhline(y=70, color='r', linestyle='--', alpha=0.7, label='Overbought')
        ax6.axhline(y=30, color='g', linestyle='--', alpha=0.7, label='Oversold')
        ax6.set_ylim([0, 100])
        ax6.legend(loc='best', fontsize=8)
    ax6.set_title('RSI (Relative Strength Index)', fontweight='bold')
    ax6.set_ylabel('RSI')
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save:
        filepath = os.path.join(FIGURES_DIR, 'dashboard.png')
        plt.savefig(filepath, dpi=DPI, bbox_inches='tight')
        print(f"✅ Dashboard saved: {filepath}")
    
    plt.show()


if __name__ == "__main__":
    print_section("Testing Visualization Module")
    
    # Load processed data
    filepath = os.path.join(DATA_PROCESSED_DIR, f"{STOCK_SYMBOL}_processed.csv")
    
    if os.path.exists(filepath):
        print(f"📂 Loading data from: {filepath}")
        df = pd.read_csv(filepath)
        
        print("\n📊 Generating visualizations...")
        
        # Generate all plots
        plot_price_history(df)
        plot_volume(df)
        plot_moving_averages(df)
        plot_correlation_matrix(df)
        create_dashboard(df)
        
        print("\n✅ All visualizations generated!")
        print(f"📁 Check the '{FIGURES_DIR}' folder for saved plots.")
    else:
        print(f"❌ Processed data not found at: {filepath}")
        print("Please run the data preprocessing pipeline first.")