"""
Visualization module for stock price prediction - Enhanced with interactive charts
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import os
import sys
from typing import Dict, Any, Optional, List

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import FIGURES_DIR, DATA_PROCESSED_DIR, STOCK_SYMBOL
from utils.helpers import print_section


class StockVisualizer:
    """
    Advanced stock visualization with interactive and static plots
    """
    
    def __init__(self, df: pd.DataFrame, symbol: str = STOCK_SYMBOL):
        self.df = df.copy()
        self.symbol = symbol
        
    def plot_price_history_interactive(self) -> go.Figure:
        """Create interactive price history chart"""
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=('Price History', 'Volume', 'Technical Indicators'),
            row_heights=[0.6, 0.2, 0.2]
        )
        
        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=self.df['Date'],
                open=self.df['Open'],
                high=self.df['High'],
                low=self.df['Low'],
                close=self.df['Close'],
                name='Price'
            ),
            row=1, col=1
        )
        
        # Moving averages
        for period, color in [(20, 'orange'), (50, 'green'), (200, 'red')]:
            ma = self.df['Close'].rolling(period).mean()
            fig.add_trace(
                go.Scatter(
                    x=self.df['Date'],
                    y=ma,
                    name=f'MA {period}',
                    line=dict(color=color, width=1)
                ),
                row=1, col=1
            )
        
        # Volume
        colors = ['green' if close >= open else 'red' 
                  for close, open in zip(self.df['Close'], self.df['Open'])]
        fig.add_trace(
            go.Bar(
                x=self.df['Date'],
                y=self.df['Volume'],
                name='Volume',
                marker_color=colors,
                opacity=0.5
            ),
            row=2, col=1
        )
        
        # RSI
        delta = self.df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        fig.add_trace(
            go.Scatter(
                x=self.df['Date'],
                y=rsi,
                name='RSI',
                line=dict(color='purple')
            ),
            row=3, col=1
        )
        
        # Add overbought/oversold lines
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
        
        # Update layout
        fig.update_layout(
            title=f'{self.symbol} Stock Analysis',
            height=800,
            template='plotly_dark',
            showlegend=True,
            xaxis_rangeslider_visible=False
        )
        
        return fig
    
    def plot_price_history(self, save=True):
        """Static price history plot"""
        fig, axes = plt.subplots(3, 1, figsize=(15, 10), dpi=100)
        
        # Price
        axes[0].plot(self.df['Date'], self.df['Close'], linewidth=2, color='steelblue')
        axes[0].set_title(f'{self.symbol} Price History', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Price ($)')
        axes[0].grid(True, alpha=0.3)
        
        # Volume
        colors = ['green' if close >= open else 'red' 
                  for close, open in zip(self.df['Close'], self.df['Open'])]
        axes[1].bar(self.df['Date'], self.df['Volume'], color=colors, alpha=0.7)
        axes[1].set_title('Trading Volume', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Volume')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        # RSI
        delta = self.df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        axes[2].plot(self.df['Date'], rsi, linewidth=2, color='purple')
        axes[2].axhline(y=70, color='red', linestyle='--', alpha=0.7)
        axes[2].axhline(y=30, color='green', linestyle='--', alpha=0.7)
        axes[2].set_title('RSI (14)', fontsize=14, fontweight='bold')
        axes[2].set_ylabel('RSI')
        axes[2].set_ylim(0, 100)
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save:
            filepath = os.path.join(FIGURES_DIR, f'{self.symbol}_price_analysis.png')
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"✅ Plot saved: {filepath}")
        
        plt.show()
    
    def plot_predictions_interactive(self, predictions_df: pd.DataFrame, 
                                     confidence_intervals: Optional[pd.DataFrame] = None) -> go.Figure:
        """Create interactive predictions chart"""
        fig = go.Figure()
        
        # Historical data (last 100 days)
        historical = self.df.tail(100)
        fig.add_trace(
            go.Scatter(
                x=historical['Date'],
                y=historical['Close'],
                name='Historical',
                line=dict(color='blue', width=2)
            )
        )
        
        # Predictions
        fig.add_trace(
            go.Scatter(
                x=predictions_df['Date'],
                y=predictions_df['Predicted_Close'],
                name='Predictions',
                line=dict(color='red', width=2, dash='dash')
            )
        )
        
        # Confidence intervals
        if confidence_intervals is not None:
            fig.add_trace(
                go.Scatter(
                    x=predictions_df['Date'],
                    y=confidence_intervals['upper'],
                    name='Upper Bound',
                    line=dict(color='rgba(255,0,0,0.2)'),
                    showlegend=True
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=predictions_df['Date'],
                    y=confidence_intervals['lower'],
                    name='Lower Bound',
                    fill='tonexty',
                    line=dict(color='rgba(255,0,0,0.2)'),
                    showlegend=True
                )
            )
        
        fig.update_layout(
            title=f'{self.symbol} Price Prediction',
            xaxis_title='Date',
            yaxis_title='Price ($)',
            template='plotly_dark',
            height=500,
            hovermode='x unified'
        )
        
        return fig
    
    def plot_feature_importance(self, importance_df: pd.DataFrame, top_n: int = 20):
        """Plot feature importance"""
        top_features = importance_df.head(top_n)
        
        fig, ax = plt.subplots(figsize=(12, 10), dpi=100)
        
        ax.barh(range(len(top_features)), top_features['importance'], 
                color='steelblue', alpha=0.8)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'], fontsize=10)
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_title(f'Top {top_n} Feature Importances', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        filepath = os.path.join(FIGURES_DIR, f'{self.symbol}_feature_importance.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"✅ Plot saved: {filepath}")
        
        plt.show()
    
    def plot_correlation_matrix(self) -> go.Figure:
        """Create interactive correlation matrix heatmap"""
        # Select numeric columns
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        # Exclude target columns
        exclude_cols = ['target', 'target_class'] + [col for col in numeric_cols if 'target_' in col]
        features = [col for col in numeric_cols if col not in exclude_cols]
        
        # Limit to top features for readability
        if len(features) > 30:
            # Select based on correlation with close
            correlations = self.df[features].corrwith(self.df['Close']).abs()
            features = correlations.sort_values(ascending=False).head(30).index.tolist()
        
        corr_matrix = self.df[features].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu_r',
            zmin=-1,
            zmax=1,
            showscale=True
        ))
        
        fig.update_layout(
            title='Feature Correlation Matrix',
            width=800,
            height=800,
            template='plotly_dark'
        )
        
        return fig
    
    def plot_dashboard_interactive(self) -> go.Figure:
        """Create comprehensive interactive dashboard"""
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Price History',
                'Trading Volume',
                'Price Distribution',
                'Returns Distribution',
                'Moving Averages',
                'RSI'
            ),
            vertical_spacing=0.1,
            horizontal_spacing=0.1
        )
        
        # Price history
        fig.add_trace(
            go.Scatter(
                x=self.df['Date'],
                y=self.df['Close'],
                name='Close Price',
                line=dict(color='blue')
            ),
            row=1, col=1
        )
        
        # Volume
        colors = ['green' if close >= open else 'red' 
                  for close, open in zip(self.df['Close'], self.df['Open'])]
        fig.add_trace(
            go.Bar(
                x=self.df['Date'],
                y=self.df['Volume'],
                name='Volume',
                marker_color=colors,
                opacity=0.7
            ),
            row=1, col=2
        )
        
        # Price distribution
        fig.add_trace(
            go.Histogram(
                x=self.df['Close'],
                nbinsx=50,
                name='Price Distribution',
                marker_color='green'
            ),
            row=2, col=1
        )
        
        # Returns distribution
        returns = self.df['Close'].pct_change().dropna() * 100
        fig.add_trace(
            go.Histogram(
                x=returns,
                nbinsx=50,
                name='Returns Distribution',
                marker_color='purple'
            ),
            row=2, col=2
        )
        
        # Moving averages
        fig.add_trace(
            go.Scatter(
                x=self.df['Date'],
                y=self.df['Close'],
                name='Close',
                line=dict(color='blue', width=1)
            ),
            row=3, col=1
        )
        
        for period, color in [(20, 'orange'), (50, 'green')]:
            ma = self.df['Close'].rolling(period).mean()
            fig.add_trace(
                go.Scatter(
                    x=self.df['Date'],
                    y=ma,
                    name=f'MA {period}',
                    line=dict(color=color, width=1)
                ),
                row=3, col=1
            )
        
        # RSI
        delta = self.df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        fig.add_trace(
            go.Scatter(
                x=self.df['Date'],
                y=rsi,
                name='RSI',
                line=dict(color='red')
            ),
            row=3, col=2
        )
        
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=2)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=2)
        
        fig.update_layout(
            title=f'{self.symbol} Dashboard',
            height=900,
            template='plotly_dark',
            showlegend=True
        )
        
        return fig


# ============================================================================
# LEGACY SUPPORT FUNCTIONS
# ============================================================================

def plot_price_history(df, save=True):
    visualizer = StockVisualizer(df)
    visualizer.plot_price_history(save=save)


def plot_volume(df, save=True):
    """Plot trading volume"""
    fig, ax = plt.subplots(figsize=(15, 6), dpi=100)
    
    colors = ['green' if close >= open else 'red' 
              for close, open in zip(df['Close'], df['Open'])]
    ax.bar(df['Date'], df['Volume'], color=colors, alpha=0.7)
    ax.set_title(f'{STOCK_SYMBOL} Trading Volume', fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Volume', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save:
        filepath = os.path.join(FIGURES_DIR, 'trading_volume.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"✅ Plot saved: {filepath}")
    
    plt.show()


def plot_moving_averages(df, save=True):
    """Plot price with moving averages"""
    fig, ax = plt.subplots(figsize=(15, 6), dpi=100)
    
    ax.plot(df['Date'], df['Close'], label='Close Price', linewidth=2, alpha=0.8)
    
    ma_cols = [col for col in df.columns if col.startswith('SMA_')]
    colors = ['orange', 'green', 'red', 'purple']
    
    for i, col in enumerate(ma_cols[:4]):
        if col in df.columns:
            ax.plot(df['Date'], df[col], label=col, linewidth=1.5, alpha=0.7, 
                   color=colors[i % len(colors)])
    
    ax.set_title(f'{STOCK_SYMBOL} Price with Moving Averages', 
                fontsize=14, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price ($)', fontsize=12)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save:
        filepath = os.path.join(FIGURES_DIR, 'moving_averages.png')
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"✅ Plot saved: {filepath}")
    
    plt.show()


def plot_correlation_matrix(df, save=True):
    """Plot correlation matrix heatmap"""
    visualizer = StockVisualizer(df)
    fig = visualizer.plot_correlation_matrix()
    fig.show()
    
    if save:
        filepath = os.path.join(FIGURES_DIR, 'correlation_matrix.html')
        fig.write_html(filepath)
        print(f"✅ Interactive plot saved: {filepath}")


def plot_predictions_vs_actual(y_true, y_pred, title='Predictions vs Actual', save=True, filename='predictions_vs_actual.png'):
    """Plot predicted vs actual values"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5), dpi=100)
    
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
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"✅ Plot saved: {filepath}")
    
    plt.show()


def plot_residuals(y_true, y_pred, save=True):
    """Plot prediction residuals"""
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), dpi=100)
    
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
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        print(f"✅ Plot saved: {filepath}")
    
    plt.show()


def plot_feature_importance(importance_df, top_n=15, save=True):
    """Plot feature importance"""
    visualizer = StockVisualizer(pd.DataFrame())
    visualizer.df = pd.DataFrame()  # Dummy
    visualizer.plot_feature_importance(importance_df, top_n)
    if save:
        # Already saved in the method
        pass


def create_dashboard(df, save=True):
    """Create comprehensive dashboard"""
    visualizer = StockVisualizer(df)
    fig = visualizer.plot_dashboard_interactive()
    fig.show()
    
    if save:
        filepath = os.path.join(FIGURES_DIR, 'dashboard.html')
        fig.write_html(filepath)
        print(f"✅ Interactive dashboard saved: {filepath}")
    
    # Also save static version
    visualizer.plot_price_history(save=save)


if __name__ == "__main__":
    print_section("Testing Enhanced Visualization")
    
    # Load processed data
    filepath = os.path.join(DATA_PROCESSED_DIR, f"{STOCK_SYMBOL}_processed.csv")
    
    if os.path.exists(filepath):
        print(f"📂 Loading data from: {filepath}")
        df = pd.read_csv(filepath)
        df['Date'] = pd.to_datetime(df['Date'])
        
        # Create visualizer
        visualizer = StockVisualizer(df)
        
        # Generate plots
        print("\n📊 Generating interactive price chart...")
        fig = visualizer.plot_price_history_interactive()
        fig.show()
        
        print("\n📊 Generating dashboard...")
        fig = visualizer.plot_dashboard_interactive()
        fig.show()
        
        print("\n📊 Generating correlation matrix...")
        fig = visualizer.plot_correlation_matrix()
        fig.show()
        
        print("\n✅ All visualizations complete!")
        
    else:
        print(f"❌ Processed data not found at: {filepath}")