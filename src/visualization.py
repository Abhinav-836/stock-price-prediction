"""
Visualization module for stock price prediction - FIXED

Fixes vs previous version:
  * plot_predictions_interactive expected a separate frame with 'upper'/'lower' keys, but
    PredictionEngine returns 'Lower_Bound'/'Upper_Bound' in the same frame -> bands never drew.
  * Upper/lower band traces were in the wrong order for fill='tonexty'.
  * Figures were never closed -> memory leak in long-running processes (Streamlit).
  * Saving failed if the output folder didn't exist; legacy plot_feature_importance ignored save.
  * Indicators are drawn only when there is enough data (no all-NaN traces).
  * Dropped unused seaborn import (slow import, not needed).
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import sys
from typing import Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.config import FIGURES_DIR, DATA_PROCESSED_DIR, STOCK_SYMBOL
from utils.helpers import print_section


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    return 100 - 100 / (1 + gain / loss)


def _save_fig(fig, filename: str):
    os.makedirs(FIGURES_DIR, exist_ok=True)
    filepath = os.path.join(FIGURES_DIR, filename)
    fig.savefig(filepath, dpi=150, bbox_inches='tight')
    print(f"✅ Plot saved: {filepath}")
    return filepath


def _finish(fig, save: bool, filename: str, show: bool = True):
    plt.tight_layout()
    if save:
        _save_fig(fig, filename)
    if show:
        plt.show()
    plt.close(fig)


class StockVisualizer:
    """Interactive (plotly) and static (matplotlib) stock charts"""

    def __init__(self, df: pd.DataFrame, symbol: str = STOCK_SYMBOL):
        self.df = df.copy()
        self.symbol = symbol
        if 'Date' in self.df.columns:
            self.df['Date'] = pd.to_datetime(self.df['Date'])

    def _volume_colors(self):
        return np.where(self.df['Close'] >= self.df['Open'], 'green', 'red')

    def plot_price_history_interactive(self) -> go.Figure:
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                            subplot_titles=('Price History', 'Volume', 'RSI (14)'),
                            row_heights=[0.6, 0.2, 0.2])
        d = self.df
        fig.add_trace(go.Candlestick(x=d['Date'], open=d['Open'], high=d['High'], low=d['Low'],
                                     close=d['Close'], name='Price'), row=1, col=1)
        for period, color in [(20, 'orange'), (50, 'green'), (200, 'red')]:
            if len(d) > period:
                fig.add_trace(go.Scatter(x=d['Date'], y=d['Close'].rolling(period).mean(), name=f'MA {period}',
                                         line=dict(color=color, width=1)), row=1, col=1)
        fig.add_trace(go.Bar(x=d['Date'], y=d['Volume'], name='Volume', marker_color=self._volume_colors(),
                             opacity=0.5), row=2, col=1)
        fig.add_trace(go.Scatter(x=d['Date'], y=_rsi(d['Close']), name='RSI', line=dict(color='purple')),
                      row=3, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
        fig.update_layout(title=f'{self.symbol} Stock Analysis', height=800, template='plotly_dark',
                          showlegend=True, xaxis_rangeslider_visible=False)
        return fig

    def plot_price_history(self, save=True, show=True):
        d = self.df
        fig, axes = plt.subplots(3, 1, figsize=(15, 10), dpi=100, sharex=True)
        axes[0].plot(d['Date'], d['Close'], linewidth=2, color='steelblue')
        axes[0].set_title(f'{self.symbol} Price History', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Price')
        axes[0].grid(True, alpha=0.3)

        axes[1].bar(d['Date'], d['Volume'], color=self._volume_colors(), alpha=0.7)
        axes[1].set_title('Trading Volume', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3, axis='y')

        axes[2].plot(d['Date'], _rsi(d['Close']), linewidth=2, color='purple')
        axes[2].axhline(y=70, color='red', linestyle='--', alpha=0.7)
        axes[2].axhline(y=30, color='green', linestyle='--', alpha=0.7)
        axes[2].set_title('RSI (14)', fontsize=14, fontweight='bold')
        axes[2].set_ylim(0, 100)
        axes[2].grid(True, alpha=0.3)
        _finish(fig, save, f'{self.symbol}_price_analysis.png', show)

    def plot_predictions_interactive(self, predictions_df: pd.DataFrame,
                                     confidence_intervals: Optional[pd.DataFrame] = None) -> go.Figure:
        fig = go.Figure()
        hist = self.df.tail(100)
        fig.add_trace(go.Scatter(x=hist['Date'], y=hist['Close'], name='Historical', line=dict(color='#4f8bf9', width=2)))

        lower = upper = None
        if confidence_intervals is not None:
            lower, upper = confidence_intervals.get('lower'), confidence_intervals.get('upper')
        elif {'Lower_Bound', 'Upper_Bound'}.issubset(predictions_df.columns):
            lower, upper = predictions_df['Lower_Bound'], predictions_df['Upper_Bound']

        if lower is not None and upper is not None:
            fig.add_trace(go.Scatter(x=predictions_df['Date'], y=upper, mode='lines', name='Upper 95%',
                                     line=dict(width=0), showlegend=False))
            fig.add_trace(go.Scatter(x=predictions_df['Date'], y=lower, mode='lines', name='95% band',
                                     fill='tonexty', fillcolor='rgba(255,99,71,0.2)', line=dict(width=0)))

        fig.add_trace(go.Scatter(x=predictions_df['Date'], y=predictions_df['Predicted_Close'],
                                 name='Forecast', line=dict(color='tomato', width=2, dash='dash')))
        fig.update_layout(title=f'{self.symbol} Price Forecast', xaxis_title='Date', yaxis_title='Price',
                          template='plotly_dark', height=500, hovermode='x unified')
        return fig

    def plot_feature_importance(self, importance_df: pd.DataFrame, top_n: int = 20, save=True, show=True):
        top = importance_df.head(top_n)
        fig, ax = plt.subplots(figsize=(12, max(4, top_n * 0.45)), dpi=100)
        ax.barh(range(len(top)), top['importance'], color='steelblue', alpha=0.8)
        ax.set_yticks(range(len(top)))
        ax.set_yticklabels(top['feature'], fontsize=10)
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_title(f'Top {top_n} Feature Importances', fontsize=14, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
        _finish(fig, save, f'{self.symbol}_feature_importance.png', show)

    def plot_correlation_matrix(self) -> go.Figure:
        numeric = self.df.select_dtypes(include=[np.number])
        features = [c for c in numeric.columns if not c.startswith('target')]
        if len(features) > 30 and 'Close' in features:
            corr_close = numeric[features].corrwith(numeric['Close']).abs()
            features = corr_close.sort_values(ascending=False).head(30).index.tolist()
        corr = numeric[features].corr()
        fig = go.Figure(go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns,
                                   colorscale='RdBu_r', zmin=-1, zmax=1))
        fig.update_layout(title='Feature Correlation Matrix', width=800, height=800, template='plotly_dark')
        return fig

    def plot_dashboard_interactive(self) -> go.Figure:
        d = self.df
        fig = make_subplots(rows=3, cols=2, vertical_spacing=0.1, horizontal_spacing=0.1,
                            subplot_titles=('Price History', 'Trading Volume', 'Price Distribution',
                                            'Returns Distribution', 'Moving Averages', 'RSI'))
        fig.add_trace(go.Scatter(x=d['Date'], y=d['Close'], name='Close', line=dict(color='#4f8bf9')), row=1, col=1)
        fig.add_trace(go.Bar(x=d['Date'], y=d['Volume'], name='Volume', marker_color=self._volume_colors(),
                             opacity=0.7), row=1, col=2)
        fig.add_trace(go.Histogram(x=d['Close'], nbinsx=50, name='Price Dist.', marker_color='green'), row=2, col=1)
        fig.add_trace(go.Histogram(x=d['Close'].pct_change().dropna() * 100, nbinsx=50, name='Returns %',
                                   marker_color='purple'), row=2, col=2)
        fig.add_trace(go.Scatter(x=d['Date'], y=d['Close'], name='Close ', line=dict(color='#4f8bf9', width=1)),
                      row=3, col=1)
        for period, color in [(20, 'orange'), (50, 'green')]:
            if len(d) > period:
                fig.add_trace(go.Scatter(x=d['Date'], y=d['Close'].rolling(period).mean(), name=f'MA {period}',
                                         line=dict(color=color, width=1)), row=3, col=1)
        fig.add_trace(go.Scatter(x=d['Date'], y=_rsi(d['Close']), name='RSI', line=dict(color='red')), row=3, col=2)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=2)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=2)
        fig.update_layout(title=f'{self.symbol} Dashboard', height=900, template='plotly_dark', showlegend=True)
        return fig


# ============================================================================
# LEGACY SUPPORT FUNCTIONS
# ============================================================================

def plot_price_history(df, save=True):
    StockVisualizer(df).plot_price_history(save=save)


def plot_volume(df, save=True):
    fig, ax = plt.subplots(figsize=(15, 6), dpi=100)
    colors = np.where(df['Close'] >= df['Open'], 'green', 'red')
    ax.bar(pd.to_datetime(df['Date']), df['Volume'], color=colors, alpha=0.7)
    ax.set_title(f'{STOCK_SYMBOL} Trading Volume', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    _finish(fig, save, 'trading_volume.png')


def plot_moving_averages(df, save=True):
    fig, ax = plt.subplots(figsize=(15, 6), dpi=100)
    dates = pd.to_datetime(df['Date'])
    ax.plot(dates, df['Close'], label='Close Price', linewidth=2, alpha=0.8)
    ma_cols = [c for c in df.columns if c.startswith('SMA_') and c.split('_')[-1].isdigit()]
    for col, color in zip(ma_cols[:4], ['orange', 'green', 'red', 'purple']):
        ax.plot(dates, df[col], label=col, linewidth=1.5, alpha=0.7, color=color)
    ax.set_title(f'{STOCK_SYMBOL} Price with Moving Averages', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    _finish(fig, save, 'moving_averages.png')


def plot_correlation_matrix(df, save=True):
    fig = StockVisualizer(df).plot_correlation_matrix()
    fig.show()
    if save:
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.write_html(os.path.join(FIGURES_DIR, 'correlation_matrix.html'))


def plot_predictions_vs_actual(y_true, y_pred, title='Predictions vs Actual', save=True,
                               filename='predictions_vs_actual.png'):
    y_true, y_pred = np.asarray(y_true, dtype=float), np.asarray(y_pred, dtype=float)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5), dpi=100)
    ax1.plot(y_true, label='Actual', linewidth=2, alpha=0.8)
    ax1.plot(y_pred, label='Predicted', linewidth=2, alpha=0.8)
    ax1.set_title(f'{title} - Time Series', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax2.scatter(y_true, y_pred, alpha=0.5, s=30)
    lo, hi = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    ax2.plot([lo, hi], [lo, hi], 'r--', linewidth=2, label='Perfect Prediction')
    ax2.set_title(f'{title} - Scatter', fontweight='bold')
    ax2.set_xlabel('Actual')
    ax2.set_ylabel('Predicted')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    _finish(fig, save, filename)


def plot_residuals(y_true, y_pred, save=True):
    residuals = np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), dpi=100)
    axes[0].plot(residuals, linewidth=1, alpha=0.8)
    axes[0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0].set_title('Residuals Over Time', fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    axes[1].hist(residuals, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    axes[1].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[1].set_title('Residuals Distribution', fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    _finish(fig, save, 'residuals.png')


def plot_feature_importance(importance_df, top_n=15, save=True):
    StockVisualizer(pd.DataFrame()).plot_feature_importance(importance_df, top_n, save=save)


def create_dashboard(df, save=True):
    visualizer = StockVisualizer(df)
    fig = visualizer.plot_dashboard_interactive()
    fig.show()
    if save:
        os.makedirs(FIGURES_DIR, exist_ok=True)
        fig.write_html(os.path.join(FIGURES_DIR, 'dashboard.html'))
    visualizer.plot_price_history(save=save)


if __name__ == "__main__":
    print_section("Testing Visualization")
    filepath = os.path.join(DATA_PROCESSED_DIR, f"{STOCK_SYMBOL}_processed.csv")
    if os.path.exists(filepath):
        viz = StockVisualizer(pd.read_csv(filepath))
        viz.plot_price_history_interactive().show()
        viz.plot_dashboard_interactive().show()
        viz.plot_correlation_matrix().show()
    else:
        print(f"❌ Processed data not found at: {filepath}")