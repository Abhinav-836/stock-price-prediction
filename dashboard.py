"""
Interactive Stock Prediction Dashboard - COMPLETE WITH ALL GRAPHS
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
from datetime import datetime, timedelta
import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from utils.config import STOCK_SYMBOL, START_DATE, END_DATE
from src.data_loader import StockDataLoader
from src.preprocess import preprocess_data
from src.train_model import prepare_features_target, split_data, ModelTrainer
from src.predict import PredictionEngine

# Page configuration
st.set_page_config(
    page_title="📈 Stock Prediction Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        text-align: center;
        margin-bottom: 1rem;
        padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
        color: white;
    }
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border: none;
        padding: 0.75rem;
        border-radius: 10px;
    }
    .stButton > button:hover {
        transform: scale(1.02);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    </style>
""", unsafe_allow_html=True)


class StockDashboard:
    """Complete Stock Prediction Dashboard with ALL Graphs"""
    
    def __init__(self):
        self.symbol = None
        self.df_raw = None
        self.df_processed = None
        self.predictions = None
        self.models = None
        self.results = None
        self.start_date = None
        self.end_date = None
        self.best_model_name = None
        self.feature_cols = None
        
    def load_data_with_date_range(self, symbol, start_date, end_date, force_download=False, use_synthetic=False):
        """Load data for specific date range"""
        try:
            loader = StockDataLoader(symbol=symbol)
            self.df_raw = loader.load_stock_data(symbol, force_download=force_download)
            
            if self.df_raw is None:
                st.error(f"Could not load data for {symbol}")
                return False
            
            # Handle timezone properly
            if not pd.api.types.is_datetime64_any_dtype(self.df_raw['Date']):
                self.df_raw['Date'] = pd.to_datetime(self.df_raw['Date'], utc=True)
                self.df_raw['Date'] = self.df_raw['Date'].dt.tz_localize(None)
            else:
                if self.df_raw['Date'].dt.tz is not None:
                    self.df_raw['Date'] = self.df_raw['Date'].dt.tz_localize(None)
            
            # Filter by date range
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            
            mask = (self.df_raw['Date'] >= start_dt) & (self.df_raw['Date'] <= end_dt)
            filtered_df = self.df_raw[mask]
            
            # If no data in range OR use_synthetic is True, use synthetic
            if len(filtered_df) < 30 or use_synthetic:
                st.info(f"Using synthetic data for {symbol} (only {len(filtered_df)} days in range)")
                from create_enough_data import StockDataGenerator
                days_needed = max(365, (pd.to_datetime(end_date) - pd.to_datetime(start_date)).days)
                generator = StockDataGenerator(symbol=symbol, days=days_needed)
                self.df_raw = generator.create_working_dataset()
                st.success(f"✅ Generated {len(self.df_raw)} days of synthetic data")
            else:
                self.df_raw = filtered_df
                st.success(f"✅ Using REAL data: {len(self.df_raw)} days from {start_date} to {end_date}")
            
            self.symbol = symbol
            self.start_date = start_date
            self.end_date = end_date
            
            with st.spinner("Preprocessing data..."):
                self.df_processed = preprocess_data(self.df_raw, save=False, scale=False)
            
            return True
            
        except Exception as e:
            st.error(f"Error loading data: {e}")
            return False
    
    def train_models_automatically(self):
        """Train models automatically on loaded data"""
        if self.df_processed is None:
            return False
        
        try:
            with st.spinner("🧠 Training models... This may take a moment..."):
                X, y, self.feature_cols = prepare_features_target(self.df_processed)
                X_train, X_test, y_train, y_test = split_data(X, y)
                
                trainer = ModelTrainer(X_train, y_train, X_test, y_test, self.feature_cols)
                self.results = trainer.train_and_evaluate_all(save_models=True)
                self.models = trainer.models
                
                # Get best model
                best_model, best_name = trainer.get_best_model()
                self.best_model_name = best_name
                
                return True
                
        except Exception as e:
            st.error(f"Error training models: {e}")
            import traceback
            st.error(traceback.format_exc())
            return False
    
    def generate_predictions(self, days=30):
        """Generate predictions using ALL features"""
        if self.df_processed is None or self.models is None:
            return None
        
        try:
            with st.spinner(f"🔮 Generating predictions for next {days} days..."):
                if self.best_model_name is None or self.best_model_name not in self.models:
                    self.best_model_name = max(self.results.keys(), key=lambda x: self.results[x]['metrics']['test_r2'])
                
                model = self.models[self.best_model_name]
                
                if self.feature_cols is None:
                    exclude_cols = ['Date', 'target', 'target_class'] + [col for col in self.df_processed.columns if 'target_' in col]
                    self.feature_cols = [col for col in self.df_processed.columns if col not in exclude_cols]
                    self.feature_cols = [col for col in self.feature_cols if not col.endswith('d')]
                    self.feature_cols = [col for col in self.feature_cols if 'target' not in col]
                
                engine = PredictionEngine(model_name=self.best_model_name, model=model)
                self.predictions = engine.predict_future(self.df_processed, self.feature_cols, days=days)
                
                return self.predictions
                
        except Exception as e:
            st.error(f"Error generating predictions: {e}")
            import traceback
            st.error(traceback.format_exc())
            return None
    
    def render_sidebar(self):
        """Render sidebar with controls"""
        with st.sidebar:
            st.markdown("### 🔧 Controls")
            
            symbol = st.text_input("Stock Symbol", value="AAPL").upper()
            
            st.markdown("### 📅 Date Range")
            
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input(
                    "Start Date",
                    value=datetime(2020, 1, 1),  # Use existing data range
                    max_value=datetime.now()
                )
            with col2:
                end_date = st.date_input(
                    "End Date",
                    value=datetime(2023, 12, 29),  # Use existing data range
                    max_value=datetime.now()
                )
            
            if start_date >= end_date:
                st.warning("⚠️ Start date must be before end date")
                st.stop()
            
            st.markdown("### 🎯 Prediction Settings")
            prediction_days = st.slider(
                "Prediction Horizon (Days)",
                min_value=1,
                max_value=90,
                value=30
            )
            
            st.markdown("### 🔄 Data Options")
            use_synthetic = st.checkbox("Use Synthetic Data", value=False)
            force_refresh = st.checkbox("Force Refresh Data", value=False)
            
            process_btn = st.button("🚀 Process & Analyze", type="primary")
            
            st.markdown("---")
            st.markdown("### ℹ️ Info")
            st.markdown(f"**Symbol:** {symbol}")
            st.markdown(f"**Date Range:** {start_date} to {end_date}")
            if self.df_raw is not None:
                st.markdown(f"**Data Points:** {len(self.df_raw)} days")
                st.markdown(f"**Data Type:** {'Synthetic' if 'synthetic' in str(self.df_raw).lower() else 'Real'}")
            if self.best_model_name is not None:
                st.markdown(f"**Best Model:** {self.best_model_name}")
            
            return {
                'symbol': symbol,
                'start_date': start_date,
                'end_date': end_date,
                'prediction_days': prediction_days,
                'use_synthetic': use_synthetic,
                'force_refresh': force_refresh,
                'process_btn': process_btn
            }
    
    def render_metrics(self):
        """Render key metrics"""
        if self.df_raw is None or len(self.df_raw) < 2:
            return
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        latest = self.df_raw.iloc[-1]
        prev = self.df_raw.iloc[-2] if len(self.df_raw) > 1 else latest
        
        with col1:
            price_change = ((latest['Close'] - prev['Close']) / prev['Close'] * 100) if prev['Close'] != 0 else 0
            st.metric(
                label="💰 Current Price",
                value=f"${latest['Close']:.2f}",
                delta=f"{price_change:+.2f}%"
            )
        
        with col2:
            vol_change = ((latest['Volume'] - self.df_raw['Volume'].mean()) / self.df_raw['Volume'].mean() * 100)
            st.metric(
                label="📊 Volume",
                value=f"{latest['Volume']/1e6:.1f}M",
                delta=f"{vol_change:+.1f}%"
            )
        
        with col3:
            st.metric(
                label="📈 Day Range",
                value=f"${latest['Low']:.2f} - ${latest['High']:.2f}",
                delta=f"Spread: ${latest['High'] - latest['Low']:.2f}"
            )
        
        with col4:
            volatility = self.df_raw['Close'].pct_change().std() * np.sqrt(252) * 100
            st.metric(
                label="📉 Volatility",
                value=f"{volatility:.1f}%"
            )
        
        with col5:
            total_return = ((latest['Close'] - self.df_raw['Close'].iloc[0]) / self.df_raw['Close'].iloc[0] * 100)
            st.metric(
                label="📊 Total Return",
                value=f"{total_return:+.1f}%"
            )
    
    def render_price_chart(self):
        """Render interactive price chart"""
        if self.df_raw is None:
            return
        
        fig = make_subplots(
            rows=4, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.05,
            subplot_titles=('Price History with Moving Averages', 'Trading Volume', 
                           'RSI (Relative Strength Index)', 'MACD'),
            row_heights=[0.5, 0.15, 0.175, 0.175]
        )
        
        # Candlestick
        fig.add_trace(
            go.Candlestick(
                x=self.df_raw['Date'],
                open=self.df_raw['Open'],
                high=self.df_raw['High'],
                low=self.df_raw['Low'],
                close=self.df_raw['Close'],
                name='Price'
            ),
            row=1, col=1
        )
        
        # Moving averages
        for period, color in [(20, 'orange'), (50, 'green'), (200, 'red')]:
            if len(self.df_raw) > period:
                ma = self.df_raw['Close'].rolling(period).mean()
                fig.add_trace(
                    go.Scatter(
                        x=self.df_raw['Date'],
                        y=ma,
                        name=f'SMA {period}',
                        line=dict(color=color, width=1)
                    ),
                    row=1, col=1
                )
        
        # Bollinger Bands
        if len(self.df_raw) > 20:
            rolling_mean = self.df_raw['Close'].rolling(20).mean()
            rolling_std = self.df_raw['Close'].rolling(20).std()
            fig.add_trace(
                go.Scatter(
                    x=self.df_raw['Date'],
                    y=rolling_mean + (rolling_std * 2),
                    name='BB Upper',
                    line=dict(color='rgba(128,128,128,0.5)', width=1, dash='dash')
                ),
                row=1, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=self.df_raw['Date'],
                    y=rolling_mean - (rolling_std * 2),
                    name='BB Lower',
                    line=dict(color='rgba(128,128,128,0.5)', width=1, dash='dash')
                ),
                row=1, col=1
            )
        
        # Volume
        colors = ['green' if close >= open else 'red' 
                  for close, open in zip(self.df_raw['Close'], self.df_raw['Open'])]
        fig.add_trace(
            go.Bar(
                x=self.df_raw['Date'],
                y=self.df_raw['Volume'],
                name='Volume',
                marker_color=colors,
                opacity=0.5
            ),
            row=2, col=1
        )
        
        # RSI
        if len(self.df_raw) > 14:
            delta = self.df_raw['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            
            fig.add_trace(
                go.Scatter(
                    x=self.df_raw['Date'],
                    y=rsi,
                    name='RSI',
                    line=dict(color='purple')
                ),
                row=3, col=1
            )
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
            fig.update_yaxes(range=[0, 100], row=3, col=1)
        
        # MACD
        if len(self.df_raw) > 26:
            exp1 = self.df_raw['Close'].ewm(span=12, adjust=False).mean()
            exp2 = self.df_raw['Close'].ewm(span=26, adjust=False).mean()
            macd = exp1 - exp2
            signal = macd.ewm(span=9, adjust=False).mean()
            
            fig.add_trace(
                go.Scatter(
                    x=self.df_raw['Date'],
                    y=macd,
                    name='MACD',
                    line=dict(color='blue')
                ),
                row=4, col=1
            )
            fig.add_trace(
                go.Scatter(
                    x=self.df_raw['Date'],
                    y=signal,
                    name='Signal',
                    line=dict(color='red')
                ),
                row=4, col=1
            )
            
            macd_diff = macd - signal
            colors_macd = ['green' if val >= 0 else 'red' for val in macd_diff]
            fig.add_trace(
                go.Bar(
                    x=self.df_raw['Date'],
                    y=macd_diff,
                    name='MACD Histogram',
                    marker_color=colors_macd,
                    opacity=0.5
                ),
                row=4, col=1
            )
        
        fig.update_layout(
            height=800,
            template='plotly_dark',
            showlegend=True,
            xaxis_rangeslider_visible=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def render_correlation_matrix(self):
        """Render correlation matrix"""
        if self.df_raw is None:
            return
        
        st.markdown("### 📊 Correlation Matrix")
        
        # Select numeric columns
        corr_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if all(col in self.df_raw.columns for col in corr_cols):
            corr_matrix = self.df_raw[corr_cols].corr()
            
            fig = px.imshow(
                corr_matrix,
                text_auto=True,
                title='Correlation Matrix of OHLC & Volume',
                color_continuous_scale='RdBu_r',
                zmin=-1,
                zmax=1
            )
            fig.update_layout(
                template='plotly_dark',
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def render_predictions(self):
        """Render predictions"""
        if self.predictions is None:
            st.info("💡 Click 'Process & Analyze' to generate predictions.")
            return
        
        st.markdown("### 🔮 Price Predictions")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            fig = go.Figure()
            
            # Historical data
            historical = self.df_raw.tail(100)
            fig.add_trace(
                go.Scatter(
                    x=historical['Date'],
                    y=historical['Close'],
                    name='Historical',
                    line=dict(color='blue', width=2)
                )
            )
            
            # Predictions
            pred_col = 'Predicted_Close'
            pred_dates = self.predictions['Date'] if 'Date' in self.predictions.columns else None
            
            fig.add_trace(
                go.Scatter(
                    x=pred_dates,
                    y=self.predictions[pred_col],
                    name='Predictions',
                    line=dict(color='red', width=2, dash='dash')
                )
            )
            
            # Confidence intervals
            if 'Lower_Bound' in self.predictions.columns:
                fig.add_trace(
                    go.Scatter(
                        x=pred_dates,
                        y=self.predictions['Upper_Bound'],
                        fill=None,
                        mode='lines',
                        name='Upper Bound',
                        line=dict(color='rgba(255,0,0,0.2)')
                    )
                )
                fig.add_trace(
                    go.Scatter(
                        x=pred_dates,
                        y=self.predictions['Lower_Bound'],
                        fill='tonexty',
                        mode='lines',
                        name='Lower Bound',
                        line=dict(color='rgba(255,0,0,0.2)')
                    )
                )
            
            fig.update_layout(
                title='Price Prediction with Confidence Intervals',
                xaxis_title='Date',
                yaxis_title='Price ($)',
                template='plotly_dark',
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            last_price = self.df_raw['Close'].iloc[-1]
            final_price = self.predictions[pred_col].iloc[-1]
            expected_return = ((final_price - last_price) / last_price) * 100
            
            st.markdown(f"""
            ### 📊 Prediction Summary
            
            | Metric | Value |
            |--------|-------|
            | **Current Price** | ${last_price:.2f} |
            | **Predicted Price** | ${final_price:.2f} |
            | **Expected Return** | {expected_return:+.2f}% |
            | **Horizon** | {len(self.predictions)} Days |
            """)
            
            if expected_return > 3:
                st.success(f"✅ **Recommendation:** BUY ({expected_return:+.2f}%)")
            elif expected_return < -3:
                st.error(f"❌ **Recommendation:** SELL ({expected_return:+.2f}%)")
            else:
                st.warning(f"⚠️ **Recommendation:** HOLD ({expected_return:+.2f}%)")
    
    def render_returns_distribution(self):
        """Render returns distribution"""
        if self.df_raw is None:
            return
        
        st.markdown("### 📊 Returns Distribution")
        
        returns = self.df_raw['Close'].pct_change().dropna() * 100
        
        fig = px.histogram(
            returns,
            title='Daily Returns Distribution',
            nbins=50,
            color_discrete_sequence=['blue']
        )
        fig.add_vline(x=0, line_dash="dash", line_color="red")
        fig.update_layout(
            template='plotly_dark',
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Returns statistics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Mean Return", f"{returns.mean():.3f}%")
        with col2:
            st.metric("Std Dev", f"{returns.std():.3f}%")
        with col3:
            st.metric("Sharpe Ratio", f"{returns.mean()/returns.std()*np.sqrt(252):.3f}")
    
    def render_risk_metrics(self):
        """Render risk metrics"""
        if self.df_raw is None or len(self.df_raw) < 2:
            return
        
        st.markdown("### 📊 Risk Metrics")
        
        returns = self.df_raw['Close'].pct_change().dropna()
        
        # Calculate metrics
        var_95 = np.percentile(returns, 5) * 100
        var_99 = np.percentile(returns, 1) * 100
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252)
        
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min() * 100
        win_rate = (returns > 0).mean() * 100
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📉 VaR (95%)", f"{var_95:.2f}%")
            st.metric("📉 VaR (99%)", f"{var_99:.2f}%")
        
        with col2:
            st.metric("📊 Sharpe Ratio", f"{sharpe:.2f}")
        
        with col3:
            st.metric("📉 Max Drawdown", f"{max_drawdown:.2f}%")
        
        with col4:
            st.metric("📈 Win Rate", f"{win_rate:.1f}%")
        
        # Drawdown chart
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=self.df_raw['Date'][1:],
                y=drawdown * 100,
                name='Drawdown',
                fill='tozeroy',
                line=dict(color='red')
            )
        )
        fig.update_layout(
            title='Drawdown History',
            xaxis_title='Date',
            yaxis_title='Drawdown (%)',
            template='plotly_dark',
            height=400
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def render_feature_importance(self):
        """Render feature importance"""
        if self.results is None:
            st.info("Train models first to see feature importance")
            return
        
        best_name = max(self.results.keys(), key=lambda x: self.results[x]['metrics']['test_r2'])
        
        if hasattr(self.models[best_name], 'feature_importances_'):
            importances = self.models[best_name].feature_importances_
            X, y, feature_cols = prepare_features_target(self.df_processed)
            
            importance_df = pd.DataFrame({
                'feature': feature_cols[:len(importances)],
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            top_20 = importance_df.head(20)
            
            fig = go.Figure()
            fig.add_trace(
                go.Bar(
                    x=top_20['importance'],
                    y=top_20['feature'],
                    orientation='h',
                    marker_color='steelblue'
                )
            )
            fig.update_layout(
                title=f'Top 20 Feature Importances ({best_name})',
                xaxis_title='Importance',
                yaxis_title='Feature',
                template='plotly_dark',
                height=500,
                yaxis=dict(autorange="reversed")
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def render_data_tab(self):
        """Render data tab"""
        if self.df_raw is None:
            return
        
        st.markdown("### 📊 Data Overview")
        st.dataframe(
            self.df_raw.tail(20).style.format({
                'Open': '${:.2f}',
                'High': '${:.2f}',
                'Low': '${:.2f}',
                'Close': '${:.2f}',
                'Volume': '{:,.0f}'
            }),
            use_container_width=True
        )
        
        csv = self.df_raw.to_csv(index=False)
        st.download_button(
            label="📥 Download Data as CSV",
            data=csv,
            file_name=f"{self.symbol}_data.csv",
            mime="text/csv"
        )
    
    def run(self):
        """Main dashboard loop"""
        st.markdown('<h1 class="main-header">📈 Stock Price Prediction Dashboard</h1>', unsafe_allow_html=True)
        
        config = self.render_sidebar()
        
        if config['process_btn']:
            # Step 1: Load data
            with st.spinner(f"Loading {config['symbol']} data..."):
                success = self.load_data_with_date_range(
                    config['symbol'],
                    config['start_date'],
                    config['end_date'],
                    force_download=config['force_refresh'],
                    use_synthetic=config['use_synthetic']
                )
            
            if success:
                st.success(f"✅ Data loaded successfully! ({len(self.df_raw)} days)")
                
                # Step 2: Train models
                train_success = self.train_models_automatically()
                if train_success:
                    st.success(f"✅ Models trained successfully! Best: {self.best_model_name}")
                    
                    # Step 3: Generate predictions
                    predictions = self.generate_predictions(config['prediction_days'])
                    if predictions is not None:
                        st.success(f"✅ Predictions generated for {len(predictions)} days!")
                
                # Step 4: Display ALL graphs
                st.markdown("---")
                self.render_metrics()
                st.markdown("---")
                
                # ALL 6 GRAPHS
                self.render_price_chart()
                st.markdown("---")
                
                self.render_returns_distribution()
                st.markdown("---")
                
                self.render_correlation_matrix()
                st.markdown("---")
                
                self.render_risk_metrics()
                st.markdown("---")
                
                # Tabs for additional content
                tab1, tab2, tab3 = st.tabs(["📈 Predictions", "🔍 Feature Importance", "📋 Data"])
                
                with tab1:
                    self.render_predictions()
                
                with tab2:
                    self.render_feature_importance()
                
                with tab3:
                    self.render_data_tab()
                    
            else:
                st.error("❌ Failed to load data. Please try again.")
        else:
            st.info("👈 Select a stock, date range, and click 'Process & Analyze' to begin!")
            
            st.markdown("### 🚀 Features:")
            st.markdown("""
            - **Real-time Data Loading** from Yahoo Finance
            - **Automatic Data Generation** if no data available
            - **Multiple ML Models** (XGBoost, Random Forest, Linear Regression)
            - **Interactive Charts** with Plotly
            - **Risk Metrics & Analysis**
            - **Feature Importance** visualization
            - **Confidence Intervals** for predictions
            - **Downloadable Data** in CSV format
            """)


if __name__ == "__main__":
    dashboard = StockDashboard()
    dashboard.run()