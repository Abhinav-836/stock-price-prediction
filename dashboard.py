"""
Interactive Stock Prediction Dashboard - FIXED for Streamlit Cloud

Fixes vs previous version:
  * Everything disappeared on ANY widget interaction (download button, tabs, slider):
    results lived on a dashboard object that Streamlit re-creates on every rerun, and the
    page only rendered while `st.button` was True (one rerun). Now the request is kept in
    st.session_state and the heavy work is cached (st.cache_resource / st.cache_data).
  * Every click re-trained 9 models + cross-validation and wrote .pkl files to disk.
    Training is cached per (symbol, dates, data mode); CV and model saving are off.
  * Selected dates were ignored: data was downloaded for config.START_DATE..END_DATE and
    then filtered, so most ranges fell back to synthetic data. Data is now fetched for the
    chosen range (+ warm-up history for 200-day indicators).
  * "Data Type: Real/Synthetic" was guessed with `'synthetic' in str(df)` (always "Real").
    The loader now reports the actual source and synthetic data shows a warning banner.
  * Predictions started ~2 weeks before the last real date and were a flat line (see predict.py).
  * Feature-importance tab crashed / mislabelled for pipeline models (SVR, Ridge).
  * Drawdown x-axis was misaligned with the returns series.
  * Price chart legend overlapped the first subplot title (legend now sits above it).
  * Added: model comparison vs a zero-return baseline, so users can see whether the model
    has any real edge, plus currency symbol for NSE/BSE tickers.
"""
import os
import sys
from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.pipeline import run_analysis, forecast, feature_importance  # noqa: E402

st.set_page_config(page_title="Stock Prediction Dashboard", page_icon="📈",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
    .main-header {
        font-size: 2.2rem; text-align: center; margin-bottom: 1rem; padding: 1rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px; color: white;
    }
    .stButton > button {
        width: 100%; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white; font-weight: bold; border: none; padding: 0.75rem; border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

TEMPLATE = 'plotly_dark'


# ============================================================================
# CACHED COMPUTATION (survives reruns; shared across sessions)
# ============================================================================

@st.cache_resource(show_spinner=False, max_entries=6, ttl=3600)
def cached_analysis(symbol: str, start: date, end: date, use_synthetic: bool):
    return run_analysis(symbol, start, end, use_synthetic=use_synthetic, run_cv=False, verbose=False)


@st.cache_data(show_spinner=False, max_entries=32, ttl=3600)
def cached_forecast(symbol: str, start: date, end: date, use_synthetic: bool, days: int, model_name: str):
    analysis = cached_analysis(symbol, start, end, use_synthetic)
    return forecast(analysis, days=days, model_name=model_name)


def currency_for(symbol: str) -> str:
    return '₹' if symbol.endswith(('.NS', '.BO')) else '$'


# ============================================================================
# SIDEBAR
# ============================================================================

def render_sidebar():
    with st.sidebar:
        st.markdown("### 🔧 Controls")
        with st.form("controls"):
            symbol = st.text_input("Stock Symbol", value="AAPL",
                                   help="Yahoo Finance ticker, e.g. AAPL, MSFT, RELIANCE.NS").upper().strip()
            today = date.today()
            col1, col2 = st.columns(2)
            start_date = col1.date_input("Start Date", value=today - timedelta(days=3 * 365), max_value=today)
            end_date = col2.date_input("End Date", value=today, max_value=today)
            prediction_days = st.slider("Forecast Horizon (business days)", 1, 90, 30)
            use_synthetic = st.checkbox("Use Synthetic Data (offline demo)", value=False)
            force_refresh = st.checkbox("Force Refresh (ignore cache)", value=False)
            submitted = st.form_submit_button("🚀 Process & Analyze", type="primary")

        if submitted:
            if not symbol:
                st.error("Enter a symbol.")
            elif start_date >= end_date:
                st.error("Start date must be before end date.")
            else:
                if force_refresh:
                    cached_analysis.clear()
                    cached_forecast.clear()
                st.session_state['params'] = dict(symbol=symbol, start=start_date, end=end_date,
                                                  use_synthetic=use_synthetic, days=prediction_days)
                st.session_state.pop('model_choice', None)

        params = st.session_state.get('params')
        if params:
            st.markdown("---\n### ℹ️ Current analysis")
            st.markdown(f"**Symbol:** {params['symbol']}  \n"
                        f"**Range:** {params['start']} → {params['end']}  \n"
                        f"**Horizon:** {params['days']} days")
        return params


# ============================================================================
# SECTIONS
# ============================================================================

def render_source_banner(a):
    if a['source'] == 'synthetic':
        st.warning("⚠️ **Synthetic data.** Yahoo Finance returned nothing (bad ticker, rate limit, or "
                   "offline mode). Charts and forecasts below are for demonstration only.")
    else:
        st.success(f"✅ Real market data from Yahoo Finance: {len(a['raw_view'])} trading days "
                   f"({a['raw_view']['Date'].iloc[0].date()} → {a['raw_view']['Date'].iloc[-1].date()})")


def render_metrics(df, cur):
    if len(df) < 2:
        return
    latest, prev = df.iloc[-1], df.iloc[-2]
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("💰 Last Close", f"{cur}{latest['Close']:.2f}", f"{(latest['Close'] / prev['Close'] - 1) * 100:+.2f}%")
    avg_vol = df['Volume'].mean()
    c2.metric("📊 Volume", f"{latest['Volume'] / 1e6:.1f}M",
              f"{(latest['Volume'] / avg_vol - 1) * 100:+.1f}% vs avg" if avg_vol else None)
    c3.metric("📈 Day Range", f"{cur}{latest['Low']:.2f} – {cur}{latest['High']:.2f}")
    c4.metric("📉 Volatility (ann.)", f"{df['Close'].pct_change().std() * np.sqrt(252) * 100:.1f}%")
    c5.metric("📊 Total Return", f"{(latest['Close'] / df['Close'].iloc[0] - 1) * 100:+.1f}%")


def render_price_chart(df):
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.04,
                        subplot_titles=('Price, Moving Averages & Bollinger Bands', 'Volume', 'RSI (14)', 'MACD'),
                        row_heights=[0.5, 0.15, 0.175, 0.175])
    x, close = df['Date'], df['Close']
    fig.add_trace(go.Candlestick(x=x, open=df['Open'], high=df['High'], low=df['Low'], close=close,
                                 name='Price'), row=1, col=1)
    for period, color in [(20, 'orange'), (50, 'lightgreen'), (200, 'red')]:
        if len(df) > period:
            fig.add_trace(go.Scatter(x=x, y=close.rolling(period).mean(), name=f'SMA {period}',
                                     line=dict(color=color, width=1)), row=1, col=1)
    if len(df) > 20:
        mid, sd = close.rolling(20).mean(), close.rolling(20).std()
        for y, name in [(mid + 2 * sd, 'BB Up'), (mid - 2 * sd, 'BB Low')]:
            fig.add_trace(go.Scatter(x=x, y=y, name=name, line=dict(color='rgba(160,160,160,0.6)', width=1,
                                                                   dash='dash')), row=1, col=1)

    colors = np.where(close >= df['Open'], 'green', 'red')
    fig.add_trace(go.Bar(x=x, y=df['Volume'], name='Volume', marker_color=colors, opacity=0.5), row=2, col=1)

    delta = close.diff()
    rsi = 100 - 100 / (1 + delta.clip(lower=0).rolling(14).mean() / (-delta.clip(upper=0)).rolling(14).mean())
    fig.add_trace(go.Scatter(x=x, y=rsi, name='RSI', line=dict(color='violet')), row=3, col=1)
    fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
    fig.update_yaxes(range=[0, 100], row=3, col=1)

    macd = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    fig.add_trace(go.Scatter(x=x, y=macd, name='MACD', line=dict(color='#4f8bf9')), row=4, col=1)
    fig.add_trace(go.Scatter(x=x, y=signal, name='Signal', line=dict(color='tomato')), row=4, col=1)
    fig.add_trace(go.Bar(x=x, y=hist, name='Hist', opacity=0.5,
                         marker_color=np.where(hist >= 0, 'green', 'red')), row=4, col=1)

    # FIX: legend was at y=1.03 (same height as the first subplot title) -> overlapping text.
    # Anchor it by its bottom edge above the title and reserve top margin for up to 2 wrapped rows.
    fig.update_layout(
        height=900, template=TEMPLATE, xaxis_rangeslider_visible=False,
        margin=dict(t=120, b=30, l=10, r=10),
        legend=dict(orientation='h', yanchor='bottom', y=1.06, xanchor='left', x=0,
                    font=dict(size=11), bgcolor='rgba(0,0,0,0)'),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_returns_distribution(df):
    st.markdown("### 📊 Returns Distribution")
    returns = df['Close'].pct_change().dropna() * 100
    fig = px.histogram(returns, nbins=60, title='Daily Returns (%)', color_discrete_sequence=['#4f8bf9'])
    fig.add_vline(x=0, line_dash="dash", line_color="red")
    fig.update_layout(template=TEMPLATE, height=380, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Mean Daily Return", f"{returns.mean():.3f}%")
    c2.metric("Std Dev", f"{returns.std():.3f}%")
    c3.metric("Sharpe (ann., rf=0)", f"{returns.mean() / returns.std() * np.sqrt(252):.2f}" if returns.std() else "n/a")


def render_correlation_matrix(df):
    st.markdown("### 📊 Correlation Matrix")
    corr = df[['Open', 'High', 'Low', 'Close', 'Volume']].corr()
    fig = px.imshow(corr, text_auto='.2f', color_continuous_scale='RdBu_r', zmin=-1, zmax=1,
                    title='OHLC & Volume')
    fig.update_layout(template=TEMPLATE, height=450)
    st.plotly_chart(fig, use_container_width=True)


def render_risk_metrics(df):
    st.markdown("### 📊 Risk Metrics (buy & hold)")
    returns = df.set_index('Date')['Close'].pct_change().dropna()
    if len(returns) < 2:
        return
    equity = (1 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📉 VaR 95% (1-day)", f"{np.percentile(returns, 5) * 100:.2f}%")
    c1.metric("📉 VaR 99% (1-day)", f"{np.percentile(returns, 1) * 100:.2f}%")
    c2.metric("📊 Sharpe", f"{returns.mean() / returns.std() * np.sqrt(252):.2f}" if returns.std() else "n/a")
    c3.metric("📉 Max Drawdown", f"{drawdown.min() * 100:.2f}%")
    c4.metric("📈 Up Days", f"{(returns > 0).mean() * 100:.1f}%")

    fig = go.Figure(go.Scatter(x=drawdown.index, y=drawdown * 100, fill='tozeroy', name='Drawdown',
                               line=dict(color='tomato')))
    fig.update_layout(title='Drawdown History', yaxis_title='Drawdown (%)', template=TEMPLATE, height=350)
    st.plotly_chart(fig, use_container_width=True)


def render_predictions(a, params, cur):
    results = a['results']
    ranked = sorted(results, key=lambda n: results[n]['metrics']['test_r2'], reverse=True)
    model_name = st.selectbox("Forecast model", ranked, key='model_choice',
                              format_func=lambda n: f"{n} (test R² {results[n]['metrics']['test_r2']:.3f})"
                              + ("  ← best" if n == a['best_name'] else ""))

    with st.spinner(f"🔮 Forecasting {params['days']} business days (features recomputed each step)..."):
        try:
            pred = cached_forecast(params['symbol'], params['start'], params['end'],
                                   params['use_synthetic'], params['days'], model_name)
        except Exception as e:
            st.error(f"Forecast failed: {e}")
            return

    hist = a['raw_view'].tail(120)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist['Date'], y=hist['Close'], name='Historical', line=dict(color='#4f8bf9', width=2)))
    fig.add_trace(go.Scatter(x=pred['Date'], y=pred['Upper_Bound'], mode='lines', line=dict(width=0),
                             showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=pred['Date'], y=pred['Lower_Bound'], mode='lines', line=dict(width=0),
                             fill='tonexty', fillcolor='rgba(255,99,71,0.18)', name='95% band'))
    # connect forecast to the last real close
    fx = pd.concat([hist['Date'].tail(1), pred['Date']])
    fy = pd.concat([hist['Close'].tail(1), pred['Predicted_Close']])
    fig.add_trace(go.Scatter(x=fx, y=fy, name='Forecast', line=dict(color='tomato', width=2, dash='dash')))
    fig.update_layout(title=f"{a['symbol']} forecast ({model_name})", yaxis_title=f'Price ({cur})',
                      template=TEMPLATE, height=500, hovermode='x unified')

    col1, col2 = st.columns([2, 1])
    col1.plotly_chart(fig, use_container_width=True)

    last_price = float(a['raw_full']['Close'].iloc[-1])
    final = pred.iloc[-1]
    exp_ret = (final['Predicted_Close'] / last_price - 1) * 100
    band_pct = (final['Upper_Bound'] / final['Predicted_Close'] - 1) * 100
    with col2:
        st.markdown(f"""
#### 📊 Forecast Summary
| | |
|---|---|
| **Last close** | {cur}{last_price:.2f} |
| **Forecast (day {len(pred)})** | {cur}{final['Predicted_Close']:.2f} |
| **Expected move** | {exp_ret:+.2f}% |
| **95% band** | {cur}{final['Lower_Bound']:.2f} – {cur}{final['Upper_Bound']:.2f} |
""")
        m = results[model_name]['metrics']
        beats_baseline = a['baseline_rmse'] is not None and m['test_rmse'] < a['baseline_rmse']
        if abs(exp_ret) < band_pct * 0.25 or not beats_baseline:
            st.info(f"⚪ **Signal: Neutral** — the expected move ({exp_ret:+.2f}%) is small next to the "
                    f"uncertainty (±{band_pct:.1f}%)" + ("" if beats_baseline else
                                                         ", and this model does not beat the zero-change baseline"))
        elif exp_ret > 0:
            st.success(f"🟢 **Signal: Bullish** ({exp_ret:+.2f}%)")
        else:
            st.error(f"🔴 **Signal: Bearish** ({exp_ret:+.2f}%)")
        st.caption("Educational project, not financial advice.")

    st.download_button("📥 Download forecast CSV", pred.to_csv(index=False).encode(),
                       file_name=f"{a['symbol']}_forecast_{model_name}.csv", mime="text/csv")


def render_model_performance(a):
    st.markdown("### 🧪 Model performance on the held-out last 20% (next-day return, %)")
    comp = a['comparison'].copy()
    st.dataframe(comp.style.format(precision=3), use_container_width=True, hide_index=True)
    if a['baseline_rmse'] is not None:
        best = a['results'][a['best_name']]['metrics']
        edge = (1 - best['test_rmse'] / a['baseline_rmse']) * 100
        st.markdown(f"**Zero-change baseline RMSE:** {a['baseline_rmse']:.3f}%  ·  "
                    f"**Best model ({a['best_name']}):** {best['test_rmse']:.3f}%  ·  "
                    f"**Improvement:** {edge:+.1f}%")
        st.caption("Daily returns are close to random; R² near 0 and ~50% direction accuracy are normal. "
                   "A model is only useful if it beats the baseline consistently.")


def render_feature_importance(a):
    imp = feature_importance(a, top_n=20)
    if imp is None:
        st.info("No tree-based model available for feature importance.")
        return
    fig = go.Figure(go.Bar(x=imp['importance'], y=imp['feature'], orientation='h', marker_color='steelblue'))
    fig.update_layout(title=f"Top 20 features ({imp['model'].iloc[0]})", template=TEMPLATE, height=560,
                      yaxis=dict(autorange="reversed"), xaxis_title='Relative importance')
    st.plotly_chart(fig, use_container_width=True)


def render_data_tab(a, cur):
    df = a['raw_view']
    st.dataframe(df.tail(30).iloc[::-1].style.format({'Open': cur + '{:.2f}', 'High': cur + '{:.2f}',
                                                      'Low': cur + '{:.2f}', 'Close': cur + '{:.2f}',
                                                      'Volume': '{:,.0f}'}),
                 use_container_width=True, hide_index=True)
    st.download_button("📥 Download price data CSV", df.to_csv(index=False).encode(),
                       file_name=f"{a['symbol']}_data.csv", mime="text/csv")


# ============================================================================
# MAIN
# ============================================================================

def main():
    st.markdown('<h1 class="main-header">📈 Stock Price Prediction Dashboard</h1>', unsafe_allow_html=True)
    params = render_sidebar()

    if not params:
        st.info("👈 Choose a stock and date range, then click **Process & Analyze**.")
        st.markdown("""
### 🚀 Features
- **Live data** from Yahoo Finance (synthetic fallback, clearly labelled)
- **340+ engineered features**, leakage-checked
- **8 ML models** compared against a zero-change baseline
- **Recursive multi-day forecast** with a widening 95% band
- **Risk metrics**, feature importance, CSV downloads
""")
        return

    with st.spinner(f"🧠 Loading {params['symbol']} and training models (first run ~10-30s, then cached)..."):
        try:
            a = cached_analysis(params['symbol'], params['start'], params['end'], params['use_synthetic'])
        except ValueError as e:
            st.error(f"❌ {e}")
            return
        except Exception as e:
            st.error(f"❌ Unexpected error: {e}")
            st.exception(e)
            return

    cur = currency_for(a['symbol'])
    render_source_banner(a)
    st.markdown("---")
    render_metrics(a['raw_view'], cur)
    st.markdown("---")
    render_price_chart(a['raw_view'])

    tab1, tab2, tab3, tab4, tab5 = st.tabs(["🔮 Forecast", "🧪 Model Performance", "🔍 Feature Importance",
                                            "📉 Risk & Returns", "📋 Data"])
    with tab1:
        render_predictions(a, params, cur)
    with tab2:
        render_model_performance(a)
    with tab3:
        render_feature_importance(a)
    with tab4:
        render_returns_distribution(a['raw_view'])
        render_correlation_matrix(a['raw_view'])
        render_risk_metrics(a['raw_view'])
    with tab5:
        render_data_tab(a, cur)


if __name__ == "__main__":
    main()