import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.filterwarnings("ignore")  # Suppress ARIMA convergence warnings

# Title
st.title("AI Investment Portfolio Manager")

# Sidebar inputs
st.sidebar.header("User Inputs")
initial_capital = st.sidebar.number_input("Initial Capital ($)", min_value=1000.0, value=10000.0, step=1000.0)
risk_level = st.sidebar.selectbox("Risk Level", ["Low", "Medium", "High"])
esg_preference = st.sidebar.checkbox("Prefer ESG Investments")

# Determine tickers based on ESG preference
if esg_preference:
    tickers = ['ESGU', 'SUSL', 'BND', 'QQQ']
    esg_note = "Using ESG-focused ETFs: ESGU (Broad US ESG), SUSL (US Large Cap ESG), BND (Bonds), QQQ (Tech/Growth)."
else:
    tickers = ['SPY', 'VTI', 'BND', 'QQQ']
    esg_note = ""

# Benchmark ticker
benchmark_ticker = 'SPY'

# Run simulation button
run_simulation = st.sidebar.button("Run Simulation")

if run_simulation:
    # Fetch current prices and 1-month momentum
    st.header("Current Market Prices")
    price_data = yf.download(tickers, period='1mo')['Close']
    current_prices = price_data.iloc[-1]
    momentum = ((price_data.iloc[-1] / price_data.iloc[0]) - 1) * 100

    prices_df = pd.DataFrame({
        'Ticker': tickers,
        'Current Price ($)': current_prices.values,
        '1-Mo Momentum (%)': momentum.values
    }).round(4)
    st.table(prices_df)

    # Fetch long historical data for forecasting
    end_date = datetime.now()
    start_date_long = end_date - timedelta(days=5*365 + 100)  # ~5 years + buffer
    hist_data_long = yf.download(tickers, start=start_date_long, end=end_date)['Close']

    # Simple ARIMA forecast function
    def forecast_expected_return(price_series, forecast_days=30):
        try:
            returns = price_series.pct_change().dropna()
            if len(returns) < 60:
                return np.nan
            model = ARIMA(returns[-252:], order=(5,1,0))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=forecast_days)
            mean_daily = forecast.mean()
            annualized = (1 + mean_daily) ** 252 - 1
            return annualized
        except Exception:
            return np.nan

    # Forecast expected returns
    st.header("Forecasted Annualized Expected Returns (Next ~1 Month Horizon)")
    expected_returns = {}
    for ticker in tickers:
        expected_returns[ticker] = forecast_expected_return(hist_data_long[ticker])

    forecast_df = pd.DataFrame({
        'Ticker': tickers,
        'Forecasted Expected Return (%)': [f"{r*100:.2f}" if not np.isnan(r) else "N/A" for r in expected_returns.values()]
    })
    st.table(forecast_df)
    st.caption("Forecasts based on ARIMA time-series model using historical daily returns. These are estimates only; actual returns may vary significantly. Past performance is not indicative of future results.")

    # Base weights based on risk level
    if risk_level == "Low":
        base_weights = [0.2, 0.2, 0.5, 0.1]
    elif risk_level == "Medium":
        base_weights = [0.3, 0.3, 0.3, 0.1]
    else:
        base_weights = [0.3, 0.3, 0.1, 0.3]

    # Simple momentum tilt (placeholder)
    rel_momentum = momentum / momentum.sum()
    tilted_weights = [w * (1 + r / 100) for w, r in zip(base_weights, rel_momentum)]
    total = sum(tilted_weights)
    allocation_percent = [w / total * 100 for w in tilted_weights]
    allocation_dollars = [p / 100 * initial_capital for p in allocation_percent]

    # Allocation table
    st.header("Recommended Portfolio Allocation")
    allocation_df = pd.DataFrame({
        'Ticker': tickers,
        'Allocation (%)': [round(p, 4) for p in allocation_percent],
        'Estimated Value ($)': [round(d, 2) for d in allocation_dollars]
    })
    st.table(allocation_df)

    # Pie chart
    fig_pie, ax_pie = plt.subplots()
    ax_pie.pie(allocation_percent, labels=tickers, autopct='%1.1f%%', startangle=90)
    ax_pie.axis('equal')
    st.pyplot(fig_pie)

    # Fetch 1-year data for normalized chart and returns
    start_date_1y = end_date - timedelta(days=365)
    hist_data = yf.download(tickers + [benchmark_ticker], start=start_date_1y, end=end_date)['Close']
    normalized = (hist_data / hist_data.iloc[0]) * 100
    one_year_returns = ((hist_data.iloc[-1] / hist_data.iloc[0]) - 1) * 100

    portfolio_return = sum(r * (w / 100) for r, w in zip(one_year_returns[tickers], allocation_percent))
    benchmark_return = one_year_returns[benchmark_ticker]

    # Historical normalized chart
    st.header("1-Year Historical Price Trends (Normalized to 100)")
    fig_hist, ax_hist = plt.subplots(figsize=(10, 6))
    for ticker in tickers:
        ax_hist.plot(normalized.index, normalized[ticker], label=ticker)
    ax_hist.plot(normalized.index, normalized[benchmark_ticker], label='S&P 500 (Benchmark)', linestyle='--', color='black')
    ax_hist.set_xlabel("Date")
    ax_hist.set_ylabel("Normalized Price")
    ax_hist.legend()
    st.pyplot(fig_hist)

    # Benchmark comparison
    st.header("Benchmark Comparison (1-Year Return %)")
    benchmark_df = pd.DataFrame({
        'Metric': ['Portfolio Return', 'S&P 500 Return'],
        'Value (%)': [round(portfolio_return, 4), round(benchmark_return, 4)]
    })
    st.table(benchmark_df)

    # Notes and disclaimers
    if esg_note:
        st.info(esg_note)
    st.warning("Data is delayed and for informational purposes only. Forecasts are model-based estimates and not guarantees. This is not financial advice. Consult a professional advisor.")
    st.caption("Powered by yfinance API.")

# End of script
