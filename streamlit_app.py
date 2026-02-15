import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA
from scipy.optimize import minimize
import warnings

warnings.filterwarnings("ignore")

st.title("AI Investment Portfolio Manager")

st.sidebar.header("User Inputs")
initial_capital = st.sidebar.number_input("Initial Capital ($)", min_value=1000.0, value=10000.0, step=1000.0)
risk_level = st.sidebar.selectbox("Risk Level", ["Low", "Medium", "High"])
esg_preference = st.sidebar.checkbox("Prefer ESG Investments")

if esg_preference:
    tickers = ['ESGU', 'SUSL', 'BND', 'QQQ']
    esg_note = "Using ESG-focused ETFs: ESGU (Broad US ESG), SUSL (US Large Cap ESG), BND (Bonds), QQQ (Tech/Growth)."
else:
    tickers = ['SPY', 'VTI', 'BND', 'QQQ']
    esg_note = ""

benchmark_ticker = 'SPY'
risk_free_rate = 0.02

def get_max_vol(risk_level):
    return {"Low": 0.10, "Medium": 0.15, "High": 0.20}[risk_level]

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

def neg_sharpe(weights, exp_returns, cov_matrix):
    port_ret = np.dot(weights, exp_returns)
    port_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return - (port_ret - risk_free_rate) / port_vol

if st.sidebar.button("Run Simulation"):
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

    end_date = datetime.now()
    start_date_long = end_date - timedelta(days=5*365 + 100)
    hist_data_long = yf.download(tickers, start=start_date_long, end=end_date)['Close']
    returns_long = hist_data_long.pct_change().dropna()
    cov_matrix = returns_long.cov() * 252

    st.header("Forecasted Annualized Expected Returns (Next ~1 Month Horizon)")
    expected_returns = pd.Series({t: forecast_expected_return(hist_data_long[t]) for t in tickers})

    forecast_df = pd.DataFrame({
        'Ticker': tickers,
        'Forecasted Expected Return (%)': [f"{r*100:.2f}" if not np.isnan(r) else "N/A" for r in expected_returns]
    })
    st.table(forecast_df)
    st.caption("Forecasts based on ARIMA time-series model using historical daily returns. These are estimates only; actual returns may vary significantly. Past performance is not indicative of future results.")

    max_vol = get_max_vol(risk_level)
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
        {'type': 'ineq', 'fun': lambda w: max_vol - np.sqrt(np.dot(w.T, np.dot(cov_matrix, w)))}
    ]
    bounds = tuple((0, 1) for _ in tickers)
    init_guess = np.array([1./len(tickers)] * len(tickers))

    exp_ret_clean = expected_returns.fillna(0).values

    result = minimize(neg_sharpe, init_guess, args=(exp_ret_clean, cov_matrix),
                      method='SLSQP', bounds=bounds, constraints=constraints)

    if result.success and not np.any(np.isnan(result.x)):
        weights = result.x
    else:
        st.warning("Optimization did not converge or produced invalid weights — using equal weights.")
        weights = init_guess

    weights = np.nan_to_num(weights, nan=0.0)
    weights = weights / np.sum(weights) if np.sum(weights) > 0 else init_guess

    alloc_pct = weights * 100
    alloc_val = weights * initial_capital

    st.header("Optimized Portfolio Allocation (Max Sharpe Ratio)")
    alloc_df = pd.DataFrame({
        'Ticker': tickers,
        'Allocation (%)': alloc_pct.round(1),
        'Estimated Value ($)': alloc_val.round(2)
    })
    st.table(alloc_df)

    fig, ax = plt.subplots()
    ax.pie(alloc_pct, labels=tickers, autopct='%1.1f%%')
    st.pyplot(fig)

    start_date_1y = end_date - timedelta(days=365)
    hist_data = yf.download(tickers + [benchmark_ticker], start=start_date_1y, end=end_date)['Close']
    normalized = (hist_data / hist_data.iloc[0]) * 100
    one_year_returns = ((hist_data.iloc[-1] / hist_data.iloc[0]) - 1) * 100

    portfolio_return = np.dot(weights, one_year_returns[tickers])
    benchmark_return = one_year_returns[benchmark_ticker]

    st.header("1-Year Historical Price Trends (Normalized to 100)")
    fig_hist, ax_hist = plt.subplots(figsize=(10, 6))
    for ticker in tickers:
        ax_hist.plot(normalized.index, normalized[ticker], label=ticker)
    ax_hist.plot(normalized.index, normalized[benchmark_ticker], label='S&P 500 (Benchmark)', linestyle='--', color='black')
    ax_hist.set_xlabel("Date")
    ax_hist.set_ylabel("Normalized Price")
    ax_hist.legend()
    st.pyplot(fig_hist)

    st.header("Benchmark Comparison (1-Year Return %)")
    benchmark_df = pd.DataFrame({
        'Metric': ['Portfolio Return', 'S&P 500 Return'],
        'Value (%)': [round(portfolio_return, 4), round(benchmark_return, 4)]
    })
    st.table(benchmark_df)

    if esg_note:
        st.info(esg_note)
    st.warning("Data is delayed and for informational purposes only. Forecasts are model-based estimates and not guarantees. This is not financial advice. Consult a professional advisor.")
    st.caption("Powered by yfinance API.")
