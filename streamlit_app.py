import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.filterwarnings("ignore")

st.title("AI Investment Portfolio Manager")

st.sidebar.header("Inputs")
capital = st.sidebar.number_input("Capital ($)", 1000.0, value=10000.0, step=1000.0)
risk = st.sidebar.selectbox("Risk", ["Low", "Medium", "High"])
esg = st.sidebar.checkbox("ESG Preference")

tickers = ['ESGU', 'SUSL', 'BND', 'QQQ'] if esg else ['SPY', 'VTI', 'BND', 'QQQ']

def forecast(prices, days=30):
    try:
        rets = prices.pct_change().dropna()
        if len(rets) < 60:
            return np.nan
        m = ARIMA(rets[-252:], order=(5,1,0))
        fit = m.fit()
        fc = fit.forecast(steps=days)
        return (1 + fc.mean()) ** 252 - 1
    except:
        return np.nan

if st.sidebar.button("Run"):
    st.write("Loading...")

    p = yf.download(tickers, period="1mo")['Close']
    prices = p.iloc[-1]
    mom = ((p.iloc[-1] / p.iloc[0]) - 1) * 100

    st.header("Prices & Momentum")
    df = pd.DataFrame({"Ticker": tickers, "Price": prices.round(2), "1M %": mom.round(2)})
    st.table(df)

    end = datetime.now()
    h = yf.download(tickers, start=end - timedelta(days=5*365 + 100))['Close']

    st.header("Forecasted Annual Returns (~1 Month)")
    fc = {t: forecast(h[t]) for t in tickers}
    fc_df = pd.DataFrame({
        "Ticker": tickers,
        "Forecast %": [f"{v*100:.2f}" if not np.isnan(v) else "N/A" for v in fc.values()]
    })
    st.table(fc_df)
    st.caption("ARIMA estimate. Actual returns vary significantly.")

    if risk == "Low":
        base = [0.2, 0.2, 0.5, 0.1]
    elif risk == "Medium":
        base = [0.3, 0.3, 0.3, 0.1]
    else:
        base = [0.3, 0.3, 0.1, 0.3]

    w = [b * (1 + m/100) for b, m in zip(base, mom)]
    w = [x / sum(w) for x in w]
    dollars = [wi * capital for wi in w]

    st.header("Allocation")
    alloc = pd.DataFrame({"Ticker": tickers, "%": [round(wi*100, 1) for wi in w], "$": [round(d, 2) for d in dollars]})
    st.table(alloc)

    fig, ax = plt.subplots()
    ax.pie([wi*100 for wi in w], labels=tickers, autopct="%1.1f%%")
    st.pyplot(fig)

    h1y = yf.download(tickers + ['SPY'], period="1y")['Close']
    norm = (h1y / h1y.iloc[0]) * 100

    st.header("1-Year Trends (Normalized)")
    fig, ax = plt.subplots(figsize=(10,5))
    for t in tickers:
        ax.plot(norm.index, norm[t], label=t)
    ax.plot(norm.index, norm['SPY'], '--k', label="S&P 500")
    ax.legend()
    st.pyplot(fig)

    ret1y = (h1y.iloc[-1] / h1y.iloc[0] - 1) * 100
    port_ret = sum(wi * ret1y[t] for wi, t in zip(w, tickers))

    st.header("1-Year Returns")
    st.table(pd.DataFrame({"": ["Portfolio", "S&P 500"], "Return %": [round(port_ret, 2), round(ret1y['SPY'], 2)]}))

    if esg:
        st.info("ESG ETFs enabled")
    st.warning("Simulation only. Not financial advice.")
    st.caption("yfinance data")
# ARIMA forecast table live - rebuild trigger 2026-02-14
