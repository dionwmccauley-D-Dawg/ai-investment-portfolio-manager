import streamlit as st
import yfinance as yf
import pandas as pd

st.title("AI Investment Portfolio Manager")

with st.sidebar:
    initial_capital = st.number_input("Initial Capital ($)", min_value=1000.0, value=10000.0, step=1000.0)
    risk_level = st.selectbox("Risk Level", options=["Low", "Medium", "High"])

if st.button("Run Simulation"):
    st.success("Fetching market data via Yahoo Finance...")

    tickers = ["SPY", "VTI", "BND", "QQQ"]  # diversified samples
    data = []

    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            price = info.get('regularMarketPrice') or info.get('currentPrice') or stock.history(period="1d")['Close'].iloc[-1]
            data.append({
                "Ticker": ticker,
                "Last Price": f"${price:.2f}",
                "Timestamp": pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            })
        except Exception as e:
            st.warning(f"Could not fetch {ticker}: {str(e)}")

    if data:
        df = pd.DataFrame(data)
        st.subheader("Latest Market Prices (Yahoo Finance)")
        st.dataframe(df, use_container_width=True)

        allocations = {
            "Low": {"BND": 60, "SPY": 25, "VTI": 15, "QQQ": 0},
            "Medium": {"SPY": 35, "VTI": 30, "BND": 20, "QQQ": 15},
            "High": {"QQQ": 40, "SPY": 35, "VTI": 25, "BND": 0}
        }

        alloc = allocations[risk_level]
        alloc_df = pd.DataFrame(list(alloc.items()), columns=["Asset", "Suggested %"])
        alloc_df["Estimated Value"] = (alloc_df["Suggested %"] / 100 * initial_capital).apply(lambda x: f"${x:,.2f}")

        st.subheader(f"Initial Allocation Suggestion ({risk_level} Risk)")
        st.dataframe(alloc_df, use_container_width=True)

        st.info("This is a simple starting allocation based on risk tolerance. Future versions will incorporate modern portfolio theory for optimal risk-adjusted returns, diversification, and ethical considerations like ESG factors. This is not financial advice; consult a professional.")
    else:
        st.error("No market data fetched. Check internet connection.")
