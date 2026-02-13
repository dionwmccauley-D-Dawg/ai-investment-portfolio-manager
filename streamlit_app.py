import streamlit as st
import os
from polygon import RESTClient
import pandas as pd

st.title("AI Investment Portfolio Manager")

with st.sidebar:
    initial_capital = st.number_input(
        "Initial Capital ($)", 
        min_value=1000.0, 
        value=10000.0, 
        step=1000.0
    )
    risk_level = st.selectbox(
        "Risk Level", 
        options=["Low", "Medium", "High"]
    )

if st.button("Run Simulation"):
    st.success("Fetching real-time market data via Polygon API...")

    # Load API key from Hugging Face secrets / environment variables
    api_key = os.getenv("POLYGON_API_KEY")
    if not api_key:
        st.error(
            "POLYGON_API_KEY not found. "
            "Please add it in Settings → Variables and secrets → Secrets (private). "
            "Name must be exactly POLYGON_API_KEY (case-sensitive)."
        )
        st.stop()

    client = RESTClient(api_key)

    # Sample diversified tickers (equity, broad market, bonds, growth)
    tickers = ["SPY", "VTI", "BND", "QQQ"]
    data = []

    for ticker in tickers:
        try:
            quote = client.get_last_trade(ticker)
            if quote:
                data.append({
                    "Ticker": ticker,
                    "Last Price": f"${quote.price:.2f}",
                    "Timestamp": pd.to_datetime(quote.timestamp, unit='ms').strftime('%Y-%m-%d %H:%M:%S')
                })
        except Exception as e:
            st.warning(f"Could not fetch {ticker}: {str(e)}")

    if data:
        df = pd.DataFrame(data)
        st.subheader("Latest Market Prices (Polygon API)")
        st.dataframe(df, use_container_width=True)

        # Simple risk-based allocation (placeholder – will improve with MPT later)
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

        st.info(
            "This is a simple starting allocation based on risk level. "
            "Future versions will use quantitative models (e.g., Modern Portfolio Theory) "
            "for optimal risk-adjusted returns and diversification."
        )
    else:
        st.error("No market data fetched. Check API key validity and internet connection.")
