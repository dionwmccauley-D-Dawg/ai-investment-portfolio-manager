import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

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
            # Fetch 1-month return for momentum
            hist = stock.history(period="1mo")
            if not hist.empty:
                momentum = (hist['Close'].iloc[-1] - hist['Close'].iloc[0]) / hist['Close'].iloc[0] * 100
            else:
                momentum = 0.0
            data.append({
                "Ticker": ticker,
                "Last Price": f"${price:.2f}",
                "1-Mo Momentum (%)": f"{momentum:.1f}%",
                "Timestamp": pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
            })
        except Exception as e:
            st.warning(f"Could not fetch {ticker}: {str(e)}")

    if data:
        df = pd.DataFrame(data)
        st.subheader("Latest Market Prices (Yahoo Finance)")
        st.dataframe(df, use_container_width=True)

        # Basic base allocations by risk
        base_allocs = {
            "Low": {"BND": 60, "SPY": 25, "VTI": 15, "QQQ": 0},
            "Medium": {"SPY": 35, "VTI": 30, "BND": 20, "QQQ": 15},
            "High": {"QQQ": 40, "SPY": 35, "VTI": 25, "BND": 0}
        }

        # Apply simple momentum tilt (adjust weights by relative momentum)
        alloc = base_allocs[risk_level]
        total_momentum = sum(float(df[df['Ticker'] == t]['1-Mo Momentum (%)'].iloc[0].strip('%') for t in alloc if alloc[t] > 0)
        if total_momentum > 0:
            for t in alloc:
                if alloc[t] > 0:
                    mom = float(df[df['Ticker'] == t]['1-Mo Momentum (%)'].iloc[0].strip('%'))
                    tilt = (mom / total_momentum) * 10  # mild tilt: +/-10% max adjustment
                    alloc[t] += tilt if mom > 0 else -tilt
            # Normalize to 100%
            total = sum(alloc.values())
            alloc = {k: v / total * 100 for k, v in alloc.items() if v > 0}

        alloc_df = pd.DataFrame(list(alloc.items()), columns=["Asset", "Suggested %"])
        alloc_df["Suggested %"] = alloc_df["Suggested %"].apply(lambda x: f"{x:.1f}%")
        alloc_df["Estimated Value"] = (alloc_df["Suggested %"].str.strip('%').astype(float) / 100 * initial_capital).apply(lambda x: f"${x:,.2f}")

        st.subheader(f"Initial Allocation Suggestion ({risk_level} Risk)")
        st.dataframe(alloc_df, use_container_width=True)

        # Pie chart visualization
        fig, ax = plt.subplots()
        ax.pie(alloc.values(), labels=alloc.keys(), autopct='%1.0f%%', startangle=90)
        ax.axis('equal')
        st.subheader("Allocation Breakdown")
        st.pyplot(fig)

        st.info("This allocation includes a simple momentum tilt for optimization. It is a starting suggestion only, not financial advice. Consult a professional advisor. Future versions will incorporate ESG factors and full modern portfolio theory.")
    else:
        st.error("No market data fetched. Check internet connection.")
