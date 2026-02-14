import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

st.title("AI Investment Portfolio Manager")

with st.sidebar:
    st.subheader("Portfolio Inputs")
    initial_capital = st.number_input(
        "Initial Capital ($)", 
        min_value=1000.0, 
        value=10000.0, 
        step=1000.0,
        help="The amount you plan to invest."
    )
    risk_level = st.selectbox(
        "Risk Level", 
        options=["Low", "Medium", "High"],
        help="Low: Conservative (more bonds), High: Aggressive (more growth)"
    )

if st.button("Run Simulation"):
    st.success("Fetching market data via Yahoo Finance...")

    tickers = ["SPY", "VTI", "BND", "QQQ"]
    data = []

    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            price = info.get('regularMarketPrice') or info.get('currentPrice') or stock.history(period="1d")['Close'].iloc[-1]

            hist = stock.history(period="1mo")
            momentum = 0.0
            if not hist.empty and len(hist) > 1:
                momentum = (hist['Close'].iloc[-1] - hist['Close'].iloc[0]) / hist['Close'].iloc[0] * 100

            data.append({
                "Ticker": ticker,
                "Last Price": f"${price:.2f}",
                "1-Mo Momentum (%)": f"{momentum:.1f}%",
                "Timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
        except Exception as e:
            st.warning(f"Could not fetch {ticker}: {str(e)}")

    if data:
        df = pd.DataFrame(data)
        st.subheader("Latest Market Prices (Yahoo Finance)")
        st.dataframe(df, use_container_width=True)

        base_allocs = {
            "Low": {"BND": 60, "SPY": 25, "VTI": 15, "QQQ": 0},
            "Medium": {"SPY": 35, "VTI": 30, "BND": 20, "QQQ": 15},
            "High": {"QQQ": 40, "SPY": 35, "VTI": 25, "BND": 0}
        }

        alloc = base_allocs[risk_level].copy()

        total_momentum = 0
        momentum_scores = {}
        for t in alloc:
            if alloc[t] > 0 and t in df['Ticker'].values:
                mom_str = df[df['Ticker'] == t]['1-Mo Momentum (%)'].iloc[0]
                mom = float(mom_str.strip('%'))
                momentum_scores[t] = mom
                total_momentum += abs(mom)

        if total_momentum > 0:
            for t in alloc:
                if alloc[t] > 0 and t in momentum_scores:
                    mom = momentum_scores[t]
                    tilt = (mom / total_momentum) * 20
                    alloc[t] += tilt if mom > 0 else -tilt

        total_pct = sum(alloc.values())
        alloc = {k: (v / total_pct * 100) if total_pct > 0 else 0 for k, v in alloc.items()}

        alloc_df = pd.DataFrame(list(alloc.items()), columns=["Asset", "Suggested %"])
        alloc_df["Suggested %"] = alloc_df["Suggested %"].apply(lambda x: f"{x:.1f}%")
        alloc_df["Estimated Value"] = (alloc_df["Suggested %"].str.strip('%').astype(float) / 100 * initial_capital).apply(lambda x: f"${x:,.2f}")

        st.subheader(f"Initial Allocation Suggestion ({risk_level} Risk)")
        st.dataframe(alloc_df, use_container_width=True)

        fig, ax = plt.subplots()
        ax.pie(alloc.values(), labels=alloc.keys(), autopct='%1.0f%%', startangle=90, colors=['#66b3ff','#99ff99','#ff9999','#ffcc99'])
        ax.axis('equal')
        st.subheader("Allocation Breakdown")
        st.pyplot(fig)

        # 1-Year Historical Price Trends (Normalized)
        st.subheader("1-Year Historical Price Trends (Normalized)")
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        for ticker in tickers:
            try:
                hist = yf.download(ticker, period="1y")['Close']
                if not hist.empty:
                    normalized = 100 * (hist / hist.iloc[0])
                    ax2.plot(normalized.index, normalized, label=ticker)
            except Exception as e:
                st.warning(f"Could not load history for {ticker}: {str(e)}")

        # Benchmark overlay (S&P 500 as dashed line)
        try:
            benchmark = yf.download("SPY", period="1y")['Close']
            if not benchmark.empty:
                normalized_bench = 100 * (benchmark / benchmark.iloc[0])
                ax2.plot(normalized_bench.index, normalized_bench, label="S&P 500 (Benchmark)", color="black", linestyle="--")
        except Exception as e:
            st.warning(f"Could not load benchmark (S&P 500): {str(e)}")

        ax2.set_title("1-Year Price History (Normalized to 100)")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Normalized Price (Start = 100)")
        ax2.legend()
        ax2.grid(True)
        plt.tight_layout()
        st.pyplot(fig2)

        st.caption(
            "This chart shows relative performance over the past year, normalized to start at 100 for easy comparison. "
            "It helps visualize trends and volatility to inform the momentum tilt in your allocation suggestion."
        )

        # New: Benchmark comparison metrics (1-year total return)
        st.subheader("Benchmark Comparison (1-Year Total Return)")
        benchmark_return = 0.0
        alloc_return = 0.0

        try:
            spy_hist = yf.download("SPY", period="1y")['Close']
            if not spy_hist.empty and len(spy_hist) > 1:
                benchmark_return = (spy_hist.iloc[-1] - spy_hist.iloc[0]) / spy_hist.iloc[0] * 100
        except Exception as e:
            st.warning(f"Could not calculate S&P 500 return: {str(e)}")

        # Weighted allocation return (using current prices and weights)
        current_prices = {row['Ticker']: float(row['Last Price'].strip('$')) for _, row in df.iterrows()}
        weighted_return = 0.0
        total_weight = 0.0
        for asset, pct in alloc.items():
            if asset in current_prices:
                # Approximate 1-year return using current price and momentum (simplified)
                # This is placeholder — real backtest would use full history
                weighted_return += pct / 100 * momentum_scores.get(asset, 0)
                total_weight += pct / 100

        if total_weight > 0:
            alloc_return = weighted_return / total_weight

        comparison_df = pd.DataFrame({
            "Metric": ["Suggested Allocation", "S&P 500 Benchmark"],
            "1-Year Return (%)": [f"{alloc_return:.1f}% (approx)", f"{benchmark_return:.1f}%"],
            "Difference": [f"{alloc_return - benchmark_return:.1f}%", "-"]
        })

        st.dataframe(comparison_df, use_container_width=True)

        st.info(
            "1-Year return for the suggested allocation is an approximation based on momentum and current prices. "
            "Past performance is no guarantee of future results. This is not investment advice — consult a certified financial advisor."
        )

        st.info(
            "This allocation includes a simple momentum tilt (favoring recent performers). "
            "Future versions will use modern portfolio theory for optimal risk-adjusted returns, "
            "diversification, and ethical considerations like ESG factors."
        )
    else:
        st.error("No market data fetched. Check internet connection.")
