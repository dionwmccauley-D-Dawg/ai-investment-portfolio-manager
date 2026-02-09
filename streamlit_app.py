import streamlit as st

st.title("AI Investment Portfolio Manager")

with st.sidebar:
    initial_capital = st.number_input("Initial Capital ($)", min_value=1000.0, value=10000.0, step=1000.0)
    risk_level = st.selectbox("Risk Level", options=["Low", "Medium", "High"])

if st.button("Run Simulation"):
    st.success("Simulation ran successfully! (Placeholder for future portfolio construction & data integration.)")
