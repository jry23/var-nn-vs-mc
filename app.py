import numpy as np 
import pandas as pd
import streamlit as st
import altair as alt
from mc import *
from lstm import *
from data import *

st.set_page_config(layout="wide")
st.title("Comparing Time-Series Methods for Market Risk Forecasts")
st.markdown("##### *Author: Jeffrey Yang, jry33@cornell.edu*")

st.sidebar.header("Configuration:")
ticker = st.sidebar.text_input("Ticker", "SPY")


st.sidebar.subheader("Monte-Carlo:")
sims = st.sidebar.slider("Simulations", 1000, 10000, 5000, step=100)
days = st.sidebar.slider("Days", 10, 250, 100, step=10)

st.sidebar.subheader("Neural Network:")
window = st.sidebar.slider("Window Size", 10, 50, 30, step=5)
hidden_size = st.sidebar.slider("Hidden Dimension Size", 16, 128, 64, step=16)
num_layers = st.sidebar.slider("Number of Layers", 1, 3, 1, step=1)
epochs = st.sidebar.slider("Training Epochs", 10, 100, 50, step=10)

st.sidebar.subheader("VaR/ES:")
alpha = st.sidebar.slider("Confidence Level", 0.90, 0.99, 0.95, step=0.01)

if st.sidebar.button("Run"):
    logReturns, lastClose = fetchLogReturns(ticker)
    mcResults = fitGARCHModel(logReturns)
    dfSim = simulatePaths(logReturns, mcResults, simulations=sims, days=days, seed=42)

    oneDayReturns, priceHorizon, VaR, ES = calculateRiskMetrics(dfSim, alpha)

    mc_var_pct = VaR * 100.0
    mc_es_pct  = ES  * 100.0

    lstmModel = trainLSTM(
        logReturns.to_numpy(),
        tau=1 - alpha,
        window=window,
        hidden_size=hidden_size,
        num_layers=num_layers,
        epochs=epochs,
        device='cpu'
    )

    lstmVaR = forecastLSTM(
        lstmModel,
        logReturns.to_numpy(),
        window=window,
        device='cpu'
    )
    lstmVaR_loss_pct = -lstmVaR * 100.0

    mcPathChart = (
        alt.Chart(priceHorizon)
        .mark_line()
        .encode(
            x=alt.X("Day:Q", title="Days"),
            y=alt.Y("Price:Q", title="Simulated Price"),
            color=alt.Color("Simulation:N", legend=None),
        )
        .properties(title="Simulated Price Paths", width=700, height=400)
    )

    returnDistribution = (
        alt.Chart(pd.DataFrame({"Return": oneDayReturns}))
        .mark_bar()
        .encode(
            x=alt.X("Return:Q", bin=alt.Bin(maxbins=60), title="One-Day Return"),
            y=alt.Y("count()", title="Frequency"),
    ))

    lines_df = pd.DataFrame({
        "value": [-VaR, -ES],
        "label": [f"VaR {int(alpha*100)}%", f"ES {int(alpha*100)}%"]
    })

    color_scale = alt.Scale(
        domain=[f"VaR {int(alpha*100)}%", f"ES {int(alpha*100)}%"],
        range=["#E63946", "#7C4DFF"] 
    )

    rules = alt.Chart(lines_df).mark_rule(size=2).encode(
        x="value:Q",
        color=alt.Color("label:N", scale=color_scale, legend=None),
        tooltip=["label:N", alt.Tooltip("value:Q", format=".2%")]
    )

    labels = alt.Chart(lines_df).mark_text(align="left", dx=5, dy=-5).encode(
        x="value:Q",
        y=alt.value(0), 
        text="label:N",
        color=alt.Color("label:N", legend=None),
    )

    chart = (returnDistribution + rules + labels).properties(
        title="Distribution of 1-Day Simulated Returns", width=700, height=400
    )

    # Display
    st.header("AR(0)-GARCH(1, 1) Monte-Carlo:")

    st.altair_chart(chart, use_container_width=False)
    c1, c2 = st.columns(2)
    with c1:
        st.metric(label=f"VaR ({int(alpha*100)}%)", value=f"{mc_var_pct:.2f}%")
    with c2:
        st.metric(label=f"ES ({int(alpha*100)}%)",  value=f"{mc_es_pct:.2f}%")
    st.altair_chart(mcPathChart, use_container_width=False)

    st.header("LSTM Recurrent Neural Network:")
    st.metric(label=f"VaR ({int(alpha*100)}%)", value=f"{lstmVaR_loss_pct:.2f}%")
