import numpy as np, pandas as pd, matplotlib.pyplot as plt, streamlit as st, yfinance as yf, torch, torch.nn as nn
from datetime import date, timedelta
from montecarlo import ARGARCHModel, ARGARCHConfig

st.set_page_config(page_title="AR-GARCH vs LSTM", layout="wide")

def to_series(x, name=None):
    if isinstance(x, pd.Series): return x if name is None else x.rename(name)
    if isinstance(x, pd.DataFrame): return (x.squeeze("columns") if name is None else x.squeeze("columns").rename(name))
    return pd.Series(np.asarray(x).ravel(), name=name)

@st.cache_data(show_spinner=False)
def load_prices_and_returns(ticker, start, end):
    df = yf.download(ticker, start=start, end=end, interval="1d", progress=False)
    col = "Adj Close" if "Adj Close" in df.columns else "Close"
    prices = to_series(df[col].dropna(), "Price")
    rets = to_series(np.log(prices / prices.shift(1)).dropna(), "Log returns")
    return prices, rets

class LSTM1(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 1)
    def forward(self, x):
        y,_ = self.lstm(x)
        return self.fc(y[:, -1, :])

def mk_seq(a, L):
    X,Y = [],[]
    for i in range(len(a)-L):
        X.append(a[i:i+L]); Y.append(a[i+L])
    X = np.array(X)[:, :, None]; Y = np.array(Y)[:, None]
    return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)

def train_lstm(rets, seq_len=30, epochs=60, lr=1e-3, hidden=32, device=None):
    r = rets.dropna().values.astype(np.float32)
    m, s = r.mean(), r.std() if r.std()>0 else 1.0
    z = (r - m) / s
    X,y = mk_seq(z, seq_len)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    X,y = X.to(device), y.to(device)
    model = LSTM1(hidden).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.MSELoss()
    model.train()
    for _ in range(epochs):
        opt.zero_grad(); loss = lossf(model(X), y); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        fit = model(X).cpu().numpy().ravel()*s + m
    idx = rets.index[seq_len:]
    fit_series = pd.Series(fit, index=idx, name="LSTM fitted")
    return model, m, s, fit_series, seq_len, device

def forecast_lstm(model, rets, mean, std, seq_len, horizon, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    r = rets.dropna().values.astype(np.float32)
    z = (r - mean) / std
    w = z[-seq_len:].tolist()
    out = []
    model.eval()
    for _ in range(int(horizon)):
        x = torch.tensor(np.array(w)[None, :, None], dtype=torch.float32, device=device)
        with torch.no_grad():
            p = model(x).item()
        out.append(p); w = w[1:] + [p]
    out = np.array(out)*std + mean
    return out

with st.sidebar:
    st.header("Settings")
    ticker = st.text_input("Ticker", value="SPY")
    start = st.date_input("Start", value=date.today() - timedelta(days=365))
    end = st.date_input("End", value=date.today())
    horizon = st.number_input("Forecast horizon (days)", 5, 252, 20)
    n_paths = st.number_input("Monte Carlo paths", 500, 100_000, 5000, step=500)
    run = st.button("Run")

st.title("AR(1)-GARCH(1,1)-t vs LSTM")

if run:
    prices, rets = load_prices_and_returns(ticker, str(start), str(end))
    c1,c2 = st.columns(2, gap="large")
    with c1: st.line_chart(prices)
    with c2: st.line_chart(rets)

    cfg = ARGARCHConfig(ar_lags=1, garch_p=1, garch_q=1, dist="student_t")
    model = ARGARCHModel(cfg).fit(rets.values)
    res = model._res

    phi = float(res.params.get("ar.L1", 0.0))
    c   = float(res.params.get("Const", 0.0))
    m   = float(rets.mean())
    s   = float(rets.std())
    mu_hist = ((1 - phi) * m + s * c + phi * rets.shift(1)).dropna().rename("AR fitted")

    model_lstm, mL, sL, lstm_fit, L, dev = train_lstm(rets, seq_len=30, epochs=60, lr=1e-3, hidden=32)

    colA, colB = st.columns(2, gap="large")
    with colA:
        fig, ax = plt.subplots()
        ax.plot(rets.index, rets.values, lw=1, label="Returns")
        ax.plot(mu_hist.index, mu_hist.values, lw=1, label="AR fitted")
        ax.set_title("Fit: AR-GARCH")
        ax.legend()
        st.pyplot(fig)
    with colB:
        fig, ax = plt.subplots()
        ax.plot(rets.index, rets.values, lw=1, label="Returns")
        ax.plot(lstm_fit.index, lstm_fit.values, lw=1, label="LSTM fitted")
        ax.set_title("Fit: LSTM")
        ax.legend()
        st.pyplot(fig)

    H = int(horizon)
    N = int(n_paths)
    paths = model.simulate_paths(horizon=H, n_paths=N, random_state=42)
    fan_q = np.percentile(paths, [5, 25, 50, 75, 95], axis=1)
    x = np.arange(1, H + 1)
    lstm_fc = forecast_lstm(model_lstm, rets, mL, sL, L, H)

    colC, colD = st.columns(2, gap="large")
    with colC:
        fig, ax = plt.subplots()
        ax.plot(x, fan_q[2], lw=1.2, label="MC median")
        ax.fill_between(x, fan_q[1], fan_q[3], alpha=0.30, label="MC IQR")
        ax.fill_between(x, fan_q[0], fan_q[4], alpha=0.15, label="MC 90%")
        ax.plot(x, lstm_fc, lw=1.2, label="LSTM")
        ax.set_xlabel("Steps ahead"); ax.set_ylabel("Return")
        ax.set_title("Forecast: MC vs LSTM (returns)")
        ax.legend()
        st.pyplot(fig)

    with colD:
        last_price = float(prices.iloc[-1])
        price_paths = last_price * np.exp(np.cumsum(paths, axis=0))
        pfan = np.percentile(price_paths, [5, 25, 50, 75, 95], axis=1)
        lstm_price = last_price * np.exp(np.cumsum(lstm_fc))
        fig, ax = plt.subplots()
        ax.plot(x, pfan[2], lw=1.2, label="MC median")
        ax.fill_between(x, pfan[1], pfan[3], alpha=0.30, label="MC IQR")
        ax.fill_between(x, pfan[0], pfan[4], alpha=0.15, label="MC 90%")
        ax.plot(x, lstm_price, lw=1.2, label="LSTM")
        ax.set_xlabel("Steps ahead"); ax.set_ylabel("Price")
        ax.set_title("Forecast: MC vs LSTM (price)")
        ax.legend()
        st.pyplot(fig)
else:
    st.info("Choose settings and click Run.")