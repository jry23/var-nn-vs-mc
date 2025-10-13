import numpy as np, pandas as pd, torch, torch.nn as nn

class LSTM1(nn.Module):
    def __init__(self, hidden=32):
        super().__init__()
        self.lstm = nn.LSTM(1, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, 1)
    def forward(self, x):
        y,_ = self.lstm(x)
        return self.fc(y[:, -1, :])

def _seq(x, L):
    X,Y = [],[]
    for i in range(len(x)-L):
        X.append(x[i:i+L]); Y.append(x[i+L])
    X = np.array(X)[:, :, None]; Y = np.array(Y)[:, None]
    return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)

def train_lstm(rets: pd.Series, seq_len=30, epochs=50, lr=1e-3, hidden=32, device=None):
    r = rets.dropna().values.astype(np.float32)
    m, s = r.mean(), r.std() if r.std()>0 else 1.0
    z = (r - m) / s
    X,y = _seq(z, seq_len)
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

def forecast_lstm(model, rets: pd.Series, mean, std, seq_len, horizon, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    r = rets.dropna().values.astype(np.float32)
    z = (r - mean) / std
    window = z[-seq_len:].tolist()
    out = []
    model.eval()
    for _ in range(int(horizon)):
        x = torch.tensor(np.array(window)[None, :, None], dtype=torch.float32, device=device)
        with torch.no_grad():
            pred = model(x).item()
        out.append(pred); window = window[1:] + [pred]
    out = np.array(out)*std + mean
    return out