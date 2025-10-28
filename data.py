import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt
import torch
from torch.utils.data import DataLoader, Dataset

# Download SPY log-returns as a pandas Series.
def fetchLogReturns(ticker='SPY', years=5):
    endDate = dt.datetime.today()
    startDate = endDate - dt.timedelta(days=365*years)
    data = yf.download(ticker, start=startDate, end=endDate, auto_adjust=True)
    logReturns = np.log(data['Close']).diff().dropna()
    lastClose = data['Close'].iloc[-1]
    return logReturns, lastClose


# Transform time-series data into supervised-pairs for LSTM training.
class prepareDataWindows(Dataset):
    def __init__(self, returns: np.ndarray, window=30):
        r = np.asarray(returns)
        if r.ndim > 1:
            r = np.squeeze(r)          
        self.returns = r.astype(np.float32).reshape(-1)  
        self.window = window

    def __len__(self): 
        return len(self.returns) - self.window
    
    def __getitem__(self, i): 
        x = self.returns[i:i+self.window]
        y = self.returns[i+self.window]
        x = torch.from_numpy(x).unsqueeze(-1)
        y = torch.tensor(y, dtype=torch.float32)
        return x, y

#Split our data into training and validation sets.
def trainValSplit(returns, window=30, batch_size=64, val_split=0.2):
    N = len(returns)
    splitIndex = int(N * (1 - val_split))

    trainingSet = returns[:splitIndex]
    validationSet = returns[splitIndex - window:]

    trainDataset = prepareDataWindows(trainingSet, window=window)
    valDataset = prepareDataWindows(validationSet, window=window)

    trainLoader = DataLoader(trainDataset, batch_size=batch_size, shuffle=True)
    valLoader = DataLoader(valDataset, batch_size=batch_size, shuffle=False)

    return trainLoader, valLoader