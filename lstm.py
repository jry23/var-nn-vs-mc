import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from data import prepareDataWindows, trainValSplit


# Define a quantile loss function for VaR estimation, penalizing under-predictions more heavily.
def quantileLoss(tau):
    def lossFunction(y_hat, y):
        error = y - y_hat
        return torch.mean(torch.maximum(tau * error, (tau - 1) * error))
    return lossFunction


# Define our LSTM model architecture. 
class LSTM(nn.Module):
    def __init__(
            self,
            input_size=1,
            hidden_size=64,
            num_layers=1,
            quantiles=1,
            dropout=0.0
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.head = nn.Linear(hidden_size, quantiles)
    
    # Define the forward pass through the network.
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :] 
        quantiles = self.head(out)
        return quantiles



# Training the LSTM model using the provided data loader and quantile loss.
def trainLSTM(
        returns,
        tau=0.05,
        window=30,
        hidden_size=64,
        num_layers=1,
        lr=1e-3,
        batch_size=64,
        epochs=50,
        device='cpu'
):
    trainLoader, valLoader = trainValSplit(returns, window=window, batch_size=batch_size)

    model = LSTM(
        input_size=1,
        hidden_size=hidden_size,
        num_layers=num_layers,
        quantiles=1,
        dropout=0.0
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = quantileLoss(tau)

    for epoch in range(epochs):
        model.train()
        trainLoss = 0.0
        for xBatch, yBatch in trainLoader:
            xBatch, yBatch = xBatch.to(device), yBatch.to(device)

            optimizer.zero_grad()
            yPred = model(xBatch)
            loss = criterion(yPred, yBatch.unsqueeze(-1))
            loss.backward()
            optimizer.step()

            trainLoss += loss.item() * xBatch.size(0)

        trainLoss /= len(trainLoader.dataset)

        model.eval()
        valLoss = 0.0
        with torch.no_grad():
            for xBatch, yBatch in valLoader:
                xBatch, yBatch = xBatch.to(device), yBatch.to(device)
                yPred = model(xBatch)
                loss = criterion(yPred, yBatch.unsqueeze(-1))
                valLoss += loss.item() * xBatch.size(0)

        valLoss /= len(valLoader.dataset)

    return model

# Generate one-day ahead return forecasts using the trained LSTM model.
def forecastLSTM(model, returns, window=30, device='cpu'):
    model.eval()
    recentWindow = np.asarray(returns[-window:], dtype=np.float32).reshape(-1)
    xInput = torch.from_numpy(recentWindow).unsqueeze(0).unsqueeze(-1).to(device)

    with torch.no_grad():
        yPred = model(xInput).squeeze().cpu().item()

    return np.exp(yPred) - 1.0

