"""
10-Day VaR and ES Pricer using Monte Carlo Simulation
Jeffrey Yang
Date: 08/15/2024

Dependencies:
pip install flask numpy pandas matplotlib alpaca-trade-api
"""

from flask import Flask, request, jsonify, render_template
import alpaca_trade_api as tradeapi
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime, timedelta

app = Flask(__name__)

# Alpaca credentials (free-tier friendly)
BASE_URL = "https://paper-api.alpaca.markets/v2"
ALPACA_API_KEY = "PKVA53DC6M2II8VQBKC8"
ALPACA_SECRET_KEY = "kknfcEo2aOiMKQBd0WmlC78FkAkeu68xofhS45LY"

api = tradeapi.REST(ALPACA_API_KEY, ALPACA_SECRET_KEY, BASE_URL, api_version='v2')


def get_stock_data(ticker):
    # Historical window: 2 years
    end_date = datetime.now().date().isoformat()
    start_date = (datetime.now() - timedelta(days=2 * 365)).date().isoformat()

    bars = api.get_bars(ticker, tradeapi.TimeFrame.Day, start=start_date, end=end_date, feed='iex').df

    if bars.empty or len(bars) < 20:
        raise ValueError(f"No sufficient historical data for {ticker}")

    # Use latest available close as S0
    S0 = bars['close'][-1]

    # Compute log returns and daily volatility
    bars['returns'] = np.log(bars['close'] / bars['close'].shift(1))
    sigma_daily = np.std(bars['returns'].dropna())

    return S0, sigma_daily


def simulate_10_day_paths(S0, r, sigma, M=10, I=10000):
    dt = 1 / 252
    paths = np.zeros((M + 1, I))
    paths[0] = S0
    for t in range(1, M + 1):
        z = np.random.standard_normal(I)
        paths[t] = paths[t - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z)
    return paths


def compute_var_es(paths, alpha=0.05):
    S_T = paths[-1]
    losses = paths[0] - S_T  # 10-day losses
    var = np.percentile(losses, alpha * 100)
    es = losses[losses >= var].mean()  # Expected Shortfall (Conditional VaR)
    return var, es


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/price', methods=['POST'])
def price():
    try:
        ticker = request.form['ticker'].upper()
        r = float(request.form['r'])
        I = int(request.form['I'])

        # Get current price and daily volatility
        S0, sigma = get_stock_data(ticker)

        M = 10  # 10-day VaR horizon
        paths = simulate_10_day_paths(S0, r, sigma, M, I)
        var, es = compute_var_es(paths, alpha=0.05)

        # Plot simulation paths
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(paths[:, :100])  # only show first 100 paths for clarity
        ax.set_title(f'Simulated 10-Day Price Paths for {ticker}')
        ax.set_xlabel('Days')
        ax.set_ylabel('Price')
        ax.grid(True)

        # Convert plot to base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.read()).decode('utf-8')
        buf.close()
        plt.close(fig)

        return jsonify({
            'ticker': ticker,
            'S0': round(S0, 2),
            'sigma_daily': round(sigma, 5),
            'VaR_95': round(var, 2),
            'ES_95': round(es, 2),
            'image_base64': image_base64
        })

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)