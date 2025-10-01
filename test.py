import yfinance as yf

ticker = 'NVDA'  # Replace with your ticker symbol
stock = yf.Ticker(ticker)

# Print all available info
info = stock.info
print("Available info keys:")
print(info.keys())

# Try fetching the stock price using different keys
possible_price_keys = ['currentPrice', 'regularMarketPrice', 'previousClose']

S0 = None
for key in possible_price_keys:
    if key in info:
        S0 = info[key]
        print(f"Stock price found using key '{key}': {S0}")
        break

if S0 is None:
    print("Stock price not found in 'stock.info' dictionary.")
