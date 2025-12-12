import yfinance as yf
import pandas as pd

# Download data
ticker = 'TCS.NS'
data = yf.download(ticker, period='1y')

# Moving Averages
data['MA10'] = data['Close'].rolling(window=10).mean()
data['MA50'] = data['Close'].rolling(window=50).mean()

# RSI
def compute_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

data['RSI'] = compute_rsi(data)

# MACD and Signal
ema_12 = data['Close'].ewm(span=12, adjust=False).mean()
ema_26 = data['Close'].ewm(span=26, adjust=False).mean()
data['MACD'] = ema_12 - ema_26
data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()

# --- 🔥 Target column: Did price go up tomorrow? ---
data['Target'] = data['Close'].shift(-1) > data['Close']  # True/False
data['Target'] = data['Target'].astype(int)  # 1 if up, 0 if down

# Drop rows with NaN (due to rolling averages)
data.dropna(inplace=True)

# Save features to CSV
features = data[['MA10', 'MA50', 'RSI', 'MACD', 'Signal', 'Target']]
features.to_csv('features.csv', index=False)

print("✅ Features saved to features.csv")
print(features.head())
