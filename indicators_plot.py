import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

# Download stock data
ticker = 'TCS.NS'
data = yf.download(ticker, period='1y')  # 1 year data

# --- Indicator 1: MA10 and MA50 ---
data['MA10'] = data['Close'].rolling(window=10).mean()
data['MA50'] = data['Close'].rolling(window=50).mean()

# --- Indicator 2: RSI (Relative Strength Index) ---
def compute_rsi(data, window=14):
    delta = data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

data['RSI'] = compute_rsi(data)

# --- Indicator 3: MACD (with Signal Line) ---
ema_12 = data['Close'].ewm(span=12, adjust=False).mean()
ema_26 = data['Close'].ewm(span=26, adjust=False).mean()
data['MACD'] = ema_12 - ema_26
data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()

# --- Plot everything ---
plt.figure(figsize=(15, 12))

# Plot 1: Price + MA10 + MA50
plt.subplot(3, 1, 1)
plt.plot(data['Close'], label='Close Price')
plt.plot(data['MA10'], label='MA10')
plt.plot(data['MA50'], label='MA50')
plt.title('Close Price with MA10 and MA50')
plt.legend()
plt.grid()

# Plot 2: RSI
plt.subplot(3, 1, 2)
plt.plot(data['RSI'], label='RSI', color='orange')
plt.axhline(70, linestyle='--', color='red')
plt.axhline(30, linestyle='--', color='green')
plt.title('RSI (Relative Strength Index)')
plt.legend()
plt.grid()

# Plot 3: MACD
plt.subplot(3, 1, 3)
plt.plot(data['MACD'], label='MACD', color='purple')
plt.plot(data['Signal'], label='Signal Line', color='gray')
plt.title('MACD and Signal Line')
plt.legend()
plt.grid()

plt.tight_layout()
plt.show()
