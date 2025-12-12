import yfinance as yf
import matplotlib.pyplot as plt

# Download stock data
ticker = 'TCS.NS'  # You can change this
data = yf.download(ticker, period='2y')  # Last 2 years

# Print last few rows
print(data.tail())

# Plot closing prices
plt.figure(figsize=(12,6))
plt.plot(data['Close'], label='Closing Price')
plt.title(f"{ticker} Stock Price")
plt.xlabel("Date")
plt.ylabel("Price (INR)")
plt.legend()
plt.grid()
plt.show()
