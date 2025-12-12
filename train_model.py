import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

def prepare_features(ticker='AAPL', period='1y'):
    # Download historical data
    data = yf.download(ticker, period=period)
    
    # Feature engineering
    data['MA10'] = data['Close'].rolling(window=10).mean()
    data['MA50'] = data['Close'].rolling(window=50).mean()
    
    delta = data['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    data['RSI'] = 100 - (100 / (1 + rs))
    
    exp1 = data['Close'].ewm(span=12, adjust=False).mean()
    exp2 = data['Close'].ewm(span=26, adjust=False).mean()
    data['MACD'] = exp1 - exp2
    data['Signal'] = data['MACD'].ewm(span=9, adjust=False).mean()
    
    # Target: 1 if next day close price goes up, else 0
    data['Target'] = (data['Close'].shift(-1) > data['Close']).astype(int)
    
    # Drop rows with NaN values (from rolling calculations)
    data = data.dropna()
    
    features = data[['MA10', 'MA50', 'RSI', 'MACD', 'Signal']]
    target = data['Target']
    
    return features, target

def train_and_save(ticker='AAPL', model_filename='stock_model.joblib'):
    X, y = prepare_features(ticker)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False)
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
    
    # Save the model
    joblib.dump(model, model_filename)
    print(f"Model saved to {model_filename}")

if __name__ == '__main__':
    train_and_save()
