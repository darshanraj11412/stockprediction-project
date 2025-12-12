# app.py — Streamlit app with LSTM forecasting integrated
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go

# Try import tensorflow (for LSTM)
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
except Exception as e:
    tf = None

st.set_page_config(layout="wide")
st.title("Stock Trend + LSTM Forecasting")

# -----------------------
# User inputs / UI
# -----------------------
col1, col2, col3 = st.columns([2,1,1])
with col1:
    ticker = st.text_input("Enter Stock Ticker (Yahoo format)", value="AAPL")
with col2:
    history_years = st.selectbox("History length (years)", [0.5, 1, 2, 3], index=1)  # 0.5 = 6 months
with col3:
    forecast_days = st.selectbox("Forecast days", [7, 15, 30], index=2)

retrain = st.checkbox("Loaded existing LSTM model for faster and accuracy prediction!", value=False)

run = st.button("Run Forecast")

# helper paths
BASE_DIR = os.path.dirname(__file__)
LSTM_MODEL_PATH = os.path.join(BASE_DIR, "lstm_model.h5")
CLASSIFIER_MODEL_PATH = os.path.join(BASE_DIR, "stock_model.joblib")

# -----------------------
# Utility functions
# -----------------------
def download_data(ticker_symbol: str, min_rows: int = 300):
    years = 3
    while True:
        end = datetime.today()
        start = end - timedelta(days=int(365 * years))
        df = yf.download(ticker_symbol, start=start, end=end, progress=False)

        if df.shape[0] >= min_rows or years > 10:
            return df

        years += 1  # Auto increase dataset until enough rows


def make_indicators(df: pd.DataFrame):
    df = df.copy()
    df['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['MACD'] = df['Close'].ewm(span=12, adjust=False).mean() - df['Close'].ewm(span=26, adjust=False).mean()
    df['MACDsig'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['RSI'] = 100 - (100 / (1 + df['Close'].pct_change().rolling(14).mean()))
    df = df.fillna(method='ffill').fillna(method='bfill')
    return df

def create_sequences(values, lookback=60):
    X, y = [], []
    for i in range(lookback, len(values)):
        X.append(values[i-lookback:i, 0])
        y.append(values[i, 0])
    X, y = np.array(X), np.array(y)
    X = X.reshape((X.shape[0], X.shape[1], 1))
    return X, y

def build_lstm(lookback=60):
    model = Sequential()
    model.add(LSTM(64, input_shape=(lookback,1), return_sequences=True))
    model.add(Dropout(0.15))
    model.add(LSTM(32, return_sequences=False))
    model.add(Dropout(0.10))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def iterative_forecast(model, last_sequence, scaler, days, lookback=60):
    seq = last_sequence.copy()  # scaled values shape (lookback, 1)
    preds = []
    for _ in range(days):
        x = seq[-lookback:].reshape(1, lookback, 1)
        p = model.predict(x, verbose=0)[0,0]
        preds.append(p)
        seq = np.append(seq, [[p]], axis=0)
    preds = np.array(preds).reshape(-1,1)
    preds = scaler.inverse_transform(preds).flatten()
    return preds

# -----------------------
# Main app logic
# -----------------------
if run:
    if ticker.strip() == "":
        st.error("Please enter a valid ticker.")
        st.stop()

    with st.spinner("Downloading price data..."):
        data = download_data(ticker.strip(), history_years)

    if data.empty:
        st.error("No data returned for ticker. Check ticker symbol / exchange suffix (eg. INFY.NS).")
        st.stop()

    # -----------------------
    # Ensure indicators exist before plotting
    data = make_indicators(data)

    # -----------------------
    # Historical Price + Indicators Graph
    # -----------------------
    st.subheader(f"{ticker.upper()} — Historical Price & Indicators")
    price_col, ind_col = st.columns([2,1])

    with price_col:
        fig_hist = go.Figure()
        # Close Price
        fig_hist.add_trace(go.Scatter(
            x=data.index, y=data['Close'],
            mode='lines', name='Close',
            line=dict(color='blue', width=2)
        ))
        # EMA20
        fig_hist.add_trace(go.Scatter(
            x=data.index, y=data['EMA20'],
            mode='lines', name='EMA20',
            line=dict(color='orange', width=1, dash='dot')
        ))
        # EMA50
        fig_hist.add_trace(go.Scatter(
            x=data.index, y=data['EMA50'],
            mode='lines', name='EMA50',
            line=dict(color='green', width=1, dash='dot')
        ))
        # RSI on secondary y-axis
        fig_hist.add_trace(go.Scatter(
            x=data.index, y=data['RSI'],
            mode='lines', name='RSI',
            yaxis='y2',
            line=dict(color='purple', width=1)
        ))
        fig_hist.update_layout(
            title=f"{ticker.upper()} — Price & Indicators",
            xaxis_title="Date",
            yaxis_title="Price",
            yaxis2=dict(title="RSI", overlaying="y", side="right"),
            template="plotly_white",
            height=500
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    # Indicators table on the side
    with ind_col:
        st.write("Latest indicators")
        st.write(data[['Close','EMA20','EMA50','RSI','MACD','MACDsig']].tail(3))


    # show price history
    st.subheader(f"Price history for {ticker.upper()} (last {history_years} years)")
    price_col, ind_col = st.columns([2,1])
    with price_col:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Close'))
        fig.update_layout(title=f"{ticker.upper()} Close Price", xaxis_title="Date", yaxis_title="Price", height=400)
        st.plotly_chart(fig, use_container_width=True)

    # indicators
    data = make_indicators(data)
    with ind_col:
        st.write("Latest indicators")
        st.write(data[['Close','EMA20','EMA50','RSI','MACD','MACDsig']].tail(3))

    # -----------------------
    # Keep your existing classifier prediction (unchanged)
    # -----------------------
    try:
        if os.path.exists(CLASSIFIER_MODEL_PATH):
            clf = joblib.load(CLASSIFIER_MODEL_PATH)
            feat = data[['EMA20','EMA50','MACD','MACDsig','RSI']].dropna().tail(1)
            if feat.shape[0] == 1:
                pred = clf.predict(feat)[0]
                st.subheader("Classifier (existing) - Current direction")
                st.success(" Bullish (UP)" if pred == 1 else "Bearish (DOWN)")
            #else:
                #st.warning("Not enough rows to feed classifier features.")
        else:
            st.warning("Classifier model not found — skipping classifier prediction.")
    except Exception as e:
        st.error(f"Error loading classifier model: {e}")

    # -----------------------
    # LSTM Forecasting
    # -----------------------
    if tf is None:
        st.error("TensorFlow not available. Install tensorflow in your environment to use LSTM forecasting.")
        st.stop()

    st.subheader(f"LSTM Forecast — next {forecast_days} days")

    # Prepare data for LSTM (use 'Close' prices)
    close_vals = data[['Close']].values  # shape (n,1)
    scaler = MinMaxScaler()
    scaled_close = scaler.fit_transform(close_vals)

    lookback = 60
    if len(scaled_close) < lookback + 10:
        st.error(f"Not enough historical data for LSTM (need at least {lookback+10} days). Increase history length.")
        st.stop()

    X, y = create_sequences(scaled_close, lookback=lookback)

    # train or load model
    model = None
    model_msg = ""
    if os.path.exists(LSTM_MODEL_PATH) and not retrain:
        try:
            model = load_model(LSTM_MODEL_PATH)
            model_msg = f"Loaded saved LSTM model from {LSTM_MODEL_PATH}"
        except Exception as e:
            st.warning(f"Could not load saved LSTM model (will train fresh): {e}")
            model = None

    if model is None:
        # Train model (simple, short training by default)
        st.info("Training LSTM model. This may take a few moments depending on environment.")
        model = build_lstm(lookback=lookback)
        epochs = 15
        batch_size = 16

        # callbacks
        cb_list = []
        ckpt_path = os.path.join(BASE_DIR, "lstm_checkpoint.h5")
        cb_list.append(EarlyStopping(monitor='loss', patience=4, restore_best_weights=True))
        cb_list.append(ModelCheckpoint(ckpt_path, save_best_only=True, monitor='loss', verbose=0))

        # Fit
        with st.spinner("Fitting LSTM..."):
            history = model.fit(X, y, epochs=epochs, batch_size=batch_size, callbacks=cb_list, verbose=0)
        # save model
        try:
            model.save(LSTM_MODEL_PATH)
            model_msg = f"Trained and saved LSTM model to {LSTM_MODEL_PATH}"
        except Exception as e:
            model_msg = f"Trained LSTM but failed to save model: {e}"

    st.write(model_msg)

    # Prepare last sequence and forecast iteratively
    last_seq = scaled_close[-lookback:]  # shape (lookback, 1)
    preds = iterative_forecast(model, last_seq, scaler, forecast_days, lookback=lookback)

    # Build forecast df
    future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=forecast_days)
    forecast_df = pd.DataFrame({"Date": future_dates, "Predicted_Close": preds})
    forecast_df.set_index("Date", inplace=True)

    # combine history + forecast for plotting
    combined = pd.concat([data['Close'], forecast_df['Predicted_Close']])
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=data.index, y=data['Close'], mode='lines', name='Historical Close'))
    fig2.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['Predicted_Close'], mode='lines+markers', name='LSTM Forecast'))
    fig2.update_layout(title=f"{ticker.upper()} — Historical + LSTM Forecast", xaxis_title="Date", yaxis_title="Price", height=450)
    st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Forecast Table")
    st.write(forecast_df)

    # Final decision: average future vs last price
    last_price = float(data['Close'].iloc[-1])
    avg_future = float(forecast_df['Predicted_Close'].mean())

    if avg_future > last_price:
        st.success(f" LSTM expects upside (Current: {last_price:.2f} → Avg Future: {avg_future:.2f})")
    else:
        st.error(f" LSTM expects downside (Current: {last_price:.2f} → Avg Future: {avg_future:.2f})")

    # Optional: show the MSE on the training tail for basic sanity check
    # compute simple one-step validation MSE on last portion if available
    try:
        # use last 20% as quick validation if dataset big enough
        split = int(len(X)*0.8)
        if split < len(X)-5:
            X_val, y_val = X[split:], y[split:]
            y_pred_val = model.predict(X_val, verbose=0).flatten()
            y_pred_val = scaler.inverse_transform(y_pred_val.reshape(-1,1)).flatten()
            y_val_true = scaler.inverse_transform(y_val.reshape(-1,1)).flatten()
            mse = np.mean((y_pred_val - y_val_true)**2)
            st.info(f"Validation MSE (one-step) ≈ {mse:.4f}")
    except Exception:
        pass

    st.write(" Forecast complete! For more accurate results, try using more historical data or training the model longer.")

