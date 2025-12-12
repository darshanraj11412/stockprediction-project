# app.py — Optimized Streamlit app with multi-step forecasting + improved UI + Fix 1 (yfinance reliability)
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import yfinance as yf
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import plotly.express as px
import plotly.graph_objects as go

# Try import TensorFlow for LSTM support — optional
try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
    TENSORFLOW_AVAILABLE = True
except Exception:
    TENSORFLOW_AVAILABLE = False

# ---------- Page config ----------
st.set_page_config(page_title="Stock Prediction App",
                   page_icon="📈",
                   layout="wide",
                   initial_sidebar_state="expanded")


# ---------- FIX 1: Improved Yahoo fetch ----------
@st.cache_data(ttl=60*60)  # cache data for one hour
def fetch_history(ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    """
    Improved Yahoo Finance fetch with retries + safe fallback.
    Fix 1 applied.
    """
    import yfinance as yf
    from yfinance import shared

    yf.pdr_override()           # enable auto retry
    shared._DEFAULT_USER_AGENT = "Mozilla/5.0"   # change UA to avoid blocks

    try:
        df = yf.download(
            ticker,
            period=period,
            interval=interval,
            progress=False,
            threads=False,       # MUST disable threads for Streamlit Cloud
            auto_adjust=False,
            repair=True
        )
    except Exception as e:
        st.error(f"Yahoo blocked the request: {e}")
        return pd.DataFrame()

    if df is None or df.empty:
        return pd.DataFrame()

    df = df.reset_index()
    df["Date"] = pd.to_datetime(df["Date"])
    return df


# ---------- Helper functions ----------
@st.cache_resource
def load_joblib_model(path: str):
    if os.path.exists(path):
        return joblib.load(path)
    return None

@st.cache_resource
def load_tf_model(path: str):
    if os.path.exists(path):
        try:
            return load_model(path)
        except Exception:
            return None
    return None

def safe_inverse_transform(scaler, arr):
    if scaler is None:
        return arr
    arr_2d = np.array(arr).reshape(-1, 1)
    return scaler.inverse_transform(arr_2d).flatten()

def prepare_series_for_model(series: pd.Series, window: int):
    arr = series.values.reshape(-1, 1)
    scaler = MinMaxScaler()
    arr_s = scaler.fit_transform(arr)

    X, y = [], []
    for i in range(window, len(arr_s)):
        X.append(arr_s[i-window:i, 0])
        y.append(arr_s[i, 0])

    X = np.array(X)
    y = np.array(y)
    X_lstm = X.reshape((X.shape[0], X.shape[1], 1))
    return X, X_lstm, y, scaler

def recursive_forecast(last_window, model, scaler, steps=7, is_lstm=False):
    preds_scaled = []
    window = len(last_window)
    window_arr = last_window.copy().tolist()

    for i in range(steps):
        if is_lstm:
            x_in = np.array(window_arr[-window:]).reshape(1, window, 1)
        else:
            x_in = np.array(window_arr[-window:]).reshape(1, window)

        p = model.predict(x_in, verbose=0)
        p_val = float(np.array(p).flatten()[0])
        preds_scaled.append(p_val)

        window_arr.append(p_val)

    preds_original = safe_inverse_transform(scaler, preds_scaled)
    return preds_original


# ---------- Sidebar ----------
st.sidebar.title("📈 Stock Prediction")
st.sidebar.markdown("Clean UI • Multi-step forecasting • LSTM optional")

section = st.sidebar.radio("Navigate to:", ["Home", "Visualizations", "Predictions", "Model Info", "About"])

ticker_input = st.sidebar.text_input("Ticker (Yahoo)", value="AAPL").upper()
period = st.sidebar.selectbox("History Period", ["6mo", "1y", "2y", "5y"], index=2)
interval = st.sidebar.selectbox("Interval", ["1d", "1wk", "1mo"], index=0)
window = st.sidebar.slider("Window (days) for model", 5, 60, 20)
n_steps = st.sidebar.slider("Forecast horizon (days)", 1, 30, 7)
method = st.sidebar.selectbox("Forecast method", ["Recursive"], index=0)
use_lstm_if_available = st.sidebar.checkbox("Use TensorFlow LSTM if available", value=True)


# ---------- Sections ----------
# HOME
if section == "Home":
    st.title("📉 Stock Prediction Dashboard")
    st.markdown("""
    This app fetches historical stock data, shows interactive visualizations,
    and produces short-term forecasts. It supports both classical ML models
    (joblib) and TensorFlow LSTM if available.
    """)

    df = fetch_history(ticker_input, period, interval)

    if df.empty:
        st.error(f"No data found for {ticker_input}. Check ticker symbol or Yahoo may be blocking requests.")
    else:
        st.subheader(f"{ticker_input} Recent Prices")
        st.dataframe(df.tail())

        fig = px.line(df, x="Date", y="Close", title=f"{ticker_input} Closing Price")
        st.plotly_chart(fig, use_container_width=True)


# VISUALIZATIONS
elif section == "Visualizations":
    st.header("📊 Visualizations")
    df = fetch_history(ticker_input, period, interval)

    if df.empty:
        st.error("No data to visualize.")
    else:
        fig = px.line(df, x="Date", y="Close", title="Close Price")
        st.plotly_chart(fig, use_container_width=True)

        df["MA20"] = df["Close"].rolling(20).mean()
        df["MA50"] = df["Close"].rolling(50).mean()

        fig2 = px.line(df, x="Date", y=["Close", "MA20", "MA50"], title="Moving Averages")
        st.plotly_chart(fig2, use_container_width=True)


# PREDICTIONS
elif section == "Predictions":
    st.header("🔮 Predictions")

    df = fetch_history(ticker_input, period, interval)
    if df.empty:
        st.error("Cannot predict — no data found.")
    else:
        series = df["Close"]
        X, X_lstm, y, scaler = prepare_series_for_model(series, window)

        # Load models
        joblib_model = load_joblib_model("stock_model.joblib")
        tf_model = load_tf_model("lstm_model.h5") if (use_lstm_if_available and TENSORFLOW_AVAILABLE) else None

        model = tf_model if tf_model else joblib_model
        is_lstm = tf_model is not None

        if model is None:
            st.error("No model found: upload stock_model.joblib or lstm_model.h5")
        else:
            last_window_scaled = X_lstm[-1].reshape(-1) if is_lstm else X[-1]

            preds = recursive_forecast(last_window_scaled, model, scaler, steps=n_steps, is_lstm=is_lstm)
            future_dates = [df["Date"].max() + timedelta(days=i+1) for i in range(n_steps)]

            pred_df = pd.DataFrame({"Date": future_dates, "Predicted": preds})
            st.write(pred_df)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], mode="lines", name="History"))
            fig.add_trace(go.Scatter(x=future_dates, y=preds, mode="lines+markers", name="Forecast"))
            fig.update_layout(title=f"{ticker_input} {n_steps}-Day Forecast", height=450)
            st.plotly_chart(fig, use_container_width=True)


# MODEL INFO
elif section == "Model Info":
    st.header("📘 Model Info")

    st.write(f"TensorFlow available: {TENSORFLOW_AVAILABLE}")
    st.write(f"LSTM model exists: {os.path.exists('lstm_model.h5')}")
    st.write(f"Joblib model exists: {os.path.exists('stock_model.joblib')}")


# ABOUT
elif section == "About":
    st.header("ℹ️ About this App")
    st.markdown("""
    Made with ❤️ for learning stock forecasting.
    Features multi-step forecasting, a clean UI, and joblib/LSTM support.
    """)

st.markdown("<hr><center>Built with Streamlit • GitHub: darshanraj11412</center>", unsafe_allow_html=True)
