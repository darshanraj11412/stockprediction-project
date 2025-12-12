# app.py — Optimized Streamlit app with multi-step forecasting + improved UI
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

# ---------- Helper cached functions ----------
@st.cache_data(ttl=60*60)  # cache fetched data for an hour
def fetch_history(ticker: str, period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    """Fetch historical OHLCV data using yfinance."""
    df = yf.download(ticker, period=period, interval=interval, progress=False)
    if df.empty:
        return pd.DataFrame()
    df = df.reset_index().rename(columns={"Date": "Date"})
    df['Date'] = pd.to_datetime(df['Date'])
    return df

@st.cache_resource
def load_joblib_model(path: str):
    if os.path.exists(path):
        return joblib.load(path)
    return None

@st.cache_resource
def load_tf_model(path: str):
    # load_model can be heavy; cache resource
    if os.path.exists(path):
        try:
            return load_model(path)
        except Exception:
            return None
    return None

def safe_inverse_transform(scaler, arr):
    """Utility to inverse transform a 1D array when scaler was fitted on 2D data."""
    if scaler is None:
        return arr
    arr_2d = np.array(arr).reshape(-1, 1)
    return scaler.inverse_transform(arr_2d).flatten()

# ---------- Prediction helpers ----------
def prepare_series_for_model(series: pd.Series, window: int):
    """Create rolling windows X,y arrays for supervised learning."""
    arr = series.values.reshape(-1, 1)
    scaler = MinMaxScaler()
    arr_s = scaler.fit_transform(arr)
    X, y = [], []
    for i in range(window, len(arr_s)):
        X.append(arr_s[i-window:i, 0])
        y.append(arr_s[i, 0])
    X = np.array(X)
    y = np.array(y)
    # For LSTM we want shape (samples, timesteps, features)
    X_lstm = X.reshape((X.shape[0], X.shape[1], 1))
    return X, X_lstm, y, scaler

def recursive_forecast(last_window, model, scaler, steps=7, is_lstm=False):
    """
    Recursive multi-step forecasting:
    - last_window: (window,) array in scaled space (0-1)
    - model: model to predict 1-step ahead
    - scaler: scaler used for inverse transform
    - steps: int number of future steps
    - is_lstm: whether the model expects LSTM input shape
    Returns scaled predictions (0-1) as list.
    """
    preds_scaled = []
    window = len(last_window)
    window_arr = last_window.copy().tolist()
    for i in range(steps):
        if is_lstm:
            x_in = np.array(window_arr[-window:]).reshape(1, window, 1)
        else:
            x_in = np.array(window_arr[-window:]).reshape(1, window)
        p = model.predict(x_in, verbose=0)
        # p might be shape (1,1) or scalar
        if isinstance(p, np.ndarray):
            p_val = float(p.flatten()[0])
        else:
            p_val = float(p)
        preds_scaled.append(p_val)
        window_arr.append(p_val)
    # inverse transform to original scale
    preds_original = safe_inverse_transform(scaler, preds_scaled)
    return preds_original

# ---------- UI: sidebar ----------
st.sidebar.title("📈 Stock Prediction")
st.sidebar.markdown("Clean UI • Multi-step forecasting • LSTM optional")
section = st.sidebar.radio("Navigate to:", ["Home", "Visualizations", "Predictions", "Model Info", "About"])

# Settings
st.sidebar.markdown("---")
ticker_input = st.sidebar.text_input("Ticker (Yahoo)", value="AAPL").upper()
period = st.sidebar.selectbox("History Period", options=["6mo", "1y", "2y", "5y"], index=2)
interval = st.sidebar.selectbox("Interval", options=["1d", "1wk", "1mo"], index=0)
window = st.sidebar.slider("Window (days) for model", min_value=5, max_value=60, value=20, step=1)
n_steps = st.sidebar.slider("Forecast horizon (days)", min_value=1, max_value=30, value=7, step=1)
method = st.sidebar.selectbox("Forecast method", options=["Recursive (default)", "Direct (not implemented)"], index=0)
use_lstm_if_available = st.sidebar.checkbox("Use TensorFlow LSTM if available", value=True)

st.sidebar.markdown("---")
st.sidebar.markdown("Quick links:")
st.sidebar.markdown("- GitHub repo")
st.sidebar.markdown("- Deployed app")

# ---------- Section: Home ----------
if section == "Home":
    st.title("📈 Stock Prediction Dashboard")
    st.markdown("""
    This app fetches historical stock data, shows interactive visualizations,
    and produces short-term forecasts. It supports both classical ML models
    (joblib) and TensorFlow LSTM if available.
    """)
    st.info("Tip: change ticker & period from the sidebar. Use multi-step forecasting to predict several days ahead.")

    # Quick fetch preview
    with st.spinner("Fetching historical data..."):
        df = fetch_history(ticker_input, period=period, interval=interval)

    if df.empty:
        st.error(f"No data found for {ticker_input}. Check ticker symbol.")
    else:
        st.subheader(f"{ticker_input} — Recent data")
        st.dataframe(df.tail(10).reset_index(drop=True))
        # Price chart
        fig = px.line(df, x="Date", y="Close", title=f"{ticker_input} Close Price", height=400)
        fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)

# ---------- Section: Visualizations ----------
elif section == "Visualizations":
    st.header("📊 Visualizations")
    df = fetch_history(ticker_input, period=period, interval=interval)
    if df.empty:
        st.error("No historical data for this ticker.")
    else:
        # Price + Volume subplots
        fig = make = go.Figure()
        fig.add_trace(go.Scatter(x=df["Date"], y=df["Close"], mode="lines", name="Close"))
        fig.update_layout(title=f"{ticker_input} Close Price", height=450)
        st.plotly_chart(fig, use_container_width=True)

        # Moving averages
        df['MA20'] = df['Close'].rolling(20).mean()
        df['MA50'] = df['Close'].rolling(50).mean()
        ma_fig = go.Figure()
        ma_fig.add_trace(go.Scatter(x=df['Date'], y=df['Close'], name='Close'))
        ma_fig.add_trace(go.Scatter(x=df['Date'], y=df['MA20'], name='MA20'))
        ma_fig.add_trace(go.Scatter(x=df['Date'], y=df['MA50'], name='MA50'))
        ma_fig.update_layout(title=f"{ticker_input} with Moving Averages", height=450)
        st.plotly_chart(ma_fig, use_container_width=True)

        st.markdown("### Volume")
        vol_fig = px.bar(df, x="Date", y="Volume", title="Trading Volume", height=300)
        st.plotly_chart(vol_fig, use_container_width=True)

# ---------- Section: Predictions ----------
elif section == "Predictions":
    st.header("🤖 Predictions & Forecasting")
    df = fetch_history(ticker_input, period=period, interval=interval)
    if df.empty:
        st.error("No historical data — cannot predict.")
    else:
        st.subheader(f"Prepared data for {ticker_input}")
        st.write(f"Data points: {len(df)} • Last date: {df['Date'].max().date()}")

        # Use Close price series
        series = df['Close']

        # Prepare windows
        X, X_lstm, y, scaler = prepare_series_for_model(series, window=window)

        # Load joblib models (classical)
        joblib_model = load_joblib_model("stock_model.joblib")
        logistic_model = load_joblib_model("logistic_model.joblib")

        # Load TensorFlow model if requested and available
        tf_model = None
        if use_lstm_if_available and TENSORFLOW_AVAILABLE:
            tf_model = load_tf_model("lstm_model.h5")  # if you have one
            if tf_model is None:
                st.warning("TensorFlow available but no `lstm_model.h5` found, falling back to joblib models.")

        # Select model to use
        model_to_use = None
        is_lstm_model = False
        if tf_model is not None:
            model_to_use = tf_model
            is_lstm_model = True
        elif joblib_model is not None:
            model_to_use = joblib_model
            is_lstm_model = False
        else:
            st.error("No model found (joblib or TensorFlow). Prediction will use naive last-value forecasting.")
            model_to_use = None

        # Single-step prediction example (last point)
        if model_to_use is not None:
            # Use last available window from scaled data
            last_window_scaled = X[-1] if not is_lstm_model else X_lstm[-1].reshape(-1)
            # For LSTM we may use X_lstm last sample
            try:
                pred_one_scaled = None
                if is_lstm_model:
                    p = model_to_use.predict(X_lstm[-1].reshape(1, window, 1), verbose=0)
                    pred_one_scaled = float(np.array(p).flatten()[0])
                else:
                    p = model_to_use.predict(X[-1].reshape(1, -1))
                    pred_one_scaled = float(np.array(p).flatten()[0])
                pred_one = safe_inverse_transform(scaler, [pred_one_scaled])[0]
                st.metric(label="Next day predicted close", value=f"{pred_one:.2f}")
            except Exception as e:
                st.warning(f"Single step prediction failed: {e}")

        # Multi-step forecasting (recursive)
        st.markdown("### Multi-step Forecast")
        method_text = "Recursive (use model repeatedly)" if method.startswith("Recursive") else "Direct (not implemented)"
        st.write(f"Method: **{method_text}** • Horizon: **{n_steps}** days")

        if model_to_use is None:
            # naive forecast: repeat last close
            last_price = series.values[-1]
            preds = [last_price] * n_steps
            future_dates = [df['Date'].max().date() + timedelta(days=i+1) for i in range(n_steps)]
            pred_df = pd.DataFrame({"Date": future_dates, "Predicted": preds})
            st.warning("No trained model — showing naive last-value forecast.")
            st.table(pred_df)
        else:
            # Build last window scaled
            if is_lstm_model:
                last_win_scaled = X_lstm[-1].reshape(-1)  # flattened
            else:
                last_win_scaled = X[-1]  # already scaled

            # Do recursive forecast
            try:
                preds = recursive_forecast(last_win_scaled, model_to_use, scaler, steps=n_steps, is_lstm=is_lstm_model)
                future_dates = [df['Date'].max().date() + timedelta(days=i+1) for i in range(n_steps)]
                pred_df = pd.DataFrame({"Date": future_dates, "Predicted": preds})
                st.success("Multi-step forecast generated.")
                st.table(pred_df)

                # Plot historical + future
                hist = df[['Date', 'Close']].copy()
                future_df = pd.DataFrame({"Date": future_dates, "Close": preds})
                combined = pd.concat([hist.tail(90), future_df], ignore_index=True, sort=False)

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=hist['Date'], y=hist['Close'], mode='lines', name='History'))
                fig.add_trace(go.Scatter(x=future_df['Date'], y=future_df['Close'], mode='lines+markers', name='Forecast'))
                fig.update_layout(title=f"{ticker_input} — Historical + {n_steps}-day Forecast", height=520)
                st.plotly_chart(fig, use_container_width=True)

                # Simple trend message
                if preds[-1] > preds[0]:
                    st.success(f"Model suggests an UP trend over the next {n_steps} days (from {preds[0]:.2f} to {preds[-1]:.2f}).")
                else:
                    st.info(f"Model suggests a DOWN/flat trend over the next {n_steps} days (from {preds[0]:.2f} to {preds[-1]:.2f}).")
            except Exception as e:
                st.error(f"Forecast generation failed: {e}")

# ---------- Section: Model Info ----------
elif section == "Model Info":
    st.header("📘 Model Info & Diagnostics")
    st.markdown("This page shows what models are available and provides diagnostics.")

    st.write(f"TensorFlow available: **{TENSORFLOW_AVAILABLE}**")
    st.write(f"TensorFlow model present: {'lstm_model.h5' if os.path.exists('lstm_model.h5') else 'No'}")
    st.write(f"Joblib model present: {'stock_model.joblib' if os.path.exists('stock_model.joblib') else 'No'}")

    # Quick model load and test
    if os.path.exists("stock_model.joblib"):
        mdl = load_joblib_model("stock_model.joblib")
        st.write("Loaded joblib model summary:")
        st.write(str(mdl))
    else:
        st.info("No joblib model found in repo root.")

    if TENSORFLOW_AVAILABLE and os.path.exists("lstm_model.h5"):
        st.info("TensorFlow LSTM model available; you can run LSTM forecasts.")
    else:
        st.info("No TensorFlow LSTM model loaded.")

# ---------- Section: About ----------
elif section == "About":
    st.header("ℹ️ About this App")
    st.markdown("""
    **Stock Prediction App** — demo app built with Streamlit.
    - Fetches data from Yahoo Finance using `yfinance`.
    - Supports joblib-based classical models and optional TensorFlow LSTM.
    - Multi-step forecasting is implemented recursively.
    - Cleaned and optimized for Streamlit deployment.
    """)
    st.markdown("---")
    st.markdown("Made with ❤️ — improve, iterate, and enjoy!")

# ---------- Footer ----------
st.markdown("""<hr style="margin-top:20px"/>""", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;color:gray;'>Built with Streamlit • GitHub: darshanraj11412</p>", unsafe_allow_html=True)

