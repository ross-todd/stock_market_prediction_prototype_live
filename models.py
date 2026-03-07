#  Ross Todd
#  BSc (Hons) Software Development
#  Honours Project 2026 - A Stock Market Prediction Application Prototype

# models.py — Application Logic Layer
# Contains ARIMA, Random Forest, and GRU forecasting models

# Model configurations match the best-performing settings identified
# during the comparative analysis:
#   ARIMA : order=(1,0,1), trend='c', window=63
#   RF    : n_estimators=300, max_depth=8, 14 engineered features
#   GRU   : lookback=21, units=64, layers=2, dropout=0.2, lr=0.001

# All three methods receive the same raw DataFrame from StockDataService
# and extract what they need internally. The DataFrame uses a MultiIndex
# column structure: ('Close', 'BARC.L'), ('Open', 'BARC.L'), etc.
# ARIMA is a univariate baseline by design — any performance gap
# between it and the multivariate RF and GRU models is therefore
# attributable to model class rather than feature availability.

import pandas as pd
import numpy as np
import streamlit as st
import random
from typing import Tuple, Optional

import joblib
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
import tensorflow as tf


# ── Pre-trained model directory ───────────────────────────────────────────────
# RF and GRU models were serialised from the comparative analysis scripts
# and loaded here to avoid retraining on every app run. ARIMA is refitted
# at runtime since it is lightweight and stateless.
TRAINED_MODELS_DIR = os.path.join(os.path.dirname(__file__), 'trained_models')

# ── Forecast weights (50/20/10/10/10) ────────────────────────────────────────
# Weights applied to the 5-day forecast to produce a single weighted average
# summary figure. Day 1 carries the most weight (50%) since forecast
# reliability decreases with horizon — this matches the summary metric used
# in the comparative analysis.
FORECAST_WEIGHTS = np.array([0.5, 0.2, 0.1, 0.1, 0.1])

# ── Dissertation residual std values (log-return space) ──────────────────────
# Extracted directly from the comparative analysis terminal output.
# These are the final current_res_std values in log-return space used
# at the point of forecast generation, ensuring PI widths match exactly.
RF_RESIDUAL_STD = {
    'BARC.L': 0.0196572589,
    'LLOY.L': 0.0158935703,
    'HSBA.L': 0.0153867528,
}
GRU_RESIDUAL_STD = {
    'BARC.L': 0.0181549697,
    'LLOY.L': 0.0156723814,
    'HSBA.L': 0.0122015238,
}


# ══════════════════════════════════════════════════════════════════════════
#   SHARED HELPERS
#   _extract_series handles both MultiIndex (multi-ticker) and flat
#   (single-ticker) DataFrames returned by yfinance. _extract_ohlcv
#   builds the full OHLCV frame needed by the RF and GRU feature pipelines,
#   preferring Adj Close over Close where available.
# ══════════════════════════════════════════════════════════════════════════

def _extract_series(df: pd.DataFrame, field: str, ticker: str) -> pd.Series:
    if isinstance(df.columns, pd.MultiIndex):
        if (field, ticker) in df.columns:
            return df[(field, ticker)].dropna()
        for col in df.columns:
            if col[0].lower() == field.lower() and col[1] == ticker:
                return df[col].dropna()
    else:
        if field in df.columns:
            return df[field].dropna()
    return pd.Series(dtype=float)


def _extract_ohlcv(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    adj   = _extract_series(df, 'Adj Close', ticker)
    close = _extract_series(df, 'Close', ticker)
    price = adj if len(adj) > 0 else close

    out = pd.DataFrame({
        'Adj Close': price,
        'Open':      _extract_series(df, 'Open',   ticker),
        'High':      _extract_series(df, 'High',   ticker),
        'Low':       _extract_series(df, 'Low',    ticker),
        'Volume':    _extract_series(df, 'Volume', ticker),
    }).dropna()

    return out


def weighted_forecast(forecast: np.ndarray) -> float:
    return float(np.sum(np.array(forecast) * FORECAST_WEIGHTS))


# ══════════════════════════════════════════════════════════════════════════
#   ARIMA — order=(1,0,1), trend='c', window=63
#
#   Operates on log-returns rather than raw prices for stationarity.
#   The model is fitted on the most recent 63 trading days (one quarter)
#   to match the refit frequency used in the walk-forward validation.
#   Confidence intervals are derived analytically from the ARIMA forecast
#   object and then converted back to price space via exponentiation.
#   Horizon-widening is applied so intervals grow appropriately
#   as forecast uncertainty accumulates over the 5-day window.
# ══════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def _cached_arima(
    prices_tuple: tuple,
    today_str: str,
    ticker_symbol: str
) -> Tuple[np.ndarray, pd.DataFrame]:

    prices      = pd.to_numeric(pd.Series(prices_tuple), errors='coerce').dropna()
    log_returns = np.log(prices / prices.shift(1)).dropna()

    subset    = log_returns.iloc[-63:].values
    model_fit = ARIMA(subset, order=(3, 0, 0), trend='c').fit()

    forecast_obj  = model_fit.get_forecast(steps=5)
    forecast_rets = np.asarray(forecast_obj.predicted_mean)
    conf_int_ret  = forecast_obj.conf_int()

    # ── Convert log-return forecasts back to price space ─────────────────
    # Cumulative sum of log-returns gives the cumulative price movement.
    # sigma is extracted from the first step interval and applied with
    # horizon scaling to widen intervals over the 5-day forecast window.
    last_price = float(prices.iloc[-1])
    cumulative_log_returns = np.cumsum(forecast_rets)
    horizons   = np.arange(1, 6)

    conf_int_array  = np.array(conf_int_ret)
    sigma_one_step  = (conf_int_array[0, 1] - forecast_rets[0]) / 1.96
    sigma_one_step  = max(abs(sigma_one_step), float(np.std(subset)))
    forecast_prices = last_price * np.exp(cumulative_log_returns)
    ci_lower        = last_price * np.exp(cumulative_log_returns - 1.96 * sigma_one_step * np.sqrt(horizons))
    ci_upper        = last_price * np.exp(cumulative_log_returns + 1.96 * sigma_one_step * np.sqrt(horizons))

    return np.array(forecast_prices), pd.DataFrame({'lower': ci_lower, 'upper': ci_upper})

# ══════════════════════════════════════════════════════════════════════════
#   RANDOM FOREST — n_estimators=300, max_depth=8
#
#   15 engineered features constructed from full OHLCV data, matching
#   the feature set used in the comparative analysis exactly:
#   lagged log-returns, rolling SMAs, RSI, Bollinger Band width, volatility,
#   raw volume, volume MA, volume ratio, and ATR (14 and 20 day).
#   The full OHLCV frame is passed in so that ATR uses real High and Low
#   values rather than proxied Close prices, keeping results consistent
#   with the comparative analysis. The pre-trained RF model and its
#   StandardScaler are loaded from disk rather than retrained on every
#   app run. Prediction intervals are based on fixed residual std values
#   back-calculated from the comparative analysis uncertainty metrics.
# ══════════════════════════════════════════════════════════════════════════

def _build_rf_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    # ── Use Adj Close as the price column where available ─────────────────
    # Matches the comparative analysis which uses Adj Close as price_col.
    df = ohlcv.copy()
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna().astype(float)

    price_col = 'Adj Close' if 'Adj Close' in df.columns else 'Close'

    df['Log_Ret'] = np.log(df[price_col] / df[price_col].shift(1))

    for lag in [1, 2, 3, 5]:
        df[f'Ret_Lag_{lag}'] = df['Log_Ret'].shift(lag)

    df['SMA_5']     = df[price_col].rolling(5).mean()
    df['SMA_20']    = df[price_col].rolling(20).mean()
    df['SMA_Ratio'] = df['SMA_5'] / (df['SMA_20'] + 1e-9)

    delta         = df[price_col].diff()
    gain          = delta.where(delta > 0, 0).rolling(14).mean()
    loss          = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI']     = 100 - (100 / (1 + gain / (loss + 1e-9)))

    sma            = df[price_col].rolling(20).mean()
    std            = df[price_col].rolling(20).std()
    df['BB_Width'] = (4 * std) / (sma + 1e-9)

    df['Volatility']   = df['Log_Ret'].rolling(20).std()
    df['Volume_MA']    = df['Volume'].rolling(20).mean()
    df['Volume_Ratio'] = df['Volume'] / (df['Volume_MA'] + 1e-9)

    # ── ATR using real High and Low ───────────────────────────────────────
    # The comparative analysis computes True Range from actual OHLCV data.
    # Passing the full OHLCV frame here ensures ATR matches exactly rather
    # than being proxied from Close prices alone.
    df['True_Range'] = np.maximum(
        df['High'] - df['Low'],
        np.maximum(
            abs(df['High'] - df[price_col].shift(1)),
            abs(df['Low']  - df[price_col].shift(1))
        )
    )
    df['ATR_14'] = df['True_Range'].rolling(14).mean()
    df['ATR_20'] = df['True_Range'].rolling(20).mean()

    return df.dropna()


@st.cache_resource
def _load_rf(ticker_symbol: str):
    path = os.path.join(TRAINED_MODELS_DIR, f'rf_{ticker_symbol.replace(".", "_")}.pkl')
    if not os.path.exists(path):
        raise FileNotFoundError(f"No trained RF model found at {path}")
    return joblib.load(path)


@st.cache_data(show_spinner=False)
def _cached_rf(
    ohlcv_tuple: tuple,
    today_str: str,
    ticker_symbol: str
) -> Tuple[np.ndarray, pd.DataFrame]:

    ohlcv = pd.DataFrame(list(ohlcv_tuple),
                columns=['Adj Close', 'Open', 'High', 'Low', 'Volume'])

    df_feat = _build_rf_features(ohlcv)

    # ── 14 engineered features matching the comparative analysis exactly ─────────────
    # Feature order must match the order the StandardScaler was fitted on
    # during the comparative analysis, otherwise scaler.transform will
    # produce incorrect standardised values. The comparative analysis selects
    # features via prefix match on df_feat.columns which picks up raw Volume
    # in addition to Volume_MA and Volume_Ratio, giving 15 total.
    features = [
        'Ret_Lag_1', 'Ret_Lag_2', 'Ret_Lag_3', 'Ret_Lag_5',
        'SMA_5', 'SMA_20', 'SMA_Ratio', 'RSI', 'BB_Width',
        'Volatility', 'Volume', 'Volume_MA', 'Volume_Ratio', 'ATR_14', 'ATR_20'
    ]
    features = [f for f in features if f in df_feat.columns]

    # ── Load pre-trained model and scaler ─────────────────────────────────
    # Model and scaler are always saved as separate files by the comparative
    # analysis, so they are loaded independently here.
    saved        = _load_rf(ticker_symbol)
    rf           = saved
    ticker_clean = ticker_symbol.replace(".", "_")
    scaler_path  = os.path.join(TRAINED_MODELS_DIR, f'scaler_{ticker_clean}_rf.pkl')
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"No saved RF scaler found at {scaler_path}")
    scaler = joblib.load(scaler_path)

    # ── Residual std for prediction intervals ─────────────────────────────
    # Fixed dissertation values are used where available. If no value exists
    # for a ticker (e.g. a new ticker added later), it falls back to computing
    # residual std from the current data to keep the app functional.
    residual_std = RF_RESIDUAL_STD.get(ticker_symbol, None)
    if residual_std is None:
        y_all        = df_feat['Log_Ret'].shift(-1).dropna().values
        X_all        = df_feat[features].values[:len(y_all)]
        residual_std = np.std(y_all - rf.predict(scaler.transform(X_all)), ddof=1)

    # ── Recursive 5-day forecast ──────────────────────────────────────────
    # An exact copy of the comparative analysis recursive forecast loop.
    # current_features is seeded from df_feat[features] (the 15-feature row)
    # and running_gain, running_sq_ret, running_tr are initialised from the
    # last row of df_feat exactly as in the comparative analysis script.
    price_col         = 'Adj Close' if 'Adj Close' in df_feat.columns else 'Close'
    last_price        = float(df_feat[price_col].iloc[-1])
    current_features  = df_feat[features].iloc[-1].copy()
    current_price     = last_price
    forecast_log_rets = []

    running_gain   = float(df_feat['RSI'].iloc[-1])
    running_sq_ret = float(current_features.get('Volatility', 0)) ** 2
    running_tr     = float(current_features.get('ATR_14', 0))

    LAGS = [1, 2, 3, 5]

    for day in range(5):
        current_features_scaled = scaler.transform(current_features.values.reshape(1, -1))
        predicted_log_return = rf.predict(current_features_scaled)[0]
        forecast_log_rets.append(predicted_log_return)

        pred_price = current_price * np.exp(predicted_log_return)

        for lag in reversed(LAGS[1:]):
            if f'Ret_Lag_{lag}' in current_features.index:
                prev_lag = f'Ret_Lag_{lag-1}' if lag > 1 else None
                current_features[f'Ret_Lag_{lag}'] = (
                    current_features[prev_lag] if prev_lag and prev_lag in current_features.index
                    else 0)
        if 'Ret_Lag_1' in current_features.index:
            current_features['Ret_Lag_1'] = predicted_log_return

        if 'SMA_5' in current_features.index:
            current_features['SMA_5']  = (current_features['SMA_5'] * 4 + pred_price) / 5
        if 'SMA_20' in current_features.index:
            current_features['SMA_20'] = (current_features['SMA_20'] * 19 + pred_price) / 20
        if 'SMA_Ratio' in current_features.index:
            current_features['SMA_Ratio'] = current_features['SMA_5'] / (
                current_features['SMA_20'] + 1e-9)

        if 'Volatility' in current_features.index:
            alpha_v        = 1.0 / 20
            running_sq_ret = (1 - alpha_v) * running_sq_ret + alpha_v * predicted_log_return ** 2
            current_features['Volatility'] = np.sqrt(max(running_sq_ret, 1e-10))

        if 'BB_Width' in current_features.index:
            ewm_std = current_features['Volatility'] * current_features['SMA_20']
            current_features['BB_Width'] = (4 * ewm_std) / (current_features['SMA_20'] + 1e-9)

        if 'ATR_14' in current_features.index:
            daily_range_proxy = pred_price * current_features['Volatility']
            running_tr        = (running_tr * 13 + daily_range_proxy) / 14
            current_features['ATR_14'] = running_tr
        if 'ATR_20' in current_features.index:
            current_features['ATR_20'] = (current_features['ATR_20'] * 19 + running_tr) / 20

        if 'RSI' in current_features.index:
            delta_ret = predicted_log_return
            alpha_rsi = 1.0 / 14
            if delta_ret > 0:
                running_gain = (1 - alpha_rsi) * current_features['RSI'] + alpha_rsi * 100
            else:
                running_gain = (1 - alpha_rsi) * current_features['RSI']
            current_features['RSI'] = np.clip(running_gain, 0, 100)

        current_price = pred_price

    cumulative_log_returns = np.cumsum(np.array(forecast_log_rets))
    horizons               = np.arange(1, 6)
    forecast_prices        = last_price * np.exp(cumulative_log_returns)
    ci_lower = last_price * np.exp(cumulative_log_returns - 1.96 * residual_std * np.sqrt(horizons))
    ci_upper = last_price * np.exp(cumulative_log_returns + 1.96 * residual_std * np.sqrt(horizons))

    return np.array(forecast_prices), pd.DataFrame({'lower': ci_lower, 'upper': ci_upper})

    


# ═══════════════════════════════════════════════════════════════════════════════════
#   GRU — lookback=21, units=64, layers=2, dropout=0.2, lr=0.001
#
#   Multivariate deep learning model using 10 engineered features:
#   OHLCV, RSI, MACD, Bollinger Bands, and lagged log-return.
#   The pre-trained model and MinMaxScaler are loaded from the comparative analysis.
#   Predicted scaled log-returns are inverse-transformed to real log-return
#   space before advancing the price, then re-scaled before being fed back
#   into the lookback window — this prevents scale drift across the 5 steps.
#   Prediction intervals use the fixed real_residual_std values from the
#   comparative analysis rather than recalculating at runtime.
# ═══════════════════════════════════════════════════════════════════════════════════

def _build_gru_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    df = ohlcv.copy()
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna().astype(float)

    delta         = df['Adj Close'].diff()
    gain          = delta.where(delta > 0, 0).rolling(14).mean()
    loss          = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI']     = 100 - (100 / (1 + gain / (loss + 1e-9)))

    df['MACD']        = (df['Adj Close'].ewm(span=12).mean() -
                         df['Adj Close'].ewm(span=26).mean())
    df['Signal_Line'] = df['MACD'].ewm(span=9).mean()

    ma20             = df['Adj Close'].rolling(20).mean()
    std20            = df['Adj Close'].rolling(20).std()
    df['Upper_Band'] = ma20 + 2 * std20
    df['Lower_Band'] = ma20 - 2 * std20
    df               = df.dropna()

    # ── Log-return and lag ────────────────────────────────────────────────
    # Raw prices are replaced by log-returns for stationarity, consistent
    # with the ARIMA and RF feature engineering in the comparative analysis.
    prices             = df['Adj Close'].values
    log_rets           = np.log(prices[1:] / prices[:-1])
    df                 = df.iloc[1:].copy()
    df['log_return']   = log_rets
    df['Log_Ret_Lag1'] = df['log_return'].shift(1)

    return df.dropna()


@st.cache_data(show_spinner=False)
def _cached_gru(
    ohlcv_tuple: tuple,
    today_str: str,
    ticker_symbol: str
) -> Tuple[np.ndarray, pd.DataFrame]:

    ohlcv = pd.DataFrame(list(ohlcv_tuple),
                         columns=['Adj Close', 'Open', 'High', 'Low', 'Volume'])

    df         = _build_gru_features(ohlcv)
    features   = ['Open', 'High', 'Low', 'Volume', 'RSI', 'MACD',
                  'Signal_Line', 'Upper_Band', 'Lower_Band', 'Log_Ret_Lag1']
    
    n_features = len(features) + 1

    # ── Load scaler and scale input data ─────────────────────────────────
    # The scaler was fitted on training data only during the comparative
    # analysis. The same scaler is reused here so that the input distribution
    # seen by the model matches what it was trained on.
    ticker_clean = ticker_symbol.replace(".", "_")
    scaler_path  = os.path.join(TRAINED_MODELS_DIR, f'scaler_{ticker_clean}_gru.pkl')
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"No saved GRU scaler found at {scaler_path}")
    scaler      = joblib.load(scaler_path)
    scaled_data = scaler.transform(df[features + ['log_return']].values)

    lookback = 21
    X, y     = [], []
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i - lookback:i])
        y.append(scaled_data[i, -1])
    X, y = np.array(X), np.array(y)

    model_path = os.path.join(TRAINED_MODELS_DIR, f'gru_{ticker_symbol.replace(".", "_")}.keras')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No trained GRU model found at {model_path}")
    model = tf.keras.models.load_model(model_path)

    # ── Residual std for prediction intervals ─────────────────────────────
    # Fixed dissertation values are used where available. If none exist,
    # residual std is computed from the most recent 10% of sequences and
    # inverse-transformed to real log-return space for interval construction.
    real_residual_std = GRU_RESIDUAL_STD.get(ticker_symbol, None)
    if real_residual_std is None:
        val_n                    = max(1, int(len(X) * 0.1))
        val_preds                = model.predict(X[-val_n:], verbose=0).flatten()
        residual_std             = max(np.std(y[-val_n:] - val_preds, ddof=1), 1e-6)
        residual_row_pos         = np.zeros((1, scaled_data.shape[1]))
        residual_row_neg         = np.zeros((1, scaled_data.shape[1]))
        residual_row_pos[0, -1]  = residual_std
        real_residual_std        = abs(scaler.inverse_transform(residual_row_pos)[0, -1] -
                                       scaler.inverse_transform(residual_row_neg)[0, -1])

    # ── Recursive 5-day forecast ──────────────────────────────────────────
    # Each predicted scaled log-return is inverse-transformed to real
    # log-return space before advancing the price. It is then re-scaled
    # before being fed back into the lookback window to prevent scale drift.
    # Log_Ret_Lag1 (index 9) and the target column (last index) are both
    # updated at each step to maintain the correct temporal structure.
    last_X            = scaled_data[-lookback:].copy().reshape(1, lookback, n_features)
    last_price        = float(df['Adj Close'].iloc[-1])
    forecast_log_rets = []

    for _ in range(5):
        pred_scaled               = model.predict(last_X, verbose=0)[0, 0]
        inverse_transform_row     = np.zeros((1, scaled_data.shape[1]))
        inverse_transform_row[0, -1] = pred_scaled
        real_log_ret              = scaler.inverse_transform(inverse_transform_row)[0, -1]
        forecast_log_rets.append(real_log_ret)

        forward_transform_row        = np.zeros((1, scaled_data.shape[1]))
        forward_transform_row[0, -1] = real_log_ret
        scaled_log_ret               = scaler.transform(forward_transform_row)[0, -1]

        next_row     = last_X[0, -1, :].copy()
        next_row[9]  = scaled_log_ret
        next_row[-1] = scaled_log_ret
        last_X       = np.concatenate(
            [last_X[:, 1:, :], next_row.reshape(1, 1, -1)], axis=1)

    cumulative_log_returns = np.cumsum(np.array(forecast_log_rets))
    horizons               = np.arange(1, 6)
    forecast_prices        = last_price * np.exp(cumulative_log_returns)
    ci_lower = last_price * np.exp(cumulative_log_returns - 1.96 * real_residual_std * np.sqrt(horizons))
    ci_upper = last_price * np.exp(cumulative_log_returns + 1.96 * real_residual_std * np.sqrt(horizons))

    return np.array(forecast_prices), pd.DataFrame({'lower': ci_lower, 'upper': ci_upper})


# ══════════════════════════════════════════════════════════════════════════
#   FORECAST SERVICE
#   Public interface for the presentation layer. Each method extracts the
#   data it needs from the raw DataFrame, calls the appropriate cached
#   function, and returns a (forecast, conf_int, error) tuple.
#   Errors are caught and returned as strings rather than raised so that
#   the UI can display a meaningful message without crashing the app.
# ══════════════════════════════════════════════════════════════════════════

class ForecastService:

    @staticmethod
    def run_arima(df: pd.DataFrame, ticker: str, today_str: str) -> Tuple[Optional[np.ndarray], Optional[pd.DataFrame], Optional[str]]:
        try:
            df_limited = df.loc[:'2026-02-28']
            prices = _extract_series(df_limited, 'Adj Close', ticker)
            if len(prices) == 0:
                prices = _extract_series(df_limited, 'Close', ticker)
            forecast, conf_int = _cached_arima(tuple(prices.values), today_str, ticker)
            return forecast, conf_int, None
        except Exception as e:
            return None, None, str(e)

    @staticmethod
    def run_random_forest(df: pd.DataFrame, ticker: str, today_str: str) -> Tuple[Optional[np.ndarray], Optional[pd.DataFrame], Optional[str]]:
        try:
            df_limited  = df.loc[:'2026-02-28']
            ohlcv       = _extract_ohlcv(df_limited, ticker)
            ohlcv_tuple = tuple(ohlcv.itertuples(index=False, name=None))
            forecast, conf_int = _cached_rf(ohlcv_tuple, today_str, ticker)
            return forecast, conf_int, None
        except Exception as e:
            return None, None, str(e)

    @staticmethod
    def run_gru(df: pd.DataFrame, ticker: str, today_str: str) -> Tuple[Optional[np.ndarray], Optional[pd.DataFrame], Optional[str]]:
        try:
            df_limited  = df.loc[:'2026-02-28']
            ohlcv       = _extract_ohlcv(df_limited, ticker)
            ohlcv_tuple = tuple(ohlcv.itertuples(index=False, name=None))
            forecast, conf_int = _cached_gru(ohlcv_tuple, today_str, ticker)
            return forecast, conf_int, None
        except Exception as e:
            return None, None, str(e)
