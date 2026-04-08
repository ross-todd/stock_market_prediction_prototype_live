#  Ross Todd
#  BSc (Hons) Software Development
#  Honours Project 2026 - A Stock Market Prediction Application Prototype

# models.py — Application Logic Layer
# Contains ARIMA, Random Forest, and GRU forecasting models

# Model configurations match the best-performing settings identified
# during the comparative analysis:
#   ARIMA : ticker-specific config — see ARIMA_BEST_CONFIG below
#             BARC.L: order=(0,0,1), trend='c', window=126
#             LLOY.L: order=(2,0,1), trend='c', window=252
#             HSBA.L: order=(1,0,1), trend='c', window=21
#   RF    : n_estimators=300, max_depth=8, 15 engineered features (pre-trained .pkl)
#             BARC.L: n_est=300, depth=8  |  LLOY.L: n_est=1200, depth=None
#             HSBA.L: n_est=300, depth=8
#   GRU   : lookback=21, units=64 — ticker-specific layers/dropout/lr (pre-trained .keras)
#             BARC.L: layers=2, dropout=0.3, lr=0.0005
#             LLOY.L: layers=2, dropout=0.2, lr=0.001
#             HSBA.L: layers=1, dropout=0.5, lr=0.001

# All three methods load data from the saved_data CSV cache created by the
# comparative analysis, guaranteeing identical input to the dissertation.
# If a cache file is missing, data_loader.py downloads and saves it.

import pandas as pd
import numpy as np
import streamlit as st
from typing import Tuple, Optional
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from data import _load_single_ticker

import joblib
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from statsmodels.tsa.arima.model import ARIMA
import tensorflow as tf


# ── Pre-trained model directory (GRU .keras files and GRU scaler) ─────────────
TRAINED_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trained_models')

# ── Saved data directory ──────────────────────────────────────────────────────
SAVED_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_data")

# ── Rolling 5-year window, with UK 17:00 market close cutoff ─────────────────
_now_uk    = datetime.now(ZoneInfo("Europe/London"))
_last_day  = _now_uk.date() if _now_uk.hour >= 17 else _now_uk.date() - timedelta(days=1)
DATA_END   = _last_day.strftime("%Y-%m-%d")
DATA_START = _last_day.replace(year=_last_day.year - 5).strftime("%Y-%m-%d")

# ── Forecast weights (50/20/10/10/10) ─────────────────────────────────────────
# Matches day_weights in rf_analysis.py / arima_analysis.py / gru_analysis.py
DAY_WEIGHTS = np.array([0.5, 0.2, 0.1, 0.1, 0.1])

# ── Lag list — matches LAGS in comparative analysis ───────────────────────────
LAGS = [1, 2, 3, 5]

# ── Walk-forward refit frequency — matches WALK_REFIT_FREQ ────────────────────
WALK_REFIT_FREQ = 63

# ── Train ratio — matches TRAIN_RATIO ─────────────────────────────────────────
TRAIN_RATIO = 0.80

# ── Dissertation residual std values (log-return space) ───────────────────────
# Fixed from validation-set results in the comparative analysis so the app
# produces identical prediction intervals to the dissertation.
RF_RESIDUAL_STD = {
    'BARC.L': 0.018373,  # was 0.0196572589
    'LLOY.L': 0.015562, 
    'HSBA.L': 0.012087, 
}
GRU_RESIDUAL_STD = {
    'BARC.L': 0.0181549697,
    'LLOY.L': 0.0156723814,
    'HSBA.L': 0.0122015238,
}

# ── Best ARIMA configs per ticker from comparative analysis grid search ────────
ARIMA_BEST_CONFIG = {
    'BARC.L': {'order': (0, 0, 1), 'trend': 'c', 'window': 126, 'end_idx': 1295},
    'LLOY.L': {'order': (2, 0, 1), 'trend': 'c', 'window': 252, 'end_idx': 1295},
    'HSBA.L': {'order': (1, 0, 1), 'trend': 'c', 'window': 21,  'end_idx': 1295},
}


# ══════════════════════════════════════════════════════════════════════════
#   SAVED DATA LOADER
# ══════════════════════════════════════════════════════════════════════════

def _load_ticker_data(ticker: str) -> pd.DataFrame:
    ticker_clean = ticker.replace(".", "_")
    start_clean  = DATA_START.replace("-", "")
    end_clean    = DATA_END.replace("-", "")
    cache_path   = os.path.join(SAVED_DATA_DIR,
                                f"{ticker_clean}_{start_clean}_{end_clean}.csv")

    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        return df.sort_index()

    return _load_single_ticker(ticker)


# ══════════════════════════════════════════════════════════════════════════
#   SHARED HELPERS
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
    """Weighted average of 5-day forecast. Matches np.sum(forecast_prices * day_weights)
    in the comparative analysis."""
    return float(np.sum(np.array(forecast) * DAY_WEIGHTS))


# ══════════════════════════════════════════════════════════════════════════
#   ARIMA — ticker-specific config from comparative analysis grid search
# ══════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def _cached_arima(
    prices_tuple: tuple,
    today_str: str,
    ticker_symbol: str
) -> Tuple[np.ndarray, pd.DataFrame]:

    config      = ARIMA_BEST_CONFIG[ticker_symbol]
    best_order  = config['order']
    best_trend  = config['trend']
    best_window = config['window']
    end_idx     = config['end_idx']

    prices      = pd.to_numeric(pd.Series(prices_tuple), errors='coerce').dropna()
    log_returns = np.log(prices / prices.shift(1)).dropna()

    subset    = log_returns.iloc[end_idx - best_window : end_idx].values
    model_fit = ARIMA(subset, order=best_order, trend=best_trend).fit()

    forecast_obj  = model_fit.get_forecast(steps=5)
    forecast_rets = np.asarray(forecast_obj.predicted_mean)
    conf_int_ret  = forecast_obj.conf_int()

    last_price          = float(prices.iloc[-1])
    cumulative_log_rets = np.cumsum(forecast_rets)
    horizons            = np.arange(1, 6)

    conf_int_array  = np.array(conf_int_ret)
    sigma_one_step  = (conf_int_array[0, 1] - forecast_rets[0]) / 1.96
    sigma_one_step  = max(abs(sigma_one_step), float(np.std(subset)))
    forecast_prices = last_price * np.exp(cumulative_log_rets)
    ci_lower        = last_price * np.exp(cumulative_log_rets - 1.96 * sigma_one_step * np.sqrt(horizons))
    ci_upper        = last_price * np.exp(cumulative_log_rets + 1.96 * sigma_one_step * np.sqrt(horizons))

    return np.array(forecast_prices), pd.DataFrame({'lower': ci_lower, 'upper': ci_upper})


# ══════════════════════════════════════════════════════════════════════════
#   RANDOM FOREST — feature engineering matches create_enhanced_features()
#   in rf_analysis.py exactly, including column names and order.
# ══════════════════════════════════════════════════════════════════════════

def create_enhanced_features(df: pd.DataFrame, price_col: str = 'Adj Close') -> pd.DataFrame:
    """
    Mirrors create_enhanced_features() in rf_analysis.py exactly.
    Builds 15 technical indicators used as RF model inputs.
    Variable names, formula order, and rolling windows are identical
    to the comparative analysis to guarantee the same feature values
    reach the pre-trained model and saved scaler.
    """
    df = df.copy()
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna().astype(float)

    df['Log_Ret'] = np.log(df[price_col] / df[price_col].shift(1))
    for lag in LAGS:
        df[f'Ret_Lag_{lag}'] = df['Log_Ret'].shift(lag)

    df['SMA_5']     = df[price_col].rolling(5).mean()
    df['SMA_20']    = df[price_col].rolling(20).mean()
    df['SMA_Ratio'] = df['SMA_5'] / df['SMA_20']

    delta         = df[price_col].diff()
    gain          = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss          = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs            = gain / (loss + 1e-9)
    df['RSI']     = 100 - (100 / (1 + rs))

    sma            = df[price_col].rolling(20).mean()
    std            = df[price_col].rolling(20).std()
    df['BB_Width'] = ((sma + 2 * std) - (sma - 2 * std)) / (sma + 1e-9)

    df['Volatility']   = df['Log_Ret'].rolling(20).std()
    df['Volume_MA']    = df['Volume'].rolling(20).mean()
    df['Volume_Ratio'] = df['Volume'] / (df['Volume_MA'] + 1e-9)

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


def get_prediction_intervals(
    rf_model,
    X: np.ndarray,
    residual_std: float,
    horizon: int = 1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Mirrors get_prediction_intervals() in rf_analysis.py.
    Returns (point_pred, lower_bound, upper_bound) for a given horizon.
    """
    point_pred  = rf_model.predict(X)
    scaled_std  = residual_std * np.sqrt(horizon)
    lower_bound = point_pred - 1.96 * scaled_std
    upper_bound = point_pred + 1.96 * scaled_std
    return point_pred, lower_bound, upper_bound


def _load_rf(ticker_symbol: str):
    """Loads pre-trained RF model from trained_models/ (output of rf_analysis.py)."""
    path = os.path.join(TRAINED_MODELS_DIR, f'rf_{ticker_symbol.replace(".", "_")}.pkl')
    if not os.path.exists(path):
        raise FileNotFoundError(f"No trained RF model found at {path}")
    return joblib.load(path)


def _load_rf_scaler(ticker_symbol: str):
    """Loads the RF scaler from trained_models/ saved by rf_analysis.py."""
    ticker_clean = ticker_symbol.replace(".", "_")
    path = os.path.join(TRAINED_MODELS_DIR, f'scaler_{ticker_clean}_rf.pkl')
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No saved RF scaler found at {path}. "
            f"Re-run rf_analysis.py to regenerate scaler_{ticker_clean}_rf.pkl."
        )
    return joblib.load(path)

@st.cache_data(show_spinner=False)
def _cached_rf(ohlcv_tuple: tuple, today_str: str, ticker_symbol: str) -> Tuple[np.ndarray, pd.DataFrame]:

    ohlcv = pd.DataFrame(
        list(ohlcv_tuple),
        columns=['Adj Close', 'Open', 'High', 'Low', 'Volume']
    )

    price_col = 'Adj Close'
    df_feat   = create_enhanced_features(ohlcv, price_col=price_col)

    # Hardcoded column order confirmed by inspecting saved scaler means:
    #   [0] ~4.97e7  → Volume
    #   [1] ~0.00052 → Ret_Lag_1  ... etc.
    # Must match the order scaler.fit_transform(X_train) was called with
    # in rf_analysis.py exactly.
    feature_columns = [
        'Volume',
        'Ret_Lag_1', 'Ret_Lag_2', 'Ret_Lag_3', 'Ret_Lag_5',
        'SMA_5', 'SMA_20', 'SMA_Ratio',
        'RSI',
        'BB_Width',
        'Volatility',
        'Volume_MA', 'Volume_Ratio',
        'ATR_14', 'ATR_20'
    ]

    # Load pre-trained model and scaler from saved_models/
    current_model        = _load_rf(ticker_symbol)
    scaler               = _load_rf_scaler(ticker_symbol)
    current_residual_std = RF_RESIDUAL_STD.get(ticker_symbol)

    if current_residual_std is None:
        X_all          = df_feat[feature_columns].values[:-1]
        y_all          = df_feat['Log_Ret'].shift(-1).dropna().values
        split_idx      = int(len(X_all) * TRAIN_RATIO)
        X_train_scaled = scaler.transform(X_all[:split_idx])
        val_start      = int(len(X_train_scaled) * 0.8)
        val_preds      = current_model.predict(X_train_scaled[val_start:])
        current_residual_std = max(
            np.std(y_all[split_idx:][:len(val_preds)] - val_preds, ddof=1), 1e-6
        )

    # ── Recursive 5-day forecast ──────────────────────────────────────────
    last_price         = float(df_feat[price_col].iloc[-1])
    last_date          = df_feat.index[-1]                     # noqa: F841
    current_features   = df_feat[feature_columns].iloc[-1].copy()
    current_price      = last_price

    running_rsi_approx = float(df_feat['RSI'].iloc[-1])
    running_sq_ret     = float(current_features.get('Volatility', 0)) ** 2
    running_true_range = float(current_features.get('ATR_14', 0))

    forecast_log_rets  = []

    for day in range(5):
        current_features_scaled = scaler.transform(current_features.values.reshape(1, -1))

        pred_logret, _, _ = get_prediction_intervals(
            current_model, current_features_scaled, current_residual_std, horizon=1
        )
        pred_logret = pred_logret[0]
        forecast_log_rets.append(pred_logret)

        pred_price = current_price * np.exp(pred_logret)

        # Lag shift — reversed LAGS[1:] = [5, 3, 2]
        for lag in reversed(LAGS[1:]):
            if f'Ret_Lag_{lag}' in current_features.index:
                prev_lag = f'Ret_Lag_{lag - 1}' if lag > 1 else None
                current_features[f'Ret_Lag_{lag}'] = (
                    current_features[prev_lag]
                    if prev_lag and prev_lag in current_features.index
                    else 0
                )
        if 'Ret_Lag_1' in current_features.index:
            current_features['Ret_Lag_1'] = pred_logret

        if 'SMA_5' in current_features.index:
            current_features['SMA_5']  = (current_features['SMA_5'] * 4 + pred_price) / 5
        if 'SMA_20' in current_features.index:
            current_features['SMA_20'] = (current_features['SMA_20'] * 19 + pred_price) / 20
        if 'SMA_Ratio' in current_features.index:
            current_features['SMA_Ratio'] = (
                current_features['SMA_5'] / (current_features['SMA_20'] + 1e-9)
            )

        if 'Volatility' in current_features.index:
            alpha_vol      = 1.0 / 20
            running_sq_ret = (1 - alpha_vol) * running_sq_ret + alpha_vol * pred_logret ** 2
            current_features['Volatility'] = np.sqrt(max(running_sq_ret, 1e-10))

        if 'BB_Width' in current_features.index:
            std_approx = current_features['Volatility'] * current_features['SMA_20']
            current_features['BB_Width'] = (4 * std_approx) / (current_features['SMA_20'] + 1e-9)

        if 'ATR_14' in current_features.index:
            daily_range_proxy  = pred_price * current_features['Volatility']
            running_true_range = (running_true_range * 13 + daily_range_proxy) / 14
            current_features['ATR_14'] = running_true_range
        if 'ATR_20' in current_features.index:
            current_features['ATR_20'] = (
                (current_features['ATR_20'] * 19 + running_true_range) / 20
            )

        if 'RSI' in current_features.index:
            alpha_rsi = 1.0 / 14
            if pred_logret > 0:
                running_rsi_approx = (
                    (1 - alpha_rsi) * current_features['RSI'] + alpha_rsi * 100
                )
            else:
                running_rsi_approx = (1 - alpha_rsi) * current_features['RSI']
            current_features['RSI'] = np.clip(running_rsi_approx, 0, 100)

        current_price = pred_price

    # ── Convert log returns to prices and build 95% PI ────────────────────
    forecast_log_rets   = np.array(forecast_log_rets)
    cumulative_log_rets = np.cumsum(forecast_log_rets)
    horizons            = np.arange(1, 6)
    forecast_prices     = last_price * np.exp(cumulative_log_rets)
    ci_lower = last_price * np.exp(
        cumulative_log_rets - 1.96 * current_residual_std * np.sqrt(horizons)
    )
    ci_upper = last_price * np.exp(
        cumulative_log_rets + 1.96 * current_residual_std * np.sqrt(horizons)
    )

    return np.array(forecast_prices), pd.DataFrame({'lower': ci_lower, 'upper': ci_upper})


# ══════════════════════════════════════════════════════════════════════════
#   GRU — lookback=21, units=64, ticker-specific layers/dropout/lr
# ══════════════════════════════════════════════════════════════════════════

def _build_gru_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Mirrors the feature engineering block in gru_analysis.py."""
    df = ohlcv.copy()
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna().astype(float)

    delta         = df['Adj Close'].diff()
    gain          = delta.where(delta > 0, 0).rolling(14).mean()
    loss          = (-delta.where(delta < 0, 0)).rolling(14).mean()
    df['RSI']     = 100 - (100 / (1 + gain / (loss + 1e-9)))

    df['MACD']        = (
        df['Adj Close'].ewm(span=12).mean() -
        df['Adj Close'].ewm(span=26).mean()
    )
    df['Signal_Line'] = df['MACD'].ewm(span=9).mean()

    ma20             = df['Adj Close'].rolling(20).mean()
    std20            = df['Adj Close'].rolling(20).std()
    df['Upper_Band'] = ma20 + 2 * std20
    df['Lower_Band'] = ma20 - 2 * std20
    df               = df.dropna()

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

    ohlcv = pd.DataFrame(
        list(ohlcv_tuple),
        columns=['Adj Close', 'Open', 'High', 'Low', 'Volume']
    )

    df       = _build_gru_features(ohlcv)
    features = [
        'Open', 'High', 'Low', 'Volume', 'RSI', 'MACD',
        'Signal_Line', 'Upper_Band', 'Lower_Band', 'Log_Ret_Lag1'
    ]

    n_features = len(features) + 1

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

    model_path = os.path.join(
        TRAINED_MODELS_DIR, f'gru_{ticker_symbol.replace(".", "_")}.keras'
    )
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"No trained GRU model found at {model_path}")
    model = tf.keras.models.load_model(model_path)

    real_residual_std = GRU_RESIDUAL_STD.get(ticker_symbol)
    if real_residual_std is None:
        val_n                   = max(1, int(len(X) * 0.1))
        val_preds               = model.predict(X[-val_n:], verbose=0).flatten()
        residual_std_scaled     = max(np.std(y[-val_n:] - val_preds, ddof=1), 1e-6)
        residual_row_pos        = np.zeros((1, scaled_data.shape[1]))
        residual_row_neg        = np.zeros((1, scaled_data.shape[1]))
        residual_row_pos[0, -1] = residual_std_scaled
        real_residual_std       = abs(
            scaler.inverse_transform(residual_row_pos)[0, -1] -
            scaler.inverse_transform(residual_row_neg)[0, -1]
        )

    last_X            = scaled_data[-lookback:].copy().reshape(1, lookback, n_features)
    last_price        = float(df['Adj Close'].iloc[-1])
    forecast_log_rets = []

    for _ in range(5):
        pred_scaled        = model.predict(last_X, verbose=0)[0, 0]
        inverse_row        = np.zeros((1, scaled_data.shape[1]))
        inverse_row[0, -1] = pred_scaled
        real_log_ret       = scaler.inverse_transform(inverse_row)[0, -1]
        forecast_log_rets.append(real_log_ret)

        forward_row        = np.zeros((1, scaled_data.shape[1]))
        forward_row[0, -1] = real_log_ret
        scaled_log_ret     = scaler.transform(forward_row)[0, -1]

        next_row     = last_X[0, -1, :].copy()
        next_row[9]  = scaled_log_ret
        next_row[-1] = scaled_log_ret
        last_X = np.concatenate(
            [last_X[:, 1:, :], next_row.reshape(1, 1, -1)], axis=1
        )

    cumulative_log_rets = np.cumsum(np.array(forecast_log_rets))
    horizons            = np.arange(1, 6)
    forecast_prices     = last_price * np.exp(cumulative_log_rets)
    ci_lower = last_price * np.exp(
        cumulative_log_rets - 1.96 * real_residual_std * np.sqrt(horizons)
    )
    ci_upper = last_price * np.exp(
        cumulative_log_rets + 1.96 * real_residual_std * np.sqrt(horizons)
    )

    return np.array(forecast_prices), pd.DataFrame({'lower': ci_lower, 'upper': ci_upper})


# ══════════════════════════════════════════════════════════════════════════
#   FORECAST SERVICE
# ══════════════════════════════════════════════════════════════════════════

class ForecastService:

    @staticmethod
    def run_arima(
        df: pd.DataFrame,
        ticker: str,
        today_str: str
    ) -> Tuple[Optional[np.ndarray], Optional[pd.DataFrame], Optional[str]]:
        try:
            saved         = _load_ticker_data(ticker)
            prices        = saved['Adj Close'].dropna()
            forecast_prices, conf_int = _cached_arima(
                tuple(prices.values), today_str, ticker
            )
            return forecast_prices, conf_int, None
        except Exception as e:
            return None, None, str(e)

    @staticmethod
    def run_random_forest(
        df: pd.DataFrame,
        ticker: str,
        today_str: str
    ) -> Tuple[Optional[np.ndarray], Optional[pd.DataFrame], Optional[str]]:
        try:
            saved       = _load_ticker_data(ticker)
            ohlcv       = saved[['Adj Close', 'Open', 'High', 'Low', 'Volume']].dropna()
            ohlcv_tuple = tuple(ohlcv.itertuples(index=False, name=None))
            forecast_prices, conf_int = _cached_rf(ohlcv_tuple, today_str, ticker)
            return forecast_prices, conf_int, None
        except Exception as e:
            return None, None, str(e)

    @staticmethod
    def run_gru(
        df: pd.DataFrame,
        ticker: str,
        today_str: str
    ) -> Tuple[Optional[np.ndarray], Optional[pd.DataFrame], Optional[str]]:
        try:
            saved       = _load_ticker_data(ticker)
            ohlcv       = saved[['Adj Close', 'Open', 'High', 'Low', 'Volume']].dropna()
            ohlcv_tuple = tuple(ohlcv.itertuples(index=False, name=None))
            forecast_prices, conf_int = _cached_gru(ohlcv_tuple, today_str, ticker)
            return forecast_prices, conf_int, None
        except Exception as e:
            return None, None, str(e)
