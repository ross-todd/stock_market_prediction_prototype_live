#  Ross Todd
#  BSc (Hons) Software Development
#  Honours Project 2026 - A Stock Market Prediction Application Prototype
#
# data.py — Data Layer
# Handles all data fetching, caching, validation, and cleaning via yfinance
#
# All three models (ARIMA, RF, GRU) rely on this layer for raw OHLCV data.
# Data is fetched once, cached for one hour, and cleaned consistently so
# that any differences in model performance are attributable to the model
# class itself rather than inconsistent input data.

import os
import yfinance as yf
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from typing import Optional


# ── Saved data folder ─────────────────────────────────────────────────────────
# Resolved relative to this file so the app finds saved_data regardless of
# the working directory it is launched from.
SAVED_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "saved_data")

# ── Rolling 5-year window ─────────────────────────────────────────────────────
# DATA_END is always set to today's date. DATA_START is 5 years prior.
_now_uk    = datetime.now(ZoneInfo("Europe/London"))
_last_day = _now_uk.date()
DATA_END   = _last_day.strftime("%Y-%m-%d")
DATA_START = _last_day.replace(year=_last_day.year - 5).strftime("%Y-%m-%d")


# ══════════════════════════════════════════════════════════════════════════
#   INTERNAL CACHE LOADER
#   Attempts to load a single ticker from the saved_data CSV created by
#   the comparative analysis (data_loader.py). The filename format is:
#       BARC_L_20210228_20260228.csv
#   If the file does not exist, falls back to downloading from yfinance
#   and saves the result so future runs use the cached version.
# ══════════════════════════════════════════════════════════════════════════

def _load_single_ticker(ticker: str) -> Optional[pd.DataFrame]:
    ticker_clean = ticker.replace(".", "_")
    start_clean  = DATA_START.replace("-", "")
    end_clean    = DATA_END.replace("-", "")
    cache_path   = os.path.join(SAVED_DATA_DIR,
                                f"{ticker_clean}_{start_clean}_{end_clean}.csv")

    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        df = df.sort_index()
        return df

    # Cache miss — download from yfinance and save for future runs
    try:
        os.makedirs(SAVED_DATA_DIR, exist_ok=True)
        end_exclusive = (_last_day + timedelta(days=1)).strftime("%Y-%m-%d")
        raw = yf.download(ticker, start=DATA_START, end=end_exclusive,
                          progress=False, auto_adjust=False)

        if raw is None or raw.empty:
            return None

        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)

        if 'Adj Close' not in raw.columns and 'Close' in raw.columns:
            raw['Adj Close'] = raw['Close']

        raw = raw.asfreq('B').ffill().bfill()
        raw.to_csv(cache_path)
        return raw

    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════
#   STOCK DATA SERVICE
#   Single entry point for all data retrieval in the application.
#   Caching is applied at this layer to avoid repeated API calls across
#   reruns — the TTL is set to 1 hour to balance freshness with performance.
#
#   get_stock_data now builds the combined DataFrame from individual cached
#   CSV files rather than a single yfinance multi-ticker download, so the
#   data is guaranteed to match the comparative analysis exactly.
# ══════════════════════════════════════════════════════════════════════════

class StockDataService:

    @staticmethod
    @st.cache_data(show_spinner=False, ttl=3600)
    def get_stock_data(tickers: list, start_date_str: str, end_date_str: str,
                       range_selection: str = None) -> Optional[pd.DataFrame]:
        try:
            frames = {}

            for ticker in tickers:
                df = _load_single_ticker(ticker)
                if df is not None and not df.empty:
                    frames[ticker] = df

            if not frames:
                return None

            if len(frames) == 1:
                # Single ticker — return flat DataFrame matching original behaviour
                ticker = list(frames.keys())[0]
                result = frames[ticker].copy()
                for col in result.columns:
                    result[col] = pd.to_numeric(result[col], errors='coerce')
                result.ffill(inplace=True)
                result.bfill(inplace=True)
                return result

            # Multiple tickers — build MultiIndex DataFrame matching yfinance output
            # so the rest of the app (models.py _extract_series) works unchanged.
            all_cols = set()
            for df in frames.values():
                all_cols.update(df.columns.tolist())

            combined_index = frames[list(frames.keys())[0]].index
            for df in frames.values():
                combined_index = combined_index.union(df.index)

            multi_frames = {}
            for col in all_cols:
                col_data = {}
                for ticker, df in frames.items():
                    if col in df.columns:
                        col_data[ticker] = df[col].reindex(combined_index)
                if col_data:
                    multi_frames[col] = pd.DataFrame(col_data)

            result = pd.concat(multi_frames, axis=1)
            result.columns = pd.MultiIndex.from_tuples(
                [(col, ticker) for col in multi_frames for ticker in multi_frames[col].columns]
            )
            result = result.sort_index()

            for col in result.columns:
                result[col] = pd.to_numeric(result[col], errors='coerce')

            result.ffill(inplace=True)
            result.bfill(inplace=True)

            return result

        except Exception:
            return None

    # ── Close price extraction ────────────────────────────────────────────────
    # Handles both MultiIndex DataFrames (multiple tickers) and flat DataFrames
    # (single ticker) returned by yfinance, which changes column structure
    # depending on whether one or multiple tickers are requested.
    @staticmethod
    def extract_close_prices(df: pd.DataFrame) -> Optional[pd.DataFrame]:
        if df is None:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            return df['Close'].copy()
        close_cols = [col for col in df.columns if 'close' in col.lower()]
        return df[close_cols].copy() if close_cols else None
