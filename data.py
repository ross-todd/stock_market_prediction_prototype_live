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

import yfinance as yf
import pandas as pd
import streamlit as st
from datetime import datetime, timedelta
from typing import Optional


# ══════════════════════════════════════════════════════════════════════════
#   STOCK DATA SERVICE
#   Single entry point for all data retrieval in the application.
#   Caching is applied at this layer to avoid repeated API calls across
#   reruns — the TTL is set to 1 hour to balance freshness with performance.
# ══════════════════════════════════════════════════════════════════════════

class StockDataService:

    @staticmethod
    @st.cache_data(show_spinner=False, ttl=3600)
    def get_stock_data(tickers: list, start_date_str: str, end_date_str: str,
                       range_selection: str = None) -> Optional[pd.DataFrame]:

        # ── Date window ───────────────────────────────────────────────────────
        # Fixed to the same 2021–2026 window used in the comparative analysis
        # so that the application forecasts are consistent with dissertation results.
        # A 5-day buffer is applied before the start date to ensure data availability
        # around weekends and UK bank holidays without losing valid trading days.
        try:
            start_date = datetime.strptime("2021-02-28", '%Y-%m-%d')
            end_date   = datetime.strptime("2026-02-28", '%Y-%m-%d')
            end_date   = end_date + timedelta(days=1)

            buffer_start_date = start_date - timedelta(days=5)

            downloaded_data = yf.download(
                tickers,
                start=buffer_start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                progress=False,
                auto_adjust=False
            )

            if downloaded_data.empty:
                return None

            # ── Cleaning pipeline ─────────────────────────────────────────────
            # Date column is converted to datetime and set as the index.
            # All price and volume columns are converted to numeric — any values
            # that cannot be parsed are replaced with NaN before gap-filling.
            # Forward-fill followed by back-fill is used to handle missing trading
            # days without introducing future prices into historical gaps.
            df = downloaded_data.copy()
            df = df.reset_index()
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df[df['Date'].notna()]
            df.set_index('Date', inplace=True)

            for col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

            df.ffill(inplace=True)
            df.bfill(inplace=True)

            return df

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
