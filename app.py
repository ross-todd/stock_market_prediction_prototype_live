#  Ross Todd
#  BSc (Hons) Software Development
#  Honours Project 2026 - A Stock Market Prediction Application Prototype
#
# app.py — Presentation Layer
# Full app: home screen, predictions screen, sidebar routing
#
# This file is the single entry point for the Streamlit application.
# It handles all user interaction, screen routing, and visualisation.
# The presentation layer is deliberately kept separate from data fetching
# (data.py) and model logic (models.py) so that changes to the UI do not
# require touching forecasting code, and vice versa.

import time

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import sys
import os

import models
sys.path.append(os.path.dirname(__file__))

from utils import (
    TICKERS, TICKER_LIST, TICKER_NAMES, COMPANY_OPTIONS,
    BANK_COLOR_MAP, MODEL_COLOR_MAP, get_start_date_from_range,
    GLOBAL_CSS, DATE_RANGE_BUTTONS_CSS, PREDICTIONS_CSS
)
from data import StockDataService, DATA_START
from models import ForecastService


# ── Page configuration ────────────────────────────────────────────────────────
# Wide layout is used to give the charts and tables as much horizontal space
# as possible. The sidebar is expanded by default so users see company and
# date controls immediately on load.

st.set_page_config(
    page_title="Stock Market Prediction - UK Banking Sector",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ── Session state initialisation ─────────────────────────────────────────────────────────
# All persistent state is stored in st.session_state so that user selections
# survive reruns triggered by button clicks or widget changes.
# Default view is the home screen with a one-month date range pre-selected for Barclays plc.

if 'start_date' not in st.session_state:
    st.session_state['start_date'] = get_start_date_from_range('1M').date()
if 'end_date' not in st.session_state:
    st.session_state['end_date'] = datetime.now().date()
if 'active_range' not in st.session_state:
    st.session_state['active_range'] = '1M'
if 'current_view' not in st.session_state:
    st.session_state['current_view'] = 'main'
if 'selected_company' not in st.session_state:
    st.session_state['selected_company'] = (
        "Barclays plc" if "Barclays plc" in COMPANY_OPTIONS else COMPANY_OPTIONS[1]
    )


st.markdown(GLOBAL_CSS, unsafe_allow_html=True)


# ── Scroll to top helper ──────────────────────────────────────────────────────
# Injected as a zero-height HTML component so it has no visible effect.
# Called after screen transitions to prevent the user landing mid-page.

def scroll_to_top():
    components.html(
        "<script>window.parent.document.querySelector('.main').scrollTo(0, 0);</script>",
        height=0, width=0
    )


# ── Shared disclaimer helpers ─────────────────────────────────────────────────
# Module-level functions so both HomeScreen and PredictionsScreen can call
# the same rendering logic without duplicating st.warning calls.

def render_data_disclaimer():
    st.warning(
        "**Data Source:** Stock price data is sourced from "
        "[Yahoo Finance](https://finance.yahoo.com) via the yfinance library. "
        "Data is for informational purposes only and may be subject to delays."
    )

def render_model_disclaimer():
    st.warning(
        "**Disclaimer:** This model is for educational purposes only. "
        "Stock prices are influenced by many factors and can be unpredictable. "
        "Seek professional financial advice before investing."
    )


# ══════════════════════════════════════════════════════════════════════════
#   HOME SCREEN
#   Displays the date range selector, interactive price chart, and
#   historical OHLCV table for the selected company.
#   All company view normalises prices to percentage change from the
#   start of the selected period so banks at different price levels
#   can be meaningfully compared on the same axis.
# ══════════════════════════════════════════════════════════════════════════

class HomeScreen:

    def __init__(self, selected_company: str):
        self.selected_company = selected_company

    def render(self):
        st.markdown(DATE_RANGE_BUTTONS_CSS, unsafe_allow_html=True)
        self._render_range_buttons()

        # ── Spinner guard ─────────────────────────────────────────────────
        # The spinner is only shown when data is not yet cached for the
        # current date range. On subsequent reruns within the 1-hour TTL
        # the call returns instantly from cache so no spinner is displayed.
        # The cache key changes when the date range changes, ensuring the
        # spinner reappears whenever a genuinely new fetch is required.
        home_cache_key = f"home_loaded_{st.session_state['start_date']}_{st.session_state['end_date']}"
        if home_cache_key not in st.session_state:
            with st.spinner("Loading market data..."):
                df = StockDataService.get_stock_data(
                    TICKER_LIST,
                    st.session_state['start_date'].strftime('%Y-%m-%d'),
                    st.session_state['end_date'].strftime('%Y-%m-%d')
                )
            st.session_state[home_cache_key] = True
        else:
            df = StockDataService.get_stock_data(
                TICKER_LIST,
                st.session_state['start_date'].strftime('%Y-%m-%d'),
                st.session_state['end_date'].strftime('%Y-%m-%d')
            )

        if df is None:
            return
        if df.empty:
            st.warning("No data available for the selected date range.")
            return

        if isinstance(df.columns, pd.MultiIndex) and 'Close' in df.columns.get_level_values(0):
            close_prices_df = df['Close'].copy()
        else:
            close_prices_df = df

        current_info = self._get_current_info(close_prices_df)
        self._render_chart(close_prices_df, current_info)
        self._render_data_table()
        self._render_data_disclaimer()

    def _get_current_info(self, close_prices_df: pd.DataFrame) -> str:
        # ── Current price extraction ──────────────────────────────────────
        # Price and date are extracted directly from the DataFrame rather
        # than making a separate API call, which avoids extra latency and
        # keeps the title consistent with the data shown in the chart.
        if self.selected_company == "All Companies":
            return "Multiple companies selected"
        ticker = TICKERS.get(self.selected_company)
        if not ticker:
            return "Company not found"
        try:
            if close_prices_df is not None and not close_prices_df.empty and ticker in close_prices_df.columns:
                close_prices = close_prices_df[ticker].dropna()
                if not close_prices.empty:
                    price_str = f"{close_prices.iloc[-1]:.2f}p"
                    date_str  = close_prices.index[-1].strftime('%d %b %Y')
                    return f"Current: {price_str} ({date_str})"
        except Exception as e:
            st.warning(f"Could not fetch current price: {e}")
        return "Current price unavailable"

    def _render_range_buttons(self):
        # ── Date range buttons ────────────────────────────────────────────
        # Seven preset ranges are offered via st.button rather than a dropdown
        # to reduce the number of interactions required. The active button is
        # styled as primary so the current selection is always visible.
        # Selecting a preset updates session state and triggers a rerun so
        # the chart refreshes immediately without a manual submit step.
        range_labels = ['1D', '1W', '1M', '3M', '6M', '1Y', '5Y']
        with st.container():
            cols = st.columns(len(range_labels))
            for col, label in zip(cols, range_labels):
                with col:
                    btn_type = "primary" if st.session_state['active_range'] == label else "secondary"
                    if st.button(label, key=f"range_{label}", type=btn_type, use_container_width=True):
                        st.session_state['active_range'] = label
                        st.session_state['start_date']   = get_start_date_from_range(label).date()
                        st.session_state['end_date']     = datetime.now().date()
                        st.rerun()

    def _render_chart(self, close_prices_df, current_info):
        if close_prices_df is None or close_prices_df.empty:
            st.error("No data available to display.")
            return

        # ── Timezone normalisation ────────────────────────────────────────
        # yfinance occasionally returns timezone-aware DatetimeIndex values.
        # These are stripped to tz-naive before filtering to avoid comparison
        # errors between aware and naive timestamps.
        start = pd.Timestamp(st.session_state['start_date']).tz_localize(None)
        end   = pd.Timestamp(st.session_state['end_date']).tz_localize(None)
        if close_prices_df.index.tz is not None:
            close_prices_df.index = close_prices_df.index.tz_localize(None)
        filtered_prices_df = close_prices_df.loc[start:end]

        if filtered_prices_df.empty:
            st.warning("No data available for the selected date range.")
            return

        chart_title = (
            ", ".join(TICKERS.keys())
            if self.selected_company == "All Companies"
            else self.selected_company
        )
        fig = go.Figure()

        if self.selected_company == "All Companies":
            # ── Percentage change normalisation ───────────────────────────
            # Raw prices cannot be compared on the same axis because HSBC
            # trades at 1393p while Lloyds trades at 102p. Normalising to
            # percentage change from the first valid price in the selected
            # window makes the relative performance of each bank visible.
            normalised_prices_df = filtered_prices_df.copy()
            for ticker in TICKERS.values():
                if ticker in normalised_prices_df.columns:
                    valid_prices = normalised_prices_df[ticker].dropna()
                    if not valid_prices.empty:
                        base_price = valid_prices.iloc[0]
                        if base_price != 0:
                            normalised_prices_df[ticker] = ((normalised_prices_df[ticker] / base_price) - 1) * 100
                        else:
                            normalised_prices_df[ticker] = 0

            for name, ticker in TICKERS.items():
                if ticker in normalised_prices_df.columns:
                    fig.add_trace(go.Scatter(
                        x=normalised_prices_df.index, y=normalised_prices_df[ticker],
                        name=name, mode='lines',
                        line=dict(color=BANK_COLOR_MAP[ticker], width=2.5),
                        text=filtered_prices_df[ticker],
                        hovertemplate='%{text:.2f} p<br>%{x|%Y-%m-%d}<extra></extra>'
                    ))

            fig.update_layout(
                title=f'Performance Comparison - {chart_title}  •  {current_info}',
                title_y=0.96, title_font=dict(size=22),
                yaxis_title='Percentage Change from Start (%)',
                margin=dict(t=80),
                xaxis=dict(title='Date', title_font=dict(weight='bold', size=14),
                           tickfont=dict(weight='bold'), zeroline=True,
                           zerolinecolor='lightgray', zerolinewidth=1),
                yaxis=dict(title_font=dict(weight='bold', size=14),
                           tickfont=dict(weight='bold'), zeroline=True,
                           zerolinecolor='lightgray', zerolinewidth=1),
                hovermode="x unified", height=670,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )
        else:
            ticker = TICKERS[self.selected_company]
            col    = ticker if ticker in filtered_prices_df.columns else 'Close'
            if col in filtered_prices_df.columns:
                fig.add_trace(go.Scatter(
                    x=filtered_prices_df.index, y=filtered_prices_df[col],
                    name=self.selected_company, mode='lines',
                    line=dict(color=BANK_COLOR_MAP[ticker], width=3)
                ))

            fig.update_layout(
                title=f'Historical Data - {chart_title}  •  {current_info}',
                title_y=0.96, title_font=dict(size=22),
                margin=dict(t=50),
                xaxis=dict(title='Date', title_font=dict(weight='bold', size=14),
                           tickfont=dict(weight='bold'), zeroline=True,
                           zerolinecolor='lightgray', zerolinewidth=1),
                yaxis=dict(title='Closing Price (p)', title_font=dict(weight='bold', size=14),
                           tickfont=dict(weight='bold'), zeroline=True,
                           zerolinecolor='lightgray', zerolinewidth=1),
                hovermode="x unified", height=670,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
            )

        st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})

    def _render_data_table(self):
        # ── OHLCV table ───────────────────────────────────────────────────
        # Only rendered for single-company selection. All Companies view
        # would produce a table too wide to be readable, and the chart
        # already shows all the relevant comparison information.
        # Data is re-fetched for the selected ticker only to keep the table
        # independent of the multi-ticker DataFrame used by the chart.
        if self.selected_company == "All Companies":
            return
        try:
            ticker    = TICKERS[self.selected_company]
            start_str = st.session_state['start_date'].strftime('%Y-%m-%d')
            end_str   = st.session_state['end_date'].strftime('%Y-%m-%d')
            range_lbl = st.session_state.get('active_range', 'Custom')
            if range_lbl == 'Custom':
                range_lbl = f"{start_str} → {end_str}"

            # ── Spinner guard ─────────────────────────────────────────────
            # Same pattern as the chart fetch above — spinner only shown on
            # the first load for this ticker and date range combination.
            table_cache_key = f"table_loaded_{ticker}_{start_str}_{end_str}"
            if table_cache_key not in st.session_state:
                with st.spinner("Loading historical data..."):
                    historical_data_df = StockDataService.get_stock_data([ticker], start_str, end_str)
                st.session_state[table_cache_key] = True
            else:
                historical_data_df = StockDataService.get_stock_data([ticker], start_str, end_str)

            if historical_data_df is None or historical_data_df.empty:
                st.info("No data available for the selected period.")
                return

            st.markdown("---")
            st.markdown(
                f"<p style='font-size:16px;font-weight:bold;'>"
                f"Historical Daily Data – {self.selected_company} ({range_lbl})</p>",
                unsafe_allow_html=True
            )

            table_data_df = historical_data_df.copy().dropna()
            if isinstance(table_data_df.columns, pd.MultiIndex):
                table_data_df.columns = table_data_df.columns.get_level_values(0)
            table_data_df = table_data_df.reset_index()
            table_data_df['Date'] = pd.to_datetime(table_data_df['Date'])

            desired_cols  = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
            table_data_df = table_data_df[[c for c in desired_cols if c in table_data_df.columns]]
            table_data_df = table_data_df.sort_values(by='Date', ascending=False)

            start_dt = pd.to_datetime(start_str)
            end_dt   = pd.to_datetime(end_str) + pd.Timedelta(days=1)
            table_data_df = table_data_df[
                (table_data_df['Date'] >= start_dt) & (table_data_df['Date'] < end_dt)
            ].copy()

            table_data_df['Date'] = table_data_df['Date'].dt.strftime('%Y-%m-%d')
            for col in ['Open', 'High', 'Low', 'Close', 'Adj Close']:
                if col in table_data_df.columns:
                    table_data_df[col] = table_data_df[col].apply(
                        lambda x: f'{x:,.2f}' if pd.notna(x) else ''
                    )

            st.dataframe(table_data_df, width='stretch')
            st.markdown("<br>", unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Could not load historical data table: {e}")

    def _render_data_disclaimer(self):
        render_data_disclaimer()


# ══════════════════════════════════════════════════════════════════════════
#   PREDICTIONS SCREEN
#   Handles model selection routing and generates 5-day forecasts for
#   a single selected company. Date inputs are disabled on this screen
#   since predictions always run on the most recent available trading day.
#   All Companies is blocked here because each model requires a single 
#   ticker as input.
# ══════════════════════════════════════════════════════════════════════════

class PredictionsScreen:

    def __init__(self, selected_company: str, selected_model: str):
        self.selected_company = selected_company
        self.selected_model   = selected_model

    def render(self):
        st.markdown(PREDICTIONS_CSS, unsafe_allow_html=True)
        if self.selected_model == "Compare All":
            st.markdown(f"## 📈 Compare All Models - {self.selected_company}")
        else:
            st.markdown(f"## 📈 {self.selected_model} Predictions - {self.selected_company}")

        if self.selected_company == "All Companies":
            st.warning("Please select a single company to see predictions.")
            return

        ticker_symbol = TICKERS[self.selected_company]

        try:
            _now_uk      = datetime.now(ZoneInfo("Europe/London"))
            _last_day    = _now_uk.date() if _now_uk.hour >= 17 else _now_uk.date() - timedelta(days=1)
            today_str    = _last_day.strftime('%Y-%m-%d')

            # ── Spinner guard ─────────────────────────────────────────────
            # Spinner only shown on the first load of the predictions screen.
            # Subsequent visits return instantly from cache.
            predictions_data_cache_key = "predictions_market_data_loaded"
            if predictions_data_cache_key not in st.session_state:
                with st.spinner("Loading market data..."):
                    df_recent = StockDataService.get_stock_data(
                        TICKER_LIST, DATA_START, today_str
                    )
                st.session_state[predictions_data_cache_key] = True
            else:
                df_recent = StockDataService.get_stock_data(
                    TICKER_LIST, DATA_START, today_str
                )

            if df_recent is None or df_recent.empty:
                st.error(f"Could not fetch recent data for {self.selected_company}")
                return

            if isinstance(df_recent.columns, pd.MultiIndex):
                close_prices = df_recent[('Close', ticker_symbol)].dropna()
            else:
                close_prices = df_recent['Close'].dropna()

            if len(close_prices) < 15:
                st.error("Not enough data to make a prediction.")
                return

            current_price = close_prices.iloc[-1]
            current_date  = close_prices.index[-1]

            # ── Forecast dates ────────────────────────────────────────────
            # Business date range is used so weekends are excluded from the
            # 5-day forecast window, matching how trading days work in practice.
            future_dates  = pd.bdate_range(start=_last_day, periods=6)[1:]

            st.divider()

            if self.selected_model == "ARIMA":
                forecast, conf_int, error = None, None, None
                # ── Spinner guard ─────────────────────────────────────────
                # Spinner only shown the first time this model is run for
                # this ticker. Subsequent visits return instantly from cache.
                arima_cache_key = f"arima_loaded_{ticker_symbol}"
                if arima_cache_key not in st.session_state:
                    with st.spinner("Running ARIMA model..."):
                        forecast, conf_int, error = ForecastService.run_arima(
                            df_recent, ticker_symbol, today_str
                        )
                    st.session_state[arima_cache_key] = True
                else:
                    forecast, conf_int, error = ForecastService.run_arima(
                        df_recent, ticker_symbol, today_str
                    )
                if error:
                    st.error(f"ARIMA model failed: {error}")
                else:
                    ModelSection(
                        "ARIMA", forecast, conf_int, future_dates,
                        current_price, current_date, ticker_symbol, close_prices
                    ).render()

            elif self.selected_model == "Random Forest":
                forecast, conf_int, error = None, None, None
                rf_cache_key = f"rf_loaded_{ticker_symbol}"
                if rf_cache_key not in st.session_state:
                    with st.spinner("Running Random Forest model..."):
                        forecast, conf_int, error = ForecastService.run_random_forest(
                            df_recent, ticker_symbol, today_str
                        )
                    st.session_state[rf_cache_key] = True
                else:
                    forecast, conf_int, error = ForecastService.run_random_forest(
                        df_recent, ticker_symbol, today_str
                    )
                if error:
                    st.error(f"Random Forest model failed: {error}")
                else:
                    ModelSection(
                        "Random Forest", forecast, conf_int, future_dates,
                        current_price, current_date, ticker_symbol, close_prices
                    ).render()

            elif self.selected_model == "GRU":
                forecast, conf_int, error = None, None, None
                gru_cache_key = f"gru_loaded_{ticker_symbol}"
                if gru_cache_key not in st.session_state:
                    with st.spinner("Running GRU model..."):
                        forecast, conf_int, error = ForecastService.run_gru(
                            df_recent, ticker_symbol, today_str
                        )
                    st.session_state[gru_cache_key] = True
                else:
                    forecast, conf_int, error = ForecastService.run_gru(
                        df_recent, ticker_symbol, today_str
                    )
                if error:
                    st.error(f"GRU model failed: {error}")
                else:
                    ModelSection(
                        "GRU", forecast, conf_int, future_dates,
                        current_price, current_date, ticker_symbol, close_prices
                    ).render()

            elif self.selected_model == "Compare All":
                # ───────────────── Compare All ─────────────────────────────
                # All three models are run inside a single st.spinner block
                # rather than three separate ones. This keeps the interface
                # clean and avoids three loading states flashing in sequence,
                # which would be distracting given the combined runtime.
                arima_forecast, arima_ci, arima_error = None, None, None
                rf_forecast, rf_ci, rf_error           = None, None, None
                gru_forecast, gru_ci, gru_error         = None, None, None
                compare_cache_key = f"compare_loaded_{ticker_symbol}"
                if compare_cache_key not in st.session_state:
                    with st.spinner("Loading all models..."):
                        arima_forecast, arima_ci, arima_error = ForecastService.run_arima(
                            df_recent, ticker_symbol, today_str
                        )
                        rf_forecast, rf_ci, rf_error = ForecastService.run_random_forest(
                            df_recent, ticker_symbol, today_str
                        )
                        gru_forecast, gru_ci, gru_error = ForecastService.run_gru(
                            df_recent, ticker_symbol, today_str
                        )
                    st.session_state[compare_cache_key] = True
                else:
                    arima_forecast, arima_ci, arima_error = ForecastService.run_arima(
                        df_recent, ticker_symbol, today_str
                    )
                    rf_forecast, rf_ci, rf_error = ForecastService.run_random_forest(
                        df_recent, ticker_symbol, today_str
                    )
                    gru_forecast, gru_ci, gru_error = ForecastService.run_gru(
                        df_recent, ticker_symbol, today_str
                    )

                if arima_error:
                    st.error(f"ARIMA model failed: {arima_error}")
                if rf_error:
                    st.error(f"Random Forest model failed: {rf_error}")
                if gru_error:
                    st.error(f"GRU model failed: {gru_error}")

                if all(e is None for e in [arima_error, rf_error, gru_error]):
                    ComparisonSection(
                        arima_forecast, arima_ci,
                        rf_forecast, rf_ci,
                        gru_forecast, gru_ci,
                        future_dates, current_price, current_date,
                        ticker_symbol, close_prices
                    ).render()
                else:
                    st.warning("One or more models failed — comparison unavailable.")

            render_model_disclaimer()

        except Exception as e:
            st.error(f"A critical error occurred while generating predictions: {e}")


# ══════════════════════════════════════════════════════════════════════════
#   MODEL SECTION
#   Renders the forecast output for a single model: a weighted average
#   summary metric, a day-by-day prediction table with direction arrows
#   and prediction intervals, and a 30-day history + 5-day forecast chart.
#   The weighted average uses the 50/20/10/10/10 scheme from the
#   comparative analysis so the summary figure is consistent with the
#   dissertation results rather than a simple mean.
# ══════════════════════════════════════════════════════════════════════════

class ModelSection:

    def __init__(self, model_name, forecast, conf_int, future_dates,
                 current_price, current_date, ticker_symbol, close_prices):
        self.model_name    = model_name
        self.forecast      = forecast
        self.conf_int      = conf_int
        self.future_dates  = future_dates
        self.current_price = current_price
        self.current_date  = current_date
        self.ticker_symbol = ticker_symbol
        self.close_prices  = close_prices

    def render(self):
        weights        = np.array([0.5, 0.2, 0.1, 0.1, 0.1])
        weighted_price = float(np.sum(np.array(self.forecast) * weights))
        delta_pct      = (weighted_price - self.current_price) / self.current_price * 100
        sign           = "+" if delta_pct > 0 else ""
        color          = "green" if delta_pct > 0 else "red"

        col1, col2 = st.columns(2)
        col1.metric(
            label=f"Current Price — {self.current_date.strftime('%d %b %Y')}",
            value=f"{self.current_price:.2f}p"
        )
        col2.metric(
            label="5-Day Weighted Avg Forecast",
            value=f"{weighted_price:.2f}p",
            delta=f"{sign}{delta_pct:.2f}%"
        )
        st.divider()
        self._render_table()
        self._render_chart()

    def _render_table(self):
        # ── Prediction table ──────────────────────────────────────────────
        # A custom HTML table is used rather than st.dataframe to give full
        # control over column widths, inline colour coding, and the direction
        # arrow layout. st.dataframe does not support per-cell styling.
        col_spec = [1.0, 1.4, 1.6, 2.6]
        h1, h2, h3, h4 = st.columns(col_spec)
        h1.markdown("<div style='font-size:15px;font-weight:bold;text-align:center;padding:12px 0;'>Date</div>", unsafe_allow_html=True)
        h2.markdown("<div style='font-size:15px;font-weight:bold;text-align:center;padding:12px 0;'>Predicted Price</div>", unsafe_allow_html=True)
        h3.markdown("<div style='font-size:15px;font-weight:bold;text-align:center;padding:12px 0;'>Expected Change</div>", unsafe_allow_html=True)
        h4.markdown(
            "<div style='text-align:center;padding:8px 0;'>"
            "<div style='font-size:15px;font-weight:bold;margin-bottom:6px;'>Prediction Interval</div>"
            "<div style='display:flex;justify-content:space-around;font-size:0.88em;font-weight:bold;color:#666;'>"
            "<div style='flex:1;'>Lower</div><div style='flex:1;'>Prediction</div><div style='flex:1;'>Upper</div>"
            "</div></div>",
            unsafe_allow_html=True
        )
        st.divider()

        for i in range(5):
            pred       = self.forecast[i]
            lower      = self.conf_int.iloc[i, 0]
            upper      = self.conf_int.iloc[i, 1]
            day_change = pred - self.current_price
            day_pct    = (day_change / self.current_price) * 100 if self.current_price != 0 else 0

            if day_change > 0:   arrow, color, sign = "↑", "green", "+"
            elif day_change < 0: arrow, color, sign = "↓", "red",   ""
            else:                arrow, color, sign = "→", "gray",  ""

            c1, c2, c3, c4 = st.columns(col_spec)
            c1.markdown(f"<div style='text-align:center;font-weight:bold;'>{self.future_dates[i].strftime('%d %b')}</div>", unsafe_allow_html=True)
            c2.markdown(f"<div style='text-align:center;font-size:1.1em;font-weight:bold;'>{pred:.2f}p</div>", unsafe_allow_html=True)
            c3.markdown(
                f"<div style='text-align:center;'>"
                f"<div style='display:inline-flex;align-items:center;justify-content:center;gap:6px;white-space:nowrap;'>"
                f"<span style='font-size:24px;line-height:1;color:{color};'>{arrow}</span>"
                f"<span style='font-size:16px;font-weight:500;'>{abs(day_change):.2f}p</span>"
                f"<span style='font-size:13px;color:{color};opacity:0.9;'>{sign}{day_pct:.2f}%</span>"
                f"</div></div>",
                unsafe_allow_html=True
            )
            c4.markdown(
                f"<div style='display:flex;justify-content:space-around;'>"
                f"<div style='flex:1;color:#444;text-align:center;'>{lower:.2f}p</div>"
                f"<div style='flex:1;font-weight:bold;color:#000;text-align:center;'>{pred:.2f}p</div>"
                f"<div style='flex:1;color:#444;text-align:center;'>{upper:.2f}p</div>"
                f"</div>",
                unsafe_allow_html=True
            )
            if i < 4:
                st.divider()

        st.divider()

    def _render_chart(self):
        # ── 30-day history + 5-day forecast chart ────────────────────────
        # The historical window is limited to 30 days to keep the forecast
        # region visible without the chart scaling to the full data range.
        # A vertical dotted line separates the historical and forecast periods.
        # The shaded confidence interval ribbon uses 18% opacity so the
        # forecast line remains readable through the fill.
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader(f"30 Day History & 5 Day {self.model_name} Prediction")

        chart_prices = self.close_prices.tail(30)
        fig = go.Figure()

        fig.add_trace(go.Scatter(
            x=chart_prices.index, y=chart_prices.values,
            name='Historical Price', mode='lines',
            line=dict(color=BANK_COLOR_MAP.get(self.ticker_symbol, "#6B7280"), width=2),
            hovertemplate="<b>Historical:</b> %{y:.2f}p<extra></extra>"
        ))
        fig.add_vline(x=self.close_prices.index[-1], line=dict(color="lightgray", dash="dot"))
        fig.add_trace(go.Scatter(
            x=[self.close_prices.index[-1]] + list(self.future_dates),
            y=[self.current_price] + list(self.forecast),
            mode='lines+markers', name='Predicted Price',
            line=dict(color="orange", width=3),
            marker=dict(size=[0, 8, 8, 8, 8, 8], color="orange"),
            customdata=np.column_stack([
                [self.current_price] + list(self.conf_int.iloc[:, 0]),
                [self.current_price] + list(self.conf_int.iloc[:, 1]),
            ]),
            hovertemplate="<b>Predicted:</b> %{y:.2f}p<br><b>Lower:</b> %{customdata[0]:.2f}p<br><b>Upper:</b> %{customdata[1]:.2f}p<extra></extra>"
        ))

        shade_x = ([self.close_prices.index[-1]] + list(self.future_dates)
                   + list(self.future_dates[::-1]) + [self.close_prices.index[-1]])
        shade_y = ([self.current_price] + list(self.conf_int.iloc[:, 1])
                   + list(self.conf_int.iloc[:, 0][::-1]) + [self.current_price])
        fig.add_trace(go.Scatter(
            x=shade_x, y=shade_y, fill='toself',
            fillcolor='rgba(255,165,0,0.18)',
            line=dict(color='rgba(0,0,0,0)'),
            name='Prediction Interval', showlegend=True, hoverinfo='skip'
        ))

        fig.update_layout(
            height=500, xaxis_title='Date', yaxis_title='Closing Price (pence)',
            hovermode='x', hoverlabel=dict(bgcolor="white", font_size=13, align="left")
        )
        st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})
        st.divider()


# ══════════════════════════════════════════════════════════════════════════
#   COMPARISON SECTION
#   Renders all three model forecasts side by side in a single HTML table
#   and an overlaid Plotly chart. Each model retains its colour from the
#   MODEL_COLOR_MAP so the visual identity is consistent with the
#   individual model views. A weighted average summary is shown at the
#   top of the table for each model using the same 50/20/10/10/10 scheme.
# ══════════════════════════════════════════════════════════════════════════

class ComparisonSection:

    def __init__(self, arima_forecast, arima_ci, rf_forecast, rf_ci,
                 gru_forecast, gru_ci, future_dates, current_price,
                 current_date, ticker_symbol, close_prices):
        self.arima_forecast = arima_forecast
        self.arima_ci       = arima_ci
        self.rf_forecast    = rf_forecast
        self.rf_ci          = rf_ci
        self.gru_forecast   = gru_forecast
        self.gru_ci         = gru_ci
        self.future_dates   = future_dates
        self.current_price  = current_price
        self.current_date   = current_date
        self.ticker_symbol  = ticker_symbol
        self.close_prices   = close_prices

    def _model_cell(self, pred, lower, upper) -> str:
        change  = pred - self.current_price
        pct     = (change / self.current_price) * 100 if self.current_price != 0 else 0
        arrow   = "↑" if change > 0 else ("↓" if change < 0 else "→")
        a_color = "green" if change > 0 else ("red" if change < 0 else "gray")
        sign    = "+" if change > 0 else ""
        return (
            f"<div style='text-align:center;'>"
            f"<div style='display:inline-flex;align-items:center;justify-content:center;gap:6px;'>"
            f"<span style='font-size:24px;color:{a_color};'>{arrow}</span>"
            f"<span style='font-size:18px;color:#000;'>{pred:.2f}p</span>"
            f"<span style='font-size:16px;color:{a_color};'>{sign}{pct:.2f}%</span>"
            f"</div>"
            f"<div style='font-size:13px;color:#888;'>[{lower:.2f} – {upper:.2f}]</div>"
            f"</div>"
        )

    def render(self):
        weights              = np.array([0.5, 0.2, 0.1, 0.1, 0.1])
        arima_weighted_avg   = float(np.sum(np.array(self.arima_forecast) * weights))
        rf_weighted_avg      = float(np.sum(np.array(self.rf_forecast)    * weights))
        gru_weighted_avg     = float(np.sum(np.array(self.gru_forecast)   * weights))
        self._render_table(arima_weighted_avg, rf_weighted_avg, gru_weighted_avg)
        self._render_chart()

    def _render_table(self, arima_weighted_avg=None, rf_weighted_avg=None, gru_weighted_avg=None):
        arima_color = MODEL_COLOR_MAP['ARIMA']
        rf_color    = MODEL_COLOR_MAP['Random Forest']
        gru_color   = MODEL_COLOR_MAP['GRU']

        def pct_html(weighted_avg_price):
            d     = (weighted_avg_price - self.current_price) / self.current_price * 100
            sign  = "+" if d > 0 else ""
            color = "#2e7d32" if d > 0 else "#c62828"
            return f"<span style='color:{color};font-size:13px;'>{sign}{d:.2f}%</span>"

        def cell_html(fc, ci, i):
            pred   = fc[i]
            lower  = ci.iloc[i, 0]
            upper  = ci.iloc[i, 1]
            change = pred - self.current_price
            pct    = (change / self.current_price) * 100 if self.current_price != 0 else 0
            arrow  = "↑" if change > 0 else ("↓" if change < 0 else "→")
            color  = "#2e7d32" if change > 0 else ("#c62828" if change < 0 else "gray")
            sign   = "+" if change > 0 else ""
            return (
                f"<td style='text-align:center;padding:12px 8px;border-bottom:1px solid #eee;'>"
                f"<span style='color:{color};font-size:20px;'>{arrow}</span> "
                f"<span style='font-size:16px;font-weight:500;'>{pred:.2f}p</span> "
                f"<span style='color:{color};font-size:13px;'>{sign}{pct:.2f}%</span><br>"
                f"<span style='color:#888;font-size:12px;'>[{lower:.2f} – {upper:.2f}]</span>"
                f"</td>"
            )

        date_rows = ""
        for i in range(5):
            bg = "#fafafa" if i % 2 == 0 else "#ffffff"
            date_rows += (
                f"<tr style='background:{bg};'>"
                f"<td style='font-weight:bold;padding:12px 8px;border-bottom:1px solid #eee;'>"
                f"{self.future_dates[i].strftime('%d %b')}</td>"
                + cell_html(self.arima_forecast, self.arima_ci, i)
                + cell_html(self.rf_forecast,    self.rf_ci,    i)
                + cell_html(self.gru_forecast,   self.gru_ci,   i)
                + "</tr>"
            )

        header_row = ""
        if arima_weighted_avg is not None:
            header_row = (
                "<tr>"
                f"<td style='width:15%;padding:8px;'>"
                f"<div style='font-size:13px;color:#888;'>Current Price — {self.current_date.strftime('%d %b %Y')}</div>"
                f"<div style='font-size:26px;font-weight:bold;'>{self.current_price:.2f}p</div>"
                f"</td>"
                f"<td style='width:28%;text-align:center;padding:8px;'>"
                f"<div style='font-size:13px;color:#888;'>ARIMA Weighted Avg Forecast</div>"
                f"<div style='font-size:24px;font-weight:bold;color:{arima_color};'>{arima_weighted_avg:.2f}p</div>"
                f"<div>{pct_html(arima_weighted_avg)}</div></td>"
                f"<td style='width:28%;text-align:center;padding:8px;'>"
                f"<div style='font-size:13px;color:#888;'>Random Forest Weighted Avg Forecast</div>"
                f"<div style='font-size:24px;font-weight:bold;color:{rf_color};'>{rf_weighted_avg:.2f}p</div>"
                f"<div>{pct_html(rf_weighted_avg)}</div></td>"
                f"<td style='width:28%;text-align:center;padding:8px;'>"
                f"<div style='font-size:13px;color:#888;'>GRU Weighted Avg Forecast</div>"
                f"<div style='font-size:24px;font-weight:bold;color:{gru_color};'>{gru_weighted_avg:.2f}p</div>"
                f"<div>{pct_html(gru_weighted_avg)}</div></td>"
                "</tr>"
            )

        html = (
            "<div style='font-size:18px;font-weight:bold;margin-bottom:12px;'>5-Day Forecast Comparison</div>"
            "<table style='width:100%;border-collapse:collapse;margin-bottom:16px;'>"
            "<thead>"
            + header_row +
            "<tr style='border-top:2px solid #ddd;border-bottom:1px solid #eee;'>"
            f"<th style='text-align:left;padding:10px 8px;font-size:17px;'>Date</th>"
            f"<th style='text-align:center;padding:10px 8px;font-size:17px;color:{arima_color};'>ARIMA</th>"
            f"<th style='text-align:center;padding:10px 8px;font-size:17px;color:{rf_color};'>Random Forest</th>"
            f"<th style='text-align:center;padding:10px 8px;font-size:17px;color:{gru_color};'>GRU</th>"
            "</tr>"
            "</thead>"
            f"<tbody>{date_rows}</tbody>"
            "</table>"
        )
        st.markdown(html, unsafe_allow_html=True)

    def _render_chart(self):
        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("30 Day History & 5 Day Model Comparison")

        chart_prices = self.close_prices.tail(30)
        fig          = go.Figure()
        anchor_x     = [self.close_prices.index[-1]] + list(self.future_dates)

        fig.add_trace(go.Scatter(
            x=chart_prices.index, y=chart_prices.values,
            name='Historical Price', mode='lines',
            line=dict(color=BANK_COLOR_MAP.get(self.ticker_symbol, "#6B7280"), width=2),
            hovertemplate="<b>Historical:</b> %{y:.2f}p<extra></extra>"
        ))
        fig.add_vline(x=self.close_prices.index[-1], line=dict(color="lightgray", dash="dot"))

        for name, forecast, ci in [
            ("ARIMA",         self.arima_forecast, self.arima_ci),
            ("Random Forest", self.rf_forecast,    self.rf_ci),
            ("GRU",           self.gru_forecast,   self.gru_ci),
        ]:
            color = MODEL_COLOR_MAP.get(name, "#000")
            fig.add_trace(go.Scatter(
                x=anchor_x,
                y=[self.current_price] + list(forecast),
                mode='lines+markers',
                name=name,
                line=dict(color=color, width=2.5),
                marker=dict(size=[0, 7, 7, 7, 7, 7], color=color),
                customdata=np.column_stack([
                    [self.current_price] + list(ci.iloc[:, 0]),
                    [self.current_price] + list(ci.iloc[:, 1]),
                ]),
                hovertemplate=f"<b>{name}:</b> %{{y:.2f}}p<br><b>Lower:</b> %{{customdata[0]:.2f}}p<br><b>Upper:</b> %{{customdata[1]:.2f}}p<extra></extra>"
            ))

        fig.update_layout(
            height=500, xaxis_title='Date', yaxis_title='Closing Price (pence)',
            hovermode='x', hoverlabel=dict(bgcolor="white", font_size=13, align="left"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig, width='stretch', config={'displayModeBar': False})
        st.divider()


# ══════════════════════════════════════════════════════════════════════════
#   SIDEBAR & ROUTING
#   The sidebar handles company selection, date inputs, screen navigation,
#   and model selection. Date inputs are disabled on the predictions screen
#   as forecasts are tied to a fixed historical window.
#   Company selection uses a separate session state key on the predictions
#   screen to prevent All Companies from being carried over from the home
#   screen, which would block the predictions view from rendering.
# ══════════════════════════════════════════════════════════════════════════

st.markdown(
    "<h1 style='text-align:center;margin-top:-60px;margin-bottom:40px;'>"
    "🏦 Stock Market Prediction - UK Banking Sector</h1>",
    unsafe_allow_html=True
)

st.sidebar.header("Data Options")

company_options_filtered = (
    [c for c in COMPANY_OPTIONS if c != "All Companies"]
    if st.session_state['current_view'] == 'predictions'
    else COMPANY_OPTIONS
)

if st.session_state['selected_company'] not in company_options_filtered:
    st.session_state['selected_company'] = company_options_filtered[0]

if st.session_state['current_view'] == 'predictions':
    default_company = st.session_state.get('selected_company', company_options_filtered[0])
    if default_company == "All Companies":
        default_company = company_options_filtered[0]
    default_index = company_options_filtered.index(default_company) if default_company in company_options_filtered else 0

    selected_company = st.sidebar.selectbox(
        "Select Company:",
        options=company_options_filtered,
        index=default_index,
        key='_selected_company_for_predictions'
    )
else:
    selected_company = st.sidebar.selectbox(
        "Select Company:",
        options=company_options_filtered,
        key='selected_company'
    )

dates_disabled = st.session_state['current_view'] == 'predictions'
today          = datetime.now().date()

custom_start = st.sidebar.date_input(
    "Start Date:",
    st.session_state.get('start_date', today - timedelta(days=365)),
    max_value=today,
    disabled=dates_disabled
)
custom_end = st.sidebar.date_input(
    "End Date:",
    st.session_state.get('end_date', today),
    max_value=today,
    disabled=dates_disabled
)

# ── Custom date change detection ──────────────────────────────────────────────
# Only triggers a rerun when dates genuinely change to avoid unnecessary
# refreshes caused by Streamlit re-evaluating widget defaults on each run.
if not dates_disabled and (
    custom_start != st.session_state.get('start_date') or
    custom_end   != st.session_state.get('end_date')
):
    st.session_state['start_date']   = custom_start
    st.session_state['end_date']     = custom_end
    st.session_state['active_range'] = None
    st.rerun()

st.sidebar.markdown("---")
st.sidebar.markdown("### Actions")

view = st.session_state.get('current_view', 'main')

if view == 'predictions':
    if st.sidebar.button("🏠 Home", use_container_width=True):
        st.session_state['current_view'] = 'main'
        st.rerun()

if st.sidebar.button("📈 Predictions", use_container_width=True):
    st.session_state['current_view'] = 'predictions'
    if 'selected_model' not in st.session_state:
        st.session_state['selected_model'] = "ARIMA"
    st.rerun()

if view == 'predictions':
    st.sidebar.markdown("---")
    st.sidebar.selectbox(
        "Select Model",
        options=["ARIMA", "Random Forest", "GRU", "Compare All"],
        key='selected_model'
    )

# ── Screen routing ────────────────────────────────────────────────────────────
# Routes to the appropriate screen class based on current_view in session state.
# All state is read from session_state rather than passed as arguments so that
# widget updates on one screen do not affect the other.
if st.session_state['current_view'] == 'main':
    HomeScreen(st.session_state['selected_company']).render()

elif st.session_state['current_view'] == 'predictions':
    PredictionsScreen(
        st.session_state.get('_selected_company_for_predictions',
                             st.session_state['selected_company']),
        st.session_state.get('selected_model', 'ARIMA')
    ).render()
