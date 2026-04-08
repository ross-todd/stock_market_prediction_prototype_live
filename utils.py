#  Ross Todd
#  BSc (Hons) Software Development
#  Honours Project 2026 - A Stock Market Prediction Application Prototype - Live Version

# # utils.py — Helpers Layer
# Constants, colours, date calculations, and custom styling

from datetime import datetime
from pandas.tseries.offsets import BDay


# ── Ticker Constants ───────────────────────────────────

TICKERS = {
    "Barclays Plc":             "BARC.L",
    "HSBC Holdings Plc":        "HSBA.L",
    "Lloyds Banking Group Plc": "LLOY.L"
}

TICKER_LIST     = list(TICKERS.values())
TICKER_NAMES    = list(TICKERS.keys())
COMPANY_OPTIONS = ["All Companies"] + TICKER_NAMES


# ── Colour Constants ───────────────────────────────────

BANK_COLOR_MAP = {
    "BARC.L": "#0000FF",   # Barclays — blue
    "HSBA.L": "#FF0000",   # HSBC — red
    "LLOY.L": "#008000",   # Lloyds — green
}

MODEL_COLOR_MAP = {
    "ARIMA":         "#9575CD",   # purple
    "Random Forest": "#4FC3F7",   # light blue
    "GRU":           "#F06292",   # pink
}


# ── Date Calculation ──────────────────────────────────────────────────────────────────────────────

# Returns a start date based on a preset range label
def get_start_date_from_range(range_selection: str):
    today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
    mapping = {
        '1D': BDay(1),
        '1W': BDay(5),
        '1M': BDay(21),
        '3M': BDay(63),
        '6M': BDay(126),
        '1Y': BDay(252),
        '5Y': BDay(5 * 252),
    }
    offset = mapping.get(range_selection, BDay(21))
    return today - offset


# ── Custom Styling ────────────────────────────────────────────────────────────

GLOBAL_CSS = """
<style>
    .stMainBlockContainer { padding-bottom: 0rem !important; }
    .stVerticalBlock { padding-bottom: 0rem !important; }
    .stPlotlyChart { margin-bottom: 0rem !important; }
    div[data-testid="stSpinner"] {
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        z-index: 9999;
        background: white;
        padding: 28px 36px;
        border-radius: 14px;
        box-shadow: 0 4px 24px rgba(0, 0, 0, 0.12);
        font-size: 16px;
        font-weight: 500;
        color: #333;
    }
</style>
"""

DATE_RANGE_BUTTONS_CSS = """
<style>
    div[data-testid="stHorizontalBlock"] {
        min-height: 50px !important;
        max-height: 50px !important;
        margin-top: -5px !important;
        margin-bottom: 10px !important;
    }
    div[data-testid="stHorizontalBlock"] > div {
        min-height: 50px !important;
    }
    button[kind="primary"] {
        background-color: #FFD700 !important;
        color: black !important;
        border: 2px solid #FFA500 !important;
        font-weight: 600 !important;
        transition: none !important;
        box-shadow: inset 0 0 0 1px rgba(0,0,0,0.1) !important;
    }
    button[kind="secondary"] {
        background-color: #f0f2f6 !important;
        color: #666 !important;
        border: 1px solid #ddd !important;
        transition: none !important;
        font-weight: 400 !important;
    }
    button[kind="secondary"]:hover {
        background-color: #e0e2e6 !important;
        border-color: #bbb !important;
    }
    div[data-testid="column"] button {
        height: 38px !important;
        min-height: 38px !important;
        max-height: 38px !important;
        padding: 0.25rem 0.75rem !important;
        margin: 5px 0 !important;
        width: 100% !important;
    }
    button:focus {
        outline: 2px solid #FFD700 !important;
        box-shadow: inset 0 0 0 1px rgba(0,0,0,0.1) !important;
    }
    div[data-testid="column"] { min-width: 0 !important; }
    .stPlotlyChart { min-height: 670px; margin-top: 10px !important; }
    button p { margin: 0 !important; white-space: nowrap !important; }
</style>
"""

PREDICTIONS_CSS = """
<style>
    [data-testid="stMetric"] { padding: 2px 0 !important; }
    [data-testid="stMetricValue"] { font-size: 28px !important; }
    [data-testid="stMetricDelta"] { font-size: 12px !important; }
    [data-testid="stMetricLabel"] { font-size: 16px !important; }
    hr { margin: 4px 0 !important; }
</style>
"""
