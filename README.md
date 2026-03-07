# Stock Market Prediction — UK Banking Sector

BSc (Hons) Software Development — Honours Project 2026
By Ross Todd

A Streamlit web application that generates five-day price forecasts for three UK banking stocks — Barclays (BARC.L), Lloyds (LLOY.L), and HSBC (HSBA.L) — using three forecasting models: ARIMA, Random Forest, and GRU.

---

## Features

- Interactive historical price charts with date range controls (1D to 5Y)
- Five-day price forecasts with 95% prediction intervals
- Weighted average forecast summary across the five-day window (50% day 1)
- Three models to choose from — ARIMA, Random Forest, and GRU
- Side-by-side model comparison view
- Performance based on a fixed 2021–2026 training window matching the comparative analysis

---

## Project Structure

```
├── app.py                  # Presentation layer — Streamlit UI and screen routing
├── data.py                 # Data layer — fetching and cleaning via yfinance
├── models.py               # Logic layer — ARIMA, Random Forest, and GRU pipelines
├── utils.py                # Helpers — constants, colours, date utilities, CSS
├── trained_models/         # Pre-trained RF and GRU model files (.pkl and .keras)
└── requirements.txt        # Python dependencies
```

---

## Installation

**1. Clone the repository**

```bash
git clone https://github.com/ross-todd/stock_market_prediction_application.git
cd stock_market_prediction_application
```

**2. Create and activate a virtual environment**

```bash
python -m venv venv
source venv/bin/activate        # macOS/Linux
venv\Scripts\activate           # Windows
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

---

## Running the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## Requirements

- Python 3.9 or higher
- See `requirements.txt` for full list of dependencies

Key packages:

- `streamlit`
- `yfinance`
- `pandas` / `numpy`
- `plotly`
- `scikit-learn`
- `statsmodels`
- `tensorflow`
- `joblib`

---

## Trained Models

The Random Forest and GRU models were trained as part of the comparative analysis and saved to the `trained_models/` directory. The following files must be present for the predictions screen to work:

| File                      | Description                      |
| ------------------------- | -------------------------------- |
| `rf_BARC_L.pkl`         | Random Forest model for Barclays |
| `rf_LLOY_L.pkl`         | Random Forest model for Lloyds   |
| `rf_HSBA_L.pkl`         | Random Forest model for HSBC     |
| `scaler_BARC_L_rf.pkl`  | StandardScaler for Barclays RF   |
| `scaler_LLOY_L_rf.pkl`  | StandardScaler for Lloyds RF     |
| `scaler_HSBA_L_rf.pkl`  | StandardScaler for HSBC RF       |
| `gru_BARC_L.keras`      | GRU model for Barclays           |
| `gru_LLOY_L.keras`      | GRU model for Lloyds             |
| `gru_HSBA_L.keras`      | GRU model for HSBC               |
| `scaler_BARC_L_gru.pkl` | MinMaxScaler for Barclays GRU    |
| `scaler_LLOY_L_gru.pkl` | MinMaxScaler for Lloyds GRU      |
| `scaler_HSBA_L_gru.pkl` | MinMaxScaler for HSBC GRU        |

---

## Disclaimer

This application is for educational purposes only. Stock prices are influenced by many factors and can be unpredictable. Do not use this application as the basis for any real investment decisions.
