# Walk-Forward Optimization Dashboard

A portfolio demo visualizing backtesting results from an algorithmic trading research system. Shows out-of-sample performance from walk-forward optimization across multiple strategy types.

## What It Shows

Real walk-forward optimization results from a multi-asset trading system:

- **Equity curves** — Cumulative out-of-sample returns across validation windows
- **Strategy comparison** — Side-by-side metrics for TCN, MLP, and rule-based strategies
- **Per-window metrics** — Sharpe ratio, returns, and win rate for each validation period
- **Raw data** — Full window-level metrics with expandable detail view

## Walk-Forward Optimization

WFO trains a model on a rolling window of historical data, then tests on the next unseen period. This process repeats across the dataset, producing out-of-sample results that reflect real trading conditions — no look-ahead bias, no data leakage.

```
[Train Window 1] → [Test 1]
    [Train Window 2] → [Test 2]
        [Train Window 3] → [Test 3]
            ...
```

## Strategy Types

- **TCN Action** — Temporal Convolutional Network with multi-timeframe features (M5, H1, D)
- **TCN Classifier** — TCN-based directional classifier ({long, flat, short})
- **MLP Classifier** — Lightweight MLP over per-bar features
- **Rule-Based** — Hand-crafted momentum/mean-reversion rules (baseline)

## Tech Stack

- **Data:** Pandas with `@st.cache_data`
- **Charts:** Plotly (interactive, hover tooltips)
- **Frontend:** Streamlit
- **Models:** PyTorch TCN and MLP architectures (results only — no model inference in the app)

## Run Locally

```bash
git clone https://github.com/vandyand/algo-trading-wfo-dashboard.git
cd algo-trading-wfo-dashboard
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

No API keys needed — the app uses bundled historical data.

## Project Structure

```
├── app.py              # Streamlit dashboard
├── data_loader.py      # WFO data parsing and DataFrame construction
├── data/               # Curated WFO run results (~11 runs)
│   ├── tcn-action-*/   # TCN strategy runs
│   ├── mlp-action-*/   # MLP strategy runs
│   └── rule-action-*/  # Rule-based baseline runs
└── test_data_loader.py # Data loading tests
```
