# Walk-Forward Optimization Dashboard — Design Document

## Purpose

Portfolio demo showcasing quantitative trading research methodology. A Streamlit dashboard that visualizes walk-forward optimization results from a real algorithmic trading system — equity curves, strategy comparisons, per-window performance metrics. Pure data visualization, no LLM calls.

Directly targets the Quantitative Developer ($20-100/hr, 6+ months) job — demonstrates backtesting, walk-forward testing, strategy evaluation, and Python data work (Pandas, Plotly).

## Architecture

### Data Pipeline

```
Curated WFO runs (static files in data/)
    ↓
load_data() with @st.cache_data
    ↓
Single Pandas DataFrame (all runs, all windows)
    ↓
Visualizations query DataFrame
```

No API calls, no database. The app ships with ~20-30 curated runs from the real rl-trading research codebase. Each run contains a `meta.json` (config) and `windows.jsonl` (per-window performance metrics).

### Data Source

Curated from `/home/kingjames/rl-trader/forex-rl/wfo/runs/` — 6,700+ historical WFO runs. We select ~20-30 representative runs spanning:

- **Strategy types:** TCN action classifier, TCN scalar threshold, MLP action classifier, rule-based
- **Instruments:** NQ, EUR_USD, EUR_JPY (where available)
- **Window counts:** Small (4-12 windows) and large (40-120 windows)
- **Date ranges:** 2023 and 2025 data

### DataFrame Schema

Columns: `run_id`, `adapter`, `instrument`, `window`, `train_start`, `train_end`, `val_start`, `val_end`, `cum_return`, `sharpe`, `sortino`, `max_dd`, `win_rate`, `profit_factor`, `trades`, `time_in_mkt`, `pos_frac_long`, `pos_frac_short`

## Page Layout

### Header
- Title: "Algorithmic Trading — Walk-Forward Optimization Dashboard"
- Caption: One sentence explaining WFO methodology

### Section 1: Summary Metrics
- `st.metric` cards in columns: total runs, total validation windows, average Sharpe, average win rate, best single-window return

### Section 2: Equity Curve
- Plotly line chart: cumulative returns across sequential validation windows
- Dropdown to select which run to display
- Shows out-of-sample performance window by window

### Section 3: Strategy Comparison
- `st.dataframe` table comparing aggregate metrics by strategy type
- Columns: strategy, mean Sharpe, mean win rate, mean profit factor, mean max drawdown, total trades

### Section 4: Per-Window Metrics
- Plotly bar chart: Sharpe ratio (or cumulative return) per validation window for selected run
- Highlights variance across windows

### Sidebar
- Run selector dropdown
- Architecture description (TCN + actor-critic pipeline)
- Strategy type legend
- GitHub source link

## Project Structure

```
algo-trading-wfo-dashboard/
├── app.py                  # Streamlit single-page dashboard
├── data/                   # Curated WFO run data
│   ├── run_001/
│   │   ├── meta.json
│   │   └── windows.jsonl
│   └── ...                 # ~20-30 curated runs
├── requirements.txt
├── .gitignore
├── .env.example
├── .streamlit/
│   └── config.toml
└── README.md
```

## Tech Stack

- **Frontend:** Streamlit
- **Data:** Pandas with @st.cache_data
- **Charts:** Plotly (hover tooltips, axis formatting, financial chart styling)
- **No LLM, no API keys** — pure data visualization
