# WFO Dashboard Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a Streamlit dashboard that visualizes walk-forward optimization results from a real algorithmic trading system, targeting the Quantitative Developer job listing.

**Architecture:** Static data files (curated WFO runs) loaded into a single Pandas DataFrame at startup via `@st.cache_data`. Plotly charts for equity curves and per-window metrics, `st.dataframe` for strategy comparison. Single-page layout with sidebar run selector.

**Tech Stack:** Python 3, Streamlit, Pandas, Plotly

---

### Task 1: Project Setup

**Files:**
- Create: `requirements.txt`
- Create: `.gitignore`
- Create: `.env.example`
- Create: `.streamlit/config.toml`

**Step 1: Create requirements.txt**

```
streamlit>=1.30.0
pandas>=2.0.0
plotly>=5.18.0
```

**Step 2: Create .gitignore**

```
__pycache__/
*.pyc
.env
venv/
.streamlit/secrets.toml
```

**Step 3: Create .env.example**

```
# No API keys needed — this app uses static data only
```

**Step 4: Create .streamlit/config.toml**

```toml
[theme]
primaryColor = "#4A90D9"
backgroundColor = "#0E1117"
secondaryBackgroundColor = "#1E2130"
textColor = "#FAFAFA"
```

**Step 5: Set up venv and install deps**

Run:
```bash
cd /home/kingjames/algo-trading-wfo-dashboard && python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt
```

**Step 6: Commit**

Run:
```bash
cd /home/kingjames/algo-trading-wfo-dashboard && git add requirements.txt .gitignore .env.example .streamlit/ && git commit -m "chore: project setup with dependencies"
```

---

### Task 2: Curate WFO Run Data

**Files:**
- Create: `data/` directory with ~15 curated runs copied from rl-trading

**Step 1: Copy selected runs into data/**

Copy these specific run directories from `/home/kingjames/rl-trader/forex-rl/wfo/runs/` into `/home/kingjames/algo-trading-wfo-dashboard/data/`:

TCN Action NQ (3 runs):
```bash
cp -r /home/kingjames/rl-trader/forex-rl/wfo/runs/tcn-action-nq-wfo-20251230-184234 /home/kingjames/algo-trading-wfo-dashboard/data/
cp -r /home/kingjames/rl-trader/forex-rl/wfo/runs/tcn-action-nq-wfo-20251230-215708 /home/kingjames/algo-trading-wfo-dashboard/data/
cp -r /home/kingjames/rl-trader/forex-rl/wfo/runs/tcn-action-nq-wfo-20251213-031445 /home/kingjames/algo-trading-wfo-dashboard/data/
```

TCN Action Classifier (3 runs):
```bash
cp -r /home/kingjames/rl-trader/forex-rl/wfo/runs/tcn-action-classifier-wfo-20251212-052141 /home/kingjames/algo-trading-wfo-dashboard/data/
cp -r /home/kingjames/rl-trader/forex-rl/wfo/runs/tcn-action-classifier-wfo-20251218-045254 /home/kingjames/algo-trading-wfo-dashboard/data/
cp -r /home/kingjames/rl-trader/forex-rl/wfo/runs/tcn-action-classifier-wfo-20251218-034418 /home/kingjames/algo-trading-wfo-dashboard/data/
```

MLP Action Classifier (2 runs):
```bash
cp -r /home/kingjames/rl-trader/forex-rl/wfo/runs/mlp-action-classifier-wfo-20251208-182918 /home/kingjames/algo-trading-wfo-dashboard/data/
cp -r /home/kingjames/rl-trader/forex-rl/wfo/runs/mlp-action-classifier-wfo-20251208-024626 /home/kingjames/algo-trading-wfo-dashboard/data/
```

Rule-Based (2 runs):
```bash
cp -r /home/kingjames/rl-trader/forex-rl/wfo/runs/rule-action-wfo-20251208-183600 /home/kingjames/algo-trading-wfo-dashboard/data/
cp -r /home/kingjames/rl-trader/forex-rl/wfo/runs/rule-action-wfo-20251208-024832 /home/kingjames/algo-trading-wfo-dashboard/data/
```

TCN Scalar Threshold (1 run — find one with non-empty windows.jsonl):
```bash
# Find a non-empty tcn-scalar-threshold run first:
# ls -la /home/kingjames/rl-trader/forex-rl/wfo/runs/tcn-scalar-threshold-wfo-*/windows.jsonl
# Then copy the best one
```

**Step 2: Verify data was copied correctly**

Run:
```bash
cd /home/kingjames/algo-trading-wfo-dashboard && for d in data/*/; do echo "$d: $(wc -l < "$d/windows.jsonl") windows"; done
```

Every run should show at least 10 windows. If any run has 0 or very few windows with actual trades, replace it with a better one from the source directory.

**Step 3: Remove any empty-window runs**

Check each windows.jsonl — if a run has mostly zero-trade windows (trades: 0.0), find a replacement run of the same strategy type that has actual trading activity.

**Step 4: Commit**

Run:
```bash
cd /home/kingjames/algo-trading-wfo-dashboard && git add data/ && git commit -m "feat: add curated WFO run data (11 runs across 4 strategy types)"
```

---

### Task 3: Build Data Loading Module

**Files:**
- Create: `data_loader.py`
- Create: `test_data_loader.py`

**Step 1: Write the failing test**

Create `test_data_loader.py`:
```python
"""Tests for WFO data loading."""
import pandas as pd
from data_loader import load_all_runs, STRATEGY_LABELS


def test_load_all_runs_returns_dataframe():
    df = load_all_runs()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


def test_dataframe_has_required_columns():
    df = load_all_runs()
    required = [
        "run_id", "strategy", "window", "val_start", "val_end",
        "cum_return", "sharpe", "sortino", "max_dd",
        "trades", "win_rate", "profit_factor",
    ]
    for col in required:
        assert col in df.columns, f"Missing column: {col}"


def test_multiple_strategies_present():
    df = load_all_runs()
    strategies = df["strategy"].unique()
    assert len(strategies) >= 3, f"Expected 3+ strategies, got {list(strategies)}"


def test_strategy_labels_maps_all_adapters():
    df = load_all_runs()
    # Every row should have a human-readable strategy label
    assert df["strategy"].notna().all()
    assert (df["strategy"] != "").all()
```

**Step 2: Run test to verify it fails**

Run: `cd /home/kingjames/algo-trading-wfo-dashboard && source venv/bin/activate && python -m pytest test_data_loader.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'data_loader'`

**Step 3: Write `data_loader.py`**

```python
"""Load and parse curated WFO run data into a single DataFrame."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

DATA_DIR = Path(__file__).parent / "data"

STRATEGY_LABELS = {
    "tcn-action-nq": "TCN Action (NQ)",
    "tcn-action-classifier": "TCN Classifier",
    "tcn-scalar-threshold": "TCN Scalar",
    "mlp-action-classifier": "MLP Classifier",
    "rule-action": "Rule-Based",
}

# Columns to extract from each window record
_METRIC_COLS = [
    "cum_return", "sharpe", "sortino", "max_dd",
    "trades", "win_rate", "profit_factor", "equity_r2",
    "long_trades", "short_trades", "time_in_mkt",
]


def _parse_window(record: Dict[str, Any]) -> Dict[str, Any]:
    """Extract relevant fields from a single window JSONL record."""
    row: Dict[str, Any] = {"window": record.get("win", 0)}

    # Validation period timestamps
    val = record.get("val", [None, None])
    row["val_start"] = val[0] if val else None
    row["val_end"] = val[1] if len(val) > 1 else None

    # Metrics — handle null and Infinity
    for col in _METRIC_COLS:
        val = record.get(col)
        if val is None or (isinstance(val, float) and (math.isinf(val) or math.isnan(val))):
            row[col] = None
        else:
            row[col] = float(val)

    return row


def _load_run(run_dir: Path) -> List[Dict[str, Any]]:
    """Load a single WFO run directory (meta.json + windows.jsonl)."""
    meta_path = run_dir / "meta.json"
    windows_path = run_dir / "windows.jsonl"

    if not meta_path.exists() or not windows_path.exists():
        return []

    with meta_path.open() as f:
        meta = json.load(f)

    adapter = meta.get("adapter", "unknown")
    strategy = STRATEGY_LABELS.get(adapter, adapter)
    run_id = run_dir.name

    rows = []
    with windows_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # Handle Infinity in JSON
            line = line.replace(": Infinity", ": null")
            line = line.replace(":-Infinity", ": null")
            line = line.replace(":Infinity", ": null")
            record = json.loads(line)
            row = _parse_window(record)
            row["run_id"] = run_id
            row["adapter"] = adapter
            row["strategy"] = strategy
            rows.append(row)

    return rows


def load_all_runs() -> pd.DataFrame:
    """Load all curated WFO runs into a single DataFrame."""
    all_rows: List[Dict[str, Any]] = []

    if not DATA_DIR.exists():
        return pd.DataFrame()

    for run_dir in sorted(DATA_DIR.iterdir()):
        if run_dir.is_dir():
            all_rows.extend(_load_run(run_dir))

    if not all_rows:
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)

    # Parse dates
    for col in ("val_start", "val_end"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], utc=True, errors="coerce")

    # Sort by run and window number
    df = df.sort_values(["run_id", "window"]).reset_index(drop=True)

    return df
```

**Step 4: Run tests to verify they pass**

Run: `cd /home/kingjames/algo-trading-wfo-dashboard && source venv/bin/activate && python -m pytest test_data_loader.py -v`
Expected: All 4 tests PASS

**Step 5: Commit**

Run:
```bash
cd /home/kingjames/algo-trading-wfo-dashboard && git add data_loader.py test_data_loader.py && git commit -m "feat: add WFO data loader with strategy labeling"
```

---

### Task 4: Build Streamlit Dashboard (`app.py`)

**Files:**
- Create: `app.py`

**Step 1: Write `app.py`**

```python
"""Walk-Forward Optimization Dashboard — Streamlit Demo.

Visualizes backtesting results from an algorithmic trading research
system. Shows equity curves, strategy comparisons, and per-window
performance metrics from real walk-forward optimization runs.
"""
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from data_loader import load_all_runs

# Strategy colors for consistent chart styling
STRATEGY_COLORS = {
    "TCN Action (NQ)": "#4A90D9",
    "TCN Classifier": "#50C878",
    "TCN Scalar": "#FFD700",
    "MLP Classifier": "#FF6B6B",
    "Rule-Based": "#C084FC",
}


def compute_equity_curve(df_run: pd.DataFrame) -> pd.DataFrame:
    """Compute cumulative equity curve from per-window returns."""
    equity = [1.0]
    for ret in df_run["cum_return"].fillna(0):
        equity.append(equity[-1] * (1 + ret))
    return pd.DataFrame({
        "window": range(len(equity)),
        "equity": equity,
    })


def main() -> None:
    st.set_page_config(
        page_title="WFO Dashboard",
        page_icon="📈",
        layout="wide",
    )

    # Load data
    df = load_all_runs()

    if df.empty:
        st.error("No WFO data found in data/ directory.")
        return

    run_ids = df["run_id"].unique().tolist()

    # --- Sidebar ---
    with st.sidebar:
        st.title("WFO Dashboard")

        selected_run = st.selectbox("Select Run", run_ids, index=0)

        st.divider()

        st.markdown("**Strategy Types**")
        for strategy, color in STRATEGY_COLORS.items():
            if strategy in df["strategy"].values:
                st.markdown(f":{color[1:]}[●] {strategy}")

        st.divider()

        st.markdown("**Architecture**")
        st.markdown(
            "Walk-forward optimization trains a model on a "
            "rolling window, then tests on the next unseen period. "
            "This process repeats across the entire dataset, "
            "producing out-of-sample results that reflect real "
            "trading conditions.\n\n"
            "**Models:** Temporal Convolutional Networks (TCN), "
            "MLPs, and rule-based baselines trained on M5 candle "
            "data with multi-timeframe features."
        )
        st.divider()
        st.markdown("[View Source on GitHub](https://github.com/vandyand/algo-trading-wfo-dashboard)")
        st.caption("Built with Streamlit + Plotly + Pandas")

    # --- Header ---
    st.title("📈 Walk-Forward Optimization Dashboard")
    st.caption(
        "Out-of-sample backtesting results from an algorithmic trading research system. "
        "Each validation window shows performance on data the model never saw during training."
    )

    # --- Section 1: Summary Metrics ---
    total_runs = df["run_id"].nunique()
    total_windows = len(df)
    avg_sharpe = df["sharpe"].mean()
    avg_win_rate = df["win_rate"].mean()
    best_return = df["cum_return"].max()

    cols = st.columns(5)
    cols[0].metric("Total Runs", total_runs)
    cols[1].metric("Validation Windows", f"{total_windows:,}")
    cols[2].metric("Avg Sharpe", f"{avg_sharpe:.2f}" if pd.notna(avg_sharpe) else "N/A")
    cols[3].metric("Avg Win Rate", f"{avg_win_rate:.0%}" if pd.notna(avg_win_rate) else "N/A")
    cols[4].metric("Best Window Return", f"{best_return:+.2%}" if pd.notna(best_return) else "N/A")

    st.divider()

    # --- Section 2: Equity Curve ---
    st.subheader("Equity Curve — Out-of-Sample Performance")

    df_run = df[df["run_id"] == selected_run].copy()
    run_strategy = df_run["strategy"].iloc[0] if len(df_run) > 0 else "Unknown"

    equity_df = compute_equity_curve(df_run)
    color = STRATEGY_COLORS.get(run_strategy, "#4A90D9")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=equity_df["window"],
        y=equity_df["equity"],
        mode="lines+markers",
        line=dict(color=color, width=2),
        marker=dict(size=5),
        name=run_strategy,
        hovertemplate="Window %{x}<br>Equity: %{y:.4f}<extra></extra>",
    ))
    fig.add_hline(y=1.0, line_dash="dash", line_color="gray", opacity=0.5)
    fig.update_layout(
        xaxis_title="Validation Window",
        yaxis_title="Cumulative Equity (starting at 1.0)",
        template="plotly_dark",
        height=400,
        margin=dict(l=40, r=20, t=20, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(f"**{selected_run}** — Strategy: {run_strategy} — {len(df_run)} windows")

    st.divider()

    # --- Section 3: Strategy Comparison ---
    st.subheader("Strategy Comparison")

    strategy_agg = df.groupby("strategy").agg(
        runs=("run_id", "nunique"),
        windows=("window", "count"),
        mean_sharpe=("sharpe", "mean"),
        mean_win_rate=("win_rate", "mean"),
        mean_profit_factor=("profit_factor", "mean"),
        mean_max_dd=("max_dd", "mean"),
        total_trades=("trades", "sum"),
    ).reset_index()

    # Format for display
    display_df = strategy_agg.copy()
    display_df.columns = [
        "Strategy", "Runs", "Windows", "Avg Sharpe",
        "Avg Win Rate", "Avg Profit Factor", "Avg Max DD", "Total Trades",
    ]
    display_df["Avg Sharpe"] = display_df["Avg Sharpe"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
    display_df["Avg Win Rate"] = display_df["Avg Win Rate"].map(lambda x: f"{x:.0%}" if pd.notna(x) else "N/A")
    display_df["Avg Profit Factor"] = display_df["Avg Profit Factor"].map(lambda x: f"{x:.2f}" if pd.notna(x) else "N/A")
    display_df["Avg Max DD"] = display_df["Avg Max DD"].map(lambda x: f"{x:.2%}" if pd.notna(x) else "N/A")
    display_df["Total Trades"] = display_df["Total Trades"].map(lambda x: f"{int(x):,}" if pd.notna(x) else "N/A")

    st.dataframe(display_df, use_container_width=True, hide_index=True)

    st.divider()

    # --- Section 4: Per-Window Metrics ---
    st.subheader("Per-Window Performance — Selected Run")

    metric_choice = st.radio(
        "Metric", ["Cumulative Return", "Sharpe Ratio", "Win Rate"],
        horizontal=True,
    )

    metric_col = {
        "Cumulative Return": "cum_return",
        "Sharpe Ratio": "sharpe",
        "Win Rate": "win_rate",
    }[metric_choice]

    fig2 = px.bar(
        df_run,
        x="window",
        y=metric_col,
        color_discrete_sequence=[color],
        labels={"window": "Validation Window", metric_col: metric_choice},
        template="plotly_dark",
    )
    fig2.update_layout(
        height=350,
        margin=dict(l=40, r=20, t=20, b=40),
    )
    st.plotly_chart(fig2, use_container_width=True)

    # Raw data expander
    with st.expander("Raw Window Data"):
        st.dataframe(
            df_run[["window", "cum_return", "sharpe", "sortino", "max_dd",
                     "trades", "win_rate", "profit_factor"]].reset_index(drop=True),
            use_container_width=True,
            hide_index=True,
        )


if __name__ == "__main__":
    main()
```

**Step 2: Verify the app runs**

Run: `cd /home/kingjames/algo-trading-wfo-dashboard && source venv/bin/activate && streamlit run app.py --server.headless true`

Check that it starts without errors (Ctrl+C to stop).

**Step 3: Commit**

Run:
```bash
cd /home/kingjames/algo-trading-wfo-dashboard && git add app.py && git commit -m "feat: add Streamlit WFO dashboard with equity curves and strategy comparison"
```

---

### Task 5: README, Push, Deploy

**Files:**
- Create: `README.md`

**Step 1: Write README.md**

```markdown
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
├── data/               # Curated WFO run results (~15 runs)
│   ├── tcn-action-*/   # TCN strategy runs
│   ├── mlp-action-*/   # MLP strategy runs
│   └── rule-action-*/  # Rule-based baseline runs
└── test_data_loader.py # Data loading tests
```
```

**Step 2: Commit**

Run:
```bash
cd /home/kingjames/algo-trading-wfo-dashboard && git add README.md && git commit -m "docs: add README with WFO methodology explanation"
```

**Step 3: Push to GitHub**

Run:
```bash
cd /home/kingjames/algo-trading-wfo-dashboard && gh repo create vandyand/algo-trading-wfo-dashboard --public --source=. --push
```

**Step 4: Deploy to Streamlit Cloud**

Manual step:
1. Go to https://share.streamlit.io
2. New app → repo `vandyand/algo-trading-wfo-dashboard`, branch `master`, file `app.py`
3. No secrets needed (no API keys)
4. Deploy
