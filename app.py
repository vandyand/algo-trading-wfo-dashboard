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

# Strategy colors for Plotly charts
STRATEGY_COLORS = {
    "TCN Action (NQ)": "#4A90D9",
    "TCN Classifier": "#50C878",
    "TCN Scalar": "#FFD700",
    "MLP Classifier": "#FF6B6B",
    "Rule-Based": "#C084FC",
}

# Streamlit named colors for sidebar legend
_STRATEGY_ST_COLORS = {
    "TCN Action (NQ)": "blue",
    "TCN Classifier": "green",
    "TCN Scalar": "orange",
    "MLP Classifier": "red",
    "Rule-Based": "violet",
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

    # Load data (cached across reruns)
    @st.cache_data
    def load_data():
        return load_all_runs()

    df = load_data()

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
        for strategy in STRATEGY_COLORS:
            if strategy in df["strategy"].values:
                st_color = _STRATEGY_ST_COLORS.get(strategy, "gray")
                st.markdown(f":{st_color}[●] {strategy}")

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
