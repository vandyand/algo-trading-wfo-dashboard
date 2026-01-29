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
