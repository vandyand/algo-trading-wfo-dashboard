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
    val_period = record.get("val") or [None, None]
    row["val_start"] = val_period[0] if len(val_period) > 0 else None
    row["val_end"] = val_period[1] if len(val_period) > 1 else None

    # Metrics — handle null and Infinity
    for col in _METRIC_COLS:
        metric_val = record.get(col)
        if metric_val is None or (isinstance(metric_val, float) and (math.isinf(metric_val) or math.isnan(metric_val))):
            row[col] = None
        else:
            row[col] = float(metric_val)

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
