#!/usr/bin/env python3
"""Analyze 40% move events exported by backtest_40pct_moves.py."""

from __future__ import annotations

import argparse
from datetime import time
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

DEFAULT_EVENTS_CSV = Path("output/40pct_moves.csv")
DEFAULT_STOP_MULTIPLIER = 1.9 # Matches backtest STOP_LOSS_MULTIPLIER; adjust if dataset differs.
DEFAULT_TAKE_PROFIT = 0.7  # For shorts: 0.7 locks a 30% gain; set to 1.0 to disable.
DEFAULT_ENTRY_CUTOFF = "14:30"  # Use None to disable time cutoff.


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize buy-at-40%-and-hold trades.")
    parser.add_argument(
        "--events",
        type=Path,
        default=DEFAULT_EVENTS_CSV,
        help="CSV exported by backtest_40pct_moves.py (default: output/40pct_moves.csv).",
    )
    parser.add_argument(
        "--entry-cutoff",
        type=str,
        default=DEFAULT_ENTRY_CUTOFF,
        help=f"Only include trades whose entry_time_et is <= HH:MM (default: {DEFAULT_ENTRY_CUTOFF}). Use 'none' to disable.",
    )
    parser.add_argument(
        "--stop-multiplier",
        type=float,
        default=DEFAULT_STOP_MULTIPLIER,
        help=f"Stop distance as multiple of entry price for short trades (default: {DEFAULT_STOP_MULTIPLIER}).",
    )
    parser.add_argument(
        "--take-profit",
        type=float,
        default=DEFAULT_TAKE_PROFIT,
        help=f"Take-profit multiple of entry price for shorts (default: {DEFAULT_TAKE_PROFIT}; set <1 for active TP).",
    )
    parser.add_argument(
        "--min-volume",
        type=float,
        default=None,
        help="Drop trades with 30-minute window volume below this value.",
    )
    parser.add_argument(
        "--min-entry-price",
        type=float,
        default=None,
        help="Drop trades with entry price below this value.",
    )
    return parser.parse_args(argv)


def load_events(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Event file not found: {path}")
    df = pd.read_csv(path)
    numeric_cols = [
        "start_price",
        "trigger_price",
        "entry_price",
        "close_price",
        "exit_price",
        "post_entry_high",
        "post_entry_low",
        "max_gain_pct",
        "close_gain_pct",
        "window_volume",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df["entry_time_et"] = pd.to_datetime(df.get("entry_time_et"), format="%H:%M", errors="coerce").dt.time
    df["exit_time_et"] = pd.to_datetime(df.get("exit_time_et"), format="%H:%M", errors="coerce").dt.time
    df["trigger_time_et"] = pd.to_datetime(df.get("trigger_time_et"), format="%H:%M", errors="coerce").dt.time
    df["start_time_et"] = pd.to_datetime(df.get("start_time_et"), format="%H:%M", errors="coerce").dt.time
    df["date"] = pd.to_datetime(df.get("date"), errors="coerce").dt.date
    if "stop_triggered" in df.columns:
        df["stop_triggered"] = df["stop_triggered"].astype(str).str.lower().isin(["true", "1", "yes"])
    df = df.dropna(subset=["entry_price", "exit_price", "entry_time_et"])
    return df


def apply_filters(
    df: pd.DataFrame,
    entry_cutoff: Optional[time],
    min_volume: Optional[float],
    min_entry_price: Optional[float],
) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    if entry_cutoff:
        mask &= df["entry_time_et"] <= entry_cutoff
    if min_volume is not None and "window_volume" in df:
        mask &= df["window_volume"] >= min_volume
    if min_entry_price is not None:
        mask &= df["entry_price"] >= min_entry_price
    return df.loc[mask].copy()


def compute_stats(
    df: pd.DataFrame,
    stop_multiplier: float,
    take_profit_multiplier: float,
) -> Dict[str, float]:
    exit_prices = df["exit_price"].copy()
    # NOTE: When tightening the stop (multiplier <= recorded exit/entry),
    # we approximate the new exit as entry_price * stop_multiplier.
    # Loosening the stop is not inferred from the dataset and keeps the recorded exit.
    stop_mask = pd.Series(False, index=df.index)
    if stop_multiplier > 0 and "post_entry_high" in df:
        stop_price = df["entry_price"] * stop_multiplier
        stop_mask = df["post_entry_high"].notna() & (df["post_entry_high"] >= stop_price)
        exit_prices = exit_prices.where(~stop_mask, stop_price)
    take_profit_mask = pd.Series(False, index=df.index)
    if 0 < take_profit_multiplier < 1 and "post_entry_low" in df:
        target_price = df["entry_price"] * take_profit_multiplier
        feasible = df["post_entry_low"].notna() & (df["post_entry_low"] <= target_price) & ~stop_mask
        take_profit_mask = feasible
        exit_prices = exit_prices.where(~take_profit_mask, target_price)
    returns = (df["entry_price"] - exit_prices) / df["entry_price"]
    df["holding_return_pct"] = returns * 100.0
    stats = {
        "trades": int(len(df)),
        "hit_rate": float((df["holding_return_pct"] > 0).mean()) if len(df) else float("nan"),
        "stop_rate": float(stop_mask.mean()) if len(df) else float("nan"),
        "take_profit_rate": float(take_profit_mask.mean()) if len(df) else float("nan"),
        "avg_return_pct": float(df["holding_return_pct"].mean()) if len(df) else float("nan"),
        "median_return_pct": float(df["holding_return_pct"].median()) if len(df) else float("nan"),
        "std_return_pct": float(df["holding_return_pct"].std(ddof=1)) if len(df) > 1 else float("nan"),
        "expected_value_per_$1": float(returns.mean()) if len(df) else float("nan"),
    }
    return stats


def print_report(df: pd.DataFrame, stop_multiplier: float, take_profit_multiplier: float) -> None:
    stats = compute_stats(df, stop_multiplier, take_profit_multiplier)
    print("=== Strategy Summary ===")
    for key, value in stats.items():
        if key.endswith("pct"):
            print(f"{key:>22}: {value:.2f}%")
        elif key in ("hit_rate", "stop_rate", "take_profit_rate"):
            print(f"{key:>22}: {value:.2%}")
        else:
            print(f"{key:>22}: {value}")
    if not len(df):
        return
    quantiles = df["holding_return_pct"].quantile([0.1, 0.25, 0.5, 0.75, 0.9])
    print("\nPercentiles (% return from entry to exit):")
    for q, val in quantiles.items():
        print(f"  {q:>4.0%}: {val:6.2f}%")


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    entry_cutoff_arg: Optional[time | str] = args.entry_cutoff
    if isinstance(entry_cutoff_arg, str):
        lower = entry_cutoff_arg.lower()
        if lower in {"none", "off", ""}:
            entry_cutoff = None
        else:
            try:
                entry_cutoff = time.fromisoformat(entry_cutoff_arg)
            except ValueError as exc:
                raise SystemExit(f"Invalid --entry-cutoff value: {args.entry_cutoff}") from exc
    else:
        entry_cutoff = entry_cutoff_arg
    df = load_events(args.events)
    df = apply_filters(df, entry_cutoff, args.min_volume, args.min_entry_price)
    print_report(df, args.stop_multiplier, args.take_profit)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
