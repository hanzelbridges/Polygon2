#!/usr/bin/env python3
"""Grid-search take-profit multipliers to find the best short setup."""

from __future__ import annotations

import argparse
from datetime import time
from pathlib import Path
from typing import List

import pandas as pd

import analyze_40pct_results as analysis


def parse_time(value: str | None) -> time | None:
    if value is None:
        return None
    lower = value.lower()
    if lower in {"", "none", "off"}:
        return None
    return time.fromisoformat(value)


def parse_args(argv: List[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize take-profit multiple for the short strategy.")
    parser.add_argument(
        "--events",
        type=Path,
        default=analysis.DEFAULT_EVENTS_CSV,
        help="Path to events CSV (default: output/40pct_moves.csv).",
    )
    parser.add_argument(
        "--entry-cutoff",
        type=str,
        default=analysis.DEFAULT_ENTRY_CUTOFF,
        help=(
            "Only include trades whose entry_time_et is <= HH:MM (default mirrors analyzer); "
            "use 'none' to keep all entries."
        ),
    )
    parser.add_argument(
        "--min-volume",
        type=float,
        default=None,
        help="Minimum window_volume to include (optional).",
    )
    parser.add_argument(
        "--min-entry-price",
        type=float,
        default=None,
        help="Minimum entry price to include (optional).",
    )
    parser.add_argument(
        "--stop-multiplier",
        type=float,
        default=analysis.DEFAULT_STOP_MULTIPLIER,
        help="Stop multiple to keep fixed while scanning take-profits (default mirrors analyzer).",
    )
    parser.add_argument(
        "--tp-min",
        type=float,
        default=0.1,
        help="Minimum take-profit multiple to test (default: 0.4).",
    )
    parser.add_argument(
        "--tp-max",
        type=float,
        default=1.0,
        help="Maximum take-profit multiple to test (default: 1.0).",
    )
    parser.add_argument(
        "--tp-step",
        type=float,
        default=0.05,
        help="Step size for take-profit sweep (default: 0.05).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to save the sweep results as CSV.",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)

    entry_cutoff = parse_time(args.entry_cutoff)
    df = analysis.load_events(args.events)
    df = analysis.apply_filters(df, entry_cutoff, args.min_volume, args.min_entry_price)
    if df.empty:
        print("No trades available after filters; nothing to optimize.")
        return 0

    tp_values = []
    current = args.tp_min
    while current <= args.tp_max + 1e-9:
        tp_values.append(round(current, 10))
        current += args.tp_step

    summary_rows = []
    best_row = None
    best_expectancy = float("-inf")

    for tp in tp_values:
        stats = analysis.compute_stats(df.copy(), args.stop_multiplier, tp)
        row = {
            "take_profit_multiplier": tp,
            "trades": stats.get("trades"),
            "hit_rate": stats.get("hit_rate"),
            "stop_rate": stats.get("stop_rate"),
            "take_profit_rate": stats.get("take_profit_rate"),
            "avg_return_pct": stats.get("avg_return_pct"),
            "median_return_pct": stats.get("median_return_pct"),
            "std_return_pct": stats.get("std_return_pct"),
            "expected_value_per_$1": stats.get("expected_value_per_$1"),
        }
        summary_rows.append(row)
        expectancy = stats.get("expected_value_per_$1", float("nan"))
        if expectancy is not None and expectancy > best_expectancy:
            best_expectancy = expectancy
            best_row = row

    results_df = pd.DataFrame(summary_rows)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(args.output, index=False)
        print(f"Saved take-profit sweep results to {args.output}")

    results_df.sort_values(by="expected_value_per_$1", ascending=False, inplace=True)
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:0.4f}"))

    if best_row:
        print("\nBest take-profit multiplier:")
        print(best_row)
    else:
        print("\nUnable to determine the best take-profit multiplier (check dataset/filters).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
