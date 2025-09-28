#!/usr/bin/env python3
"""Grid-search stop multipliers to find the best short setup."""

from __future__ import annotations

import argparse
from datetime import time
from pathlib import Path
from typing import List, Tuple

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
    parser = argparse.ArgumentParser(description="Optimize stop-loss multiple for the short strategy.")
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
        "--take-profit",
        type=float,
        default=analysis.DEFAULT_TAKE_PROFIT,
        help=(
            "Take-profit multiple of entry (for shorts). Use 1.0 to disable. Defaults to analyzer setting."
        ),
    )
    parser.add_argument(
        "--stop-min",
        type=float,
        default=1.1,
        help="Minimum stop multiplier to test (default: 1.1).",
    )
    parser.add_argument(
        "--stop-max",
        type=float,
        default=3.0,
        help="Maximum stop multiplier to test (default: 3.0).",
    )
    parser.add_argument(
        "--stop-step",
        type=float,
        default=0.05,
        help="Step size for stop multiplier sweep (default: 0.1).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to save the sweep results as CSV.",
    )
    return parser.parse_args(argv)


def sweep_stops(
    df: pd.DataFrame,
    stop_values: List[float],
    take_profit: float,
) -> List[Tuple[float, dict]]:
    results: List[Tuple[float, dict]] = []
    for stop in stop_values:
        stats = analysis.compute_stats(df.copy(), stop, take_profit)
        stats["stop_multiplier"] = stop
        results.append((stop, stats))
    return results


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv)

    entry_cutoff = parse_time(args.entry_cutoff)
    df = analysis.load_events(args.events)
    df = analysis.apply_filters(df, entry_cutoff, args.min_volume, args.min_entry_price)
    if df.empty:
        print("No trades available after filters; nothing to optimize.")
        return 0

    stops = []
    current = args.stop_min
    while current <= args.stop_max + 1e-9:
        stops.append(round(current, 10))
        current += args.stop_step

    summary_rows = []
    best_row = None
    best_expectancy = float("-inf")

    for stop, stats in sweep_stops(df, stops, args.take_profit):
        row = {
            "stop_multiplier": stop,
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
        print(f"Saved stop sweep results to {args.output}")

    results_df.sort_values(by="expected_value_per_$1", ascending=False, inplace=True)
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:0.4f}"))

    if best_row:
        print("\nBest stop multiplier:")
        print(best_row)
    else:
        print("\nUnable to determine the best stop multiplier (check dataset/filters).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
