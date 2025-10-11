#!/usr/bin/env python3
"""Grid-search stop-loss, take-profit, and entry cut-off combinations for the short strategy."""

from __future__ import annotations

import argparse
from datetime import date, datetime, time, timedelta
from itertools import product
from pathlib import Path
from typing import Dict, List, Sequence

import pandas as pd

# ===== User Config =====
# Adjust these defaults when running directly (CLI flags still override them).
# Set both min and max equal to lock a range to a single value; keep step positive.
DEFAULT_EVENTS_CSV = Path('output/100pct_moves_incl_pm_3y.csv')
ENABLE_STOP_SWEEP = True
ENABLE_TP_SWEEP = True
ENABLE_CUTOFF_SWEEP = True

FIXED_STOP_MULTIPLIER = 1.60
FIXED_TAKE_PROFIT_MULTIPLIER = 0.70
FIXED_ENTRY_CUTOFF = '16:00'

DEFAULT_STOP_RANGE = (1.40, 3.0, 0.10)   # stop sweep (min, max, step)
DEFAULT_TP_RANGE = (0.30, 1.00, 0.05)    # take-profit sweep (min, max, step)
DEFAULT_ENTRY_CUTOFF_START = '04:00'    # earliest entry time considered
DEFAULT_ENTRY_CUTOFF_END = '09:30'      # latest entry time considered
DEFAULT_CUTOFF_STEP_MINUTES = 30        # increment between entry cut-offs (minutes)
DEFAULT_CAPITAL_PER_TRADE = 5000.0      # dollars risked per trade when sizing total P/L
DEFAULT_OPTIMIZE_BY = 'total_pl'        # choose 'total_pl' or 'ev_pct'
DEFAULT_MIN_EV_PCT = -999.0                # drop combos whose EV% is below this
DEFAULT_TOP_RESULTS = 200                # number of rows printed in the summary table
DEFAULT_SORT_ORDER = 'desc'            # 'desc' (high-to-low) or 'asc' (low-to-high) sort order
DEFAULT_REQUIRE_ENTRY_AFTER_START = True  # True to drop trades before the cutoff start



import para_analysis as analysis
try:
    from colorama import Fore, Style, init as colorama_init

    colorama_init()
    EMPHASIS = Fore.MAGENTA + Style.BRIGHT
    RESET = Style.RESET_ALL
except Exception:  # colorama not available or init failed
    EMPHASIS = "\033[95;1m"
    RESET = "\033[0m"


def parse_time_arg(raw: str, flag: str) -> time:
    try:
        return time.fromisoformat(raw)
    except ValueError as exc:
        raise SystemExit(f"{flag} expects HH:MM (got {raw!r}).") from exc


def frange(start: float, stop: float, step: float) -> List[float]:
    values: List[float] = []
    current = start
    while current <= stop + 1e-9:
        values.append(round(current, 10))
        current += step
    return values


def generate_cutoff_times(start: time, end: time, step_minutes: int) -> List[time]:
    if step_minutes <= 0:
        raise SystemExit("--cutoff-step-minutes must be a positive integer.")

    start_dt = datetime.combine(date.today(), start)
    end_dt = datetime.combine(date.today(), end)
    if end_dt < start_dt:
        raise SystemExit("--cutoff-end must be equal to or after --cutoff-start.")

    delta = timedelta(minutes=step_minutes)
    times: List[time] = []
    current = start_dt
    while current <= end_dt + timedelta(seconds=1):  # inclusive upper bound
        times.append(current.time())
        current += delta
    return times


def format_time_label(value: time) -> str:
    return value.strftime("%H:%M")


def format_total_pl(value: float) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{int(round(value)):,}"


def format_expected_pct(value: float) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{value:0.2f}"


def format_target_value(target_key: str, value: float) -> str:
    if target_key == "total_pl":
        display = format_total_pl(value)
        return "n/a" if display == "n/a" else f"${display}"
    if target_key == "ev_pct":
        display = format_expected_pct(value)
        return "n/a" if display == "n/a" else f"{display}%"
    if value is None or pd.isna(value):
        return "n/a"
    return str(value)

def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = value.strip().lower()
    if lowered in {"true", "t", "yes", "y", "1", "on"}:
        return True
    if lowered in {"false", "f", "no", "n", "0", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value!r}")


OPTIMIZATION_TARGETS: Dict[str, Dict[str, str]] = {
    "total_pl": {"field": "total_expected_pl", "label": "Total expected P/L"},
    "ev_pct": {"field": "expected_value_pct", "label": "Expected value %"},
}

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Joint optimization of stop-loss, take-profit, and entry cut-off settings."
    )
    parser.add_argument(
        "--events",
        type=Path,
        default=DEFAULT_EVENTS_CSV,
        help="Path to events CSV (default: output/40pct_moves_incl_pm_3y.csv).",
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
        "--stop-min",
        type=float,
        default=DEFAULT_STOP_RANGE[0],
        help=f"Minimum stop multiplier to test (default: {DEFAULT_STOP_RANGE[0]}).",
    )
    parser.add_argument(
        "--stop-max",
        type=float,
        default=DEFAULT_STOP_RANGE[1],
        help=f"Maximum stop multiplier to test (default: {DEFAULT_STOP_RANGE[1]}).",
    )
    parser.add_argument(
        "--stop-step",
        type=float,
        default=DEFAULT_STOP_RANGE[2],
        help=f"Step size for stop multiplier sweep (default: {DEFAULT_STOP_RANGE[2]}).",
    )
    parser.add_argument(
        "--tp-min",
        type=float,
        default=DEFAULT_TP_RANGE[0],
        help=f"Minimum take-profit multiple to test (default: {DEFAULT_TP_RANGE[0]}).",
    )
    parser.add_argument(
        "--tp-max",
        type=float,
        default=DEFAULT_TP_RANGE[1],
        help=f"Maximum take-profit multiple to test (default: {DEFAULT_TP_RANGE[1]}).",
    )
    parser.add_argument(
        "--tp-step",
        type=float,
        default=DEFAULT_TP_RANGE[2],
        help=f"Step size for take-profit sweep (default: {DEFAULT_TP_RANGE[2]}).",
    )
    parser.add_argument(
        "--cutoff-start",
        type=str,
        default=DEFAULT_ENTRY_CUTOFF_START,
        help=f"Earliest entry cut-off (HH:MM ET, default: {DEFAULT_ENTRY_CUTOFF_START}).",
    )
    parser.add_argument(
        "--cutoff-end",
        type=str,
        default=DEFAULT_ENTRY_CUTOFF_END,
        help=f"Latest entry cut-off (HH:MM ET, default: {DEFAULT_ENTRY_CUTOFF_END}).",
    )
    parser.add_argument(
        "--cutoff-step-minutes",
        type=int,
        default=DEFAULT_CUTOFF_STEP_MINUTES,
        help=f"Step size in minutes between cut-offs (default: {DEFAULT_CUTOFF_STEP_MINUTES}).",
    )
    parser.add_argument(
        "--capital-per-trade",
        type=float,
        default=DEFAULT_CAPITAL_PER_TRADE,
        help=f"Dollar size allocated to each trade when computing total P/L (default: {DEFAULT_CAPITAL_PER_TRADE:.0f}).",
    )
    parser.add_argument(
        "--optimize-by",
        choices=sorted(OPTIMIZATION_TARGETS.keys()),
        default=DEFAULT_OPTIMIZE_BY,
        help=f"Metric to maximize when ranking combos (default: {DEFAULT_OPTIMIZE_BY}).",
    )
    parser.add_argument(
        "--min-ev-pct",
        type=float,
        default=DEFAULT_MIN_EV_PCT,
        help=f"Minimum expected value percentage a combination must achieve (default: {DEFAULT_MIN_EV_PCT}).",
    )
    parser.add_argument(
        "--sort-order",
        choices=["asc", "desc"],
        default=DEFAULT_SORT_ORDER,
        help=f"Sort order for results table (default: {DEFAULT_SORT_ORDER}).",
    )
    parser.add_argument(
        "--enforce-entry-after-start",
        type=parse_bool,
        default=DEFAULT_REQUIRE_ENTRY_AFTER_START,
        help=f"Require entry_time_et >= cutoff start? (default: {DEFAULT_REQUIRE_ENTRY_AFTER_START}).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path to save the grid results as CSV.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    df = analysis.load_events(args.events)
    df = analysis.apply_filters(df, None, args.min_volume, args.min_entry_price)
    if df.empty:
        print("No trades available after filters; nothing to optimize.")
        return 0

    cutoff_start = parse_time_arg(args.cutoff_start, "--cutoff-start")
    cutoff_end = parse_time_arg(args.cutoff_end, "--cutoff-end")
    if args.enforce_entry_after_start:
        df = df[df["entry_time_et"] >= cutoff_start]
        if df.empty:
            print("No trades remain after enforcing entry lower bound; nothing to optimize.")
            return 0
    if ENABLE_CUTOFF_SWEEP:
        cutoff_values = generate_cutoff_times(cutoff_start, cutoff_end, args.cutoff_step_minutes)
    else:
        fixed_cutoff_val = FIXED_ENTRY_CUTOFF or args.cutoff_end
        if isinstance(fixed_cutoff_val, time):
            fixed_cutoff = fixed_cutoff_val
        else:
            fixed_cutoff = parse_time_arg(str(fixed_cutoff_val), "FIXED_ENTRY_CUTOFF")
        cutoff_values = [fixed_cutoff]

    if ENABLE_STOP_SWEEP:
        stop_values = frange(args.stop_min, args.stop_max, args.stop_step)
    else:
        fallback_stop = FIXED_STOP_MULTIPLIER if FIXED_STOP_MULTIPLIER is not None else args.stop_min
        stop_values = [float(fallback_stop)]

    if ENABLE_TP_SWEEP:
        tp_values = frange(args.tp_min, args.tp_max, args.tp_step)
    else:
        fallback_tp = FIXED_TAKE_PROFIT_MULTIPLIER if FIXED_TAKE_PROFIT_MULTIPLIER is not None else args.tp_min
        tp_values = [float(fallback_tp)]

    if not stop_values or not tp_values or not cutoff_values:
        print("Parameter sweep produced no combinations; adjust the ranges.")
        return 0

    cutoff_frames: Dict[str, pd.DataFrame] = {}
    for cutoff in cutoff_values:
        label = format_time_label(cutoff)
        subset = df[df["entry_time_et"] <= cutoff].copy()
        cutoff_frames[label] = subset

    target_info = OPTIMIZATION_TARGETS[args.optimize_by]
    metric_field = target_info["field"]
    metric_label = target_info["label"]
    sort_ascending = args.sort_order == "asc"

    cutoff_labels = list(cutoff_frames.keys())
    total_combos = len(stop_values) * len(tp_values) * len(cutoff_labels)
    print(f"Evaluating {total_combos:,} combinations...", flush=True)

    summary_rows = []
    best_row = None
    best_metric = float("-inf")
    progress_step = max(1, total_combos // 20)

    min_ev_pct = args.min_ev_pct

    for idx, (stop, tp, cutoff_label) in enumerate(product(stop_values, tp_values, cutoff_labels), start=1):
        subset_df = cutoff_frames[cutoff_label]
        stats = analysis.compute_stats(subset_df.copy(), stop, tp)
        trades = stats.get("trades") or 0
        expectancy = stats.get("expected_value_per_$1")
        expectancy_valid = expectancy is not None and not pd.isna(expectancy)
        if trades and expectancy_valid:
            total_pl = expectancy * args.capital_per_trade * trades
        else:
            total_pl = 0.0
        expected_value_pct = expectancy * 100.0 if expectancy_valid else float("nan")

        if trades == 0:
            if idx % progress_step == 0 or idx == total_combos:
                print(f"  Processed {idx:,}/{total_combos:,} combinations", flush=True)
            continue

        if expected_value_pct < min_ev_pct:
            if idx % progress_step == 0 or idx == total_combos:
                print(f"  Processed {idx:,}/{total_combos:,} combinations", flush=True)
            continue

        row = {
            "stop_multiplier": stop,
            "take_profit_multiplier": tp,
            "entry_cutoff": cutoff_label,
            "trades": trades,
            "win_rate": stats.get("win_rate"),
            "stop_rate": stats.get("stop_rate"),
            "take_profit_rate": stats.get("take_profit_rate"),
            "avg_return_pct": stats.get("avg_return_pct"),
            "median_return_pct": stats.get("median_return_pct"),
            "std_return_pct": stats.get("std_return_pct"),
            "avg_win_pct": stats.get("avg_win_pct"),
            "avg_loss_pct": stats.get("avg_loss_pct"),
            "risk_reward": stats.get("risk_reward"),
            "expected_value_per_$1": expectancy,
            "expected_value_pct": expected_value_pct,
            "total_expected_pl": total_pl,
        }
        summary_rows.append(row)

        metric_value = row.get(metric_field)
        compare_value = metric_value
        if compare_value is None or pd.isna(compare_value):
            compare_value = float("-inf")
        if compare_value >= best_metric:
            best_metric = compare_value
            best_row = row

        if idx % progress_step == 0 or idx == total_combos:
            print(f"  Processed {idx:,}/{total_combos:,} combinations", flush=True)

    if not summary_rows:
        print(
            f"No combinations met the minimum EV% threshold of {min_ev_pct:.2f}. Try lowering --min-ev-pct."
        )
        return 0

    results_df = pd.DataFrame(summary_rows)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(args.output, index=False)
        print(f"Saved optimization grid to {args.output}")

    results_df.sort_values(by=metric_field, ascending=sort_ascending, inplace=True)
    print("")
    output_df = results_df.head(DEFAULT_TOP_RESULTS)
    formatters = {}
    if "total_expected_pl" in output_df.columns:
        formatters["total_expected_pl"] = format_total_pl
    if "expected_value_pct" in output_df.columns:
        formatters["expected_value_pct"] = format_expected_pct

    if not output_df.empty and EMPHASIS:
        lines = output_df.to_string(
            index=False,
            float_format=lambda x: f"{x:0.4f}",
            formatters=formatters or None,
        ).splitlines()
        for line_idx, line in enumerate(lines):
            if line_idx == 1:
                print(f"{EMPHASIS}{line}{RESET}")
            else:
                print(line)
    else:
        print(
            output_df.to_string(
                index=False,
                float_format=lambda x: f"{x:0.4f}",
                formatters=formatters or None,
            )
        )

    if best_row:
        metric_display = format_target_value(args.optimize_by, best_row.get(metric_field))
        print("\nBest combo:")
        print("-----------")
        print(f"  stop_multiplier:        {best_row['stop_multiplier']:.2f}")
        print(f"  take_profit_multiplier: {best_row['take_profit_multiplier']:.2f}")
        print(f"  entry_cutoff:           {best_row['entry_cutoff']}")
        print(f"  trades:                 {best_row['trades']}")
        print(f"  win_rate:               {best_row['win_rate']:.2%}")
        print(f"  avg_win_pct:            {best_row['avg_win_pct']:0.2f}%")
        print(f"  avg_loss_pct:           {best_row['avg_loss_pct']:0.2f}%")
        print(f"  risk_reward:            {best_row['risk_reward']:0.2f}")
        print(f"  expected_value_per_$1:  {best_row['expected_value_per_$1']:0.4f}")
        print(f"  optimized_metric:       {metric_label} = {metric_display}")
        total_display = format_total_pl(best_row['total_expected_pl'])
        if total_display == "n/a":
            print("  total_expected_pl:      n/a")
        else:
            print(f"  total_expected_pl:      ${total_display}")
        ev_display = format_expected_pct(best_row['expected_value_pct'])
        if ev_display == "n/a":
            print("  expected_value_pct:     n/a")
        else:
            print(f"  expected_value_pct:     {ev_display}%")
        print()
    else:
        print("\nUnable to determine best combination (check dataset/filters).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

