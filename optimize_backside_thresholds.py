#!/usr/bin/env python3
"""Optimize backside short rise/drop thresholds using generated event data."""

from __future__ import annotations

import argparse
from datetime import date, time
from pathlib import Path
from typing import Callable, List, Sequence

import pandas as pd

# Default configuration
DEFAULT_EVENTS_PATH = Path("output/backside_short_events_1b.csv")
DEFAULT_RISE_THRESHOLDS = [1.00, 1.10, 1.20, 1.30, 1.40, 1.50]
DEFAULT_DROP_THRESHOLDS = [0.10, 0.12, 0.14, 0.16, 0.18, 0.20]
DEFAULT_STOP_MULTIPLIER = 1.40
DEFAULT_TAKE_PROFIT = 1.00
DEFAULT_CAPITAL_PER_TRADE = 5000.0
DEFAULT_STOP_ITERATIONS = 25
DEFAULT_STOP_STEP_PCT = 5.0
DEFAULT_ENTRY_CUTOFF = "09:29"
DEFAULT_START_DATE = "2023-09-11"  # e.g. "2020-01-01"
DEFAULT_END_DATE = "2025-10-11"    # e.g. "2024-12-31"
DEFAULT_SORT_ORDER = "desc"  # "asc" or "desc"
DEFAULT_OPTIMIZE_BY = "total_pl"  # "ev_pct" or "total_pl"

NUMERIC_COLUMNS = [
    "prev_close",
    "day_open_price",
    "window_start_price",
    "rise_threshold_decimal",
    "drop_threshold_decimal",
    "rise_price",
    "peak_price",
    "trigger_price",
    "entry_price",
    "close_price",
    "post_entry_high",
    "post_entry_low",
    "rise_pct",
    "drop_pct",
    "window_max_gain_pct",
    "window_close_gain_pct",
    "window_volume",
    "volume_since_rise",
]

TIME_COLUMNS = (
    "rise_time_et",
    "peak_time_et",
    "drop_time_et",
    "trigger_time_et",
    "entry_time_et",
    "window_start_time_et",
)

DATE_COLUMNS = ("date",)


def load_events(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Event file not found: {path}")
    df = pd.read_csv(path)
    for column in NUMERIC_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    for column in TIME_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], format="%H:%M", errors="coerce").dt.time
    for column in DATE_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors="coerce").dt.date
    required = [col for col in ("entry_price", "close_price") if col in df.columns]
    if required:
        df = df.dropna(subset=required)
    return df


def apply_date_range(df: pd.DataFrame, start_date: date | None, end_date: date | None) -> pd.DataFrame:
    if start_date is None and end_date is None:
        return df
    mask = pd.Series(True, index=df.index)
    if "date" in df:
        dates = pd.to_datetime(df["date"], errors="coerce").dt.date
        if start_date is not None:
            mask &= dates >= start_date
        if end_date is not None:
            mask &= dates <= end_date
    return df.loc[mask].copy()


def apply_filters(
    df: pd.DataFrame,
    entry_cutoff: time | None,
    min_volume: float | None,
    min_entry_price: float | None,
) -> pd.DataFrame:
    mask = pd.Series(True, index=df.index)
    if entry_cutoff is not None and "entry_time_et" in df:
        mask &= df["entry_time_et"].notna() & (df["entry_time_et"] <= entry_cutoff)
    if min_volume is not None:
        volume_col = None
        for candidate in ("window_volume", "volume_since_rise"):
            if candidate in df.columns:
                volume_col = candidate
                break
        if volume_col:
            mask &= df[volume_col].fillna(0) >= float(min_volume)
    if min_entry_price is not None and "entry_price" in df:
        mask &= df["entry_price"].fillna(0) >= float(min_entry_price)
    return df.loc[mask].copy()


def compute_stats(
    df: pd.DataFrame,
    stop_multiplier: float,
    take_profit_multiplier: float,
) -> dict:
    working = df.copy()
    exit_prices = working["close_price"].copy()
    stop_mask = pd.Series(False, index=working.index)
    if stop_multiplier > 0 and "post_entry_high" in working:
        stop_price = working["entry_price"] * stop_multiplier
        stop_mask = working["post_entry_high"].notna() & (working["post_entry_high"] >= stop_price)
        exit_prices = exit_prices.where(~stop_mask, stop_price)
    take_profit_mask = pd.Series(False, index=working.index)
    if 0 < take_profit_multiplier < 1 and "post_entry_low" in working:
        target_price = working["entry_price"] * take_profit_multiplier
        feasible = working["post_entry_low"].notna() & (working["post_entry_low"] <= target_price) & ~stop_mask
        take_profit_mask = feasible
        exit_prices = exit_prices.where(~take_profit_mask, target_price)
    returns = (working["entry_price"] - exit_prices) / working["entry_price"]
    working["holding_return_pct"] = returns * 100.0
    wins = working["holding_return_pct"][working["holding_return_pct"] > 0]
    losses = working["holding_return_pct"][working["holding_return_pct"] < 0]
    avg_win_pct = float(wins.mean()) if len(wins) else float("nan")
    avg_loss_pct = float(losses.mean()) if len(losses) else float("nan")
    risk_reward = float("nan")
    if len(wins) and len(losses):
        denom = abs(avg_loss_pct)
        risk_reward = avg_win_pct / denom if denom else float("nan")
    stats = {
        "trades": int(len(working)),
        "win_rate": float((working["holding_return_pct"] > 0).mean()) if len(working) else float("nan"),
        "stop_rate": float(stop_mask.mean()) if len(working) else float("nan"),
        "take_profit_rate": float(take_profit_mask.mean()) if len(working) else float("nan"),
        "avg_return_pct": float(working["holding_return_pct"].mean()) if len(working) else float("nan"),
        "median_return_pct": float(working["holding_return_pct"].median()) if len(working) else float("nan"),
        "std_return_pct": float(working["holding_return_pct"].std(ddof=1)) if len(working) > 1 else float("nan"),
        "avg_win_pct": avg_win_pct,
        "avg_loss_pct": avg_loss_pct,
        "risk_reward": risk_reward,
        "expected_value_per_$1": float(returns.mean()) if len(working) else float("nan"),
    }
    return stats


def _parse_threshold_value(token: str) -> float:
    cleaned = token.strip().rstrip("%")
    if not cleaned:
        raise argparse.ArgumentTypeError("Empty threshold value")
    value = float(cleaned)
    if value >= 1.0:
        value /= 100.0
    return value


def expand_thresholds(raw: str, default: Sequence[float]) -> List[float]:
    if not raw:
        return list(default)
    values: List[float] = []
    for part in raw.split(","):
        token = part.strip()
        if not token:
            continue
        if ":" in token:
            pieces = [p.strip() for p in token.split(":") if p.strip()]
            if len(pieces) not in {2, 3}:
                raise argparse.ArgumentTypeError(
                    f"Invalid range token {token!r}; expected start:end[:step]."
                )
            start = _parse_threshold_value(pieces[0])
            end = _parse_threshold_value(pieces[1])
            step = _parse_threshold_value(pieces[2]) if len(pieces) == 3 else 0.05
            if step <= 0:
                raise argparse.ArgumentTypeError("Range step must be positive.")
            current = start
            while current <= end + 1e-9:
                values.append(round(current, 6))
                current += step
        else:
            values.append(round(_parse_threshold_value(token), 6))
    unique = sorted({v for v in values if v > 0})
    return unique or list(default)


def parse_time_arg(value: str | None) -> time | None:
    if value is None:
        return None
    lower = value.strip().lower()
    if lower in {"", "none", "off"}:
        return None
    try:
        return time.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid time value: {value!r}") from exc


def parse_thresholds_arg(raw: str, *, default: Sequence[float]) -> List[float]:
    return expand_thresholds(raw, default)


def parse_date_arg(raw: str | None) -> date | None:
    if raw is None:
        return None
    lower = raw.strip().lower()
    if lower in {"", "none", "off"}:
        return None
    try:
        return date.fromisoformat(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid date value: {raw!r}") from exc


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate backside short performance across rise/drop thresholds using stop/TP settings.",
    )
    parser.add_argument(
        "--events",
        type=Path,
        default=DEFAULT_EVENTS_PATH,
        help=f"Path to backside event CSV (default: {DEFAULT_EVENTS_PATH}).",
    )
    parser.add_argument(
        "--rise-values",
        type=str,
        default="",
        help="Override rise thresholds (e.g. '0.8,1.0,1.2' or '80,120').",
    )
    parser.add_argument(
        "--drop-values",
        type=str,
        default="",
        help="Override drop thresholds (e.g. '0.1:0.3:0.05').",
    )
    parser.add_argument(
        "--entry-cutoff",
        type=str,
        default=DEFAULT_ENTRY_CUTOFF,
        help="Only include trades entered by HH:MM ET (use 'none' to disable).",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        default=DEFAULT_START_DATE,
        help="Earliest trade date to include (YYYY-MM-DD, optional).",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default=DEFAULT_END_DATE,
        help="Latest trade date to include (YYYY-MM-DD, optional).",
    )
    parser.add_argument(
        "--min-volume",
        type=float,
        default=None,
        help="Minimum volume metric to include (optional).",
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
        default=DEFAULT_STOP_MULTIPLIER,
        help="Base stop multiple to apply when computing stats.",
    )
    parser.add_argument(
        "--stop-iterations",
        type=int,
        default=DEFAULT_STOP_ITERATIONS,
        help="Number of times to rerun the sweep, scaling the stop each pass (default: 1).",
    )
    parser.add_argument(
        "--stop-step-pct",
        type=float,
        default=DEFAULT_STOP_STEP_PCT,
        help="Percent increase applied to the stop after each iteration (default: 5.0).",
    )
    parser.add_argument(
        "--take-profit",
        type=float,
        default=DEFAULT_TAKE_PROFIT,
        help="Take-profit multiple (set 1.0 to disable).",
    )
    parser.add_argument(
        "--capital-per-trade",
        type=float,
        default=DEFAULT_CAPITAL_PER_TRADE,
        help="Dollar amount per trade when translating EV into expected P/L (default: 5000).",
    )
    parser.add_argument(
        "--optimize-by",
        choices=["ev_pct", "total_pl"],
        default=DEFAULT_OPTIMIZE_BY,
        help="Sort combos by expected value percentage or total expected P/L (default: ev_pct).",
    )
    parser.add_argument(
        "--sort-order",
        choices=["asc", "desc"],
        default=DEFAULT_SORT_ORDER,
        help="Sort order for the optimization metric (default: desc).",
    )
    parser.add_argument(
        "--min-trades",
        type=int,
        default=5,
        help="Skip combinations with fewer than this many trades (default: 5).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional CSV path to write the full optimization table.",
    )
    return parser.parse_args(argv)


def compute_summary(
    df: pd.DataFrame,
    rise_values: Sequence[float],
    drop_values: Sequence[float],
    stop_multiplier: float,
    take_profit: float,
    capital_per_trade: float,
    min_trades: int,
    progress_callback: Callable[[int], None] | None = None,
) -> pd.DataFrame:
    rows: List[dict] = []
    for rise in rise_values:
        rise_mask = df["rise_threshold_decimal"].sub(rise).abs() <= 1e-6
        if not rise_mask.any():
            continue
        subset_by_rise = df[rise_mask]
        for drop in drop_values:
            if progress_callback is not None:
                progress_callback(1)
            drop_mask = subset_by_rise["drop_threshold_decimal"].sub(drop).abs() <= 1e-6
            subset = subset_by_rise[drop_mask]
            trades = int(len(subset))
            if trades < min_trades:
                continue
            stats = compute_stats(subset.copy(), stop_multiplier, take_profit)
            expectancy = stats.get("expected_value_per_$1")
            valid_expectancy = expectancy is not None and not pd.isna(expectancy)
            total_pl = expectancy * capital_per_trade * trades if valid_expectancy else float("nan")
            ev_pct = expectancy * 100.0 if valid_expectancy else float("nan")
            stop_rate = stats.get("stop_rate")
            stop_hits = float("nan")
            if stop_rate is not None and not pd.isna(stop_rate):
                stop_hits = int(round(stop_rate * trades))
            rows.append(
                {
                    "stop_multiplier": stop_multiplier,
                    "rise_threshold": rise,
                    "drop_threshold": drop,
                    "trades": trades,
                    "win_rate": stats.get("win_rate"),
                    "stop_rate": stop_rate,
                    "stopped_trades": stop_hits,
                    "avg_return_pct": stats.get("avg_return_pct"),
                    "median_return_pct": stats.get("median_return_pct"),
                    "std_return_pct": stats.get("std_return_pct"),
                    "avg_win_pct": stats.get("avg_win_pct"),
                    "avg_loss_pct": stats.get("avg_loss_pct"),
                    "risk_reward": stats.get("risk_reward"),
                    "expected_value_per_$1": expectancy,
                    "expected_value_pct": ev_pct,
                    "total_expected_pl": total_pl,
                }
            )
    return pd.DataFrame(rows)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)

    entry_cutoff = parse_time_arg(args.entry_cutoff)
    start_date = parse_date_arg(args.start_date)
    end_date = parse_date_arg(args.end_date)

    df = load_events(args.events)
    df = apply_date_range(df, start_date, end_date)
    if df.empty:
        print("Dataset is empty or missing required columns.")
        return 0

    df["rise_threshold_decimal"] = pd.to_numeric(df.get("rise_threshold_decimal"), errors="coerce")
    df["drop_threshold_decimal"] = pd.to_numeric(df.get("drop_threshold_decimal"), errors="coerce")
    df = df.dropna(subset=["rise_threshold_decimal", "drop_threshold_decimal"])

    df = apply_filters(df, entry_cutoff, args.min_volume, args.min_entry_price)
    if df.empty:
        print("No trades left after applying filters.")
        return 0

    rise_values = (
        parse_thresholds_arg(args.rise_values, default=DEFAULT_RISE_THRESHOLDS)
        if args.rise_values
        else sorted(df["rise_threshold_decimal"].dropna().unique().tolist())
    )
    drop_values = (
        parse_thresholds_arg(args.drop_values, default=DEFAULT_DROP_THRESHOLDS)
        if args.drop_values
        else sorted(df["drop_threshold_decimal"].dropna().unique().tolist())
    )

    if not rise_values or not drop_values:
        print("No thresholds available to optimize.")
        return 0

    iterations = max(1, int(args.stop_iterations or 1))
    if args.stop_multiplier <= 0:
        raise SystemExit("--stop-multiplier must be positive.")
    if args.stop_step_pct < 0:
        raise SystemExit("--stop-step-pct must be zero or positive.")

    stop_values: List[float] = []
    current_stop = args.stop_multiplier
    step_increment = args.stop_step_pct / 100.0
    for _ in range(iterations):
        stop_values.append(round(current_stop, 6))
        current_stop += step_increment

    total_combos = len(stop_values) * len(rise_values) * len(drop_values)
    if total_combos == 0:
        print("No rise/drop/stop combinations to evaluate.")
        return 0

    print(
        f"Evaluating {len(stop_values)} stop level(s) x {len(rise_values)} rise x {len(drop_values)} drop = {total_combos:,} combinations...",
        flush=True,
    )

    processed = 0
    progress_step = max(1, total_combos // 20)
    next_report = progress_step

    def progress_callback(count: int) -> None:
        nonlocal processed, next_report
        processed += max(0, count)
        if processed >= next_report or processed >= total_combos:
            print(f"  Processed {processed:,}/{total_combos:,} combinations", flush=True)
            next_report = min(total_combos, next_report + progress_step)

    summaries: List[pd.DataFrame] = []
    for stop_value in stop_values:
        partial = compute_summary(
            df,
            rise_values,
            drop_values,
            stop_value,
            args.take_profit,
            args.capital_per_trade,
            args.min_trades,
            progress_callback=progress_callback,
        )
        if not partial.empty:
            partial["stop_multiplier"] = stop_value
            summaries.append(partial)

    if not summaries:
        print("No combinations met the trade threshold; adjust --min-trades or input data.")
        return 0

    summary = pd.concat(summaries, ignore_index=True)

    if processed < total_combos:
        print(f"  Processed {processed:,}/{total_combos:,} combinations", flush=True)

    metric_field = "expected_value_pct" if args.optimize_by == "ev_pct" else "total_expected_pl"
    sort_ascending = args.sort_order == "asc"
    summary.sort_values(by=metric_field, ascending=sort_ascending, inplace=True)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        summary.to_csv(args.output, index=False)
        print(f"Saved optimization table to {args.output}")

    display = summary.copy()
    display["rise_pct"] = display["rise_threshold"] * 100.0
    display["drop_pct"] = display["drop_threshold"] * 100.0
    display["stop_multiplier"] = display.get("stop_multiplier")
    if "win_rate" in display.columns:
        display["win_rate_pct"] = display["win_rate"] * 100.0
    if "stop_rate" in display.columns:
        display["stop_rate_pct"] = display["stop_rate"] * 100.0
    cols = [
        "stop_multiplier",
        "rise_pct",
        "drop_pct",
        "trades",
        "win_rate_pct",
        "stop_rate_pct",
        "stopped_trades",
        "avg_return_pct",
        "expected_value_pct",
        "total_expected_pl",
    ]
    existing_cols = [c for c in cols if c in display.columns]
    print(display[existing_cols].head(200).to_string(index=False, float_format=lambda x: f"{x:0.4f}"))

    best_row = summary.iloc[0]
    rise_pct = best_row["rise_threshold"] * 100.0
    drop_pct = best_row["drop_threshold"] * 100.0
    best_stop = best_row.get("stop_multiplier")
    win_rate = best_row.get("win_rate")
    stop_rate = best_row.get("stop_rate")
    stop_hits = best_row.get("stopped_trades")
    avg_return = best_row.get("avg_return_pct")
    ev_per_dollar = best_row.get("expected_value_per_$1")
    ev_pct = best_row.get("expected_value_pct")
    total_pl = best_row.get("total_expected_pl")

    stop_display = "n/a" if pd.isna(best_stop) else f"{best_stop:0.3f}"
    win_rate_display = "n/a" if pd.isna(win_rate) else f"{win_rate:.2%}"
    stop_rate_display = "n/a" if pd.isna(stop_rate) else f"{stop_rate:.2%}"
    stop_hits_display = "n/a" if pd.isna(stop_hits) else f"{int(stop_hits)}"
    avg_return_display = "n/a" if pd.isna(avg_return) else f"{avg_return:0.2f}%"
    ev_per_dollar_display = "n/a" if pd.isna(ev_per_dollar) else f"{ev_per_dollar:0.4f}"
    ev_pct_display = "n/a" if pd.isna(ev_pct) else f"{ev_pct:0.2f}%"
    total_pl_display = "n/a" if pd.isna(total_pl) else f"${total_pl:0.2f}"

    print("\nBest combo:")
    print("-----------")
    print(f"  stop_multiplier: {stop_display}")
    print(f"  rise_threshold:  {rise_pct:0.2f}% | drop_threshold: {drop_pct:0.2f}%")
    print(f"  trades:         {int(best_row['trades'])}")
    print(f"  win_rate:       {win_rate_display}")
    print(f"  stop_rate:      {stop_rate_display} ({stop_hits_display} stops)")
    print(f"  avg_return:     {avg_return_display}")
    print(f"  EV per $1:      {ev_per_dollar_display}")
    print(f"  EV percent:     {ev_pct_display}")
    print(f"  Total P/L:      {total_pl_display}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


