#!/usr/bin/env python3
"""Generate backside short events by scanning Polygon minute data."""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, time as dtime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set

try:
    from zoneinfo import ZoneInfo  # type: ignore
except ImportError:  # pragma: no cover
    ZoneInfo = None  # type: ignore

from src.polygon_client import PolygonError, api_get

EASTERN_TZ_NAME = "America/New_York"
DEFAULT_SESSION_START = dtime(4, 0)
DEFAULT_SESSION_END = dtime(16, 0)
DEFAULT_MONTHS = 1
DEFAULT_OUTPUT_PATH = Path("output/backside_short_events_1a.csv")
DEFAULT_RISE_THRESHOLDS = [
    1.50, 1.60, 1.70, 1.80, 1.90, 2.00, 2.10, 2.20, 2.30
]
DEFAULT_DROP_THRESHOLDS = [
    0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.24, 0.26, 0.28
]
DEFAULT_CANDIDATE_MIN_RISE = 0.50

HARD_CODED_API_KEY = "CEHW_iOpbogk0Dcjdh3bwXQvHMjzdMkP"


@dataclass
class MinuteBar:
    dt: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class BacksideEvent:
    ticker: str
    trade_date: date
    prev_close: float
    day_open_price: float
    rise_threshold: float
    drop_threshold: float
    rise_dt: datetime
    peak_dt: datetime
    drop_dt: datetime
    entry_dt: Optional[datetime]
    duration_minutes: int
    rise_price: float
    peak_price: float
    drop_price: float
    entry_price: Optional[float]
    close_price: float
    post_entry_high: Optional[float]
    post_entry_low: Optional[float]
    rise_pct: float
    drop_pct: float
    volume_since_rise: float


# ----- Utility helpers -----

def log(msg: str) -> None:
    print(msg, file=sys.stderr)


def months_ago(today: date, months: int) -> date:
    year = today.year
    month = today.month - months
    while month <= 0:
        month += 12
        year -= 1
    day = min(today.day, _days_in_month(year, month))
    return date(year, month, day)


def _days_in_month(year: int, month: int) -> int:
    if month in (1, 3, 5, 7, 8, 10, 12):
        return 31
    if month in (4, 6, 9, 11):
        return 30
    is_leap = year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
    return 29 if is_leap else 28


def daterange_weekdays(start: date, end: date) -> Iterable[date]:
    cur = start
    step = timedelta(days=1)
    while cur <= end:
        if cur.weekday() < 5:
            yield cur
        cur += step


def get_eastern_tz() -> timezone:
    if ZoneInfo is not None:
        try:
            return ZoneInfo(EASTERN_TZ_NAME)
        except Exception:  # pragma: no cover
            pass
    return timezone(timedelta(hours=-5))


# ----- Polygon fetch helpers -----

def polygon_call(path: str, params: Optional[Dict[str, object]] = None, *, retries: int = 4) -> Dict[str, object]:
    attempt = 0
    while True:
        try:
            return api_get(path, params)
        except PolygonError as exc:
            attempt += 1
            if attempt > retries:
                raise
            delay = 1.5 ** attempt
            log(f"Polygon error ({exc}); retrying in {delay:.1f}s")
            time.sleep(delay)


def fetch_grouped(day: date, adjusted: bool = True) -> List[Dict[str, object]]:
    path = f"/v2/aggs/grouped/locale/us/market/stocks/{day.isoformat()}"
    payload = polygon_call(path, {"adjusted": str(adjusted).lower()})
    results = payload.get("results")
    if not isinstance(results, list):
        return []
    return results


def fetch_minute_bars(
    ticker: str,
    day: date,
    adjusted: bool = True,
) -> List[Dict[str, object]]:
    safe_ticker = ticker.replace("/", "%2F")
    path = f"/v2/aggs/ticker/{safe_ticker}/range/1/minute/{day.isoformat()}/{day.isoformat()}"
    payload = polygon_call(
        path,
        {
            "adjusted": str(adjusted).lower(),
            "sort": "asc",
            "limit": 5000,
        },
    )
    results = payload.get("results")
    if not isinstance(results, list):
        return []
    return results


# ----- Data wrangling -----

def normalize_bars(
    raw_bars: Sequence[Dict[str, object]],
    session_start: dtime,
    session_end: dtime,
) -> List[MinuteBar]:
    tz = get_eastern_tz()
    normalized: List[MinuteBar] = []
    for item in raw_bars:
        try:
            ts = float(item.get("t")) / 1000.0
            dt_utc = datetime.fromtimestamp(ts, tz=timezone.utc)
        except Exception:
            continue
        dt_local = dt_utc.astimezone(tz)
        bar_time = dt_local.time()
        if bar_time < session_start or bar_time > session_end:
            continue
        try:
            bar = MinuteBar(
                dt=dt_local,
                open=float(item.get("o")),
                high=float(item.get("h")),
                low=float(item.get("l")),
                close=float(item.get("c")),
                volume=float(item.get("v")),
            )
        except (TypeError, ValueError):
            continue
        normalized.append(bar)
    normalized.sort(key=lambda b: b.dt)
    return normalized


def fetch_common_stock_tickers(active: Optional[bool] = True, *, max_pages: int = 1000) -> Set[str]:
    tickers: Set[str] = set()
    params: Dict[str, str] = {"market": "stocks", "type": "CS", "limit": "1000"}
    if active is True:
        params["active"] = "true"
    elif active is False:
        params["active"] = "false"
    cursor: Optional[str] = None
    for _ in range(max_pages):
        call_params = dict(params)
        if cursor:
            call_params["cursor"] = cursor
        payload = polygon_call("/v3/reference/tickers", call_params)
        results = payload.get("results")
        if not isinstance(results, list):
            break
        for item in results:
            ticker = item.get("ticker")
            if isinstance(ticker, str):
                tickers.add(ticker.upper())
        next_url = payload.get("next_url") or payload.get("next")
        if not next_url:
            break
        try:
            from urllib.parse import parse_qs, urlparse

            query = urlparse(str(next_url)).query
            cursor_vals = parse_qs(query).get("cursor") or []
            cursor = cursor_vals[0] if cursor_vals else None
        except Exception:
            cursor = None
        if not cursor:
            break
    return tickers

def load_common_stock_allowlist(include_inactive: bool) -> Set[str]:
    allowlist = fetch_common_stock_tickers(active=True)
    if include_inactive:
        try:
            inactive = fetch_common_stock_tickers(active=False)
        except PolygonError:
            inactive = set()
        allowlist.update(inactive)
    return allowlist


def extract_closes(grouped: Sequence[Dict[str, object]]) -> Dict[str, float]:
    closes: Dict[str, float] = {}
    for item in grouped:
        ticker = item.get("T")
        if not isinstance(ticker, str):
            continue
        try:
            close_price = float(item.get("c"))
        except (TypeError, ValueError):
            continue
        if close_price <= 0:
            continue
        closes[ticker.upper()] = close_price
    return closes


def filter_backside_candidates(
    grouped: Sequence[Dict[str, object]],
    prev_closes: Dict[str, float],
    min_rise_threshold: float,
    allowlist: Optional[Set[str]] = None,
) -> List[str]:
    cutoff_multiplier = 1.0 + min_rise_threshold
    candidates: List[str] = []
    for item in grouped:
        ticker = item.get("T")
        if not isinstance(ticker, str):
            continue
        ticker_key = ticker.upper()
        if allowlist is not None and ticker_key not in allowlist:
            continue
        prev_close = prev_closes.get(ticker_key)
        if not prev_close:
            continue
        try:
            day_high = float(item.get("h"))
        except (TypeError, ValueError):
            continue
        if day_high <= 0:
            continue
        if day_high / prev_close >= cutoff_multiplier:
            candidates.append(ticker)
    return candidates


def seed_previous_closes(current_day: date, max_lookback: int = 7) -> Dict[str, float]:
    cur = current_day - timedelta(days=1)
    attempts = 0
    while attempts < max_lookback:
        if cur.weekday() >= 5:
            cur -= timedelta(days=1)
            attempts += 1
            continue
        try:
            grouped = fetch_grouped(cur)
        except PolygonError:
            cur -= timedelta(days=1)
            attempts += 1
            continue
        closes = extract_closes(grouped)
        if closes:
            return closes
        cur -= timedelta(days=1)
        attempts += 1
    return {}


def detect_backside_events(
    ticker: str,
    trade_date: date,
    bars: Sequence[MinuteBar],
    *,
    prev_close: float,
    day_open: float,
    rise_thresholds: Sequence[float],
    drop_thresholds: Sequence[float],
) -> List[BacksideEvent]:
    if prev_close <= 0 or not bars:
        return []
    sorted_rises = sorted({thr for thr in rise_thresholds if thr > 0})
    sorted_drops = sorted({thr for thr in drop_thresholds if 0 < thr < 1})
    if not sorted_rises or not sorted_drops:
        return []
    session_close_bar = bars[-1]
    events: List[BacksideEvent] = []
    for rise in sorted_rises:
        target_price = prev_close * (1.0 + rise)
        rise_index: Optional[int] = None
        for idx, bar in enumerate(bars):
            if bar.high >= target_price:
                rise_index = idx
                break
        if rise_index is None:
            continue
        triggered_drops: Set[float] = set()
        max_high = bars[rise_index].high
        max_high_dt = bars[rise_index].dt
        volume_since_rise = 0.0
        for idx in range(rise_index, len(bars)):
            bar = bars[idx]
            volume_since_rise += bar.volume
            if bar.high > max_high:
                max_high = bar.high
                max_high_dt = bar.dt
            if max_high <= 0:
                continue
            for drop in sorted_drops:
                if drop in triggered_drops:
                    continue
                drop_target = max_high * (1.0 - drop)
                if drop_target <= 0:
                    continue
                if bar.low <= drop_target:
                    actual_drop_price = min(bar.low, drop_target)
                    entry_index = idx + 1
                    if entry_index >= len(bars):
                        triggered_drops.add(drop)
                        continue
                    entry_bar = bars[entry_index]
                    if entry_bar.open <= 0:
                        triggered_drops.add(drop)
                        continue
                    following_bars = bars[entry_index:]
                    post_entry_high = max((b.high for b in following_bars), default=None)
                    post_entry_low = min((b.low for b in following_bars), default=None)
                    duration_minutes = int(round((bar.dt - bars[rise_index].dt).total_seconds() / 60.0))
                    if duration_minutes < 0:
                        duration_minutes = 0
                    rise_price = bars[rise_index].high
                    peak_price = max_high
                    rise_pct = ((peak_price / prev_close) - 1.0) * 100.0 if prev_close > 0 else 0.0
                    drop_pct = ((peak_price - actual_drop_price) / peak_price) * 100.0 if peak_price > 0 else 0.0
                    events.append(
                        BacksideEvent(
                            ticker=ticker,
                            trade_date=trade_date,
                            prev_close=prev_close,
                            day_open_price=day_open,
                            rise_threshold=rise,
                            drop_threshold=drop,
                            rise_dt=bars[rise_index].dt,
                            peak_dt=max_high_dt,
                            drop_dt=bar.dt,
                            entry_dt=entry_bar.dt,
                            duration_minutes=duration_minutes,
                            rise_price=rise_price,
                            peak_price=peak_price,
                            drop_price=actual_drop_price,
                            entry_price=entry_bar.open,
                            close_price=session_close_bar.close,
                            post_entry_high=post_entry_high,
                            post_entry_low=post_entry_low,
                            rise_pct=rise_pct,
                            drop_pct=drop_pct,
                            volume_since_rise=volume_since_rise,
                        )
                    )
                    triggered_drops.add(drop)
            if len(triggered_drops) == len(sorted_drops):
                break
    return events


# ----- Output -----

def format_time_et(dt_value: Optional[datetime]) -> str:
    if dt_value is None:
        return ""
    tz = get_eastern_tz()
    return dt_value.astimezone(tz).strftime("%H:%M")


def write_results(path: Path, events: Sequence[BacksideEvent]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "date",
        "ticker",
        "prev_close",
        "day_open_price",
        "window_start_price",
        "rise_threshold_decimal",
        "drop_threshold_decimal",
        "rise_threshold_pct",
        "drop_threshold_pct",
        "window_start_time_et",
        "peak_time_et",
        "trigger_time_et",
        "entry_time_et",
        "minutes_rise_to_drop",
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
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for event in events:
            close_gain_pct = (
                ((event.drop_price / event.prev_close) - 1.0) * 100.0 if event.prev_close > 0 else 0.0
            )
            writer.writerow(
                {
                    "date": event.trade_date.isoformat(),
                    "ticker": event.ticker,
                    "prev_close": f"{event.prev_close:.4f}",
                    "day_open_price": f"{event.day_open_price:.4f}",
                    "window_start_price": f"{event.prev_close:.4f}",
                    "rise_threshold_decimal": f"{event.rise_threshold:.4f}",
                    "drop_threshold_decimal": f"{event.drop_threshold:.4f}",
                    "rise_threshold_pct": f"{event.rise_threshold * 100.0:.2f}",
                    "drop_threshold_pct": f"{event.drop_threshold * 100.0:.2f}",
                    "window_start_time_et": format_time_et(event.rise_dt),
                    "peak_time_et": format_time_et(event.peak_dt),
                    "trigger_time_et": format_time_et(event.drop_dt),
                    "entry_time_et": format_time_et(event.entry_dt),
                    "minutes_rise_to_drop": event.duration_minutes,
                    "rise_price": f"{event.rise_price:.4f}",
                    "peak_price": f"{event.peak_price:.4f}",
                    "trigger_price": f"{event.drop_price:.4f}",
                    "entry_price": f"{event.entry_price:.4f}" if event.entry_price is not None else "",
                    "close_price": f"{event.close_price:.4f}",
                    "post_entry_high": f"{event.post_entry_high:.4f}" if event.post_entry_high is not None else "",
                    "post_entry_low": f"{event.post_entry_low:.4f}" if event.post_entry_low is not None else "",
                    "rise_pct": f"{event.rise_pct:.2f}",
                    "drop_pct": f"{event.drop_pct:.2f}",
                    "window_max_gain_pct": f"{event.rise_pct:.2f}",
                    "window_close_gain_pct": f"{close_gain_pct:.2f}",
                    "window_volume": f"{event.volume_since_rise:.0f}",
                    "volume_since_rise": f"{event.volume_since_rise:.0f}",
                }
            )


# ----- CLI parsing -----# ----- CLI parsing -----

def parse_threshold_value(raw: str) -> float:
    token = raw.strip().rstrip("%")
    if not token:
        raise ValueError("Empty threshold value")
    value = float(token)
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
            start = parse_threshold_value(pieces[0])
            end = parse_threshold_value(pieces[1])
            step = parse_threshold_value(pieces[2]) if len(pieces) == 3 else 0.05
            if step <= 0:
                raise argparse.ArgumentTypeError("Range step must be positive.")
            current = start
            while current <= end + 1e-9:
                values.append(round(current, 6))
                current += step
        else:
            values.append(round(parse_threshold_value(token), 6))
    unique_sorted = sorted({v for v in values if v > 0})
    return unique_sorted or list(default)


def parse_time_arg(label: str, raw: str) -> dtime:
    try:
        return dtime.fromisoformat(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{label} expects HH:MM format.") from exc


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Detect backside short events where price first rallies off the prior close "
            "and then sells off by a configurable amount."
        )
    )
    parser.add_argument(
        "--months",
        type=int,
        default=DEFAULT_MONTHS,
        help="Number of months of history to scan (default: 24).",
    )
    parser.add_argument(
        "--rise-thresholds",
        type=str,
        default="",
        help=(
            "Rise thresholds as decimals (0.80 = 80%%). Use commas for explicit values or start:end:step for ranges."
        ),
    )
    parser.add_argument(
        "--drop-thresholds",
        type=str,
        default="",
        help=(
            "Drop thresholds as decimals (0.10 = 10%% decline). Use commas for explicit values or start:end:step for ranges."
        ),
    )
    parser.add_argument(
        "--candidate-min-rise",
        type=float,
        default=DEFAULT_CANDIDATE_MIN_RISE,
        help="Daily prefilter: minimum rise vs prior close to fetch minute bars (decimal, default: 0.50).",
    )
    parser.add_argument(
        "--session-start",
        type=str,
        default=DEFAULT_SESSION_START.strftime("%H:%M"),
        help="Earliest ET time to include (default: 04:00).",
    )
    parser.add_argument(
        "--session-end",
        type=str,
        default=DEFAULT_SESSION_END.strftime("%H:%M"),
        help="Latest ET time to include (default: 20:00).",
    )
    parser.add_argument(
        "--no-common-filter",
        action="store_true",
        help="Process all tickers without restricting to Polygon common stock universe.",
    )
    parser.add_argument(
        "--include-inactive-common",
        action="store_true",
        help="Include inactive CS tickers when building the common stock allowlist.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="Polygon API key (overrides environment variable).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help=f"CSV destination for the detected events (default: {DEFAULT_OUTPUT_PATH}).",
    )
    return parser.parse_args(argv)


# ----- Main -----

def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    rise_thresholds = expand_thresholds(args.rise_thresholds, DEFAULT_RISE_THRESHOLDS)
    drop_thresholds = expand_thresholds(args.drop_thresholds, DEFAULT_DROP_THRESHOLDS)
    if not rise_thresholds or not drop_thresholds:
        raise SystemExit("Need at least one rise and drop threshold.")

    try:
        session_start = parse_time_arg("--session-start", args.session_start)
        session_end = parse_time_arg("--session-end", args.session_end)
    except argparse.ArgumentTypeError as exc:
        raise SystemExit(str(exc))
    if session_end <= session_start:
        raise SystemExit("--session-end must be after --session-start")

    api_key = args.api_key or HARD_CODED_API_KEY
    if api_key:
        os.environ["POLYGON_API_KEY"] = api_key

    only_common = not args.no_common_filter
    include_inactive_common = args.include_inactive_common
    common_allowlist: Optional[Set[str]] = None
    if only_common:
        try:
            log("Loading Polygon common stock universe (type=CS)...")
            common_allowlist = load_common_stock_allowlist(include_inactive_common)
            log(f"  Loaded {len(common_allowlist)} symbols")
        except PolygonError as exc:
            log(f"Warning: failed to load common stock universe: {exc}")
            common_allowlist = None
            only_common = False

    today = date.today()
    start_date = months_ago(today, max(1, args.months))
    log(
        f"Scanning {start_date.isoformat()} to {today.isoformat()} with rise>={args.candidate_min_rise * 100:.1f}% "
        f"and drop>={min(drop_thresholds) * 100:.1f}% using {session_start.strftime('%H:%M')} - {session_end.strftime('%H:%M')} ET"
    )

    all_events: List[BacksideEvent] = []
    prev_day_closes: Dict[str, float] = {}
    trading_days = list(daterange_weekdays(start_date, today))
    for idx, day in enumerate(trading_days, start=1):
        log(f"[{idx}/{len(trading_days)}] {day.isoformat()} - loading grouped results")
        try:
            grouped = fetch_grouped(day)
        except PolygonError as exc:
            log(f"  Failed to fetch grouped data: {exc}")
            continue
        if not grouped:
            log("  No grouped data returned")
            continue
        if not prev_day_closes:
            prev_day_closes = seed_previous_closes(day)
            if not prev_day_closes:
                log("  Unable to seed previous closes; skipping day")
                prev_day_closes = extract_closes(grouped)
                continue
        candidates = filter_backside_candidates(
            grouped,
            prev_day_closes,
            args.candidate_min_rise,
            common_allowlist if only_common else None,
        )
        log(f"  Candidates: {len(candidates)}")
        for ticker in candidates:
            ticker_key = ticker.upper()
            prev_close = prev_day_closes.get(ticker_key)
            if not prev_close:
                continue
            try:
                raw_bars = fetch_minute_bars(ticker, day)
            except PolygonError as exc:
                log(f"    {ticker}: failed to fetch minute bars ({exc})")
                continue
            session_bars = normalize_bars(raw_bars, session_start, session_end)
            if not session_bars:
                continue
            day_open = session_bars[0].open
            events = detect_backside_events(
                ticker,
                day,
                session_bars,
                prev_close=prev_close,
                day_open=day_open,
                rise_thresholds=rise_thresholds,
                drop_thresholds=drop_thresholds,
            )
            if events:
                all_events.extend(events)
                log(f"    {ticker}: recorded {len(events)} event(s)")
        prev_day_closes = extract_closes(grouped)

    if not all_events:
        log("No qualifying backside events detected.")
        return 0

    write_results(args.output, all_events)
    log(f"Wrote {len(all_events)} events to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())









