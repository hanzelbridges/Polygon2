#!/usr/bin/env python3
"""Backtest 40%+ moves within a sliding 30-bar window using Polygon minute data."""

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
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover
    ZoneInfo = None

from src.polygon_client import PolygonError, api_get

EASTERN_TZ_NAME = "America/New_York"
SESSION_START = dtime(9, 30)
SESSION_END = dtime(16, 0)
DEFAULT_THRESHOLD = 0.40
DEFAULT_WINDOW = 30
DEFAULT_MONTHS = 1

HARD_CODED_API_KEY = ""  # Set your Polygon API key here to embed it in the script.


@dataclass
class MinuteBar:
    dt: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class IntradayMove:
    ticker: str
    trade_date: date
    start_dt: datetime
    trigger_dt: datetime
    entry_dt: Optional[datetime]
    exit_dt: Optional[datetime]
    duration_minutes: int
    start_price: float
    trigger_price: float
    entry_price: Optional[float]
    close_price: float
    exit_price: float
    post_entry_high: Optional[float]
    post_entry_low: Optional[float]
    stop_triggered: bool
    max_gain_pct: float
    close_gain_pct: float
    window_volume: float


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


def fetch_minute_bars(ticker: str, day: date, adjusted: bool = True) -> List[Dict[str, object]]:
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


def get_eastern_tz() -> timezone:
    if ZoneInfo is not None:
        try:
            return ZoneInfo(EASTERN_TZ_NAME)
        except Exception:  # pragma: no cover
            pass
    return timezone(timedelta(hours=-5))


def normalize_bars(raw_bars: Sequence[Dict[str, object]], include_premarket: bool) -> List[MinuteBar]:
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
        if not include_premarket and (bar_time < SESSION_START or bar_time > SESSION_END):
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


def filter_candidates(
    grouped: Sequence[Dict[str, object]],
    threshold: float,
    allowlist: Optional[Set[str]] = None,
) -> List[str]:
    tickers: List[str] = []
    cutoff = 1.0 + threshold
    for item in grouped:
        ticker = item.get("T")
        if not isinstance(ticker, str):
            continue
        if allowlist is not None and ticker.upper() not in allowlist:
            continue
        try:
            opening = float(item.get("o"))
            day_high = float(item.get("h"))
        except (TypeError, ValueError):
            continue
        if opening <= 0:
            continue
        if day_high / opening >= cutoff:
            tickers.append(ticker)
    return tickers


def detect_intraday_moves(
    ticker: str,
    trade_date: date,
    bars: Sequence[MinuteBar],
    *,
    threshold: float,
    window: int,
) -> List[IntradayMove]:
    if len(bars) < window:
        return []
    target_multiplier = 1.0 + threshold
    events: List[IntradayMove] = []
    idx = 0
    session_close_bar = bars[-1]
    while idx <= len(bars) - window:
        window_slice = bars[idx : idx + window]
        start_bar = window_slice[0]
        if start_bar.open <= 0:
            idx += 1
            continue
        target_price = start_bar.open * target_multiplier
        cumulative_volume = 0.0
        max_high = start_bar.open
        trigger_bar: Optional[MinuteBar] = None
        trigger_offset: Optional[int] = None
        for offset, bar in enumerate(window_slice):
            cumulative_volume += bar.volume
            if bar.high > max_high:
                max_high = bar.high
            if trigger_bar is None and bar.high >= target_price:
                trigger_bar = bar
                trigger_offset = offset
        if trigger_bar is not None and trigger_offset is not None:
            duration_minutes = int(round((trigger_bar.dt - start_bar.dt).total_seconds() / 60))
            if duration_minutes < 0:
                duration_minutes = 0
            close_gain_pct = ((window_slice[-1].close / start_bar.open) - 1.0) * 100.0
            max_gain_pct = ((max_high / start_bar.open) - 1.0) * 100.0
            entry_bar: Optional[MinuteBar] = None
            global_trigger_index = idx + trigger_offset
            if global_trigger_index + 1 < len(bars):
                entry_candidate = bars[global_trigger_index + 1]
                if entry_candidate.open > 0:
                    entry_bar = entry_candidate
            exit_dt: Optional[datetime] = session_close_bar.dt
            exit_price = session_close_bar.close
            post_entry_high: Optional[float] = None
            post_entry_low: Optional[float] = None
            stop_triggered = False
            if entry_bar is not None:
                trailing_high = entry_bar.high
                trailing_low = entry_bar.low
                for bar in bars[global_trigger_index + 1 :]:
                    if bar.high > trailing_high:
                        trailing_high = bar.high
                    if bar.low < trailing_low:
                        trailing_low = bar.low
                post_entry_high = trailing_high
                post_entry_low = trailing_low
            events.append(
                IntradayMove(
                    ticker=ticker,
                    trade_date=trade_date,
                    start_dt=start_bar.dt,
                    trigger_dt=trigger_bar.dt,
                    entry_dt=entry_bar.dt if entry_bar else None,
                    exit_dt=exit_dt,
                    duration_minutes=duration_minutes,
                    start_price=start_bar.open,
                    trigger_price=trigger_bar.high,
                    entry_price=entry_bar.open if entry_bar else None,
                    close_price=session_close_bar.close,
                    exit_price=exit_price,
                    post_entry_high=post_entry_high,
                    post_entry_low=post_entry_low,
                    stop_triggered=stop_triggered,
                    max_gain_pct=max_gain_pct,
                    close_gain_pct=close_gain_pct,
                    window_volume=cumulative_volume,
                )
            )
            idx += window
        else:
            idx += 1
    return events


def write_results(path: Path, events: Sequence[IntradayMove]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "date",
        "ticker",
        "start_time_et",
        "trigger_time_et",
        "entry_time_et",
        "exit_time_et",
        "minutes_to_trigger",
        "start_price",
        "trigger_price",
        "entry_price",
        "close_price",
        "exit_price",
        "post_entry_high",
        "post_entry_low",
        "stop_triggered",
        "max_gain_pct",
        "close_gain_pct",
        "window_volume",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for event in events:
            writer.writerow(
                {
                    "date": event.trade_date.isoformat(),
                    "ticker": event.ticker,
                    "start_time_et": event.start_dt.strftime("%H:%M"),
                    "trigger_time_et": event.trigger_dt.strftime("%H:%M"),
                    "entry_time_et": event.entry_dt.strftime("%H:%M") if event.entry_dt else "",
                    "exit_time_et": event.exit_dt.strftime("%H:%M") if event.exit_dt else "",
                    "minutes_to_trigger": event.duration_minutes,
                    "start_price": f"{event.start_price:.4f}",
                    "trigger_price": f"{event.trigger_price:.4f}",
                    "entry_price": f"{event.entry_price:.4f}" if event.entry_price is not None else "",
                    "close_price": f"{event.close_price:.4f}",
                    "exit_price": f"{event.exit_price:.4f}",
                    "post_entry_high": f"{event.post_entry_high:.4f}" if event.post_entry_high is not None else "",
                    "post_entry_low": f"{event.post_entry_low:.4f}" if event.post_entry_low is not None else "",
                    "stop_triggered": "true" if event.stop_triggered else "false",
                    "max_gain_pct": f"{event.max_gain_pct:.2f}",
                    "close_gain_pct": f"{event.close_gain_pct:.2f}",
                    "window_volume": f"{event.window_volume:.0f}",
                }
            )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect stocks that gained a target percentage within a rolling 30-bar window.",
    )
    parser.add_argument(
        "--months",
        type=int,
        default=DEFAULT_MONTHS,
        help="Number of months to scan ending today (default: 1).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Gain threshold as a decimal (default: 0.40 for 40%%).",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=DEFAULT_WINDOW,
        help="Window size in bars (default: 30).",
    )
    parser.add_argument(
        "--include-premarket",
        action="store_true",
        help="Include 04:00-09:30 ET data when evaluating windows.",
    )
    parser.add_argument(
        "--no-common-filter",
        action="store_true",
        help="Process all tickers without restricting to common stock.",
    )
    parser.add_argument(
        "--include-inactive-common",
        action="store_true",
        help="Include inactive common stock tickers in the common-stock filter.",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        help="Polygon API key (overrides environment variable).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/40pct_moves.csv"),
        help="CSV destination for the detected moves.",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
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
            log(f"  Loaded {len(common_allowlist)} tickers")
        except PolygonError as exc:
            log(f"Warning: failed to load common stock universe: {exc}")
            only_common = False
            common_allowlist = None

    today = date.today()
    start_date = months_ago(today, max(1, args.months))
    log(
        f"Scanning from {start_date.isoformat()} to {today.isoformat()} "
        f"for {args.threshold * 100:.0f}% moves within {args.window}-bar windows..."
    )

    all_events: List[IntradayMove] = []
    trading_days = list(daterange_weekdays(start_date, today))
    for idx, day in enumerate(trading_days, start=1):
        log(f"[{idx}/{len(trading_days)}] {day.isoformat()} - loading grouped results")
        try:
            grouped = fetch_grouped(day)
        except PolygonError as exc:
            log(f"  Failed to fetch grouped data: {exc}")
            continue
        candidates = filter_candidates(
            grouped,
            args.threshold,
            common_allowlist if only_common else None,
        )
        if not candidates:
            continue
        log(f"  {len(candidates)} candidate tickers")
        for ticker in candidates:
            try:
                raw_bars = fetch_minute_bars(ticker, day)
            except PolygonError as exc:
                log(f"    {ticker}: failed to fetch minute bars ({exc})")
                continue
            session_bars = normalize_bars(raw_bars, include_premarket=args.include_premarket)
            if not session_bars:
                continue
            events = detect_intraday_moves(
                ticker,
                day,
                session_bars,
                threshold=args.threshold,
                window=args.window,
            )
            if events:
                all_events.extend(events)
                log(f"    {ticker}: found {len(events)} event(s)")
    if not all_events:
        log("No qualifying moves detected.")
    else:
        write_results(args.output, all_events)
        log(f"Wrote {len(all_events)} events to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
