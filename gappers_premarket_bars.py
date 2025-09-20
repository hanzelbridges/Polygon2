"""
Collect pre-market 5-minute bars for gapper tickers to support backtesting.

This script scans a date range, identifies stocks whose regular-session open price
gapped above the previous close by at least a configurable threshold, and writes
out JSON lines containing the pre-market 5-minute bar history alongside summary
statistics. The goal is to analyse whether entries during pre-market could
outperform waiting for the regular trading session open.
"""

import csv
import json
import math
import os
import sys
import time
from datetime import date, datetime, timedelta, time as dtime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

import http.client
import ssl
import urllib.error
import urllib.parse
import urllib.request

import requests
import yaml

try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except Exception:  # pragma: no cover
    ZoneInfo = None

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
_REF_TYPE_CACHE: Dict[str, Optional[str]] = {}
_CS_TICKERS_CACHE_ACTIVE: Optional[Set[str]] = None
_CS_TICKERS_CACHE_INACTIVE: Optional[Set[str]] = None

DEFAULT_CONFIG_PATH = "config_gappers.yaml"
EASTERN_TZ = "America/New_York"


def load_config(config_path: str = DEFAULT_CONFIG_PATH) -> dict:
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
            return data if isinstance(data, dict) else {}
    except FileNotFoundError:
        return {}
    except Exception as exc:
        sys.stderr.write(f"Warning: failed to read {config_path}: {exc}\n")
        return {}


def parse_date_arg(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def years_ago(d: date, years: float) -> date:
    days_back = int(years * 365.25)
    return d - timedelta(days=days_back)


def daterange_weekdays(start: date, end: date) -> Iterable[date]:
    cur = start
    step = timedelta(days=1)
    while cur <= end:
        if cur.weekday() < 5:
            yield cur
        cur += step


def http_get_json(url: str, max_retries: int = 5, backoff: float = 1.5) -> Optional[dict]:
    attempt = 0
    while attempt <= max_retries:
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                if resp.status != 200:
                    raise urllib.error.HTTPError(url, resp.status, resp.reason, resp.headers, None)
                data = resp.read()
                return json.loads(data)
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503, 504):
                sleep_s = backoff ** max(1, attempt)
                time.sleep(sleep_s)
                attempt += 1
                continue
            sys.stderr.write(f"HTTPError {e.code} for {url}: {e.reason}\n")
            return None
        except (urllib.error.URLError, TimeoutError, http.client.RemoteDisconnected, ConnectionResetError, ssl.SSLEOFError):
            sleep_s = backoff ** max(1, attempt)
            time.sleep(sleep_s)
            attempt += 1
            continue
        except json.JSONDecodeError:
            sys.stderr.write(f"Invalid JSON from {url}\n")
            return None
    sys.stderr.write(f"Failed after {max_retries} retries: {url}\n")
    return None


def fetch_grouped_for_date(d: date, adjusted: bool = True) -> Optional[List[dict]]:
    if not POLYGON_API_KEY:
        raise RuntimeError("Missing POLYGON_API_KEY environment variable.")
    date_str = d.strftime("%Y-%m-%d")
    url = (
        f"https://api.polygon.io/v2/aggs/grouped/locale/us/market/stocks/{date_str}?"
        f"adjusted={'true' if adjusted else 'false'}&apiKey={POLYGON_API_KEY}"
    )
    payload = http_get_json(url)
    if not payload:
        return None
    if payload.get("status") != "OK":
        return payload.get("results") or []
    return payload.get("results") or []


def _fetch_ohlcv_for_ticker_on(ticker: str, d: date) -> Optional[Dict[str, float]]:
    grouped = fetch_grouped_for_date(d)
    if not grouped:
        return None
    for g in grouped:
        if g.get("T") == ticker:
            try:
                return {
                    "open": float(g.get("o")),
                    "high": float(g.get("h")),
                    "low": float(g.get("l")),
                    "close": float(g.get("c")),
                    "volume": float(g.get("v")),
                }
            except (TypeError, ValueError):
                return None
    return None


def to_utc(dt_local: datetime, tz_name: str = EASTERN_TZ) -> datetime:
    if ZoneInfo is None:
        return dt_local.replace(tzinfo=timezone.utc)
    tz = ZoneInfo(tz_name)
    return dt_local.replace(tzinfo=tz).astimezone(timezone.utc)


def premarket_window_utc(d: date) -> tuple:
    start_local = datetime.combine(d, dtime(4, 0, 0))
    end_local = datetime.combine(d, dtime(9, 30, 0))
    return to_utc(start_local), to_utc(end_local)


def fetch_aggs_range(
    ticker: str,
    start_utc: datetime,
    end_utc: datetime,
    timespan: str = "minute",
    multiplier: int = 1,
    adjusted: bool = True,
    limit: int = 50000,
) -> List[dict]:
    if not POLYGON_API_KEY:
        raise RuntimeError("Missing POLYGON_API_KEY environment variable.")
    base = "https://api.polygon.io/v2/aggs/ticker"
    from_str = start_utc.date().isoformat()
    to_str = end_utc.date().isoformat()
    safe_ticker = urllib.parse.quote(ticker, safe="")
    url = (
        f"{base}/{safe_ticker}/range/{multiplier}/{timespan}/"
        f"{from_str}/{to_str}?adjusted={'true' if adjusted else 'false'}&sort=asc&limit={limit}&apiKey={POLYGON_API_KEY}"
    )
    payload = http_get_json(url)
    if not payload:
        return []
    return payload.get("results") or []


def is_probably_non_common(ticker: str) -> bool:
    if not ticker:
        return False
    t = ticker.upper()
    if ".WS" in t or t.endswith(".W"):
        return True
    if len(t) == 5 and t.isalpha() and t[-1] in {"W", "R", "U"}:
        return True
    return False


def get_ticker_type_cached(ticker: str) -> Optional[str]:
    if ticker in _REF_TYPE_CACHE:
        return _REF_TYPE_CACHE[ticker]
    if not POLYGON_API_KEY:
        _REF_TYPE_CACHE[ticker] = None
        return None
    try:
        safe = urllib.parse.quote(ticker, safe="")
        url = f"https://api.polygon.io/v3/reference/tickers/{safe}?apiKey={POLYGON_API_KEY}"
        resp = requests.get(url, timeout=20)
        if not resp.ok:
            _REF_TYPE_CACHE[ticker] = None
            return None
        data = resp.json().get("results", {})
        t = data.get("type")
        _REF_TYPE_CACHE[ticker] = t
        return t
    except Exception:
        _REF_TYPE_CACHE[ticker] = None
        return None


def is_etf_or_etn_type(t: Optional[str]) -> bool:
    if not t:
        return False
    ts = str(t).strip().upper()
    return ts in {"ETF", "ETN", "ETP"}


def fetch_common_stock_tickers(active: Optional[bool] = True, max_pages: int = 1000, backoff: float = 1.5) -> Set[str]:
    global _CS_TICKERS_CACHE_ACTIVE, _CS_TICKERS_CACHE_INACTIVE
    if active is True and _CS_TICKERS_CACHE_ACTIVE is not None:
        return _CS_TICKERS_CACHE_ACTIVE
    if active is False and _CS_TICKERS_CACHE_INACTIVE is not None:
        return _CS_TICKERS_CACHE_INACTIVE
    tickers: Set[str] = set()
    if not POLYGON_API_KEY:
        if active is True:
            _CS_TICKERS_CACHE_ACTIVE = tickers
        elif active is False:
            _CS_TICKERS_CACHE_INACTIVE = tickers
        return tickers
    base_url = "https://api.polygon.io/v3/reference/tickers"
    params = {
        "market": "stocks",
        "type": "CS",
        **({"active": "true"} if active is True else ({"active": "false"} if active is False else {})),
        "limit": 1000,
        "apiKey": POLYGON_API_KEY,
    }
    cursor: Optional[str] = None
    pages = 0
    while pages < max_pages:
        try:
            call_params = dict(params)
            if cursor:
                call_params["cursor"] = cursor
            resp = requests.get(base_url, params=call_params, timeout=30)
            if resp.status_code == 429:
                sleep_s = backoff ** max(1, pages)
                time.sleep(sleep_s)
                continue
            if not resp.ok:
                break
            data = resp.json() or {}
            for r in data.get("results", []) or []:
                t = r.get("ticker")
                if t:
                    tickers.add(str(t).upper())
            next_url = data.get("next_url") or data.get("next")
            if next_url:
                try:
                    from urllib.parse import urlparse, parse_qs

                    q = parse_qs(urlparse(next_url).query)
                    cursor_vals = q.get("cursor") or []
                    cursor = cursor_vals[0] if cursor_vals else None
                except Exception:
                    cursor = None
            else:
                cursor = None
            pages += 1
            if not cursor:
                break
        except Exception:
            break
    if active is True:
        _CS_TICKERS_CACHE_ACTIVE = tickers
    elif active is False:
        _CS_TICKERS_CACHE_INACTIVE = tickers
    return tickers


def fetch_premarket_5min_bars(ticker: str, session_date: date) -> List[Dict[str, float]]:
    start_utc, end_utc = premarket_window_utc(session_date)
    raw_bars = fetch_aggs_range(ticker, start_utc, end_utc, timespan="minute", multiplier=5)
    if not raw_bars:
        return []
    start_ms = int(start_utc.timestamp() * 1000)
    end_ms = int(end_utc.timestamp() * 1000)
    bars: List[Dict[str, float]] = []
    for b in raw_bars:
        t_val = b.get("t")
        if not isinstance(t_val, (int, float)):
            continue
        ts = int(t_val)
        if ts < start_ms or ts >= end_ms:
            continue
        try:
            o = float(b.get("o"))
            h = float(b.get("h"))
            l = float(b.get("l"))
            c = float(b.get("c"))
        except (TypeError, ValueError):
            continue
        v = b.get("v")
        n = b.get("n")
        try:
            vol = float(v) if v is not None else 0.0
        except (TypeError, ValueError):
            vol = 0.0
        try:
            trades = int(n) if n is not None else 0
        except (TypeError, ValueError):
            trades = 0
        bars.append({"t": ts, "o": o, "h": h, "l": l, "c": c, "v": vol, "n": trades})
    bars.sort(key=lambda item: item["t"])
    return bars


def summarize_premarket(bars: List[Dict[str, float]]) -> Dict[str, Optional[float]]:
    if not bars:
        return {
            "pm_open": None,
            "pm_high": None,
            "pm_low": None,
            "pm_close": None,
            "pm_vol": 0.0,
            "pm_trades": 0,
        }
    pm_open = bars[0]["o"]
    pm_close = bars[-1]["c"]
    pm_high = max(b["h"] for b in bars if b.get("h") is not None)
    pm_low = min(b["l"] for b in bars if b.get("l") is not None)
    pm_vol = sum(float(b.get("v") or 0.0) for b in bars)
    pm_trades = sum(int(b.get("n") or 0) for b in bars)
    return {
        "pm_open": pm_open,
        "pm_high": pm_high,
        "pm_low": pm_low,
        "pm_close": pm_close,
        "pm_vol": pm_vol,
        "pm_trades": pm_trades,
    }


def format_premarket_bars(bars: List[Dict[str, float]]) -> List[Dict[str, object]]:
    formatted: List[Dict[str, object]] = []
    if ZoneInfo is not None:
        try:
            eastern = ZoneInfo(EASTERN_TZ)
        except Exception:
            eastern = None
    else:
        eastern = None
    for bar in bars:
        ts_utc = datetime.fromtimestamp(bar["t"] / 1000.0, tz=timezone.utc)
        if eastern is not None:
            ts_local = ts_utc.astimezone(eastern)
        else:
            ts_local = ts_utc
        formatted.append(
            {
                "start": ts_local.isoformat(),
                "open": round(float(bar["o"]), 6),
                "high": round(float(bar["h"]), 6),
                "low": round(float(bar["l"]), 6),
                "close": round(float(bar["c"]), 6),
                "volume": round(float(bar.get("v", 0.0)), 3),
                "trades": int(bar.get("n", 0) or 0),
            }
        )
    return formatted


def log_status(message: str) -> None:
    """Print a timestamped status line."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def _format_eta(seconds: float) -> str:
    if not seconds or seconds <= 0 or not math.isfinite(seconds):
        return "N/A"
    seconds_int = int(seconds)
    hours, remainder = divmod(seconds_int, 3600)
    minutes, secs = divmod(remainder, 60)
    parts: List[str] = []
    if hours:
        parts.append(f"{hours}h")
    if minutes or hours:
        parts.append(f"{minutes}m")
    parts.append(f"{secs}s")
    return " ".join(parts)


def _print_progress(processed_days: int, total_days: int, current_date: date, gappers_today: int, total_gappers: int, started_at: float) -> None:
    pct = (processed_days / total_days * 100.0) if total_days else 100.0
    elapsed = time.time() - started_at
    eta_seconds = 0.0
    if processed_days and total_days and processed_days < total_days:
        eta_seconds = (elapsed / processed_days) * (total_days - processed_days)
    eta_str = _format_eta(eta_seconds)
    message = (
        f"Progress: {processed_days}/{total_days} weekdays ({pct:.1f}%) through {current_date} - "
        f"Gappers today: {gappers_today} - Total gappers: {total_gappers} - ETA: {eta_str}"
    )
    log_status(message)


def collect_gap_records(
    start: date,
    end: date,
    gap_threshold_pct: float,
    *,
    min_price: float,
    exclude_by_suffix: bool,
    only_common_stock: bool,
    cs_allowlist: Optional[Set[str]],
    exclude_etf_etn: bool,
    total_days: int,
) -> Iterable[Dict[str, object]]:
    prev_close_by_ticker: Dict[str, float] = {}
    allowlist = {t.upper() for t in cs_allowlist} if cs_allowlist else None
    allowlist_available = bool(allowlist)
    threshold_decimal = gap_threshold_pct / 100.0
    processed_days = 0
    total_gappers = 0
    started_at = time.time()
    fallback_common_logged = False

    for current_date in daterange_weekdays(start, end):
        day_index = processed_days + 1
        log_status(f"Processing {current_date} ({day_index}/{total_days})")
        grouped = fetch_grouped_for_date(current_date)
        if grouped is None:
            log_status(f"No grouped data returned for {current_date}; skipping.")
            sys.stderr.write(f"Warning: failed to fetch grouped data for {current_date}, skipping.\n")
            processed_days += 1
            _print_progress(processed_days, total_days, current_date, 0, total_gappers, started_at)
            continue

        gappers_today = 0
        for g in grouped:
            ticker = g.get("T")
            if not ticker:
                continue
            try:
                o = float(g.get("o"))
                h = float(g.get("h"))
                l = float(g.get("l"))
                c = float(g.get("c"))
            except (TypeError, ValueError):
                continue

            ticker_upper = ticker.upper()
            if o <= min_price:
                prev_close_by_ticker[ticker] = c
                continue
            if exclude_by_suffix and is_probably_non_common(ticker):
                prev_close_by_ticker[ticker] = c
                continue

            prev_c = prev_close_by_ticker.get(ticker)
            if prev_c is None or prev_c <= 0:
                prev_close_by_ticker[ticker] = c
                continue

            gap_decimal = (o - prev_c) / prev_c
            if gap_decimal < threshold_decimal:
                prev_close_by_ticker[ticker] = c
                continue

            ticker_type: Optional[str] = None
            if only_common_stock:
                if allowlist_available:
                    if ticker_upper not in allowlist:
                        prev_close_by_ticker[ticker] = c
                        continue
                else:
                    if not fallback_common_logged:
                        log_status("Polygon common-stock allowlist unavailable; skipping type filter for gappers.")
                        fallback_common_logged = True
                    ticker_type = get_ticker_type_cached(ticker)
                    if not ticker_type or ticker_type.upper() != "CS":
                        prev_close_by_ticker[ticker] = c
                        continue

            if exclude_etf_etn:
                if ticker_type is None:
                    ticker_type = get_ticker_type_cached(ticker)
                if is_etf_or_etn_type(ticker_type):
                    prev_close_by_ticker[ticker] = c
                    continue

            pm_bars = fetch_premarket_5min_bars(ticker, current_date)
            pm_bar_count = len(pm_bars)
            pm_summary = summarize_premarket(pm_bars)
            record: Dict[str, object] = {
                "date": current_date.isoformat(),
                "ticker": ticker,
                "prev_close": round(prev_c, 6),
                "open": round(o, 6),
                "high": round(h, 6),
                "low": round(l, 6),
                "close": round(c, 6),
                "gap_pct": round(gap_decimal * 100.0, 6),
                "pm_open": round(pm_summary["pm_open"], 6) if pm_summary["pm_open"] is not None else None,
                "pm_high": round(pm_summary["pm_high"], 6) if pm_summary["pm_high"] is not None else None,
                "pm_low": round(pm_summary["pm_low"], 6) if pm_summary["pm_low"] is not None else None,
                "pm_close": round(pm_summary["pm_close"], 6) if pm_summary["pm_close"] is not None else None,
                "pm_vol": round(pm_summary["pm_vol"], 3),
                "pm_trades": pm_summary["pm_trades"],
                "pm_prices": format_premarket_bars(pm_bars),
            }
            log_status(
                f"Collected pre-market data for {ticker_upper} on {current_date}: "
                f"{pm_bar_count} bars, gap {gap_decimal * 100.0:.2f}%"
            )
            yield record
            gappers_today += 1
            total_gappers += 1
            prev_close_by_ticker[ticker] = c

        processed_days += 1
        _print_progress(processed_days, total_days, current_date, gappers_today, total_gappers, started_at)
        log_status(
            f"Completed {current_date}: {gappers_today} gappers captured; running total {total_gappers}"
        )

    elapsed = time.time() - started_at
    total_str = _format_eta(elapsed)
    log_status(
        f"Finished scanning {processed_days}/{total_days} weekdays. Total gappers: {total_gappers}. Elapsed: {total_str}"
    )

def prepare_output_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if path.parent and not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    return path


def main(argv: List[str]) -> int:
    config = load_config(DEFAULT_CONFIG_PATH)
    global POLYGON_API_KEY
    polygon_key = (
        os.getenv("POLYGON_API_KEY")
        or config.get("polygon_api_key")
        or config.get("POLYGON_API_KEY")
    )
    if not polygon_key:
        sys.stderr.write(
            "Error: Polygon API key is missing. Set POLYGON_API_KEY or add polygon_api_key to config_gappers.yaml.\n"
        )
        return 2
    POLYGON_API_KEY = polygon_key

    gap_threshold_pct = float(config.get("gap_threshold_pct", 20.0))
    min_price = float(config.get("min_price", 0.0))
    only_common_stock = bool(config.get("only_common_stock", False))
    include_inactive_common = bool(config.get("include_inactive_common", True))
    exclude_by_suffix = bool(config.get("exclude_non_common_by_suffix", False))
    exclude_etf_etn = bool(config.get("exclude_etf_etn", False))
    default_output = (
        config.get("premarket_output_path")
        or config.get("premarket_output")
        or "output/gappers_premarket_5min.csv"
    )

    today = date.today()
    default_years_back = float(config.get("premarket_years_back", 2.0))
    use_years_back = bool(config.get("use_years_back", True))
    if use_years_back:
        start = years_ago(today, default_years_back)
        end = today
    else:
        start = years_ago(today, default_years_back)
        end = today
        cfg_start = config.get("start_date")
        cfg_end = config.get("end_date")
        if cfg_start:
            try:
                start = parse_date_arg(str(cfg_start))
            except Exception:
                pass
        if cfg_end:
            try:
                end = parse_date_arg(str(cfg_end))
            except Exception:
                pass

    if len(argv) >= 2 and argv[1]:
        start = parse_date_arg(argv[1])
    if len(argv) >= 3 and argv[2]:
        end = parse_date_arg(argv[2])
    if len(argv) >= 4 and argv[3]:
        gap_threshold_pct = float(argv[3])
    if len(argv) >= 5 and argv[4]:
        output_path_str = argv[4]
    else:
        output_path_str = default_output

    if start > end:
        sys.stderr.write("Start date must be <= end date.\n")
        return 2

    output_path = prepare_output_path(output_path_str)

    allowlist: Optional[Set[str]] = None
    if only_common_stock:
        print("Loading Polygon common stock universe (type=CS)...", flush=True)
        try:
            active = fetch_common_stock_tickers(active=True)
            allowlist = set(active)
            if include_inactive_common:
                inactive = fetch_common_stock_tickers(active=False)
                allowlist.update(inactive)
            if not allowlist:
                allowlist = None
            else:
                print(f"Loaded {len(allowlist)} tickers into common-stock allowlist.", flush=True)
        except Exception as exc:
            sys.stderr.write(f"Warning: failed to load common stock universe: {exc}\n")
            allowlist = None

    total_days = sum(1 for _ in daterange_weekdays(start, end))
    if total_days == 0:
        print("No weekdays in the requested range; nothing to do.")
        return 0

    print(
        f"Collecting pre-market 5-minute bars for gappers between {start} and {end} "
        f"(gap >= {gap_threshold_pct:.2f}%)...",
        flush=True,
    )

    records_written = 0
    writer: Optional[csv.DictWriter] = None
    fieldnames = [
        "date",
        "ticker",
        "prev_close",
        "open",
        "high",
        "low",
        "close",
        "gap_pct",
        "pm_open",
        "pm_high",
        "pm_low",
        "pm_close",
        "pm_vol",
        "pm_trades",
        "pm_prices",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as fh:
        for record in collect_gap_records(
            start,
            end,
            gap_threshold_pct,
            min_price=min_price,
            exclude_by_suffix=exclude_by_suffix,
            only_common_stock=only_common_stock,
            cs_allowlist=allowlist,
            exclude_etf_etn=exclude_etf_etn,
            total_days=total_days,
        ):
            if writer is None:
                writer = csv.DictWriter(fh, fieldnames=fieldnames)
                writer.writeheader()
            row = dict(record)
            row["pm_prices"] = json.dumps(row.get("pm_prices")) if row.get("pm_prices") is not None else "[]"
            writer.writerow(row)
            records_written += 1

    print(f"Wrote {records_written} records to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
