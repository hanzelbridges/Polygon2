
"""
Collect 1-minute pre-market and regular-session bars for high-gap stocks.

The script scans a date range, finds tickers gapping >= threshold at the open,
and writes JSONL records with full 1-minute OHLCV data for the pre-market
(04:00-09:30 ET) and regular session (09:30-16:00 ET), along with the previous
close for reference.
"""

import json
import os
import sys
import time
from datetime import date, datetime, timedelta, time as dtime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Set

import requests
import http.client
import urllib.error
import urllib.parse
import urllib.request
import yaml

try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except Exception:  # pragma: no cover
    ZoneInfo = None

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
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
        except (urllib.error.URLError, TimeoutError, http.client.IncompleteRead, http.client.RemoteDisconnected, ConnectionResetError):
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


def to_utc(dt_local: datetime, tz_name: str = EASTERN_TZ) -> datetime:
    if ZoneInfo is None:
        return dt_local.replace(tzinfo=timezone.utc)
    tz = ZoneInfo(tz_name)
    return dt_local.replace(tzinfo=tz).astimezone(timezone.utc)


def premarket_window_utc(d: date) -> tuple:
    start_local = datetime.combine(d, dtime(4, 0, 0))
    end_local = datetime.combine(d, dtime(9, 30, 0))
    return to_utc(start_local), to_utc(end_local)


def regular_session_window_utc(d: date) -> tuple:
    start_local = datetime.combine(d, dtime(9, 30, 0))
    end_local = datetime.combine(d, dtime(16, 0, 0))
    return to_utc(start_local), to_utc(end_local)


def prev_weekday(d: date) -> date:
    cur = d - timedelta(days=1)
    while cur.weekday() >= 5:
        cur -= timedelta(days=1)
    return cur


def is_probably_non_common(ticker: str) -> bool:

    if not ticker:

        return False

    t = ticker.upper()

    if ".WS" in t or t.endswith(".W"):

        return True

    if len(t) == 5 and t.isalpha() and t[-1] in {"W", "R", "U"}:

        return True

    return False

def format_bars(bars: List[dict]) -> List[dict]:
    out: List[dict] = []
    eastern = None
    if ZoneInfo is not None:
        try:
            eastern = ZoneInfo(EASTERN_TZ)
        except Exception:
            eastern = None
    for bar in bars:
        ts_val = bar.get("t")
        if not isinstance(ts_val, (int, float)):
            continue
        ts_utc = datetime.fromtimestamp(ts_val / 1000.0, tz=timezone.utc)
        ts_local = ts_utc.astimezone(eastern) if eastern else ts_utc
        try:
            out.append(
                {
                    "start": ts_local.isoformat(),
                    "open": round(float(bar.get("o")), 6),
                    "high": round(float(bar.get("h")), 6),
                    "low": round(float(bar.get("l")), 6),
                    "close": round(float(bar.get("c")), 6),
                    "volume": round(float(bar.get("v", 0.0)), 3),
                    "trades": int(bar.get("n", 0) or 0),
                }
            )
        except (TypeError, ValueError):
            continue
    return out


def filter_bars_to_window(bars: List[dict], start_utc: datetime, end_utc: datetime) -> List[dict]:
    start_ms = int(start_utc.timestamp() * 1000)
    end_ms = int(end_utc.timestamp() * 1000)
    filtered: List[dict] = []
    for bar in bars:
        ts_val = bar.get("t")
        if isinstance(ts_val, (int, float)) and start_ms <= int(ts_val) < end_ms:
            filtered.append(bar)
    filtered.sort(key=lambda b: b.get("t", 0))
    return filtered


def fetch_minute_bars_for_windows(ticker: str, session_date: date) -> tuple:
    full_start, full_end = to_utc(datetime.combine(session_date, dtime(4, 0, 0))), to_utc(datetime.combine(session_date, dtime(16, 0, 0)))
    raw = fetch_aggs_range(ticker, full_start, full_end, timespan="minute", multiplier=1)
    pm_start, pm_end = premarket_window_utc(session_date)
    rth_start, rth_end = regular_session_window_utc(session_date)
    pm_raw = filter_bars_to_window(list(raw), pm_start, pm_end)
    rth_raw = filter_bars_to_window(list(raw), rth_start, rth_end)
    return format_bars(pm_raw), format_bars(rth_raw)



def _format_eta(seconds: float) -> str:
    if not seconds or seconds <= 0:
        return "N/A"
    seconds = int(seconds)
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes}m {secs}s"
    if minutes:
        return f"{minutes}m {secs}s"
    return f"{secs}s"

def log_status(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def build_prev_close_lookup(grouped: Optional[List[dict]]) -> Dict[str, float]:
    lookup: Dict[str, float] = {}
    if not grouped:
        return lookup
    for g in grouped:
        ticker = g.get("T")
        close_val = g.get("c")
        if ticker is None or close_val is None:
            continue
        try:
            lookup[str(ticker)] = float(close_val)
        except (TypeError, ValueError):
            continue
    return lookup


def fetch_grouped_cached(d: date, cache: Dict[date, Optional[List[dict]]]) -> Optional[List[dict]]:
    if d in cache:
        return cache[d]
    data = fetch_grouped_for_date(d)
    cache[d] = data
    return data


def find_prev_grouped(current_date: date, cache: Dict[date, Optional[List[dict]]]) -> Optional[List[dict]]:
    cursor = current_date
    attempts = 0
    while attempts < 7:
        cursor = prev_weekday(cursor)
        attempts += 1
        data = fetch_grouped_cached(cursor, cache)
        if data:
            return data
    return None


def collect_gap_records(
    start: date,
    end: date,
    gap_threshold_pct: float,
    *,
    min_price: float,
    exclude_by_suffix: bool,
    only_common_stock: bool,
    cs_allowlist: Optional[Set[str]],
) -> Iterable[Dict[str, object]]:
    allowlist = {t.upper() for t in cs_allowlist} if cs_allowlist else None
    grouped_cache: Dict[date, Optional[List[dict]]] = {}

    total_weekdays = sum(1 for _ in daterange_weekdays(start, end))
    if total_weekdays == 0:
        log_status("No weekdays found in range; nothing to do.")
        return
    processed_days = 0
    total_gappers = 0
    started_at = time.time()
    for current_date in daterange_weekdays(start, end):
        processed_days += 1
        eta_seconds = 0.0
        if processed_days < total_weekdays:
            elapsed = time.time() - started_at
            eta_seconds = elapsed / processed_days * (total_weekdays - processed_days)
        log_status(
            f"Processing {current_date} ({processed_days}/{total_weekdays}) - ETA {_format_eta(eta_seconds)}"
        )
        grouped = fetch_grouped_cached(current_date, grouped_cache)
        if grouped is None:
            log_status(f"No grouped data for {current_date}; skipping.")
            continue
        prev_grouped = find_prev_grouped(current_date, grouped_cache)
        prev_lookup = build_prev_close_lookup(prev_grouped)
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
            if o <= min_price:
                continue
            ticker_upper = str(ticker).upper()
            if exclude_by_suffix and is_probably_non_common(ticker_upper):
                continue
            prev_c = prev_lookup.get(ticker_upper)
            if prev_c is None or prev_c <= 0:
                continue
            gap_decimal = (o - prev_c) / prev_c
            if gap_decimal < gap_threshold_pct / 100.0:
                continue
            if only_common_stock and allowlist and ticker_upper not in allowlist:
                continue
            pm_bars, rth_bars = fetch_minute_bars_for_windows(ticker_upper, current_date)
            record = {
                "date": current_date.isoformat(),
                "ticker": ticker_upper,
                "prev_close": round(prev_c, 6),
                "open": round(o, 6),
                "high": round(h, 6),
                "low": round(l, 6),
                "close": round(c, 6),
                "gap_pct": round(gap_decimal * 100.0, 6),
                "pm_bars": pm_bars,
                "rth_bars": rth_bars,
            }
            yield record
            gappers_today += 1
        log_status(f"Finished {current_date}: {gappers_today} gappers")


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
        sys.stderr.write("Error: Polygon API key is missing. Set POLYGON_API_KEY or add polygon_api_key to config_gappers.yaml.\n")
        return 2
    POLYGON_API_KEY = polygon_key

    gap_threshold_pct = float(config.get("gap_threshold_pct", 30.0))
    min_price = float(config.get("min_price", 2.0))
    only_common_stock = bool(config.get("only_common_stock", False))
    include_inactive_common = bool(config.get("include_inactive_common", True))
    exclude_by_suffix = bool(config.get("exclude_non_common_by_suffix", False))
    default_output = config.get("minute_output", "output/gapper_minute_bars.jsonl")

    today = date.today()
    default_years_back = float(config.get("minute_years_back", 2.0))
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
        try:
            log_status("Loading Polygon common stock universe (type=CS)...")
            active = fetch_common_stock_tickers(active=True)
            allowlist = set(active)
            if include_inactive_common:
                inactive = fetch_common_stock_tickers(active=False)
                allowlist.update(inactive)
            log_status(f"Loaded {len(allowlist)} tickers into common-stock allowlist.")
        except Exception as exc:
            sys.stderr.write(f"Warning: failed to load common stock universe: {exc}\n")
            allowlist = None

    total_days = sum(1 for _ in daterange_weekdays(start, end))
    if total_days == 0:
        print("No weekdays in the requested range; nothing to do.")
        return 0

    log_status(
        f"Collecting 1-minute bars for gappers between {start} and {end} (gap >= {gap_threshold_pct:.2f}%, min price {min_price})"
    )

    records_written = 0
    with output_path.open("w", encoding="utf-8") as fh:
        for record in collect_gap_records(
            start,
            end,
            gap_threshold_pct,
            min_price=min_price,
            exclude_by_suffix=exclude_by_suffix,
            only_common_stock=only_common_stock,
            cs_allowlist=allowlist,
        ):
            fh.write(json.dumps(record) + "\n")
            records_written += 1

    log_status(f"Wrote {records_written} records to {output_path}")
    return 0


def fetch_common_stock_tickers(active: Optional[bool] = True, max_pages: int = 1000, backoff: float = 1.5) -> Set[str]:
    tickers: Set[str] = set()
    if not POLYGON_API_KEY:
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
    return tickers


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
