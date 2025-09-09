import csv
import math
import os
import re
import sys
import time
from datetime import date, datetime, timedelta, time as dtime, timezone
from typing import Dict, List, Optional, Tuple, Set

import json
import requests
import urllib.request
import urllib.error
import urllib.parse
import yaml  # Add this import
try:
    from zoneinfo import ZoneInfo  # Python 3.9+
except Exception:  # pragma: no cover
    ZoneInfo = None

# Google Sheets imports
from google.oauth2 import service_account
from googleapiclient.discovery import build

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
_REF_TYPE_CACHE: Dict[str, Optional[str]] = {}
_CS_TICKERS_CACHE_ACTIVE: Optional[Set[str]] = None
_CS_TICKERS_CACHE_INACTIVE: Optional[Set[str]] = None
SHORT_INTEREST_ENDPOINTS = (
    "https://api.polygon.io/v3/reference/shorts",
    "https://api.polygon.io/vX/reference/shorts",
    "https://api.polygon.io/v2/reference/shorts",
    "https://api.polygon.io/stocks/v1/short-interest",
)


def years_ago(d: date, years: float) -> date:
    """Return the calendar date approximately `years` prior, supporting fractions."""
    days_back = int(years * 365.25)
    return d - timedelta(days=days_back)


def daterange_weekdays(start: date, end: date):
    """Yield weekdays from start to end inclusive (Mon-Fri)."""
    cur = start
    step = timedelta(days=1)
    while cur <= end:
        if cur.weekday() < 5:  # 0=Mon, 6=Sun
            yield cur
        cur += step


def http_get_json(url: str, max_retries: int = 5, backoff: float = 1.5) -> Optional[dict]:
    """GET JSON with basic retry and rate-limit handling."""
    attempt = 0
    while attempt <= max_retries:
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                if resp.status != 200:
                    raise urllib.error.HTTPError(url, resp.status, resp.reason, resp.headers, None)
                data = resp.read()
                return json.loads(data)
        except urllib.error.HTTPError as e:
            # 429 or transient 5xx: retry with backoff
            if e.code in (429, 500, 502, 503, 504):
                sleep_s = backoff ** attempt
                time.sleep(sleep_s)
                attempt += 1
                continue
            # Other HTTP errors: don't retry
            sys.stderr.write(f"HTTPError {e.code} for {url}: {e.reason}\n")
            return None
        except (urllib.error.URLError, TimeoutError) as e:
            sleep_s = backoff ** attempt
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
    # Polygon returns: { queryCount, resultsCount, adjusted, results: [...], status }
    if payload.get("status") != "OK":
        # Non-trading days often return status OK with resultsCount=0; some may differ.
        return payload.get("results") or []
    return payload.get("results") or []


def load_config(config_path: str = "config_gappers.yaml") -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def is_probably_non_common(ticker: str) -> bool:
    """Heuristic filter for non-common stock tickers.
    Rules (conservative):
    - Contains '.WS' anywhere (warrants), or ends with '.W'.
    - Exactly 5 letters and ends with one of {W, R, U} (NASDAQ suffixes for Warrant/Right/Unit).
    - Ends with 'ZZT' or similar test patterns (e.g., ZXZZT, ZWZZT).
    - Ends with lowercase 'p' followed by one or more uppercase letters (preferred shares, e.g., pF, pG).
    Does NOT exclude class share suffixes like '.A', '.B', etc.
    """
    if not ticker:
        return False
    # Case-insensitive checks for most rules
    t_upper = ticker.upper()
    if ".WS" in t_upper or t_upper.endswith(".W") or t_upper.endswith(".U"):
        return True
    if len(t_upper) == 5 and t_upper.isalpha() and t_upper[-1] in {"W", "R", "U"}:
        return True
    # Add check for test tickers (case-insensitive)
    if 'ZZT' in t_upper:
        return True
    # Add check for preferred shares (exact case: lowercase 'p' + uppercase letters)
    if re.search(r'p[A-Z]+$', ticker):
        return True
    return False


def get_ticker_type_cached(ticker: str) -> Optional[str]:
    """Fetch and cache Polygon reference 'type' for a ticker."""
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
    """Return a set of active US common stock tickers from Polygon reference API.
    Uses /v3/reference/tickers with market=stocks&type=CS&active=true and paginates via next_url.
    Caches results in-process for the duration of the run.
    """
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
            # Always call base_url with apiKey; use cursor for pagination when present
            call_params = dict(params)
            if cursor:
                call_params["cursor"] = cursor
            resp = requests.get(base_url, params=call_params, timeout=30)
            if resp.status_code == 429:
                # basic backoff then retry this page
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
            # Get cursor from next_url if present
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


def _a1_sheet_ref(sheet_title: str, cell: str = "A1") -> str:
    """Return a safe A1 reference for a sheet title, quoting as needed.
    Escapes single quotes inside the title by doubling them per A1 rules.
    """
    if sheet_title is None:
        # Fallback to plain cell (let API default or fail clearly)
        return cell
    needs_quote = any(ch in sheet_title for ch in [" ", "[", "]", "*", "?", "'", "!", ",", ":"])
    if needs_quote:
        return f"'{sheet_title.replace("'", "''")}'!{cell}"
    return f"{sheet_title}!{cell}"


def _get_sheet_title(service, spreadsheet_id: str, configured_title: Optional[str]) -> Optional[str]:
    """Resolve the sheet tab title to use for A1 ranges.
    Priority: configured_title -> first sheet title -> None
    """
    if configured_title:
        return configured_title
    try:
        meta = service.spreadsheets().get(
            spreadsheetId=spreadsheet_id,
            fields="sheets(properties(title,index))",
        ).execute()
        sheets = meta.get("sheets") or []
        if not sheets:
            return None
        # Find sheet with smallest index (first tab)
        first = sorted(
            (s.get("properties") or {} for s in sheets),
            key=lambda p: p.get("index", 0)
        )[0]
        return first.get("title")
    except Exception:
        return None


# Ownership/insider/institutional via CSV overrides removed per user request.


def fetch_short_interest_asof(ticker: str, target_date: date) -> Tuple[Optional[int], Optional[float], Optional[float], Optional[str]]:
    """Best-effort short interest as of the latest settlement date <= target_date.
    Returns: (short_interest_shares, short_percent_float, days_to_cover, settlement_date_str)
    Tries multiple API versions for compatibility.
    """
    if not POLYGON_API_KEY:
        return None, None, None, None
    params_common = {
        "ticker": ticker,
        "order": "desc",
        "limit": 100,
        "apiKey": POLYGON_API_KEY,
    }
    for base in SHORT_INTEREST_ENDPOINTS:
        try:
            resp = requests.get(base, params=params_common, timeout=30)
            if not resp.ok:
                continue
            data = resp.json()
            results = data.get("results") or []
            best = None
            best_dt = None
            for r in results:
                # normalize keys
                si = r.get("short_interest") or r.get("shortInterest") or r.get("short_interest_shares")
                spf = r.get("short_percent_float") or r.get("shortPercentFloat") or r.get("short_percent_of_float")
                dtc = r.get("days_to_cover") or r.get("daysToCover") or r.get("short_ratio")
                sdate = r.get("settlement_date") or r.get("settlementDate") or r.get("date")
                # choose only rows with a usable date
                if not sdate:
                    continue
                try:
                    sdt = datetime.strptime(str(sdate)[:10], "%Y-%m-%d").date()
                except Exception:
                    continue
                if sdt <= target_date:
                    if best_dt is None or sdt > best_dt:
                        best_dt = sdt
                        best = (si, spf, dtc, sdate)
            if best:
                return best
        except Exception:
            continue
    return None, None, None, None


def to_utc(dt_local: datetime, tz_name: str = "America/New_York") -> datetime:
    """Convert a naive local datetime to timezone-aware UTC datetime."""
    if ZoneInfo is None:
        # Fallback: treat input as UTC
        return dt_local.replace(tzinfo=timezone.utc)
    tz = ZoneInfo(tz_name)
    return dt_local.replace(tzinfo=tz).astimezone(timezone.utc)


def fetch_aggs_range(
    ticker: str,
    start_utc: datetime,
    end_utc: datetime,
    timespan: str = "minute",
    multiplier: int = 1,
    adjusted: bool = True,
    limit: int = 50000,
) -> List[dict]:
    """Fetch aggregate bars for a ticker between UTC datetimes."""
    if not POLYGON_API_KEY:
        raise RuntimeError("Missing POLYGON_API_KEY environment variable.")
    base = "https://api.polygon.io/v2/aggs/ticker"
    # Polygon expects from/to as dates (YYYY-MM-DD) or ms timestamps; using dates is most compatible.
    from_str = start_utc.date().isoformat()
    to_str = end_utc.date().isoformat()
    safe_ticker = urllib.parse.quote(ticker, safe="")
    url = (
        f"{base}/{safe_ticker}/range/{multiplier}/{timespan}/"
        f"{from_str}/{to_str}?"
        f"adjusted={'true' if adjusted else 'false'}&sort=asc&limit={limit}&apiKey={POLYGON_API_KEY}"
    )
    payload = http_get_json(url)
    if not payload:
        return []
    return payload.get("results") or []


def premarket_window_utc(d: date) -> Tuple[datetime, datetime]:
    """UTC window for US premarket (04:00–09:30 ET) on date d."""
    start_local = datetime.combine(d, dtime(4, 0, 0))
    end_local = datetime.combine(d, dtime(9, 30, 0))
    return to_utc(start_local), to_utc(end_local)


def rth_open_windows_utc(d: date) -> Dict[str, Tuple[datetime, datetime]]:
    """UTC windows for 5m/15m/30m from regular session open."""
    open_local = datetime.combine(d, dtime(9, 30, 0))
    start = to_utc(open_local)
    return {
        "5m": (start, start + timedelta(minutes=5)),
        "15m": (start, start + timedelta(minutes=15)),
        "30m": (start, start + timedelta(minutes=30)),
    }


def compute_premarket_stats(ticker: str, d: date) -> Dict[str, Optional[float]]:
    """Premarket stats between 04:00–09:30 ET for date d."""
    start_utc, end_utc = premarket_window_utc(d)
    bars = fetch_aggs_range(ticker, start_utc, end_utc, timespan="minute")
    if not bars:
        return {"pm_high": None, "pm_low": None, "pm_vwap": None, "pm_volume": 0.0, "pm_trades": 0}
    # Filter to exact time window using bar timestamps (ms since epoch)
    start_ms = int(start_utc.timestamp() * 1000)
    end_ms = int(end_utc.timestamp() * 1000)
    bars = [b for b in bars if isinstance(b.get("t"), (int, float)) and start_ms <= int(b["t"]) < end_ms]
    if not bars:
        return {"pm_high": None, "pm_low": None, "pm_vwap": None, "pm_volume": 0.0, "pm_trades": 0}
    pm_high = max(b.get("h") for b in bars if b.get("h") is not None)
    pm_low = min(b.get("l") for b in bars if b.get("l") is not None)
    total_vol = sum(b.get("v", 0) or 0 for b in bars)
    total_trades = sum(b.get("n", 0) or 0 for b in bars)
    num = 0.0
    den = 0.0
    for b in bars:
        v = b.get("v") or 0.0
        vw = b.get("vw")
        px = vw if vw is not None else b.get("c")
        if px is None or v is None:
            continue
        num += float(px) * float(v)
        den += float(v)
    pm_vwap = (num / den) if den > 0 else None
    return {
        "pm_high": float(pm_high) if pm_high is not None else None,
        "pm_low": float(pm_low) if pm_low is not None else None,
        "pm_vwap": float(pm_vwap) if pm_vwap is not None else None,
        "pm_volume": float(total_vol),
        "pm_trades": int(total_trades),
    }


def compute_rth_open_stats(ticker: str, d: date, today_open: float) -> Dict[str, Optional[float]]:
    """Early-session stats from open over 5/15/30 minutes."""
    windows = rth_open_windows_utc(d)
    out: Dict[str, Optional[float]] = {}
    for label, (start, end) in windows.items():
        bars = fetch_aggs_range(ticker, start, end, timespan="minute")
        # Filter to exact window
        start_ms = int(start.timestamp() * 1000)
        end_ms = int(end.timestamp() * 1000)
        bars = [b for b in bars if isinstance(b.get("t"), (int, float)) and start_ms <= int(b["t"]) < end_ms]
        if not bars:
            out[f"ret_{label}"] = None
            out[f"high_{label}_pct"] = None
            out[f"low_{label}_pct"] = None
            out[f"vol_{label}"] = 0.0
            continue
        last_close = bars[-1].get("c")
        high = max(b.get("h") for b in bars if b.get("h") is not None)
        low = min(b.get("l") for b in bars if b.get("l") is not None)
        vol = sum(b.get("v", 0) or 0 for b in bars)
        ret = (float(last_close) - today_open) / today_open if last_close and today_open else None
        high_pct = (float(high) - today_open) / today_open if high and today_open else None
        low_pct = (float(low) - today_open) / today_open if low and today_open else None
        out[f"ret_{label}"] = round(ret, 6) if ret is not None else None
        out[f"high_{label}_pct"] = round(high_pct, 6) if high_pct is not None else None
        out[f"low_{label}_pct"] = round(low_pct, 6) if low_pct is not None else None
        out[f"vol_{label}"] = float(vol)
    return out


def fetch_daily_history(ticker: str, start: date, end: date, adjusted: bool = True) -> List[dict]:
    start_dt = datetime.combine(start, dtime(0, 0)).replace(tzinfo=timezone.utc)
    end_dt = datetime.combine(end, dtime(23, 59, 59)).replace(tzinfo=timezone.utc)
    return fetch_aggs_range(ticker, start_dt, end_dt, timespan="day", adjusted=adjusted)


def rth_session_window_utc(d: date) -> Tuple[datetime, datetime]:
    """UTC window for regular session (09:30–16:00 ET) on date d."""
    start_local = datetime.combine(d, dtime(9, 30, 0))
    end_local = datetime.combine(d, dtime(16, 0, 0))
    return to_utc(start_local), to_utc(end_local)


def compute_rth_hod_lod_times(ticker: str, d: date) -> Dict[str, Optional[str]]:
    """Compute time of high-of-day and low-of-day during RTH in ET (HH:MM)."""
    start_utc, end_utc = rth_session_window_utc(d)
    bars = fetch_aggs_range(ticker, start_utc, end_utc, timespan="minute")
    if not bars:
        return {"hod_time": None, "lod_time": None}
    # Filter by exact window
    start_ms = int(start_utc.timestamp() * 1000)
    end_ms = int(end_utc.timestamp() * 1000)
    bars = [b for b in bars if isinstance(b.get("t"), (int, float)) and start_ms <= int(b["t"]) < end_ms]
    if not bars:
        return {"hod_time": None, "lod_time": None}
    # Find earliest time for max high and min low
    max_h = None; max_t = None
    min_l = None; min_t = None
    for b in bars:
        bh = b.get("h"); bl = b.get("l"); bt = b.get("t")
        if bh is not None:
            if (max_h is None) or (bh > max_h) or (bh == max_h and bt < max_t):
                max_h = bh; max_t = bt
        if bl is not None:
            if (min_l is None) or (bl < min_l) or (bl == min_l and bt < min_t):
                min_l = bl; min_t = bt
    def fmt_et(ms: Optional[int]) -> Optional[str]:
        if ms is None:
            return None
        dt_utc = datetime.fromtimestamp(int(ms)/1000, tz=timezone.utc)
        if ZoneInfo is not None:
            dt_local = dt_utc.astimezone(ZoneInfo("America/New_York"))
        else:
            dt_local = dt_utc
        return dt_local.strftime("%H:%M")
    return {"hod_time": fmt_et(max_t), "lod_time": fmt_et(min_t)}


def compute_indicators_prior_to(d: date, ticker: str) -> Dict[str, Optional[float]]:
    """ATR14, 30D avg vol, 52w position/dist using data strictly before d."""
    lookback_52w = 370
    start = d - timedelta(days=lookback_52w)
    end_prev = d - timedelta(days=1)
    bars = fetch_daily_history(ticker, start, end_prev)
    if not bars:
        return {
            "atr14": None,
            "avg_vol_30d": None,
            "pos_52w": None,
            "dist_52w_high_pct": None,
            "sma200": None,
            "prevclose_to_sma200_pct": None,
        }
    closes = [b.get("c") for b in bars if b.get("c") is not None]
    highs = [b.get("h") for b in bars if b.get("h") is not None]
    lows = [b.get("l") for b in bars if b.get("l") is not None]
    vols = [b.get("v") or 0 for b in bars]
    atr14 = None
    if len(bars) >= 15:
        trs: List[float] = []
        prev_close = bars[0].get("c")
        for b in bars[1:16]:
            h = b.get("h"); l = b.get("l"); c = b.get("c")
            if None in (h, l, prev_close):
                continue
            tr = max(float(h) - float(l), abs(float(h) - float(prev_close)), abs(float(l) - float(prev_close)))
            trs.append(tr)
            prev_close = c
        if trs:
            atr = sum(trs) / len(trs)
            for b in bars[16:]:
                h = b.get("h"); l = b.get("l"); c = b.get("c")
                if None in (h, l, prev_close):
                    continue
                tr = max(float(h) - float(l), abs(float(h) - float(prev_close)), abs(float(l) - float(prev_close)))
                atr = (atr * 13 + tr) / 14
                prev_close = c
            atr14 = atr
    avg_vol_30d = None
    if len(vols) >= 30:
        avg_vol_30d = float(sum(vols[-30:]) / 30.0)
    pos_52w = None
    dist_52w_high_pct = None
    sma200 = None
    prevclose_to_sma200_pct = None
    if highs and lows and closes:
        hi_52w = max(highs)
        lo_52w = min(lows)
        prev_c = closes[-1]
        if hi_52w is not None and lo_52w is not None and prev_c is not None and hi_52w > lo_52w:
            pos_52w = (float(prev_c) - float(lo_52w)) / (float(hi_52w) - float(lo_52w))
            dist_52w_high_pct = (float(hi_52w) - float(prev_c)) / float(prev_c)
        # 200-day simple moving average from last 200 closes up to prev day
        if len(closes) >= 200:
            last200 = [float(x) for x in closes[-200:]]
            sma200 = sum(last200) / 200.0
            if prev_c:
                prevclose_to_sma200_pct = (float(prev_c) - float(sma200)) / float(sma200)
    return {
        "atr14": round(atr14, 6) if atr14 is not None else None,
        "avg_vol_30d": round(avg_vol_30d, 3) if avg_vol_30d is not None else None,
        "pos_52w": round(pos_52w, 6) if pos_52w is not None else None,
        "dist_52w_high_pct": round(dist_52w_high_pct, 6) if dist_52w_high_pct is not None else None,
        "sma200": round(sma200, 6) if sma200 is not None else None,
        "prevclose_to_sma200_pct": round(prevclose_to_sma200_pct, 6) if prevclose_to_sma200_pct is not None else None,
    }


def compute_gappers(
    start: date,
    end: date,
    gap_threshold_pct: float,
    sheet_id: str,
    fieldnames: List[str],
) -> str:
    # Authenticate with Google Sheets
    creds = service_account.Credentials.from_service_account_file('service_account.json')
    service = build('sheets', 'v4', credentials=creds)

    prev_close_by_ticker: Dict[str, float] = {}
    rows: List[Dict[str, object]] = []  # Keep for counting, but write to Sheets immediately

    total_days = 0
    processed_days = 0
    for _ in daterange_weekdays(start, end):
        total_days += 1

    # Load exclusion flags and min_price once per run
    _cfg = load_config("config_gappers.yaml")
    exclude_by_suffix = bool(_cfg.get("exclude_non_common_by_suffix", False))
    exclude_etf_etn = bool(_cfg.get("exclude_etf_etn", False))
    min_price = float(_cfg.get("min_price", 0.0))  # New: Minimum price filter
    only_common_stock = bool(_cfg.get("only_common_stock", True))
    sheet_name_cfg = _cfg.get("sheet_name")
    sheet_title = sheet_name_cfg if sheet_name_cfg else _get_sheet_title(service, sheet_id, None)
    a1_anchor = _a1_sheet_ref(sheet_title, "A1")

    cs_allowlist: Set[str] = set()
    cs_active_count = cs_inactive_count = 0
    include_inactive_common = bool(_cfg.get("include_inactive_common", True))
    if only_common_stock and POLYGON_API_KEY:
        # Build allowlist of active and optionally inactive US common stocks (type=CS).
        try:
            active_set = fetch_common_stock_tickers(active=True)
            cs_active_count = len(active_set)
        except Exception:
            active_set = set()
        inactive_set: Set[str] = set()
        if include_inactive_common:
            try:
                inactive_set = fetch_common_stock_tickers(active=False)
                cs_inactive_count = len(inactive_set)
            except Exception:
                inactive_set = set()
        cs_allowlist = active_set | inactive_set
    # Log the size of the common stock filter universe (or fallback)
    if only_common_stock:
        if cs_allowlist:
            if include_inactive_common:
                print(f"Loaded common stock universe (type=CS): active={cs_active_count}, inactive={cs_inactive_count}, total={len(cs_allowlist)} tickers")
            else:
                print(f"Loaded common stock universe (type=CS): {len(cs_allowlist)} tickers")
        else:
            if not POLYGON_API_KEY:
                print("POLYGON_API_KEY not set; cannot load common stock universe.")
            else:
                print("Common stock universe unavailable; no tickers will be considered (only_common_stock enabled).")

    total_gappers = 0
    start_time = time.time()

    # Write header to Google Sheets
    body = {'values': [fieldnames]}
    service.spreadsheets().values().append(
        spreadsheetId=sheet_id,
        range=a1_anchor,
        valueInputOption='RAW',
        body=body
    ).execute()

    for i, d in enumerate(daterange_weekdays(start, end), start=1):
        grouped = fetch_grouped_for_date(d)
        if grouped is None:
            sys.stderr.write(f"Warning: failed to fetch {d}, skipping.\n")
            continue

        if not grouped:
            continue

        gappers_this_day = 0

        for g in grouped:
            try:
                ticker = g.get("T")
                o = float(g.get("o"))
                h = float(g.get("h"))
                l = float(g.get("l"))
                c = float(g.get("c"))
            except (TypeError, ValueError):
                continue

            # New: Skip if open price is below minimum
            if o <= min_price:
                continue

            if exclude_by_suffix and is_probably_non_common(ticker):
                continue

            prev_c = prev_close_by_ticker.get(ticker)
            if prev_c and prev_c > 0:
                gap_pct = (o - prev_c) / prev_c
                if gap_pct >= gap_threshold_pct / 100.0:
                    # Apply common-stock-only filtering now (after gap check) to reduce API lookups
                    if only_common_stock:
                        if not ticker or ticker.upper() not in cs_allowlist:
                            continue
                    if exclude_etf_etn:
                        # If we strictly filtered to CS via allowlist, ETF/ETN check is redundant; skip to avoid API calls
                        if not only_common_stock:
                            t_type2 = get_ticker_type_cached(ticker)
                            if is_etf_or_etn_type(t_type2):
                                prev_close_by_ticker[ticker] = c
                                continue
                        # else: already restricted to CS, proceed

                    open_to_high = (h - o) / o if o else None
                    open_to_low = (l - o) / o if o else None
                    open_to_close = (c - o) / o if o else None
                    # Premarket stats for the gap date
                    pm = compute_premarket_stats(ticker, d) if any(k in fieldnames for k in ["pm_high","pm_low","pm_vwap","pm_volume","pm_trades"]) else {}
                    # Early-session (from open) stats
                    rth = compute_rth_open_stats(ticker, d, o) if any(k in fieldnames for k in [
                        "ret_5m","ret_15m","ret_30m","high_5m_pct","low_5m_pct","high_15m_pct","low_15m_pct","high_30m_pct","low_30m_pct","vol_5m","vol_15m","vol_30m"
                    ]) else {}
                    # Historical indicators prior to gap day
                    ind = compute_indicators_prior_to(d, ticker) if any(k in fieldnames for k in ["atr14","avg_vol_30d","pos_52w","dist_52w_high_pct","gap_vs_atr"]) else {}
                    # Fundamentals as-of (best effort)
                    float_val = None
                    market_cap_val = None
                    if any(k in fieldnames for k in ["float_proxy","shares_outstanding","market_cap","float_rotation_pm","float_rotation_30m"]):
                        float_val, market_cap_val = fetch_fundamentals_asof(ticker, d)
                    # Derived metrics
                    gap_vs_atr = (abs(o - prev_c) / ind["atr14"]) if ind.get("atr14") else None
                    float_rotation_pm = (pm.get("pm_volume", 0) / float_val) if float_val and pm else None
                    float_rotation_30m = (rth.get("vol_30m", 0) / float_val) if float_val and rth else None
                    # Distances to 200SMA
                    open_to_sma200_pct = None
                    if ind.get("sma200"):
                        open_to_sma200_pct = (o - ind["sma200"]) / ind["sma200"] if ind["sma200"] else None
                    # Short interest (if available via Polygon)
                    si_shares = si_pct_float = si_days_to_cover = None
                    si_settlement = None
                    if any(k in fieldnames for k in [
                        "short_interest_shares","short_percent_float","days_to_cover","short_settlement_date"
                    ]):
                        si_shares, si_pct_float, si_days_to_cover, si_settlement = fetch_short_interest_asof(ticker, d)
                        # If API does not provide short_percent_float, compute from proxy float
                        if (si_pct_float is None) and si_shares and float_val and float_val > 0:
                            try:
                                si_pct_float = float(si_shares) / float(float_val)
                            except Exception:
                                si_pct_float = None
                    # Times of RTH high/low
                    hodlod = compute_rth_hod_lod_times(ticker, d) if any(k in fieldnames for k in ["hod_time","lod_time"]) else {}

                    row = {
                        "date": d.isoformat(),
                        "ticker": ticker,
                        "prev_close": round(prev_c, 6),
                        "open": round(o, 6),
                        "gap_pct": round(gap_pct, 6),
                        "high": round(h, 6),
                        "low": round(l, 6),
                        "open_to_high_pct": round(open_to_high, 6) if open_to_high is not None else None,
                        "open_to_low_pct": round(open_to_low, 6) if open_to_low is not None else None,
                        "open_to_close_pct": round(open_to_close, 6) if open_to_close is not None else None,
                        "close": round(c, 6),
                        # Fundamentals (as-of best effort)
                        "float_proxy": float_val,
                        "shares_outstanding": float_val,
                        "market_cap": market_cap_val,
                        # Premarket
                        "pm_high": pm.get("pm_high"),
                        "pm_low": pm.get("pm_low"),
                        "pm_vwap": pm.get("pm_vwap"),
                        "pm_volume": pm.get("pm_volume"),
                        "pm_trades": pm.get("pm_trades"),
                        # Early session
                        "ret_5m": rth.get("ret_5m"),
                        "ret_15m": rth.get("ret_15m"),
                        "ret_30m": rth.get("ret_30m"),
                        "high_5m_pct": rth.get("high_5m_pct"),
                        "low_5m_pct": rth.get("low_5m_pct"),
                        "high_15m_pct": rth.get("high_15m_pct"),
                        "low_15m_pct": rth.get("low_15m_pct"),
                        "high_30m_pct": rth.get("high_30m_pct"),
                        "low_30m_pct": rth.get("low_30m_pct"),
                        "vol_5m": rth.get("vol_5m"),
                        "vol_15m": rth.get("vol_15m"),
                        "vol_30m": rth.get("vol_30m"),
                        # Historical indicators
                        "atr14": ind.get("atr14"),
                        "avg_vol_30d": ind.get("avg_vol_30d"),
                        "pos_52w": ind.get("pos_52w"),
                        "dist_52w_high_pct": ind.get("dist_52w_high_pct"),
                        "sma200": ind.get("sma200"),
                        "prevclose_to_sma200_pct": ind.get("prevclose_to_sma200_pct"),
                        "open_to_sma200_pct": round(open_to_sma200_pct, 6) if open_to_sma200_pct is not None else None,
                        "gap_vs_atr": round(gap_vs_atr, 6) if gap_vs_atr is not None else None,
                        # Float rotation
                        "float_rotation_pm": round(float_rotation_pm, 6) if float_rotation_pm is not None else None,
                        "float_rotation_30m": round(float_rotation_30m, 6) if float_rotation_30m is not None else None,
                        # Short interest (Polygon, if available)
                        "short_interest_shares": si_shares,
                        "short_percent_float": round(si_pct_float, 6) if isinstance(si_pct_float, (int, float)) and si_pct_float is not None else si_pct_float,
                        "days_to_cover": si_days_to_cover,
                        "short_settlement_date": si_settlement,
                        # RTH time of high/low
                        "hod_time": hodlod.get("hod_time"),
                        "lod_time": hodlod.get("lod_time"),
                    }
                    filtered_row = {k: row[k] for k in fieldnames if k in row}
                    rows.append(filtered_row)
                    gappers_this_day += 1

                    # Append row to Google Sheets immediately
                    values = [list(filtered_row.values())]
                    body = {'values': values}
                    service.spreadsheets().values().append(
                        spreadsheetId=sheet_id,
                        range=a1_anchor,  # Append to the end
                        valueInputOption='RAW',
                        insertDataOption='INSERT_ROWS',
                        body=body
                    ).execute()

            prev_close_by_ticker[ticker] = c

        processed_days += 1
        total_gappers += gappers_this_day

        # Progress print remains the same
        pct = (processed_days / total_days) * 100
        current_time = time.time()
        elapsed = current_time - start_time
        eta_seconds = elapsed / pct * (100 - pct) if pct > 0 else 0
        eta_str = f"{int(eta_seconds // 3600)}h {int((eta_seconds % 3600) // 60)}m {int(eta_seconds % 60)}s" if eta_seconds > 0 else "N/A"
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] Progress: {pct:.1f}% ({processed_days}/{total_days} weekdays through {d}) - Gappers found: {total_gappers} - ETA: {eta_str}")

    # Final status
    end_time = time.time()
    total_elapsed = end_time - start_time
    total_str = f"{int(total_elapsed // 3600)}h {int((total_elapsed % 3600) // 60)}m {int(total_elapsed % 60)}s"
    print(f"Progress: 100.0% ({total_days}/{total_days} weekdays completed) - Total gappers: {total_gappers} - Total time: {total_str}")

    print(f"Wrote {len(rows)} rows to Google Sheets")
    return sheet_id


def parse_date_arg(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def fetch_float(ticker: str) -> float:
    url = f"https://api.polygon.io/v3/reference/tickers/{ticker}?apiKey={POLYGON_API_KEY}"
    resp = requests.get(url)
    if resp.ok:
        data = resp.json()
        return data.get("results", {}).get("share_class_shares_outstanding")
    return None


def fetch_market_cap_on_date(ticker: str, target_date: date) -> float:
    url = (
        f"https://api.polygon.io/vX/reference/financials"
        f"?ticker={ticker}&order=desc&limit=100&apiKey={POLYGON_API_KEY}"
    )
    resp = requests.get(url)
    if not resp.ok:
        return None
    data = resp.json()
    results = data.get("results", [])
    for report in results:
        report_date = report.get("reportPeriod") or report.get("filing_date")
        if report_date and report_date <= target_date.isoformat():
            # Check for market_cap in various places
            if "market_cap" in report:
                return report["market_cap"]
            if "metrics" in report and "market_cap" in report["metrics"]:
                return report["metrics"]["market_cap"]
            if "financials" in report and "market_cap" in report["financials"]:
                return report["financials"]["market_cap"]
    return None


def fetch_fundamentals_asof(ticker: str, target_date: date) -> Tuple[Optional[float], Optional[float]]:
    """Return (shares_outstanding, market_cap) as of latest filing before target_date.
    Falls back to current reference values if not available.
    """
    url = (
        f"https://api.polygon.io/vX/reference/financials?"
        f"ticker={ticker}&order=desc&limit=100&apiKey={POLYGON_API_KEY}"
    )
    try:
        resp = requests.get(url)
        if resp.ok:
            data = resp.json()
            results = data.get("results", [])
            for report in results:
                report_date = report.get("reportPeriod") or report.get("filing_date")
                if report_date and report_date <= target_date.isoformat():
                    fin = report.get("financials") or {}
                    metrics = report.get("metrics") or {}
                    shares = (
                        metrics.get("weighted_shares_outstanding")
                        or metrics.get("weighted_average_shares_outstanding_basic")
                        or fin.get("income_statement", {}).get("weighted_average_shares_outstanding_basic")
                        or fin.get("income_statement", {}).get("weighted_average_shares_outstanding_diluted")
                    )
                    mcap = (
                        report.get("market_cap")
                        or metrics.get("market_cap")
                    )
                    if shares or mcap:
                        return shares, mcap
    except Exception:
        pass
    # Fallback to current reference values
    url_ref = f"https://api.polygon.io/v3/reference/tickers/{ticker}?apiKey={POLYGON_API_KEY}"
    try:
        resp2 = requests.get(url_ref)
        if resp2.ok:
            data2 = resp2.json().get("results", {})
            shares2 = data2.get("share_class_shares_outstanding")
            mcap2 = data2.get("market_cap")
            return shares2, mcap2
    except Exception:
        pass
    return None, None


def main(argv: List[str]) -> int:
    if not POLYGON_API_KEY:
        sys.stderr.write(
            "Error: POLYGON_API_KEY not set. Export it in your environment.\n"
        )
        return 2

    config = load_config("config_gappers.yaml")
    today = date.today()
    years_back = float(config.get("years_back", 5))
    default_start = years_ago(today, years_back)
    start = default_start
    end = today
    threshold = float(config.get("gap_threshold_pct", 20.0))
    sheet_id = '1BuJoTLMW8h8lYibU2YL6uS6rGk3mQNIV6FS4rn9KP-g'  # Use provided Sheet ID
    fieldnames = config.get("dimensions", []) + config.get("metrics", [])

    # Args: [start] [end] [threshold]
    if len(argv) >= 2 and argv[1]:
        start = parse_date_arg(argv[1])
    if len(argv) >= 3 and argv[2]:
        end = parse_date_arg(argv[2])
    if len(argv) >= 4 and argv[3]:
        threshold = float(argv[3])

    if start > end:
        sys.stderr.write("Start date must be <= end date.\n")
        return 2

    print(
        f"Scanning weekdays {start} -> {end} for gap >= {threshold:.2f}%..."
    )
    compute_gappers(start, end, threshold, sheet_id, fieldnames)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
