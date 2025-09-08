import json
import os
import sys
import urllib.parse
import urllib.request
from typing import Any, Dict, Optional


BASE_URL = "https://api.polygon.io"
ENV_API_KEY = "POLYGON_API_KEY"


class PolygonError(RuntimeError):
    pass


def _get_api_key() -> str:
    api_key = os.getenv(ENV_API_KEY)
    if not api_key:
        raise PolygonError(
            f"Missing API key. Set environment variable {ENV_API_KEY}."
        )
    return api_key


def api_get(path: str, params: Optional[Dict[str, Any]] = None, timeout: int = 30) -> Dict[str, Any]:
    """
    Perform a GET request to the Polygon.io API using only the standard library.

    Args:
        path: The API path beginning with '/'. Example: '/v2/aggs/ticker/AAPL/prev'
        params: Query parameters excluding the 'apiKey' which is injected automatically.
        timeout: Request timeout in seconds.

    Returns:
        Parsed JSON response as a dict.
    """
    if not path.startswith("/"):
        path = "/" + path

    api_key = _get_api_key()
    q = dict(params or {})
    q["apiKey"] = api_key
    query = urllib.parse.urlencode(q)
    url = f"{BASE_URL}{path}?{query}"

    req = urllib.request.Request(url, headers={"User-Agent": "polygon-cli/0.1"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec B310
            data = resp.read()
            return json.loads(data)
    except urllib.error.HTTPError as e:
        try:
            err_body = e.read().decode("utf-8", errors="replace")
        except Exception:
            err_body = "<no body>"
        raise PolygonError(f"HTTP {e.code} for {path}: {err_body}") from e
    except urllib.error.URLError as e:
        raise PolygonError(f"Network error for {path}: {e}") from e


# Convenience wrappers for common endpoints

def get_previous_close(ticker: str, adjusted: bool = True) -> Dict[str, Any]:
    """GET /v2/aggs/ticker/{ticker}/prev"""
    path = f"/v2/aggs/ticker/{urllib.parse.quote(ticker)}/prev"
    return api_get(path, {"adjusted": str(adjusted).lower()})


def get_aggregates(
    ticker: str,
    multiplier: int,
    timespan: str,
    from_date: str,
    to_date: str,
    *,
    adjusted: bool = True,
    sort: str = "asc",
    limit: int = 50000,
) -> Dict[str, Any]:
    """GET /v2/aggs/ticker/{ticker}/range/{multiplier}/{timespan}/{from}/{to}"""
    path = \
        f"/v2/aggs/ticker/{urllib.parse.quote(ticker)}/range/{multiplier}/{urllib.parse.quote(timespan)}/" \
        f"{urllib.parse.quote(from_date)}/{urllib.parse.quote(to_date)}"
    return api_get(
        path,
        {
            "adjusted": str(adjusted).lower(),
            "sort": sort,
            "limit": limit,
        },
    )


def get_daily_open_close(ticker: str, date: str, adjusted: bool = True) -> Dict[str, Any]:
    """GET /v1/open-close/{ticker}/{date}"""
    path = f"/v1/open-close/{urllib.parse.quote(ticker)}/{urllib.parse.quote(date)}"
    return api_get(path, {"adjusted": str(adjusted).lower()})


def get_last_trade(ticker: str) -> Dict[str, Any]:
    """GET /v2/last/trade/{ticker}"""
    path = f"/v2/last/trade/{urllib.parse.quote(ticker)}"
    return api_get(path)


def pretty_print(obj: Any) -> None:
    json.dump(obj, sys.stdout, indent=2, sort_keys=False)
    sys.stdout.write("\n")

