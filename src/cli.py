import argparse
import sys

from polygon_client import (
    get_aggregates,
    get_daily_open_close,
    get_last_trade,
    get_previous_close,
    pretty_print,
    PolygonError,
)


def cmd_prev_close(args: argparse.Namespace) -> int:
    data = get_previous_close(args.ticker, adjusted=(not args.unadjusted))
    pretty_print(data)
    return 0


def cmd_aggs(args: argparse.Namespace) -> int:
    data = get_aggregates(
        args.ticker,
        args.multiplier,
        args.timespan,
        args.start,
        args.end,
        adjusted=(not args.unadjusted),
        sort=args.sort,
        limit=args.limit,
    )
    pretty_print(data)
    return 0


def cmd_open_close(args: argparse.Namespace) -> int:
    data = get_daily_open_close(args.ticker, args.date, adjusted=(not args.unadjusted))
    pretty_print(data)
    return 0


def cmd_last_trade(args: argparse.Namespace) -> int:
    data = get_last_trade(args.ticker)
    pretty_print(data)
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="polygon-cli",
        description="Minimal Polygon.io CLI using standard library only",
    )
    sub = p.add_subparsers(dest="command", required=True)

    # prev-close
    sp = sub.add_parser("prev-close", help="Previous close aggregate for a ticker")
    sp.add_argument("ticker", help="Ticker symbol, e.g. AAPL")
    sp.add_argument("--unadjusted", action="store_true", help="Use unadjusted prices")
    sp.set_defaults(func=cmd_prev_close)

    # aggs
    sp = sub.add_parser("aggs", help="Aggregates (candles) for a ticker and date range")
    sp.add_argument("ticker", help="Ticker symbol, e.g. AAPL")
    sp.add_argument("multiplier", type=int, help="Aggregate multiplier, e.g. 1")
    sp.add_argument("timespan", choices=[
        "minute", "hour", "day", "week", "month", "quarter", "year"
    ], help="Aggregate timespan unit")
    sp.add_argument("start", help="Start date/time (YYYY-MM-DD or ISO8601)")
    sp.add_argument("end", help="End date/time (YYYY-MM-DD or ISO8601)")
    sp.add_argument("--unadjusted", action="store_true", help="Use unadjusted prices")
    sp.add_argument("--sort", choices=["asc", "desc"], default="asc", help="Sort order")
    sp.add_argument("--limit", type=int, default=50000, help="Max results (API limit applies)")
    sp.set_defaults(func=cmd_aggs)

    # open-close
    sp = sub.add_parser("open-close", help="Daily open/close for a given date")
    sp.add_argument("ticker", help="Ticker symbol, e.g. AAPL")
    sp.add_argument("date", help="Date (YYYY-MM-DD)")
    sp.add_argument("--unadjusted", action="store_true", help="Use unadjusted prices")
    sp.set_defaults(func=cmd_open_close)

    # last-trade
    sp = sub.add_parser("last-trade", help="Last trade for a ticker")
    sp.add_argument("ticker", help="Ticker symbol, e.g. AAPL")
    sp.set_defaults(func=cmd_last_trade)

    return p


def main(argv: list[str] | None = None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
        return args.func(args)
    except PolygonError as e:
        sys.stderr.write(f"Error: {e}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

