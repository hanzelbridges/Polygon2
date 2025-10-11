import csv
import os
import random
from datetime import datetime, time
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

import para_backtest_data as backtest

OUTPUT_DIR = Path('output/premarket_charts')
EVENTS_CSV = Path('output/40pct_moves_incl_pm.csv')
NUM_CHARTS = 10

PREMARKET_END = time(9, 30)


def is_premarket(time_str: str) -> bool:
    if not isinstance(time_str, str) or ':' not in time_str:
        return False
    try:
        hour, minute = map(int, time_str.split(':', 1))
        return (hour, minute) < (PREMARKET_END.hour, PREMARKET_END.minute)
    except ValueError:
        return False


def load_events() -> pd.DataFrame:
    if not EVENTS_CSV.exists():
        raise SystemExit(f"Events file not found: {EVENTS_CSV}")
    df = pd.read_csv(EVENTS_CSV)
    df['is_premarket_trigger'] = df['trigger_time_et'].apply(is_premarket)
    df = df[df['is_premarket_trigger']].copy()
    if df.empty:
        raise SystemExit("No pre-market events found in the CSV.")
    return df


def ensure_api_key():
    if not os.getenv('POLYGON_API_KEY') and getattr(backtest, 'HARD_CODED_API_KEY', ''):
        os.environ['POLYGON_API_KEY'] = backtest.HARD_CODED_API_KEY


def to_datetime_local(date_str: str, time_str: str) -> datetime | None:
    if not isinstance(time_str, str) or not time_str:
        return None
    try:
        t = datetime.strptime(time_str, '%H:%M').time()
    except ValueError:
        return None
    dt = datetime.strptime(date_str, '%Y-%m-%d')
    return datetime.combine(dt, t)


def plot_event(row: pd.Series, output_path: Path) -> None:
    trade_date = datetime.strptime(row['date'], '%Y-%m-%d').date()
    ticker = row['ticker']
    raw_bars = backtest.fetch_minute_bars(ticker, trade_date)
    session_bars = backtest.normalize_bars(raw_bars)
    if not session_bars:
        print(f"Skipping {ticker} {trade_date}: no minute bars")
        return

    records = []
    for bar in session_bars:
        dt_local = bar.dt.astimezone(backtest.get_eastern_tz()).replace(tzinfo=None)
        records.append({
            'dt': dt_local,
            'open': bar.open,
            'high': bar.high,
            'low': bar.low,
            'close': bar.close,
            'volume': bar.volume,
        })
    frame = pd.DataFrame.from_records(records)
    frame.sort_values('dt', inplace=True)

    fig, ax_price = plt.subplots(figsize=(11, 5))
    ax_price.plot(frame['dt'], frame['close'], label='Close', color='steelblue')

    # Shade premarket vs RTH
    pm_end_dt = datetime.combine(trade_date, PREMARKET_END)
    ax_price.axvspan(frame['dt'].min(), pm_end_dt, color='lightgray', alpha=0.3, label='Premarket')

    # Vertical markers
    markers = [
        ('Window Start', row['window_start_time_et'], 'tab:orange'),
        ('Trigger', row['trigger_time_et'], 'tab:red'),
        ('Entry', row.get('entry_time_et', ''), 'tab:green'),
    ]
    for label, time_str, color in markers:
        dt_marker = to_datetime_local(row['date'], time_str)
        if dt_marker is None:
            continue
        ax_price.axvline(dt_marker, color=color, linestyle='--', alpha=0.8, label=label)

    ax_price.set_title(f"{ticker} on {trade_date} (Premarket 40% Move)")
    ax_price.set_ylabel('Price')
    ax_price.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    fig.autofmt_xdate()
    ax_price.legend(loc='upper left', fontsize='small')

    ax_volume = ax_price.twinx()
    ax_volume.bar(frame['dt'], frame['volume'], width=0.0006, color='lightsteelblue', alpha=0.4, label='Volume')
    ax_volume.set_ylabel('Volume')
    ax_volume.set_ylim(0, frame['volume'].max() * 4)

    ax_price.grid(True, linestyle=':', alpha=0.4)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved chart to {output_path}")


def main() -> None:
    ensure_api_key()
    df = load_events()
    sample_size = min(NUM_CHARTS, len(df))
    sample_rows = df.sample(n=sample_size, random_state=random.randint(0, 10_000))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for _, row in sample_rows.iterrows():
        filename = f"{row['date']}_{row['ticker']}.png"
        output_path = OUTPUT_DIR / filename
        try:
            plot_event(row, output_path)
        except Exception as exc:
            print(f"Failed to plot {row['ticker']} on {row['date']}: {exc}")


if __name__ == '__main__':
    main()
