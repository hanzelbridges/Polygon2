# Gapper CSV Columns

Simple explanations of every column the script can output. Unless stated otherwise:

- All prices are in USD.
- Percentage-like fields are decimal fractions (e.g., 0.20 = 20%).
- Times refer to US Eastern Time (ET). Premarket is 04:00–09:30 ET; Regular session opens 09:30 ET.

## Dimensions

- date: Trading date for the row (YYYY-MM-DD).
- ticker: Stock symbol (may include suffixes like ".WS" for warrants).

## Core Daily Prices

- prev_close: Previous trading day’s official close price.
- open: Current day’s regular-session opening price (09:30 ET).
- gap_pct: Size of the opening gap = (open - prev_close) / prev_close.
- high: Current day’s high price (full session high for that date).
- low: Current day’s low price (full session low for that date).
- open_to_high_pct: Intraday stretch above the open = (high - open) / open.
- open_to_low_pct: Intraday drop below the open = (low - open) / open.
- open_to_close_pct: Full-session move from open to close = (close - open) / open.
- close: Current day’s official close price.
- hod_time: Time of the RTH high-of-day (09:30–16:00 ET), formatted HH:MM in ET.
- lod_time: Time of the RTH low-of-day (09:30–16:00 ET), formatted HH:MM in ET.

## Premarket Structure (as of the gap day)

- pm_high: Highest trade price during premarket (04:00–09:30 ET).
- pm_low: Lowest trade price during premarket.
- pm_vwap: Volume-weighted average price over premarket minutes.
- pm_volume: Total share volume traded during premarket.
- pm_trades: Total number of trades (sum of per-minute transactions) during premarket.

## Early-Session From Open (as of the gap day)

- ret_5m: Return from the opening price to the last price in the first 5 minutes.
- ret_15m: Return from the opening price to the last price in the first 15 minutes.
- ret_30m: Return from the opening price to the last price in the first 30 minutes.
- high_5m_pct: Max stretch above the open during the first 5 minutes.
- low_5m_pct: Max drop below the open during the first 5 minutes.
- high_15m_pct: Max stretch above the open during the first 15 minutes.
- low_15m_pct: Max drop below the open during the first 15 minutes.
- high_30m_pct: Max stretch above the open during the first 30 minutes.
- low_30m_pct: Max drop below the open during the first 30 minutes.
- vol_5m: Total share volume in the first 5 minutes.
- vol_15m: Total share volume in the first 15 minutes.
- vol_30m: Total share volume in the first 30 minutes.

## Historical Context (computed strictly up to the day before the gap)

- atr14: 14‑period Average True Range on daily bars (Wilder’s method), indicates typical daily movement.
- avg_vol_30d: Average daily share volume over the prior 30 trading days.
- pos_52w: Position of yesterday’s close within the 52‑week range, from 0 (at 52‑week low) to 1 (at 52‑week high).
- dist_52w_high_pct: Distance from yesterday’s close up to the 52‑week high, relative to yesterday’s close.
- sma200: 200‑day simple moving average of closing prices as of yesterday.
- prevclose_to_sma200_pct: Distance from yesterday’s close to the 200‑day SMA, relative to the SMA.
- open_to_sma200_pct: Distance from today’s open to the 200‑day SMA, relative to the SMA.

## Derived Ratios

- gap_vs_atr: Absolute gap size divided by ATR14; gauges how extreme the gap is versus typical range.

<!-- Ownership/insider/institutional override fields removed: not sourced from Polygon. -->

## Short Interest (from Polygon, if entitled)

- short_interest_shares: Shares sold short, as of the latest available settlement date on/before the gap day.
- days_to_cover: Days to cover (short interest divided by average daily volume as defined by the data source).
- short_settlement_date: Settlement date associated with the short interest values (YYYY-MM-DD).

Notes: Availability depends on your Polygon plan and entitlements. If the API is unavailable or returns no results prior to the gap date, these fields will be blank. The `short_percent_float` field is intentionally omitted because Polygon does not provide it and computing it from present-day float proxies would not be point-in-time accurate. We also omit `float_proxy`, `shares_outstanding`, `market_cap`, and float-rotation metrics because those values are not point-in-time accurate with our current data sources.

## Earnings (approximate; Polygon financials)

- earnings_flag: True if Polygon financials show a filing_date on the gap day or previous weekday.
- earnings_report_date: The filing_date that triggered the flag (YYYY-MM-DD).

Notes: filing_date is a proxy for the earnings announcement timing and may not perfectly align with press-release times. We do not include EPS estimates/surprises or revenue here.

## Point-in-Time Fundamentals (Sharadar)

- shares_outstanding_pit: Shares outstanding as of the latest filing date on/before the gap day (Sharadar SF1 MRQ, using datekey).
- market_cap_pit: Market capitalization from the same SF1 record.
 
Notes: For backtesting, we query SF1 via Nasdaq Data Link datatables using As Reported dimensions (ARQ, then ARY) and select the latest filing with datekey <= gap_date. This avoids restatement look-ahead bias.

## Notes & Conventions

- Point‑in‑time logic: Historical indicators use only data available up to the day before the gap.
- Decimal formatting: If `decimal_separator` in `config_gappers.yaml` is set to `comma`, numeric values will use `,` instead of `.` in the CSV.
- Data sources: Prices/volumes from Polygon aggregates.
