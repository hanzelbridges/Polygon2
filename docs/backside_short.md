# Backside Short Backtest

## Data extraction

Use `backside_backtest.py` to scan Polygon minute data for names that spike off the prior close and then break down. The script fetches grouped aggregates to find candidates, downloads minute bars once per ticker/day, and emits one row per (rise, drop) threshold combination that triggers.

```powershell
py backside_backtest.py \
  --months 18 \
  --rise-thresholds 0.8:1.4:0.1 \
  --drop-thresholds 0.1:0.3:0.05 \
  --session-start 04:00 \
  --session-end 20:00 \
  --output output/backside_short_events.csv
```

Key CLI options:
- `--rise-thresholds`: decimal or percent inputs (`0.8` or `80`) and optional ranges (`start:end:step`).
- `--drop-thresholds`: decline thresholds from the intraday peak after the rise triggers.
- `--session-start` / `--session-end`: earliest and latest Eastern times to keep (premarket through after-hours by default).
- `--no-common-filter`: include all tickers; otherwise the script restricts to Polygon `type=CS` equities.

The CSV includes both the thresholds used (`rise_threshold_decimal`, `drop_threshold_decimal`) and the realized stats (rise/drop percentages, entry price, post-entry high/low, volume since the rise, etc.). Columns such as `window_start_price`, `trigger_price`, `entry_price`, `post_entry_high`, `post_entry_low`, and `window_volume` line up with `para_analysis.py` so the existing analytics pipeline can be reused.

## Threshold optimization

`optimize_backside_thresholds.py` groups the dataset by rise/drop settings and evaluates the short performance with the usual stop-loss and take-profit parameters.

```powershell
py optimize_backside_thresholds.py \
  --events output/backside_short_events.csv \
  --stop-multiplier 1.6 \
  --take-profit 0.7 \
  --entry-cutoff 09:29 \
  --min-trades 10 \
  --optimize-by total_pl
```

Useful flags:
- `--rise-values` / `--drop-values` to test a custom subset rather than every combination in the file.
- `--capital-per-trade` to translate expectancy into total dollar P/L.
- `--output` to persist the full optimization grid for further analysis.

Because the exports retain `rise_threshold_decimal` and `drop_threshold_decimal`, you can also filter directly in pandas or plug the data into `para_analysis.apply_filters` for deeper drill-downs before running stop-loss or take-profit sweeps.

