# Data drop folder

Put your OHLCV CSV files here, one file per symbol + timeframe, e.g.:

```
data/
  eurusd_1h.csv
  btcusdt_15m.csv
  nq_5m.csv
```

## Required format

Plain CSV with a header row. Column names are case-insensitive and may be in
any order:

| column | required | notes |
|---|---|---|
| `timestamp` (or `datetime`/`date`/`time`) | recommended | any pandas-parseable format, e.g. `2024-01-15 09:00:00` |
| `open` | yes | |
| `high` | yes | |
| `low` | yes | |
| `close` | yes | |
| `volume` | yes | use tick volume if that's all your broker gives you |

Example:

```csv
timestamp,open,high,low,close,volume
2024-01-15 09:00:00,1.0951,1.0963,1.0948,1.0960,18234
2024-01-15 10:00:00,1.0960,1.0971,1.0955,1.0968,20112
```

## Requirements

- **One consistent timeframe per file** (don't mix 1h and 4h rows).
- **Chronological, no gaps beyond normal market closures.** Rows are sorted
  on load, but duplicated timestamps should be removed first.
- **Enough history:** 10,000+ bars is comfortable, 5,000 is a workable
  minimum. Below that the walk-forward folds get too small to trust.
  (10k hourly bars ≈ 1.5 years of 24h markets, ≈ 5 years of equity RTH.)
- **Raw prices, not adjusted-to-death:** back-adjusted continuous futures
  are fine; percent-adjusted data that distorts high/low ranges is not.

## Then train

```bash
python -m magnitude_detector.train --csv data/eurusd_1h.csv --out models/eurusd
python -m magnitude_detector.predict --model models/eurusd --csv data/eurusd_1h.csv
```
