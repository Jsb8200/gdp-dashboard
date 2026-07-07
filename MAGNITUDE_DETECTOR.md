# Magnitude Detector (LightGBM)

Detects — **before the move starts** — bars where a high-confluence,
high-magnitude move is imminent. Instead of predicting direction, it answers:
*"is price about to travel far?"* That makes it a pre-breakout / expansion
filter you can layer on top of a directional strategy or zone analysis.

## How it works

**Label.** For each bar `t` we measure the maximum excursion (up or down)
price achieves over the next `horizon` bars, normalized by ATR at `t`:

```
fwd_mag[t] = max( max(high[t+1..t+H]) - close[t],
                  close[t] - min(low[t+1..t+H]) ) / ATR[t]
```

A bar is a positive when `fwd_mag >= magnitude_threshold` (default: 3 ATRs
within the next 12 bars). Features only look backward, labels only look
forward — the split point is the close of bar `t`, so the model can be
scored live on the most recent closed bar.

**Features** — the confluence conditions that tend to precede expansion:

| Group | Features |
|---|---|
| Volatility compression | Bollinger bandwidth + percentile, fast/slow ATR ratio, ATR percentile, Keltner squeeze flag + duration, range contraction, realized vol + slope |
| Trend alignment | EMA stack alignment (8/21/50/200), distance to slow EMA, ADX, RSI, MACD histogram |
| Range / zones | Donchian position + width, distance to range high/low, number of recent tests of the range edges |
| Volume | Volume z-score, volume dry-up ratio, OBV slope |
| Candle structure | Mean body/range ratio, inside-bar streak |
| Composite | `confluence_score` — count of concurrent setup conditions |

**Model.** Two LightGBM heads sharing the same features:

- **classifier** → `p_big_move`: probability the next `H` bars contain a
  ≥ `k`·ATR excursion
- **regressor** (L1) → `expected_mag_atr`: expected forward magnitude in ATRs

**Evaluation.** Expanding-window walk-forward splits with an embargo gap of
`horizon` bars between train and test, so overlapping forward labels never
leak. Reported per fold: AUC, average precision, precision in the top-10%
of scores vs. the base rate, and magnitude MAE.

## Quick start

```bash
pip install -r requirements.txt

# demo on synthetic data (compression→expansion regimes)
python -m magnitude_detector.train --synthetic

# real data: csv with timestamp,open,high,low,close,volume
python -m magnitude_detector.train --csv data/eurusd_1h.csv \
    --horizon 12 --threshold 3.0 --out models/eurusd

# score the most recent closed bar
python -m magnitude_detector.predict --model models/eurusd --csv data/eurusd_1h.csv

# score every bar and export signals
python -m magnitude_detector.predict --model models/eurusd \
    --csv data/eurusd_1h.csv --all --out signals.csv
```

Example output (synthetic demo, 20k hourly bars):

```
 fold  test_bars  base_rate    auc  avg_precision  precision_top10pct
    1       3957     0.2209 0.8306         0.6315              0.7570
    2       3957     0.2267 0.8091         0.6292              0.7747
    3       3957     0.2345 0.7863         0.5783              0.7139
    4       3957     0.2360 0.8316         0.6545              0.8127
```

Top-decile precision ≈ 0.76 vs. a 0.23 base rate — the model concentrates
the true pre-move bars into its highest scores.

## Tuning

- `--horizon` — bars the move has to develop in. Match it to your holding
  period (e.g. 12 × 1h bars ≈ half a trading day).
- `--threshold` — how big "big" is, in ATRs. Higher = rarer, cleaner
  positives; lower = more signals, more noise.
- `--alert-threshold` (predict) — probability cutoff for flagging a bar.
  Pick it from the walk-forward precision numbers to hit your desired
  hit-rate/frequency trade-off.

## Project layout

```
magnitude_detector/
  config.py     # all knobs in one dataclass
  data.py       # csv loader + regime-switching synthetic generator
  features.py   # backward-looking confluence features
  labels.py     # forward max-excursion labels (ATR-normalized)
  model.py      # LightGBM two-head wrapper + walk-forward splitter
  train.py      # CLI: walk-forward eval + final model
  predict.py    # CLI: live scoring / batch signals
tests/
  test_pipeline.py  # leakage checks, label math, end-to-end train
```

## Notes

- Direction is deliberately out of scope: `fwd_direction` is included in the
  label frame for analysis, but the detector only sizes the move. Combine
  with your directional logic (zone break direction, trend filter, etc.).
- All features are computed on closed bars only; score after each bar close.
- Run tests with `python -m pytest tests/ -q`.
