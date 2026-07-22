# :chart_with_upwards_trend: Market forecast dashboard (LightGBM)

A Streamlit dashboard that forecasts **where the market is going** over the next
5 / 10 / 15 bars (horizons are configurable): the **direction**, the **move size
in points**, the **from → to price range** with an 80% uncertainty fan, and
**where the move likely starts and ends** — the bar/date and price level at
which the move likely begins, and where it is likely exhausted (or a "sideways,
no clear move" call when the forecast stays within noise).

Under the hood, per horizon it trains four small LightGBM models on
close-price-derived features (lagged returns, volatility, RSI, MACD, momentum):

- quantile regressors (10th / 50th / 90th percentile) predicting the forward
  move in points — the median gives the headline forecast, the outer quantiles
  the shaded fan;
- a classifier giving the probability the move is up.

A chronological 80/20 backtest reports out-of-sample directional accuracy and
mean absolute error, so the forecast is never presented without its honest
track record.

Data comes from a built-in synthetic market series (geometric Brownian motion
with regime shifts), or upload your own data:

- a simple CSV with `Date` and `Close` columns, or
- **options order-flow exports** (QuantData style, one or more files) — trades
  are bucketed into intraday bars and summarized into order-flow features
  (bullish/bearish premium imbalance, delta-weighted flow, put/call premium
  ratio) that feed the model alongside the price features, with horizons in
  minutes.

A **signal confidence filter** turns forecasts into LONG / SHORT / STAND ASIDE
calls: it only fires when the direction model clears your chosen confidence
bar, and reports the *measured* out-of-sample hit rate at that bar — trading
frequency for accuracy honestly, rather than promising unrealistic win rates.

> Educational demo — not financial advice.

### How to run it on your own machine

1. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```

2. Run the app

   ```
   $ streamlit run streamlit_app.py
   ```
