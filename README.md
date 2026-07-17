# ⚖️ Dealer Positioning Dashboard

A Streamlit dashboard that detects **market-maker (dealer) hedging obligations**
from options open interest: net gamma exposure (GEX), whether dealers are
*forced* to hedge short or long, and the price levels where that behavior flips
or concentrates — the **gamma flip**, **call wall**, and **put wall**. It also
generates a downloadable **dealer-positioning report** (markdown or HTML).

## How to run it

1. Install the requirements

   ```
   pip install -r requirements.txt
   ```

2. Run the app

   ```
   streamlit run streamlit_app.py
   ```

The app opens with a bundled synthetic SPY-like sample chain so everything works
without any data. Switch the sidebar to **Upload CSV** to analyze your own
option chains.

## Input data

Upload option-chain CSV exports from your broker or CBOE. Two shapes are
auto-detected:

- **Long format** — one row per contract, with columns like
  `Expiration Date, Type, Strike, Open Interest, Volume, Implied Volatility`
  (column names are matched case-insensitively against common synonyms:
  `OI`/`Open Int`, `IV`/`Impl Vol`, `Exp`/`Expiration`, `Call/Put`/`Right`, …).
  See `data/sample_option_chain.csv` for a working example.
- **CBOE side-by-side** — calls and puts on the same row per strike (the CBOE
  `quotedata` download format, including its metadata preamble lines).

If auto-detection fails, the sidebar shows a manual column-mapping form.
A `Gamma` column is used when present; otherwise gamma is computed with
Black-Scholes from each contract's implied volatility. If the file includes an
underlying/spot price it is picked up automatically; otherwise set it in the
sidebar. Multiple files (e.g. one per expiry) can be uploaded together.

## What it computes

| Output | Meaning |
|---|---|
| **Net GEX** | Dollar dealer gamma per 1% move: `gamma × OI × 100 × spot² × 1%`, calls positive / puts negative |
| **Verdict** | Dealers **long gamma** → obliged to buy dips and sell rips (stabilizing, mean-reverting). Dealers **short gamma** → forced to sell weakness and buy strength (destabilizing, trending) |
| **Gamma flip** | The spot level where net GEX crosses zero — recomputed across a ±15% spot grid and interpolated |
| **Call / put wall** | Strikes with the largest positive / negative dealer gamma — pin and acceleration levels |
| **Max pain** | Strike minimizing option-holder payout at expiry |
| **Report** | Downloadable `.md` / `.html` summary of all of the above with a per-expiry breakdown |

## Assumptions & caveats

Dealer positioning uses the standard GEX convention: dealers are assumed long
customer-sold calls and short customer-bought puts. Actual dealer books can
differ, open interest updates only once daily, and the flip curve holds IV
fixed (sticky-strike). Treat every level as an estimate — this is positioning
analysis, **not trading advice**.

## Development

```
pip install pytest
python -m pytest tests/
python scripts/make_sample_chain.py   # regenerate the bundled sample chain
```
