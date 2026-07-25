# ⚖️ Dealer Positioning Dashboard

A Streamlit dashboard that reads **market-maker (dealer) hedging obligations**
from options data and fuses every level system into one ranked read. It detects
whether dealers are *forced* to hedge with or against the market, finds the
price levels where that behavior concentrates or flips, cross-checks them
against raw open interest, institutional block flow, and dark-pool prints, and
— given several days of data — validates whether those levels actually held.

Everything is driven by CSV uploads (no live feed). It opens on a bundled
synthetic sample so it works with zero setup.

## How to run it

```
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Switch the sidebar to **Upload CSV** to analyze your own data.

To host it on [Streamlit Community Cloud](https://share.streamlit.io): point a
new app at this repo with `streamlit_app.py` as the main file. `requirements.txt`
and `packages.txt` (`libgomp1`, the OpenMP runtime LightGBM links against) are
picked up automatically. No secrets or API keys — the app reads uploads only.

## Input data

Four file shapes are auto-detected — no configuration:

- **Long-format chain** — one row per contract (broker export). Column names are
  matched against synonyms (`OI`/`Open Int`, `IV`/`Impl Vol`, `Exp`, `Call/Put`…).
  See `data/sample_option_chain.csv`.
- **CBOE side-by-side** — calls and puts per strike (`quotedata` download,
  preamble and all).
- **Trade-level order flow** (QuantData "Options Order Flow") — one row per
  print. Prints are collapsed to a correct per-contract chain (open interest is
  taken as a max, never summed), ticker/spot/date are inferred, and the ask/bid
  side codes power the signed-flow, block, and conviction features. Both type
  columns are read: `Consolidation Type` (SWEEP / BLOCK / SPLIT — the shape)
  and `Trade Type` (AUTO, FLR, CROSS, COB, AUCT, ISO, with `SPRD_`/`SPRD_LEG_`
  and `TIED_` prefixes — the mechanism).
- **Dark-pool / equity blocks** — price + size per print, no strike. Used as a
  confluence *overlay* on an options analysis (upload alongside a chain).

If auto-detection fails, a manual column-mapping form appears. Missing greeks
are filled with Black-Scholes gamma from each contract's IV. Multiple files
combine into one book — or, when they cover different days, can be **compared as
history** instead.

## What it computes

**Headline read**

| Output | Meaning |
|---|---|
| **Verdict** | Dealers **long gamma** → buy dips / sell rips (stabilizing, mean-reverting) vs **short gamma** → sell weakness / buy strength (destabilizing, trending) |
| **TL;DR** | One-line synthesis of regime, top level, and expected move |
| **Master levels (confluence)** | Every level system fused into one 0-100 ranking; a level confirmed by more independent systems ranks higher, with a high/medium/low confidence flag |

**Levels (all pinpoint, not strike-rounded)**

| Level | Meaning |
|---|---|
| **Gamma flip** | Spot where net GEX crosses zero — bisected to the cent |
| **Gamma call / put wall** | Peak aggregate dealer gamma per side |
| **OI call / put wall** | Peak raw open interest per side |
| **Magnets & accelerators** | Every positive (pin) / negative (repel) net-gamma peak, ranked |
| **OI clusters** | Kernel-smoothed open-interest concentration, call/put labeled |
| **Block commitment levels** | Where negotiated institutional premium concentrated |
| **Dark-pool levels** | Where off-exchange equity size concentrated |
| **Max pain / expected move** | Expiry gravitation; 1σ straddle range |

**Flow intelligence** (order-flow files)

- **Net GEX / DEX / vanna / charm** — dollar hedging demand per 1% move, dealer
  delta inventory, and forced re-hedging from IV drops and time decay.
- **Block intelligence** — the blocks-only vs sweeps-only books side by side
  (smart vs fast money) with an automatic aligned/divergent verdict, plus a
  floor-only book when the file distinguishes floor prints.
- **Block types** — a "block" can mean very different things. Block prints are
  split by *how they printed* and ranked into tiers: 🤝 **negotiated** (floor,
  cross — size someone had to find a counterparty for), 📣 **facilitated**
  (complex-order book, auction — real size worked publicly), ⚡ **electronic**
  (the default route), 🧩 **fragment** (one leg of a spread package). On a real
  QuantData export the fragments are ~60% of block *prints* and ~3% of block
  *premium*, so they are excluded from levels and the block book by default.
  Stock-tied (delta-hedged) prints are flagged 🔗 — a volatility position, not
  a directional one — and cancelled/busted prints are dropped outright. Each
  type carries the **premium-weighted strike it traded at** and that level's
  distance from spot: a $25M block 5% out is a different trade from the same
  size at the money.
- **Conviction filter** — rebuild *every* level from flagged prints only
  (sweeps / blocks / floor / cross / auto / splits / golden / unusual /
  opening).
- **Consolidated flow by type** — premium, contracts, prints, average size and
  net direction per *precise* execution type: `floor block` 🏛️🧱, `auto sweep`
  ⚡🌊, `cross` 🔁, `block`, `sweep`… Venue (floor / auto / cross) and shape
  (block / sweep / split / multi) are read as **separate axes**, so a
  floor-negotiated block is never averaged in with the electronic default.
  Floor and cross prints count as institutional size; plain `auto` does not.
  A code the mapping doesn't know keeps its raw text instead of being silently
  bucketed.
- **Notable flow** — the largest premium prints with flags.
- **Intraday timeline** — cumulative signed flow through the session, split by
  blocks vs sweeps: *when* and *who*.

**Multi-day (upload several daily files)**

- **Level migration** — how the flip, walls, and regime moved day over day.
- **Block campaigns** — contracts hit by blocks across multiple days.
- **Level hit-rate** — each prior day's levels tested against the next day's
  range (reconstructed from print reference prices): did they actually hold?
- **Adaptive confluence** — the hit-rate feeds back to re-weight the confluence
  scoring by what held on this ticker (gated to avoid tuning on noise).
- **Expected-move model (LightGBM)** — the implied expected move is a *price*,
  not a forecast. A gradient-boosted-tree regression learns the map from the
  day's positioning state (GEX regime, distance to flip, wall width, DEX,
  vanna/charm, recent realized vol…) to the **next session's realized move**,
  and reports whether that beats the option market. Scored by expanding-window
  walk-forward — every evaluated prediction is out of sample — and blended into
  the implied move at exactly its measured skill: beat implied by 20% and the
  model gets 20% of the number, fail to beat it and the model gets zero. Needs
  ~10 daily files before it will say anything; until then it shows the implied
  move and how many more days it wants. Without `lightgbm` installed the same
  pipeline runs on a ridge fallback, labelled as such.

**Tools & output**

- **Scenario simulator** — re-price the whole book at a hypothetical spot/IV to
  see the regime, flip, and forced hedge flow *before* the market goes there.
- **0DTE mode** — restrict to same-day expiry (per-hour charm, move to the close).
- **Contract multiplier** — 100 for equities/index, 50 ES, 20 NQ… (levels are
  scale-invariant; only dollar figures scale).
- **Weighting** — open interest / volume / signed order flow.
- **Report** — downloadable `.md` / `.html` with the TL;DR, master levels,
  playbook, and every table.

## Assumptions & caveats

Convention-based positioning assumes dealers are long customer-sold calls and
short customer-bought puts; signed-flow mode instead reads actual trade
direction. Open interest updates once daily, the flip curve holds IV fixed
(sticky-strike), dark-pool and hit-rate ranges are reconstructions, and small
samples are flagged. The expected-move model trains on a handful of
reconstructed session closes from one ticker — its skill number is honest but
noisy, which is exactly why it only gets the weight it earns. Every level is an
estimate — this is positioning analysis, **not trading advice**.

## Development

```
pip install pytest
python -m pytest tests/                 # 105 tests
python scripts/make_sample_chain.py     # regenerate the bundled sample chain
```

Layout: `dealer_gex/` (parsing, analytics, forecast, report) ·
`streamlit_app.py` (UI) · `tests/` · `data/` (bundled sample).

`lightgbm` is only needed for the expected-move model; the rest of the app —
and the test suite — runs without it.
