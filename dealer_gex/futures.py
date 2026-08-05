"""Quote the book's levels on the futures contract you actually trade.

The option chain is priced on an underlying — SPX, QQQ, GLD. The screen in
front of you is usually a future — ES, NQ, GC. Every level the dashboard
produces (flip, walls, max pain, magnets, the strike ladder) is a price on
the *chain's* scale, so pasting it into a futures chart is off by the basis.

Two kinds of relationship show up, and they are not interchangeable:

* **offset** — an index and its own future differ by carry: ES is SPX plus
  a few points that decay to zero at expiry. ``ES = SPX + 26.98``.
* **ratio** — an ETF and a future track the same thing at different unit
  sizes: QQQ is a fraction of the Nasdaq-100, GLD a fraction of an ounce.
  ``NQ = QQQ x 41.765``, ``GC = GLD x 10.87``.

Using an offset where a ratio belongs is not a small error — adding 41.765
to QQQ at 688 is a 6% basis, which is nonsense, while multiplying is right.
So the mode is part of the conversion, not an assumption baked into it.

This is deliberately a *display* transform applied after the analysis, not
a re-pricing of the chain. Shifting spot and strikes and re-running
Black-Scholes would move every gamma slightly, because gamma depends on
ln(S/K) and a parallel shift does not preserve it. Converting the finished
levels keeps the math honest and the arithmetic exact.

What converts and what does not follows from dimension:

* a **price level** converts (spot, flip, walls, max pain, strikes)
* a **distance** converts by the scale factor only, never the offset — a
  one-sigma move of 74 points is 74 on ES too, but 74 QQQ points is
  74 x 41.765 on NQ
* a **dollar amount or a ratio** does not convert at all — net GEX is
  already money, and a percentage of spot is unitless
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import pandas as pd

#: Seeded from the desk's own numbers. The basis moves daily, so these are
#: starting points to be overwritten, not constants.
PRESETS = {
    "ES": {"underlying": "SPX", "mode": "offset", "value": 26.98,
           "note": "S&P 500 future = SPX + carry"},
    "NQ": {"underlying": "QQQ", "mode": "ratio", "value": 41.765,
           "note": "Nasdaq-100 future = QQQ x unit size"},
    "GC": {"underlying": "GLD", "mode": "ratio", "value": 10.87,
           "note": "Gold future = GLD x unit size"},
}

MODES = ("offset", "ratio")

#: Columns that hold a price level, by frame. Anything not listed here is
#: left alone — see the module docstring on what converts and what does not.
LEVEL_COLUMNS = ("level", "anchor_strike", "strike", "spot_level")


@dataclass(frozen=True)
class Conversion:
    """How to express an underlying price on a futures contract."""
    target: str = ""
    mode: str = "offset"
    value: float = 0.0

    @property
    def active(self) -> bool:
        """A ratio of 1 or an offset of 0 is the identity — not worth
        relabelling the whole dashboard for."""
        if not self.target:
            return False
        return self.value != (1.0 if self.mode == "ratio" else 0.0)

    @property
    def scale(self) -> float:
        """The multiplier a *distance* picks up. An offset does not stretch
        the axis, so a one-sigma move keeps its size."""
        return float(self.value) if self.mode == "ratio" else 1.0

    def level(self, x):
        """Convert a price level."""
        if x is None or not self.active:
            return x
        return x * self.value if self.mode == "ratio" else x + self.value

    def distance(self, x):
        """Convert a width — an expected move, a wall-to-wall range."""
        if x is None or not self.active:
            return x
        return x * self.scale

    def back(self, x):
        """Futures price -> underlying. Used to read futures candles on the
        chain's own scale."""
        if x is None or not self.active:
            return x
        return x / self.value if self.mode == "ratio" else x - self.value

    def label(self) -> str:
        op = f"x {self.value:g}" if self.mode == "ratio" else f"{self.value:+g}"
        return f"{self.target} = underlying {op}"


def convert_frame(df: pd.DataFrame | None, conv: Conversion,
                  spot: float | None = None) -> pd.DataFrame | None:
    """Return a copy with every price column converted.

    Strengths, scores and open interest are left alone — they were never
    prices. ``distance_pct`` needs more care than it looks: a *ratio* scales
    level and spot alike, so ``L/S`` is preserved exactly, but an *offset*
    does not — ``(L+b)/(S+b)`` is not ``L/S``. Pass the converted ``spot``
    and the distances are recomputed against it rather than carried over
    slightly stale.
    """
    if df is None or not conv.active or df.empty:
        return df
    out = df.copy()
    for col in LEVEL_COLUMNS:
        if col in out.columns:
            out[col] = out[col].astype(float).map(conv.level)
    if (spot and conv.mode == "offset"
            and "distance_pct" in out.columns and "level" in out.columns):
        out["distance_pct"] = out["level"].astype(float) / float(spot) - 1.0
    return out


def convert_analysis(a, conv: Conversion):
    """Return a copy of the Analysis with its levels quoted on ``target``.

    Dollar figures — net GEX, DEX, vanna, charm — are money and stay as they
    are. A ratio conversion is a change of *units*, not of value: the same
    dealer exposure described against a bigger tick.
    """
    if not conv.active:
        return a
    out = replace(
        a,
        spot=conv.level(a.spot),
        gamma_flip=conv.level(a.gamma_flip),
        call_wall=conv.level(a.call_wall),
        put_wall=conv.level(a.put_wall),
        call_wall_strike=conv.level(a.call_wall_strike),
        put_wall_strike=conv.level(a.put_wall_strike),
        max_pain=conv.level(a.max_pain),
        expected_move=conv.distance(a.expected_move),
        flip_levels=[conv.level(x) for x in (a.flip_levels or [])],
        by_strike=convert_frame(a.by_strike, conv),
        curve=convert_frame(a.curve, conv),
    )
    return out


def candle_basis(candle_close: float, chain_spot: float,
                 mode: str = "offset") -> float:
    """The conversion implied by a futures candle sitting beside the chain's
    own spot for the same session — so the number can be measured instead of
    typed in, and checked against what was typed."""
    if not chain_spot:
        return 0.0
    return (candle_close / chain_spot if mode == "ratio"
            else candle_close - chain_spot)
