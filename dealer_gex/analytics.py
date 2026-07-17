"""Dealer gamma-exposure analytics.

Sign convention (standard GEX / SqueezeMetrics-style): dealers are assumed
to be long the calls and short the puts held by customers, so call open
interest contributes positive dealer gamma and put open interest negative.
GEX is quoted as dollar gamma per 1% move in the underlying:

    GEX_contract = gamma * OI * 100 * S^2 * 0.01

That assumption is a market convention, not an observation — it is stated
in the UI and the report.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

TRADING_DAYS = 252
MIN_T = 0.5 / TRADING_DAYS  # floor expiring contracts at half a trading day


def bs_gamma(spot, strike, t, iv, rate: float = 0.045, div_yield: float = 0.0):
    """Black-Scholes gamma (same for calls and puts). Vectorized.

    ``t`` in years; ``iv`` as a decimal. Expired/zero-vol inputs return 0.
    """
    spot = np.asarray(spot, dtype=float)
    strike = np.asarray(strike, dtype=float)
    t = np.maximum(np.asarray(t, dtype=float), 0.0)
    iv = np.asarray(iv, dtype=float)

    valid = (t > 0) & (iv > 0) & (spot > 0) & (strike > 0)
    t_ = np.where(valid, t, 1.0)
    iv_ = np.where(valid, iv, 1.0)
    s_ = np.where(valid, spot, 1.0)
    k_ = np.where(valid, strike, 1.0)

    d1 = (np.log(s_ / k_) + (rate - div_yield + 0.5 * iv_**2) * t_) / (iv_ * np.sqrt(t_))
    pdf = np.exp(-0.5 * d1**2) / np.sqrt(2.0 * np.pi)
    gamma = np.exp(-div_yield * t_) * pdf / (s_ * iv_ * np.sqrt(t_))
    return np.where(valid, gamma, 0.0)


@dataclass
class Analysis:
    spot: float
    asof: date
    rate: float
    total_gex: float          # $ per 1% move, net dealer gamma
    regime: str               # 'long_gamma' | 'short_gamma'
    gamma_flip: float | None  # zero-gamma spot level (None if no crossing)
    call_wall: float
    put_wall: float
    max_pain: float
    by_strike: pd.DataFrame   # strike, call_gex, put_gex, net_gex, call_oi, put_oi
    curve: pd.DataFrame       # spot_level, total_gex
    by_expiry: pd.DataFrame   # expiry, net_gex, call_oi, put_oi
    n_contracts: int = 0
    expiries: list = field(default_factory=list)


def _years_to_expiry(expiry: pd.Series, asof: date) -> pd.Series:
    days = (expiry.dt.date - pd.Timestamp(asof).date()).map(
        lambda d: d.days if pd.notna(d) else np.nan
    )
    t = pd.to_numeric(days, errors="coerce") / 365.0
    return t.fillna(30 / 365.0).clip(lower=MIN_T)  # undated rows: assume ~1 month


def _fill_gamma(chain: pd.DataFrame, spot: float, asof: date, rate: float) -> pd.DataFrame:
    df = chain.copy()
    df["t"] = _years_to_expiry(df["expiry"], asof)
    computed = bs_gamma(spot, df["strike"], df["t"], df["iv"].fillna(0.0), rate)
    df["gamma"] = df["gamma"].where(df["gamma"].notna() & (df["gamma"] >= 0), computed)
    return df


def _dollar_gex(gamma, oi, spot: float):
    return gamma * oi * 100.0 * spot**2 * 0.01


def gex_by_strike(df: pd.DataFrame, spot: float) -> pd.DataFrame:
    df = df.assign(gex=_dollar_gex(df["gamma"], df["open_interest"], spot))
    grouped = df.pivot_table(
        index="strike", columns="type", values=["gex", "open_interest"],
        aggfunc="sum", fill_value=0.0,
    )
    out = pd.DataFrame(index=grouped.index)
    out["call_gex"] = grouped.get(("gex", "C"), 0.0)
    out["put_gex"] = -grouped.get(("gex", "P"), 0.0)  # dealer-short puts: negative
    out["net_gex"] = out["call_gex"] + out["put_gex"]
    out["call_oi"] = grouped.get(("open_interest", "C"), 0.0)
    out["put_oi"] = grouped.get(("open_interest", "P"), 0.0)
    return out.reset_index().sort_values("strike").reset_index(drop=True)


def total_gex_at(df: pd.DataFrame, spot_level: float, rate: float) -> float:
    """Net dealer GEX with every contract's gamma re-priced at a hypothetical spot.

    Always recomputed from IV (file-supplied gamma is only valid at the
    current spot), holding each contract's IV fixed — the standard
    sticky-strike simplification.
    """
    gamma = bs_gamma(spot_level, df["strike"], df["t"], df["iv"].fillna(0.0), rate)
    signed = np.where(df["type"] == "C", 1.0, -1.0)
    return float(np.sum(signed * _dollar_gex(gamma, df["open_interest"], spot_level)))


def gex_curve(df: pd.DataFrame, spot: float, rate: float,
              span: float = 0.15, n: int = 121) -> pd.DataFrame:
    levels = np.linspace(spot * (1 - span), spot * (1 + span), n)
    totals = [total_gex_at(df, s, rate) for s in levels]
    return pd.DataFrame({"spot_level": levels, "total_gex": totals})


def gamma_flip(curve: pd.DataFrame, spot: float) -> float | None:
    """Interpolated zero crossing of the GEX curve nearest to current spot."""
    lv = curve["spot_level"].to_numpy()
    gx = curve["total_gex"].to_numpy()
    sign_change = np.nonzero(np.diff(np.sign(gx)) != 0)[0]
    if sign_change.size == 0:
        return None
    crossings = []
    for i in sign_change:
        x0, x1, y0, y1 = lv[i], lv[i + 1], gx[i], gx[i + 1]
        crossings.append(x0 if y1 == y0 else x0 - y0 * (x1 - x0) / (y1 - y0))
    return float(min(crossings, key=lambda x: abs(x - spot)))


def max_pain(chain: pd.DataFrame) -> float:
    """Strike minimizing the total intrinsic payout to option holders."""
    strikes = np.sort(chain["strike"].unique())
    calls = chain[chain["type"] == "C"]
    puts = chain[chain["type"] == "P"]
    payouts = [
        float(
            (np.maximum(s - calls["strike"], 0) * calls["open_interest"]).sum()
            + (np.maximum(puts["strike"] - s, 0) * puts["open_interest"]).sum()
        )
        for s in strikes
    ]
    return float(strikes[int(np.argmin(payouts))])


def analyze(chain: pd.DataFrame, spot: float, asof: date,
            rate: float = 0.045) -> Analysis:
    # Expired contracts carry no hedging obligation; clipping them to a tiny
    # time-to-expiry would instead explode their gamma, so drop them.
    expired = chain["expiry"].notna() & (chain["expiry"].dt.date < asof)
    chain = chain[~expired]
    if chain.empty:
        raise ValueError("All contracts are expired as of the analysis date.")
    df = _fill_gamma(chain, spot, asof, rate)

    strikes = gex_by_strike(df, spot)
    curve = gex_curve(df, spot, rate)
    total = total_gex_at(df, spot, rate)
    flip = gamma_flip(curve, spot)

    pos = strikes[strikes["call_gex"] > 0]
    neg = strikes[strikes["put_gex"] < 0]
    call_wall = float(pos.loc[pos["call_gex"].idxmax(), "strike"]) if not pos.empty else spot
    put_wall = float(neg.loc[neg["put_gex"].idxmin(), "strike"]) if not neg.empty else spot

    signed = np.where(df["type"] == "C", 1.0, -1.0)
    df = df.assign(net_gex=signed * _dollar_gex(df["gamma"], df["open_interest"], spot))
    by_exp = (
        df.groupby(df["expiry"].dt.date)
        .agg(
            net_gex=("net_gex", "sum"),
            call_oi=("open_interest", lambda s: s[df.loc[s.index, "type"] == "C"].sum()),
            put_oi=("open_interest", lambda s: s[df.loc[s.index, "type"] == "P"].sum()),
        )
        .reset_index()
        .rename(columns={"expiry": "expiry"})
    )

    return Analysis(
        spot=spot,
        asof=asof,
        rate=rate,
        total_gex=total,
        regime="long_gamma" if total >= 0 else "short_gamma",
        gamma_flip=flip,
        call_wall=call_wall,
        put_wall=put_wall,
        max_pain=max_pain(chain),
        by_strike=strikes,
        curve=curve,
        by_expiry=by_exp,
        n_contracts=len(chain),
        expiries=sorted(d for d in chain["expiry"].dt.date.dropna().unique()),
    )


def fmt_dollars(x: float) -> str:
    """$1.23B-style formatting for GEX magnitudes."""
    sign = "-" if x < 0 else ""
    x = abs(x)
    for div, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if x >= div:
            return f"{sign}${x / div:.2f}{suffix}"
    return f"{sign}${x:.0f}"
