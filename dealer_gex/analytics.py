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
DEFAULT_MULTIPLIER = 100.0  # shares per contract; ES=50, NQ=20, equity/index=100


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


def _norm_cdf(x):
    """Standard normal CDF (Abramowitz & Stegun 26.2.17, |err| < 7.5e-8) —
    vectorized without a scipy dependency."""
    x = np.asarray(x, dtype=float)
    t = 1.0 / (1.0 + 0.2316419 * np.abs(x))
    poly = t * (0.319381530 + t * (-0.356563782 + t * (1.781477937
           + t * (-1.821255978 + t * 1.330274429))))
    pdf = np.exp(-0.5 * x**2) / np.sqrt(2.0 * np.pi)
    upper = 1.0 - pdf * poly
    return np.where(x >= 0, upper, 1.0 - upper)


def _d1_d2(spot, strike, t, iv, rate):
    d1 = (np.log(spot / strike) + (rate + 0.5 * iv**2) * t) / (iv * np.sqrt(t))
    return d1, d1 - iv * np.sqrt(t)


@dataclass
class Analysis:
    spot: float
    asof: date
    rate: float
    total_gex: float          # $ per 1% move, net dealer gamma
    regime: str               # 'long_gamma' | 'short_gamma'
    gamma_flip: float | None  # zero-gamma spot level (None if no crossing)
    call_wall: float          # pinpoint: spot level of peak aggregate call gamma
    put_wall: float           # pinpoint: spot level of peak aggregate put gamma
    call_wall_strike: float   # anchor strike the call wall sits on
    put_wall_strike: float    # anchor strike the put wall sits on
    max_pain: float
    by_strike: pd.DataFrame   # strike, call_gex, put_gex, net_gex, call_oi, put_oi
    curve: pd.DataFrame       # spot_level, total_gex
    by_expiry: pd.DataFrame   # expiry, net_gex, call_oi, put_oi
    n_contracts: int = 0
    expiries: list = field(default_factory=list)
    weight_mode: str = "open_interest"   # 'open_interest' | 'volume'
    multiplier: float = DEFAULT_MULTIPLIER  # shares/units per contract
    dex: float = 0.0          # net dealer delta, $ notional (positive = long stock equiv.)
    vanna_flow: float = 0.0   # $ dealers must trade if IV drops 1 pt (positive = buy)
    charm_flow: float = 0.0   # $ dealers must trade per calendar day (positive = buy)
    expected_move: float | None = None   # 1-sigma $ move to the nearest expiry
    nearest_expiry: date | None = None


def _years_to_expiry(expiry: pd.Series, asof: date) -> pd.Series:
    days = (expiry.dt.date - pd.Timestamp(asof).date()).map(
        lambda d: d.days if pd.notna(d) else np.nan
    )
    t = pd.to_numeric(days, errors="coerce") / 365.0
    return t.fillna(30 / 365.0).clip(lower=MIN_T)  # undated rows: assume ~1 month


def _fill_gamma(chain: pd.DataFrame, spot: float, asof: date, rate: float,
                weight_col: str = "open_interest") -> pd.DataFrame:
    df = chain.copy()
    df["t"] = _years_to_expiry(df["expiry"], asof)
    computed = bs_gamma(spot, df["strike"], df["t"], df["iv"].fillna(0.0), rate)
    df["gamma"] = df["gamma"].where(df["gamma"].notna() & (df["gamma"] >= 0), computed)
    df["weight"] = df[weight_col].clip(lower=0).fillna(0.0)
    return df


def _dollar_gex(gamma, qty, spot: float, multiplier: float = DEFAULT_MULTIPLIER):
    return gamma * qty * multiplier * spot**2 * 0.01


def _weights(df: pd.DataFrame) -> pd.Series:
    return df["weight"] if "weight" in df.columns else df["open_interest"]


def gex_by_strike(df: pd.DataFrame, spot: float,
                  multiplier: float = DEFAULT_MULTIPLIER) -> pd.DataFrame:
    df = df.assign(gex=_dollar_gex(df["gamma"], _weights(df), spot, multiplier))
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


def total_gex_at(df: pd.DataFrame, spot_level: float, rate: float,
                 multiplier: float = DEFAULT_MULTIPLIER) -> float:
    """Net dealer GEX with every contract's gamma re-priced at a hypothetical spot.

    Always recomputed from IV (file-supplied gamma is only valid at the
    current spot), holding each contract's IV fixed — the standard
    sticky-strike simplification.
    """
    gamma = bs_gamma(spot_level, df["strike"], df["t"], df["iv"].fillna(0.0), rate)
    signed = np.where(df["type"] == "C", 1.0, -1.0)
    return float(np.sum(signed * _dollar_gex(gamma, _weights(df), spot_level, multiplier)))


def gex_curve(df: pd.DataFrame, spot: float, rate: float,
              span: float = 0.15, n: int = 121,
              multiplier: float = DEFAULT_MULTIPLIER) -> pd.DataFrame:
    levels = np.linspace(spot * (1 - span), spot * (1 + span), n)
    totals = [total_gex_at(df, s, rate, multiplier) for s in levels]
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


def refine_flip(df: pd.DataFrame, curve: pd.DataFrame, spot: float,
                rate: float, tol: float = 0.005) -> float | None:
    """Bisect the actual GEX function around the curve's crossing to cent
    precision (the curve alone is only as precise as its grid step)."""
    approx = gamma_flip(curve, spot)
    if approx is None:
        return None
    lv = curve["spot_level"].to_numpy()
    i = int(np.clip(np.searchsorted(lv, approx) - 1, 0, len(lv) - 2))
    lo, hi = lv[i], lv[i + 1]
    f_lo = total_gex_at(df, lo, rate)
    if f_lo == 0:
        return float(lo)
    while hi - lo > tol:
        mid = (lo + hi) / 2
        f_mid = total_gex_at(df, mid, rate)
        if f_mid == 0:
            return float(mid)
        if (f_lo < 0) == (f_mid < 0):
            lo, f_lo = mid, f_mid
        else:
            hi = mid
    return float((lo + hi) / 2)


def side_gamma_density(by_strike: pd.DataFrame, x, side: str, bandwidth: float) -> float:
    """Kernel-smoothed dealer gamma concentration for one side as a
    continuous function of price level (Gaussian kernel per strike,
    weighted by that strike's dollar gamma at the current spot)."""
    col = "call_gex" if side == "C" else "put_gex"
    w = by_strike[col].abs().to_numpy()
    k = by_strike["strike"].to_numpy()
    return float(np.sum(w * np.exp(-0.5 * ((x - k) / bandwidth) ** 2)))


def wall_level(by_strike: pd.DataFrame, seed_strike: float, spacing: float,
               side: str, tol: float = 0.005) -> float:
    """Pinpoint wall: the price level where that side's smoothed gamma
    density peaks, searched around the heaviest strike (golden-section).

    Neighboring strikes pull the peak off the grid, so the wall lands
    between strikes when the concentration is lopsided."""
    inv_phi = (np.sqrt(5.0) - 1.0) / 2.0
    a, b = seed_strike - spacing, seed_strike + spacing
    c = b - inv_phi * (b - a)
    d = a + inv_phi * (b - a)
    f_c = side_gamma_density(by_strike, c, side, spacing)
    f_d = side_gamma_density(by_strike, d, side, spacing)
    while b - a > tol:
        if f_c > f_d:
            b, d, f_d = d, c, f_c
            c = b - inv_phi * (b - a)
            f_c = side_gamma_density(by_strike, c, side, spacing)
        else:
            a, c, f_c = c, d, f_d
            d = a + inv_phi * (b - a)
            f_d = side_gamma_density(by_strike, d, side, spacing)
    return float((a + b) / 2)


def hedge_flows(df: pd.DataFrame, spot: float, rate: float,
                multiplier: float = DEFAULT_MULTIPLIER) -> tuple[float, float, float]:
    """Dealer hedge inventory and forced flows beyond gamma.

    Returns (dex, vanna_flow, charm_flow), all in $ notional:

    * ``dex`` — net dealer delta under the same convention (long calls,
      short puts). Its *changes* are what force flow.
    * ``vanna_flow`` — dollars dealers must trade to re-hedge if IV drops
      one point across the board (positive = forced buying).
    * ``charm_flow`` — dollars dealers must trade per calendar day as
      deltas decay (positive = forced buying).
    """
    t = df["t"].to_numpy()
    iv = df["iv"].fillna(0.0).to_numpy()
    k = df["strike"].to_numpy()
    w = _weights(df).to_numpy()
    is_call = (df["type"] == "C").to_numpy()

    valid = (t > 0) & (iv > 0) & (k > 0)
    t_ = np.where(valid, t, 1.0)
    iv_ = np.where(valid, iv, 1.0)
    k_ = np.where(valid, k, 1.0)

    d1, d2 = _d1_d2(spot, k_, t_, iv_, rate)
    pdf = np.exp(-0.5 * d1**2) / np.sqrt(2.0 * np.pi)

    # dealer-held per-contract greeks: +call greek, -put greek
    delta = np.where(is_call, _norm_cdf(d1), _norm_cdf(d1) - 1.0)
    vanna = -pdf * d2 / iv_                       # dVega/dSpot = dDelta/dVol
    charm = -pdf * (2.0 * rate * t_ - d2 * iv_ * np.sqrt(t_)) / (2.0 * t_ * iv_ * np.sqrt(t_))

    sign = np.where(is_call, 1.0, -1.0)
    scale = np.where(valid, w, 0.0) * multiplier * spot
    dex = float(np.sum(sign * delta * scale))
    vanna_total = float(np.sum(sign * vanna * scale))   # per 1.00 change in vol
    charm_total = float(np.sum(sign * charm * scale))   # per year

    # IV -1pt changes dealer delta by -0.01*vanna_total; hedging trades the
    # opposite. One day of decay changes it by charm_total/365; same logic.
    vanna_flow = 0.01 * vanna_total
    charm_flow = -charm_total / 365.0
    return dex, vanna_flow, charm_flow


def expected_move(df: pd.DataFrame, spot: float) -> tuple[float | None, date | None]:
    """1-sigma expected move to the nearest expiry, from near-the-money IV
    (weighted by contract weight)."""
    dated = df[df["expiry"].notna()]
    if dated.empty:
        return None, None
    nearest = dated["expiry"].min()
    sub = dated[
        (dated["expiry"] == nearest)
        & (dated["strike"].between(spot * 0.97, spot * 1.03))
        & dated["iv"].notna()
    ]
    if sub.empty:
        return None, nearest.date()
    w = _weights(sub).to_numpy()
    w = w if w.sum() > 0 else np.ones_like(w)
    iv_atm = float(np.average(sub["iv"].to_numpy(), weights=w))
    t = float(sub["t"].iloc[0])
    return spot * iv_atm * np.sqrt(t), nearest.date()


def max_pain(chain: pd.DataFrame) -> float:
    """Level minimizing the total intrinsic payout to option holders.

    Evaluated on the strike grid, then refined off-grid with a parabola
    through the minimum and its neighbors (the payout function is convex
    and piecewise-linear between strikes, so this stays inside them).
    """
    strikes = np.sort(chain["strike"].unique())
    calls = chain[chain["type"] == "C"]
    puts = chain[chain["type"] == "P"]
    payouts = np.array([
        float(
            (np.maximum(s - calls["strike"], 0) * calls["open_interest"]).sum()
            + (np.maximum(puts["strike"] - s, 0) * puts["open_interest"]).sum()
        )
        for s in strikes
    ])
    i = int(np.argmin(payouts))
    if 0 < i < len(strikes) - 1:
        x0, x1, x2 = strikes[i - 1], strikes[i], strikes[i + 1]
        y0, y1, y2 = payouts[i - 1], payouts[i], payouts[i + 1]
        denom = (x1 - x0) * (y1 - y2) - (x1 - x2) * (y1 - y0)
        if denom != 0:
            vertex = x1 - 0.5 * (
                (x1 - x0) ** 2 * (y1 - y2) - (x1 - x2) ** 2 * (y1 - y0)
            ) / denom
            return float(np.clip(vertex, x0, x2))
    return float(strikes[i])


def analyze(chain: pd.DataFrame, spot: float, asof: date,
            rate: float = 0.045, weight: str = "open_interest",
            multiplier: float = DEFAULT_MULTIPLIER) -> Analysis:
    # Expired contracts carry no hedging obligation; clipping them to a tiny
    # time-to-expiry would instead explode their gamma, so drop them.
    expired = chain["expiry"].notna() & (chain["expiry"].dt.date < asof)
    chain = chain[~expired]
    if chain.empty:
        raise ValueError("All contracts are expired as of the analysis date.")
    df = _fill_gamma(chain, spot, asof, rate, weight_col=weight)

    strikes = gex_by_strike(df, spot, multiplier)
    curve = gex_curve(df, spot, rate, multiplier=multiplier)
    total = total_gex_at(df, spot, rate, multiplier)
    flip = refine_flip(df, curve, spot, rate)  # zero crossing is scale-invariant

    uniq = np.sort(strikes["strike"].unique())
    spacing = float(np.median(np.diff(uniq))) if len(uniq) > 1 else spot * 0.01

    pos = strikes[strikes["call_gex"] > 0]
    neg = strikes[strikes["put_gex"] < 0]
    call_seed = float(pos.loc[pos["call_gex"].idxmax(), "strike"]) if not pos.empty else spot
    put_seed = float(neg.loc[neg["put_gex"].idxmin(), "strike"]) if not neg.empty else spot
    call_wall = wall_level(strikes, call_seed, spacing, "C") if not pos.empty else spot
    put_wall = wall_level(strikes, put_seed, spacing, "P") if not neg.empty else spot

    dex, vanna_flow, charm_flow = hedge_flows(df, spot, rate, multiplier)
    em, nearest = expected_move(df, spot)

    signed = np.where(df["type"] == "C", 1.0, -1.0)
    df = df.assign(net_gex=signed * _dollar_gex(df["gamma"], _weights(df), spot, multiplier))
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
        call_wall_strike=call_seed,
        put_wall_strike=put_seed,
        max_pain=max_pain(chain),
        by_strike=strikes,
        curve=curve,
        by_expiry=by_exp,
        n_contracts=len(chain),
        expiries=sorted(d for d in chain["expiry"].dt.date.dropna().unique()),
        weight_mode=weight,
        multiplier=multiplier,
        dex=dex,
        vanna_flow=vanna_flow,
        charm_flow=charm_flow,
        expected_move=em,
        nearest_expiry=nearest,
    )


def fmt_dollars(x: float) -> str:
    """$1.23B-style formatting for GEX magnitudes."""
    sign = "-" if x < 0 else ""
    x = abs(x)
    for div, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if x >= div:
            return f"{sign}${x / div:.2f}{suffix}"
    return f"{sign}${x:.0f}"
