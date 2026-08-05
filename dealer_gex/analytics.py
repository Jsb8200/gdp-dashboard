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
from typing import NamedTuple

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
    # every zero-gamma crossing the grid resolves, nearest to spot first;
    # gamma_flip is flip_levels[0]. More than one means the regime switches
    # back — spot can sit in a pocket with the opposite regime either side.
    flip_levels: list = field(default_factory=list)


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
    if weight_col == "net_customer_size":
        # signed-flow mode: dealers hold the other side of customer flow, so
        # the weight itself carries the dealer's sign — no C/P convention.
        df["weight"] = -df[weight_col].fillna(0.0)
        df["gsign"] = 1.0
    else:
        df["weight"] = df[weight_col].clip(lower=0).fillna(0.0)
        df["gsign"] = np.where(df["type"] == "C", 1.0, -1.0)
    return df


def _gsign(df: pd.DataFrame):
    return df["gsign"] if "gsign" in df.columns else np.where(df["type"] == "C", 1.0, -1.0)


def _dollar_gex(gamma, qty, spot: float, multiplier: float = DEFAULT_MULTIPLIER):
    return gamma * qty * multiplier * spot**2 * 0.01


def _weights(df: pd.DataFrame) -> pd.Series:
    return df["weight"] if "weight" in df.columns else df["open_interest"]


def gex_by_strike(df: pd.DataFrame, spot: float,
                  multiplier: float = DEFAULT_MULTIPLIER) -> pd.DataFrame:
    # dealer-signed gex: in convention modes gsign is +C/−P (puts come out
    # negative as before); in flow mode the sign lives in the weight itself
    df = df.assign(gex=_gsign(df) * _dollar_gex(df["gamma"], _weights(df), spot, multiplier))
    grouped = df.pivot_table(
        index="strike", columns="type", values=["gex", "open_interest"],
        aggfunc="sum",
    ).astype(float).fillna(0.0)
    out = pd.DataFrame(index=grouped.index)
    out["call_gex"] = grouped.get(("gex", "C"), 0.0)
    out["put_gex"] = grouped.get(("gex", "P"), 0.0)
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
    return float(np.sum(_gsign(df) * _dollar_gex(gamma, _weights(df), spot_level, multiplier)))


def gex_at_levels(df: pd.DataFrame, levels, rate: float,
                  multiplier: float = DEFAULT_MULTIPLIER) -> np.ndarray:
    """Net dealer GEX at many hypothetical spots at once.

    Same maths as ``total_gex_at`` per level, evaluated as one
    (levels x contracts) block instead of a Python loop — an order of
    magnitude faster, which is what makes a genuinely fine grid affordable
    for the flip scan. Chunked so a long grid on a big chain cannot blow
    up memory.
    """
    levels = np.atleast_1d(np.asarray(levels, dtype=float))
    k = df["strike"].to_numpy()[None, :]
    t = df["t"].to_numpy()[None, :]
    iv = df["iv"].fillna(0.0).to_numpy()[None, :]
    w = np.asarray(_weights(df), dtype=float)[None, :]
    sign = np.asarray(_gsign(df), dtype=float)[None, :]

    out = np.empty(len(levels), dtype=float)
    chunk = max(1, int(2_000_000 // max(df.shape[0], 1)))   # ~2M cells at a time
    for i in range(0, len(levels), chunk):
        s = levels[i:i + chunk][:, None]
        gamma = bs_gamma(s, k, t, iv, rate)
        out[i:i + chunk] = np.sum(sign * gamma * w * multiplier * s**2 * 0.01, axis=1)
    return out


def gex_curve(df: pd.DataFrame, spot: float, rate: float,
              span: float = 0.15, n: int = 601,
              multiplier: float = DEFAULT_MULTIPLIER) -> pd.DataFrame:
    """Net GEX across a grid of hypothetical spot levels.

    The grid is what the flip is found on, so its step is a precision
    floor: a crossing that lives entirely between two samples is invisible.
    601 points over +/-15% is ~0.05% of spot per step — fine enough that
    the crossings a trader can act on are all resolved, and cheap because
    the whole grid is evaluated in one vectorized block.
    """
    levels = np.linspace(spot * (1 - span), spot * (1 + span), n)
    return pd.DataFrame({"spot_level": levels,
                         "total_gex": gex_at_levels(df, levels, rate, multiplier)})


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


def _bisect_zero(df: pd.DataFrame, lo: float, hi: float, rate: float,
                 tol: float = 1e-4) -> float:
    """Bisect the true GEX function to a hundredth of a cent inside a
    bracket already known to change sign."""
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


def flip_levels(df: pd.DataFrame, curve: pd.DataFrame, spot: float,
                rate: float, tol: float = 1e-4) -> list[float]:
    """**Every** spot level where net dealer gamma crosses zero, bisected,
    ordered by distance from spot.

    The regime does not have to switch exactly once. A book can flip back
    within a few points — spot sits in a short-gamma pocket with long-gamma
    on both sides — and reporting only the nearest crossing hides the far
    edge of that pocket, which is where the behaviour changes back.

    Only crossings the grid resolves are found; the curve's step is the
    floor on what is visible, which is why ``gex_curve`` samples finely.
    """
    lv = curve["spot_level"].to_numpy()
    g = curve["total_gex"].to_numpy()
    idx = np.nonzero(np.diff(np.sign(g)) != 0)[0]
    roots = []
    for i in idx:
        if g[i] == 0:
            roots.append(float(lv[i]))
            continue
        roots.append(_bisect_zero(df, float(lv[i]), float(lv[i + 1]), rate, tol))
    # a shared bracket edge can yield the same root twice
    uniq: list[float] = []
    for r in sorted(roots):
        if not uniq or abs(r - uniq[-1]) > max(tol * 10, 1e-3):
            uniq.append(r)
    return sorted(uniq, key=lambda x: abs(x - spot))


def refine_flip(df: pd.DataFrame, curve: pd.DataFrame, spot: float,
                rate: float, tol: float = 1e-4) -> float | None:
    """The zero-gamma level nearest spot, bisected on the true GEX function
    (the curve alone is only as precise as its grid step)."""
    roots = flip_levels(df, curve, spot, rate, tol)
    return roots[0] if roots else None


def _kernel_density(strikes: np.ndarray, weights: np.ndarray, x, bandwidth: float) -> float:
    """Gaussian-kernel concentration of `weights` over the strike axis."""
    return float(np.sum(weights * np.exp(-0.5 * ((x - strikes) / bandwidth) ** 2)))


def side_gamma_density(by_strike: pd.DataFrame, x, side: str, bandwidth: float) -> float:
    """Kernel-smoothed dealer gamma concentration for one side as a
    continuous function of price level (Gaussian kernel per strike,
    weighted by that strike's dollar gamma at the current spot).

    Sides: "C"/"P" (convention modes) or "POS"/"NEG" — the positive/negative
    parts of signed net GEX (flow mode)."""
    if side == "C":
        w = by_strike["call_gex"].abs().to_numpy()
    elif side == "P":
        w = by_strike["put_gex"].abs().to_numpy()
    elif side == "POS":
        w = by_strike["net_gex"].clip(lower=0).to_numpy()
    else:
        w = (-by_strike["net_gex"]).clip(lower=0).to_numpy()
    return _kernel_density(by_strike["strike"].to_numpy(), w, x, bandwidth)


def _golden_max(f, lo: float, hi: float, tol: float = 1e-4) -> float:
    """Golden-section search for the maximum of a unimodal f on [lo, hi].

    ``tol`` is the bracket width at exit, so the returned level is good
    to half of it — a hundredth of a cent by default."""
    inv_phi = (np.sqrt(5.0) - 1.0) / 2.0
    a, b = lo, hi
    c = b - inv_phi * (b - a)
    d = a + inv_phi * (b - a)
    f_c, f_d = f(c), f(d)
    while b - a > tol:
        if f_c > f_d:
            b, d, f_d = d, c, f_c
            c = b - inv_phi * (b - a)
            f_c = f(c)
        else:
            a, c, f_c = c, d, f_d
            d = a + inv_phi * (b - a)
            f_d = f(d)
    return float((a + b) / 2)


def wall_level(by_strike: pd.DataFrame, seed_strike: float, spacing: float,
               side: str, tol: float = 1e-4, reach: float = 2.0) -> float:
    """Pinpoint wall: the price level where that side's smoothed gamma
    density peaks, searched around the heaviest strike (golden-section).

    Neighboring strikes pull the peak off the grid, so the wall lands
    between strikes when the concentration is lopsided. The search spans
    ``reach`` strikes either side rather than one — a heavy pair of
    neighbours can carry the true peak past the next strike, and a bracket
    of exactly one strike would clamp the answer back onto the seed. The
    bracket is scanned coarsely first so golden-section starts on a
    genuinely unimodal interval.
    """
    lo, hi = seed_strike - reach * spacing, seed_strike + reach * spacing
    f = lambda x: side_gamma_density(by_strike, x, side, spacing)
    grid = np.linspace(lo, hi, int(reach * 40) + 1)
    vals = np.array([f(x) for x in grid])
    i = int(vals.argmax())
    a = grid[max(i - 1, 0)]
    b = grid[min(i + 1, len(grid) - 1)]
    return _golden_max(f, a, b, tol)


def magnet_levels(a: "Analysis", top_n: int = 5) -> pd.DataFrame:
    """Ranked magnet map: every local peak of dealer gamma concentration
    within ±10% of spot, pinpointed and scored.

    * ``magnet`` — a positive net-gamma peak: dealer hedging fades moves
      around it, pulling price toward it (pinning), strongest near expiry.
    * ``accelerator`` — a negative net-gamma peak: hedging pushes price
      away from it, so moves through it tend to extend.

    Strength is 0-100, normalized to the strongest level found, comparable
    across the two kinds. Columns: level, kind, strength, distance_pct,
    anchor_strike.
    """
    ks = a.by_strike["strike"].to_numpy()
    if len(ks) < 2:
        return pd.DataFrame(columns=["level", "kind", "strength", "distance_pct", "anchor_strike"])
    spacing = float(np.median(np.diff(np.sort(np.unique(ks)))))
    lo, hi = a.spot * 0.90, a.spot * 1.10
    grid = np.arange(lo, hi, spacing / 10.0)

    rows = []
    for kind, side in (("magnet", "POS"), ("accelerator", "NEG")):
        dens = np.array([side_gamma_density(a.by_strike, x, side, spacing) for x in grid])
        if dens.max() <= 0:
            continue
        floor = 0.05 * dens.max()
        for i in range(1, len(grid) - 1):
            if dens[i] > floor and dens[i] > dens[i - 1] and dens[i] >= dens[i + 1]:
                level = _golden_max(
                    lambda x: side_gamma_density(a.by_strike, x, side, spacing),
                    grid[i] - spacing, grid[i] + spacing,
                )
                strength = side_gamma_density(a.by_strike, level, side, spacing)
                anchor = float(ks[np.argmin(np.abs(ks - level))])
                rows.append((level, kind, strength, (level / a.spot - 1) * 100, anchor))

    if not rows:
        return pd.DataFrame(columns=["level", "kind", "strength", "distance_pct", "anchor_strike"])
    df = pd.DataFrame(rows, columns=["level", "kind", "strength", "distance_pct", "anchor_strike"])
    # golden-section refinement can converge to the same peak from adjacent
    # grid maxima — keep the strongest within half a strike spacing
    df = df.sort_values("strength", ascending=False)
    kept: list[int] = []
    for idx, row in df.iterrows():
        if all(abs(row["level"] - df.loc[k, "level"]) > spacing / 2 for k in kept):
            kept.append(idx)
    df = df.loc[kept].head(top_n).reset_index(drop=True)
    df["strength"] = (100 * df["strength"] / df["strength"].max()).round(0)
    return df


class OIWalls(NamedTuple):
    call: float | None          # pinpoint: peak of smoothed call-OI density
    put: float | None           # pinpoint: peak of smoothed put-OI density
    call_strike: float | None   # anchor strike (largest raw call OI)
    put_strike: float | None    # anchor strike (largest raw put OI)


def oi_walls(a: "Analysis") -> OIWalls:
    """The classic OI walls — where the largest raw call/put open interest
    sits — pinpointed like every other level: the anchor is the max-OI
    strike, and the level is the peak of the kernel-smoothed OI density
    around it (neighboring size pulls it off the grid). Distinct from the
    gamma-weighted walls on the Analysis: these mark sheer position size,
    regardless of today's hedging sensitivity."""
    bs = a.by_strike
    ks = bs["strike"].to_numpy()
    spacing = float(np.median(np.diff(np.sort(np.unique(ks))))) if len(ks) > 1 else 0.0

    def _wall(col: str) -> tuple[float | None, float | None]:
        if bs[col].max() <= 0:
            return None, None
        anchor = float(bs.loc[bs[col].idxmax(), "strike"])
        if spacing <= 0:
            return anchor, anchor
        w = bs[col].to_numpy(dtype=float)
        level = _golden_max(
            lambda x: _kernel_density(ks, w, x, spacing),
            anchor - spacing, anchor + spacing,
        )
        return level, anchor

    call, call_strike = _wall("call_oi")
    put, put_strike = _wall("put_oi")
    return OIWalls(call, put, call_strike, put_strike)


def oi_levels(a: "Analysis", top_n: int = 5) -> pd.DataFrame:
    """Ranked raw open-interest concentration levels — where positions SIT,
    independent of current gamma sensitivity.

    Heavy OI is the classic support/resistance and expiry-pin marker: a
    call-heavy cluster above spot tends to cap rallies, a put-heavy cluster
    below tends to catch selloffs, and price gravitates to the biggest
    clusters into expiry. Complements :func:`magnet_levels`, which weights
    the same OI by its hedging force *today*.

    Columns: level, side ('call'/'put'/'mixed'), strength (0-100),
    distance_pct, anchor_strike, call_oi, put_oi.
    """
    bs = a.by_strike
    ks = bs["strike"].to_numpy()
    if len(ks) < 2:
        return pd.DataFrame(columns=["level", "side", "strength", "distance_pct",
                                     "anchor_strike", "call_oi", "put_oi"])
    spacing = float(np.median(np.diff(np.sort(np.unique(ks)))))
    call_w = bs["call_oi"].to_numpy(dtype=float)
    put_w = bs["put_oi"].to_numpy(dtype=float)
    total_w = call_w + put_w

    lo, hi = a.spot * 0.90, a.spot * 1.10
    grid = np.arange(lo, hi, spacing / 10.0)
    dens = np.array([_kernel_density(ks, total_w, x, spacing) for x in grid])
    if dens.max() <= 0:
        return pd.DataFrame(columns=["level", "side", "strength", "distance_pct",
                                     "anchor_strike", "call_oi", "put_oi"])

    floor = 0.05 * dens.max()
    rows = []
    for i in range(1, len(grid) - 1):
        if dens[i] > floor and dens[i] > dens[i - 1] and dens[i] >= dens[i + 1]:
            level = _golden_max(
                lambda x: _kernel_density(ks, total_w, x, spacing),
                grid[i] - spacing, grid[i] + spacing,
            )
            strength = _kernel_density(ks, total_w, level, spacing)
            c_d = _kernel_density(ks, call_w, level, spacing)
            p_d = _kernel_density(ks, put_w, level, spacing)
            side = "call" if c_d > 1.5 * p_d else "put" if p_d > 1.5 * c_d else "mixed"
            j = int(np.argmin(np.abs(ks - level)))
            rows.append((level, side, strength, (level / a.spot - 1) * 100,
                         float(ks[j]), float(call_w[j]), float(put_w[j])))

    if not rows:
        return pd.DataFrame(columns=["level", "side", "strength", "distance_pct",
                                     "anchor_strike", "call_oi", "put_oi"])
    df = pd.DataFrame(rows, columns=["level", "side", "strength", "distance_pct",
                                     "anchor_strike", "call_oi", "put_oi"])
    df = df.sort_values("strength", ascending=False)
    kept: list[int] = []
    for idx, row in df.iterrows():
        if all(abs(row["level"] - df.loc[k, "level"]) > spacing / 2 for k in kept):
            kept.append(idx)
    df = df.loc[kept].head(top_n).reset_index(drop=True)
    df["strength"] = (100 * df["strength"] / df["strength"].max()).round(0)
    return df


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

    # dealer-held per-contract greeks: +call greek, -put greek (convention
    # modes); in flow mode gsign is 1 and the weight carries the sign
    delta = np.where(is_call, _norm_cdf(d1), _norm_cdf(d1) - 1.0)
    vanna = -pdf * d2 / iv_                       # dVega/dSpot = dDelta/dVol
    charm = -pdf * (2.0 * rate * t_ - d2 * iv_ * np.sqrt(t_)) / (2.0 * t_ * iv_ * np.sqrt(t_))

    sign = np.asarray(_gsign(df), dtype=float)
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
    # Weight the IV blend by *activity*, not direction. In signed-flow mode
    # the weights carry a sign, and mixed signs summing positive slip past a
    # "sum > 0" guard straight into np.average — which then returns a mean
    # outside the range of the inputs. Two strikes at 20% and 60% IV with
    # weights +1000 and -900 produced an implied vol of -3.4 and an expected
    # move of -105 on a spot of 100.
    w = np.abs(_weights(sub).to_numpy(dtype=float))
    w = w if np.isfinite(w).all() and w.sum() > 0 else np.ones(len(sub))
    iv_atm = float(np.average(sub["iv"].to_numpy(), weights=w))
    t = float(sub["t"].iloc[0])
    if not np.isfinite(iv_atm) or iv_atm <= 0:
        return None, nearest.date()
    return spot * iv_atm * np.sqrt(t), nearest.date()


def block_levels(prints: pd.DataFrame, spot: float, top_n: int = 5) -> pd.DataFrame:
    """Block commitment levels: price levels where negotiated institutional
    (block) premium concentrated, from kernel-smoothed block-premium density
    over strikes. These mark where big money committed size — entry zones to
    cross-check against the walls and magnets.

    Columns: level, side ('call'/'put'/'mixed'), direction ('buy'/'sell'/
    'mixed' from signed block flow near the level), strength (0-100),
    premium ($ within one bandwidth), distance_pct, anchor_strike.
    """
    empty = pd.DataFrame(columns=["level", "side", "direction", "strength",
                                  "premium", "distance_pct", "anchor_strike"])
    b = prints[institutional_mask(prints) & prints["strike"].notna()].copy()
    b = b[b["strike"].between(spot * 0.90, spot * 1.10) & (b["premium"] > 0)]
    if b.empty:
        return empty
    b["cp"] = b["type"].astype(str).str.strip().str.upper().str[0]

    ks_all = np.sort(prints["strike"].dropna().unique())
    spacing = float(np.median(np.diff(ks_all))) if len(ks_all) > 1 else spot * 0.005

    per = b.groupby("strike").agg(
        premium=("premium", "sum"),
        call_prem=("premium", lambda s: s[b.loc[s.index, "cp"] == "C"].sum()),
        put_prem=("premium", lambda s: s[b.loc[s.index, "cp"] == "P"].sum()),
        net_signed=("signed_size", "sum"),
    ).reset_index()
    ks = per["strike"].to_numpy()
    w = per["premium"].to_numpy(dtype=float)

    lo, hi = spot * 0.90, spot * 1.10
    grid = np.arange(lo, hi, spacing / 10.0)
    dens = np.array([_kernel_density(ks, w, x, spacing) for x in grid])
    if dens.max() <= 0:
        return empty

    floor = 0.05 * dens.max()
    rows = []
    for i in range(1, len(grid) - 1):
        if dens[i] > floor and dens[i] > dens[i - 1] and dens[i] >= dens[i + 1]:
            level = _golden_max(
                lambda x: _kernel_density(ks, w, x, spacing),
                grid[i] - spacing, grid[i] + spacing,
            )
            strength = _kernel_density(ks, w, level, spacing)
            near = per[np.abs(per["strike"] - level) <= spacing]
            c_p, p_p = near["call_prem"].sum(), near["put_prem"].sum()
            side = "call" if c_p > 1.5 * p_p else "put" if p_p > 1.5 * c_p else "mixed"
            signed = near["net_signed"].sum()
            traded = b[np.abs(b["strike"] - level) <= spacing]["size"].sum()
            direction = ("buy" if signed > 0.1 * traded
                         else "sell" if signed < -0.1 * traded else "mixed")
            j = int(np.argmin(np.abs(ks - level)))
            rows.append((level, side, direction, strength,
                         float(near["premium"].sum()),
                         (level / spot - 1) * 100, float(ks[j])))
    if not rows:
        return empty
    df = pd.DataFrame(rows, columns=["level", "side", "direction", "strength",
                                     "premium", "distance_pct", "anchor_strike"])
    df = df.sort_values("strength", ascending=False)
    kept: list[int] = []
    for idx, row in df.iterrows():
        if all(abs(row["level"] - df.loc[k, "level"]) > spacing / 2 for k in kept):
            kept.append(idx)
    df = df.loc[kept].head(top_n).reset_index(drop=True)
    df["strength"] = (100 * df["strength"] / df["strength"].max()).round(0)
    return df


def flow_type_breakdown(prints: pd.DataFrame, min_share: float = 0.0) -> pd.DataFrame:
    """Consolidated premium and quantity per precise flow type.

    One row per execution type as the file tagged it — ``floor block``,
    ``auto sweep``, ``block``, ``cross``, or the raw code when it matched
    nothing — so floor-negotiated size is never averaged in with the
    electronic default.

    Columns: flow_type, venue, shape, prints, contracts, premium,
    avg_premium (per print), net_contracts (signed: + customer bought),
    direction, premium_share, institutional.
    """
    cols = ["flow_type", "venue", "shape", "prints", "contracts", "premium",
            "avg_premium", "net_contracts", "direction", "premium_share",
            "institutional"]
    if prints is None or prints.empty or "flow_type" not in prints:
        return pd.DataFrame(columns=cols)

    p = prints.copy()
    p["premium"] = pd.to_numeric(p.get("premium", 0.0), errors="coerce").fillna(0.0)
    p["size"] = pd.to_numeric(p.get("size", 0.0), errors="coerce").fillna(0.0)
    signed = pd.to_numeric(p.get("signed_size", 0.0), errors="coerce").fillna(0.0)
    p["signed_size"] = signed

    grouped = p.groupby("flow_type", dropna=False).agg(
        venue=("flow_venue", "first"),
        shape=("flow_shape", "first"),
        prints=("flow_type", "size"),
        contracts=("size", "sum"),
        premium=("premium", "sum"),
        net_contracts=("signed_size", "sum"),
        institutional=("is_institutional", "any"),
    ).reset_index()

    grouped["avg_premium"] = np.where(
        grouped["prints"] > 0, grouped["premium"] / grouped["prints"], 0.0)
    total = float(grouped["premium"].sum())
    grouped["premium_share"] = grouped["premium"] / total if total > 0 else 0.0
    # direction from signed size relative to the type's own traded size
    tilt = np.where(grouped["contracts"] > 0,
                    grouped["net_contracts"] / grouped["contracts"], 0.0)
    grouped["direction"] = np.where(tilt > 0.15, "bought",
                            np.where(tilt < -0.15, "sold", "two-way"))
    out = grouped[cols].sort_values("premium", ascending=False).reset_index(drop=True)
    return out[out["premium_share"] >= min_share] if min_share > 0 else out


#: Horizon buckets for a weighted days-to-expiration.
HORIZON_BUCKETS = ((1, "0DTE"), (8, "weekly"), (46, "monthly"),
                   (181, "quarterly"), (float("inf"), "LEAP"))


def horizon_label(dte) -> str:
    """Name the horizon a days-to-expiration falls in."""
    try:
        d = float(dte)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(d) or d < 0:
        return ""
    for edge, name in HORIZON_BUCKETS:
        if d < edge:
            return name
    return "LEAP"


#: Block tiers, most meaningful first — see parsing.BLOCK_TIERS.
BLOCK_TIER_ORDER = ["negotiated", "facilitated", "electronic", "fragment"]
BLOCK_TIER_NOTE = {
    "negotiated": "Agreed upstairs, off the public book — floor and cross "
                  "prints. The size someone had to find a counterparty for.",
    "facilitated": "Exposed to the complex-order book or an auction for price "
                   "improvement — real size, publicly worked.",
    "electronic": "Routed the default electronic way. Size, but no evidence "
                  "anyone negotiated it.",
    "fragment": "One leg of a spread package. The premium here is a fraction "
                "of a trade whose other legs are elsewhere in the file — not "
                "an outright position.",
}


def block_type_breakdown(prints: pd.DataFrame,
                         spot: float | None = None) -> pd.DataFrame:
    """Block prints only, split by *how they printed*.

    A file can tag thousands of prints BLOCK and mean very different things
    by it: a floor-negotiated cross and one leg of an auto-executed spread
    are both "blocks". This ranks the block book by execution mechanism and
    tags each row with the tier that says how much it means.

    ``level`` is the premium-weighted strike each type traded at — where
    that money actually sat — and ``distance_pct`` places it against
    ``spot``. ``ref_price`` is the premium-weighted underlying price when
    those prints hit, which is the spot the trades were struck against and
    can differ from the current spot.

    Columns: block_type, tier, mechanism, tied, spread_leg, prints,
    contracts, premium, median_premium, avg_premium, net_contracts,
    direction, premium_share, level, ref_price, distance_pct.
    """
    cols = ["block_type", "code", "tier", "mechanism", "tied", "spread_leg",
            "prints", "contracts", "premium", "median_premium", "avg_premium",
            "net_contracts", "direction", "premium_share", "level",
            "ref_price", "distance_pct", "dte", "dte_median", "horizon",
            "otm_pct", "moneyness"]
    if prints is None or prints.empty or "is_block" not in prints:
        return pd.DataFrame(columns=cols)
    b = prints[prints["is_block"].fillna(False)].copy()
    if b.empty:
        return pd.DataFrame(columns=cols)

    b["premium"] = pd.to_numeric(b.get("premium", 0.0), errors="coerce").fillna(0.0)
    b["size"] = pd.to_numeric(b.get("size", 0.0), errors="coerce").fillna(0.0)
    b["signed_size"] = pd.to_numeric(b.get("signed_size", 0.0),
                                     errors="coerce").fillna(0.0)
    if "flow_type" not in b:
        b["flow_type"] = "block"
    for c, default in (("block_tier", ""), ("mechanism", ""),
                       ("is_tied", False), ("is_spread_leg", False),
                       ("trade_type", "")):
        if c not in b:
            b[c] = default

    # premium-weighted strike / reference price: where this type's money sat
    b["_w"] = b["premium"].where(b["premium"] > 0, 0.0)
    b["_wk"] = b["_w"] * pd.to_numeric(b["strike"], errors="coerce")
    b["_wr"] = b["_w"] * pd.to_numeric(b.get("ref_price", np.nan), errors="coerce")
    b["_dte"] = pd.to_numeric(b.get("dte", np.nan), errors="coerce")
    b["_wd"] = b["_w"] * b["_dte"]
    b["_otm"] = pd.to_numeric(b.get("otm_pct", np.nan), errors="coerce")
    b["_wo"] = b["_w"] * b["_otm"]

    g = b.groupby("flow_type", dropna=False).agg(
        code=("trade_type", lambda s: " / ".join(sorted(set(s.astype(str))
                                                        - {""})) or "—"),
        tier=("block_tier", "first"),
        mechanism=("mechanism", "first"),
        tied=("is_tied", "any"),
        spread_leg=("is_spread_leg", "any"),
        prints=("flow_type", "size"),
        contracts=("size", "sum"),
        premium=("premium", "sum"),
        median_premium=("premium", "median"),
        net_contracts=("signed_size", "sum"),
        _w=("_w", "sum"), _wk=("_wk", "sum"), _wr=("_wr", "sum"),
        _wd=("_wd", "sum"), dte_median=("_dte", "median"),
        _wo=("_wo", "sum"), _n_otm=("_otm", "count"),
    ).reset_index().rename(columns={"flow_type": "block_type"})

    with np.errstate(invalid="ignore", divide="ignore"):
        g["level"] = np.where(g["_w"] > 0, g["_wk"] / g["_w"], np.nan)
        g["ref_price"] = np.where(g["_w"] > 0, g["_wr"] / g["_w"], np.nan)
        # premium-weighted DTE: where the money's horizon is, which is not
        # the median print's horizon when the size sits further out
        g["dte"] = np.where(g["_w"] > 0, g["_wd"] / g["_w"], np.nan)
        # premium-weighted distance out of the money, and the bucket it lands in
        g["otm_pct"] = np.where((g["_w"] > 0) & (g["_n_otm"] > 0),
                                g["_wo"] / g["_w"], np.nan)
    g["horizon"] = [horizon_label(d) for d in g["dte"]]
    band = (moneyness_band(float(spot), None) / float(spot) * 100.0
            if spot else ATM_BAND_PCT * 100.0)
    g["moneyness"] = classify_moneyness(g["otm_pct"], band).to_numpy()
    ref = float(spot) if spot else float(np.nanmedian(g["ref_price"])) \
        if np.isfinite(g["ref_price"]).any() else np.nan
    g["distance_pct"] = ((g["level"] / ref - 1.0) * 100.0
                         if ref and np.isfinite(ref) else np.nan)

    g["avg_premium"] = np.where(g["prints"] > 0, g["premium"] / g["prints"], 0.0)
    total = float(g["premium"].sum())
    g["premium_share"] = g["premium"] / total if total > 0 else 0.0
    tilt = np.where(g["contracts"] > 0, g["net_contracts"] / g["contracts"], 0.0)
    g["direction"] = np.where(tilt > 0.15, "bought",
                       np.where(tilt < -0.15, "sold", "two-way"))
    return g[cols].sort_values("premium", ascending=False).reset_index(drop=True)


def block_oi_breakdown(prints: pd.DataFrame,
                       spot: float | None = None) -> pd.DataFrame:
    """The same block types, measured against the **standing book**.

    Premium says how much money printed; this says how much of the existing
    open interest that flow landed on, and how far it moved it. Open
    interest is a property of the contract, not of the print — several
    blocks on one strike all carry the same OI — so it is taken as a max
    per contract and summed across contracts, never summed over prints. On
    a real export the naive version overstates it several-fold.

    ``add_ratio`` is traded size over that open interest: 0.5 means this
    type traded half the standing book on the contracts it touched, which
    is a position being built; 0.01 is noise landing on a crowded strike.
    ``opening_share`` is the size fraction the file flagged as opening a
    position rather than closing one.

    Columns: block_type, tier, contracts_touched, open_interest, traded,
    add_ratio, opening_share, oi_share, level, distance_pct.

    A contract touched by two block types is counted under both — the
    shares are per type, so they describe composition, not a partition.
    """
    cols = ["block_type", "code", "tier", "contracts_touched", "open_interest",
            "traded", "add_ratio", "opening_share", "oi_share", "level",
            "distance_pct", "dte", "horizon"]
    if prints is None or prints.empty or "is_block" not in prints:
        return pd.DataFrame(columns=cols)
    b = prints[prints["is_block"].fillna(False)].copy()
    if b.empty or "open_interest" not in b:
        return pd.DataFrame(columns=cols)

    b["open_interest"] = pd.to_numeric(b["open_interest"], errors="coerce").fillna(0.0)
    b["size"] = pd.to_numeric(b.get("size", 0.0), errors="coerce").fillna(0.0)
    b["_open_sz"] = b["size"] * b.get("is_opening", False).astype(float)
    if "flow_type" not in b:
        b["flow_type"] = "block"
    if "block_tier" not in b:
        b["block_tier"] = ""
    if "trade_type" not in b:
        b["trade_type"] = ""
    keys = ["flow_type", "block_tier"]
    for c in ("ticker", "expiry", "strike", "type"):
        if c in b:
            keys.append(c)

    # one row per (type, contract): OI is the contract's, size is this
    # type's traded volume on it
    per = b.groupby(keys, dropna=False).agg(
        open_interest=("open_interest", "max"),
        traded=("size", "sum"),
        opening=("_open_sz", "sum"),
        code=("trade_type", "first"),
        dte=("dte", "median") if "dte" in b else ("open_interest", "size"),
    ).reset_index()
    if "dte" not in b:
        per["dte"] = np.nan
    per["_wk"] = per["open_interest"] * pd.to_numeric(
        per.get("strike", np.nan), errors="coerce")
    per["_wd"] = per["open_interest"] * pd.to_numeric(per["dte"], errors="coerce")

    g = per.groupby(["flow_type", "block_tier"], dropna=False).agg(
        code=("code", lambda s: " / ".join(sorted(set(s.astype(str))
                                                  - {""})) or "—"),
        contracts_touched=("open_interest", "size"),
        open_interest=("open_interest", "sum"),
        traded=("traded", "sum"),
        opening=("opening", "sum"),
        _wk=("_wk", "sum"), _wd=("_wd", "sum"),
    ).reset_index().rename(columns={"flow_type": "block_type",
                                    "block_tier": "tier"})

    with np.errstate(invalid="ignore", divide="ignore"):
        g["add_ratio"] = np.where(g["open_interest"] > 0,
                                  g["traded"] / g["open_interest"], np.nan)
        g["opening_share"] = np.where(g["traded"] > 0,
                                      g["opening"] / g["traded"], np.nan)
        g["level"] = np.where(g["open_interest"] > 0,
                              g["_wk"] / g["open_interest"], np.nan)
        g["dte"] = np.where(g["open_interest"] > 0,
                            g["_wd"] / g["open_interest"], np.nan)
    g["horizon"] = [horizon_label(d) for d in g["dte"]]
    total = float(g["open_interest"].sum())
    g["oi_share"] = g["open_interest"] / total if total > 0 else 0.0
    ref = float(spot) if spot else np.nan
    g["distance_pct"] = ((g["level"] / ref - 1.0) * 100.0
                         if np.isfinite(ref) else np.nan)
    return g[cols].sort_values("open_interest", ascending=False).reset_index(drop=True)


def block_tier_summary(breakdown: pd.DataFrame) -> pd.DataFrame:
    """Roll a ``block_type_breakdown`` up to one row per tier, ranked by how
    much the tier means rather than by how many prints it has."""
    cols = ["tier", "prints", "contracts", "premium", "premium_share", "note"]
    if breakdown is None or breakdown.empty:
        return pd.DataFrame(columns=cols)
    g = breakdown.groupby("tier", dropna=False).agg(
        prints=("prints", "sum"), contracts=("contracts", "sum"),
        premium=("premium", "sum"),
    ).reset_index()
    total = float(g["premium"].sum())
    g["premium_share"] = g["premium"] / total if total > 0 else 0.0
    g["note"] = g["tier"].map(BLOCK_TIER_NOTE).fillna("")
    g["_rank"] = g["tier"].map(
        {t: i for i, t in enumerate(BLOCK_TIER_ORDER)}).fillna(len(BLOCK_TIER_ORDER))
    return g.sort_values("_rank")[cols].reset_index(drop=True)


#: What each execution mechanism means for dealer hedging and price. These
#: are readings of the mechanism, not of any one day's tape — the measured
#: numbers in ``block_type_behaviour`` are what test them on your file.
MECHANISM_BEHAVIOUR = {
    "floor": "Negotiated upstairs and printed on the floor: someone had to "
             "find the other side, and a dealer is usually on it. Expect a "
             "real hedging obligation anchored at that strike — pinning "
             "toward it in a long-gamma book, acceleration through it in a "
             "short-gamma one. The highest-information block there is.",
    "cross": "Both sides arranged before the print (facilitation). The bank "
             "may be flat rather than warehousing the risk, so the forced-"
             "hedging read is weaker than the size suggests — treat the "
             "strike as a marker of institutional interest, not of flow the "
             "dealer still has to cover.",
    "cob": "Worked through the complex-order book. Real size, but the market "
           "saw it and priced it as it filled — the reaction is usually "
           "already in the tape by the time the print lands.",
    "cob auction": "A complex order exposed to auction for price improvement. "
                   "Advertised size: the move tends to happen at the print "
                   "and fade rather than build.",
    "auction": "Exposed to an auction before filling. The size was public, so "
               "read it as confirmation of a level rather than a catalyst.",
    "auto": "The default electronic route — size without evidence anyone "
            "negotiated it. Read it with the rest of the tape, not on its own.",
    "iso": "Intermarket sweep: took liquidity across exchanges to get filled "
           "now. Urgency rather than patience — closer in character to a "
           "sweep than to a block.",
}
MODIFIER_BEHAVIOUR = {
    "fragment": "One leg of a spread package, with the offsetting legs "
                "elsewhere in this file. Do not read it directionally, and "
                "do not add its premium to the outright total.",
    "tied": "Stock-tied — the delta was hedged on the trade itself, so there "
            "is no follow-on hedging flow to chase. This is a volatility "
            "position: it changes gamma at the strike, not direction.",
    "spread": "Printed as a package, so the premium covers more than one leg "
              "and the net exposure is smaller than the headline number.",
}


def block_behaviour_note(row) -> str:
    """The behavioural read for one ``block_type_breakdown`` row."""
    parts = []
    if row.get("spread_leg"):
        parts.append(MODIFIER_BEHAVIOUR["fragment"])
    else:
        base = MECHANISM_BEHAVIOUR.get(str(row.get("mechanism", "")), "")
        if base:
            parts.append(base)
        if row.get("tied"):
            parts.append(MODIFIER_BEHAVIOUR["tied"])
        if str(row.get("block_type", "")).endswith("spread"):
            parts.append(MODIFIER_BEHAVIOUR["spread"])
    return " ".join(parts)


def _bullish_intent(df: pd.DataFrame) -> pd.Series:
    """+1 when the print expresses a bullish view, -1 bearish, 0 unknown:
    bought calls and sold puts are bullish, the mirrors bearish."""
    signed = pd.to_numeric(df.get("signed_size", 0.0), errors="coerce").fillna(0.0)
    cp = df["type"].astype(str).str.strip().str.upper().str[0]
    return np.sign(signed) * np.where(cp == "P", -1.0, 1.0)


def block_type_behaviour(prints: pd.DataFrame, horizon_min: int = 30,
                         min_prints: int = 3) -> pd.DataFrame:
    """Did the tape follow each block type, on this file?

    For every block print, the underlying's move from that print's
    reference price is measured twice — ``horizon_min`` minutes later and
    at the session's last print — and signed by what the print expressed
    (bought calls / sold puts = bullish). Positive means price went the way
    the block leaned.

    Columns: block_type, tier, prints, premium, scored, followed_horizon,
    followed_close, hit_rate, sample, behaviour.

    Honest limits: the reference price is what the flow file stamps on each
    print, not exchange OHLC; a session or two is a tiny sample; and a
    block that prints at 15:58 has no horizon left to be measured over
    (those rows are excluded from the horizon column, not counted as zero).
    Types under ``min_prints`` are marked ``thin`` and left unscored.
    """
    cols = ["block_type", "tier", "prints", "premium", "scored",
            "followed_horizon", "followed_close", "hit_rate", "sample",
            "behaviour"]
    if prints is None or prints.empty or "is_block" not in prints:
        return pd.DataFrame(columns=cols)

    base = block_type_breakdown(prints)
    if base.empty:
        return pd.DataFrame(columns=cols)
    notes = {r["block_type"]: block_behaviour_note(r) for _, r in base.iterrows()}

    need = {"trade_time", "ref_price", "flow_type", "type"}
    p = prints.copy()
    if not need <= set(p.columns):
        out = base[["block_type", "tier", "prints", "premium"]].copy()
        out["scored"] = 0
        for c in ("followed_horizon", "followed_close", "hit_rate"):
            out[c] = np.nan
        out["sample"] = "no timestamps"
        out["behaviour"] = out["block_type"].map(notes)
        return out[cols].reset_index(drop=True)

    p["trade_time"] = pd.to_datetime(p["trade_time"], errors="coerce", utc=True)
    p["ref_price"] = pd.to_numeric(p["ref_price"], errors="coerce")
    p = p[p["trade_time"].notna() & (p["ref_price"] > 0)]
    if p.empty:
        return pd.DataFrame(columns=cols)
    p["_day"] = p["trade_time"].dt.date
    p["_tkr"] = p["ticker"].astype(str) if "ticker" in p else ""

    # the price path is every print's reference price, per ticker per day
    path = (p[["trade_time", "ref_price", "_day", "_tkr"]]
            .sort_values("trade_time").reset_index(drop=True))
    session = path.groupby(["_tkr", "_day"]).agg(
        close=("ref_price", "last"), last_time=("trade_time", "max"))

    b = p[p["is_block"].fillna(False)].copy()
    if b.empty:
        return pd.DataFrame(columns=cols)
    b["_target"] = b["trade_time"] + pd.Timedelta(minutes=horizon_min)
    b = b.sort_values("_target")
    later = pd.merge_asof(
        b[["_target", "_day", "_tkr"]], path.rename(columns={"ref_price": "_ref_h"}),
        left_on="_target", right_on="trade_time", by=["_tkr", "_day"],
        direction="backward",
    )
    b["_ref_h"] = later["_ref_h"].to_numpy()
    keys = pd.MultiIndex.from_arrays([b["_tkr"], b["_day"]])
    b["_close"] = session["close"].reindex(keys).to_numpy()
    b["_last_time"] = session["last_time"].reindex(keys).to_numpy()
    # a print with less than the horizon left in the session cannot be scored
    b.loc[b["_target"] > b["_last_time"], "_ref_h"] = np.nan

    intent = _bullish_intent(b)
    b["_ft_h"] = intent * (b["_ref_h"] / b["ref_price"] - 1.0) * 100.0
    b["_ft_c"] = intent * (b["_close"] / b["ref_price"] - 1.0) * 100.0
    b.loc[intent == 0, ["_ft_h", "_ft_c"]] = np.nan   # mid prints have no side
    b["_w"] = b["premium"].where(b["premium"] > 0, 0.0)

    def _wmean(sub: pd.DataFrame, col: str) -> float:
        ok = sub[sub[col].notna()]
        w = ok["_w"]
        if ok.empty or w.sum() <= 0:
            return float(ok[col].mean()) if not ok.empty else np.nan
        return float(np.average(ok[col], weights=w))

    rows = []
    for name, sub in b.groupby("flow_type", dropna=False):
        scored = int(sub["_ft_c"].notna().sum())
        thin = scored < min_prints
        rows.append({
            "block_type": name,
            "scored": scored,
            "followed_horizon": np.nan if thin else _wmean(sub, "_ft_h"),
            "followed_close": np.nan if thin else _wmean(sub, "_ft_c"),
            "hit_rate": np.nan if thin else
                        float((sub["_ft_c"].dropna() > 0).mean()),
            "sample": "thin" if thin else ("ok" if scored >= 3 * min_prints
                                           else "small"),
        })
    measured = pd.DataFrame(rows)
    out = base[["block_type", "tier", "prints", "premium"]].merge(
        measured, on="block_type", how="left")
    out["scored"] = out["scored"].fillna(0).astype(int)
    out["sample"] = out["sample"].fillna("thin")
    out["behaviour"] = out["block_type"].map(notes).fillna("")
    return out[cols].reset_index(drop=True)


#: Bucket order for the term-structure view, near-dated first.
HORIZON_ORDER = ["0DTE", "weekly", "monthly", "quarterly", "LEAP"]
HORIZON_RANGE = {"0DTE": "0", "weekly": "1–7", "monthly": "8–45",
                 "quarterly": "46–180", "LEAP": "181+"}


def block_dte_breakdown(prints: pd.DataFrame,
                        spot: float | None = None) -> pd.DataFrame:
    """Block flow by **when it expires**, near-dated first.

    The per-type tables answer "what is this type's horizon"; this answers
    the other direction — how the block book is distributed across tenor,
    and which type owns each bucket. Rows are ordered by horizon rather
    than by size, because a term structure read out of order is not a term
    structure.

    Open interest follows the same max-per-contract rule as
    ``block_oi_breakdown``, so a contract traded by several prints in one
    bucket contributes its book once.

    Columns: horizon, dte_range, prints, contracts, premium,
    premium_share, open_interest, add_ratio, opening_share, net_contracts,
    direction, level, distance_pct, top_type, top_share.
    """
    cols = ["horizon", "dte_range", "prints", "contracts", "premium",
            "premium_share", "open_interest", "add_ratio", "opening_share",
            "net_contracts", "direction", "level", "distance_pct",
            "top_type", "top_share"]
    if prints is None or prints.empty or "is_block" not in prints:
        return pd.DataFrame(columns=cols)
    b = prints[prints["is_block"].fillna(False)].copy()
    if b.empty or "dte" not in b:
        return pd.DataFrame(columns=cols)

    b["_dte"] = pd.to_numeric(b["dte"], errors="coerce")
    b = b[b["_dte"].notna()]
    if b.empty:
        return pd.DataFrame(columns=cols)
    b["horizon"] = [horizon_label(d) for d in b["_dte"]]
    b = b[b["horizon"] != ""]
    if b.empty:
        return pd.DataFrame(columns=cols)

    b["premium"] = pd.to_numeric(b.get("premium", 0.0), errors="coerce").fillna(0.0)
    b["size"] = pd.to_numeric(b.get("size", 0.0), errors="coerce").fillna(0.0)
    b["signed_size"] = pd.to_numeric(b.get("signed_size", 0.0),
                                     errors="coerce").fillna(0.0)
    b["open_interest"] = pd.to_numeric(b.get("open_interest", 0.0),
                                       errors="coerce").fillna(0.0)
    b["_open_sz"] = b["size"] * b.get("is_opening", False).astype(float)
    b["_wk"] = b["premium"] * pd.to_numeric(b["strike"], errors="coerce")

    g = b.groupby("horizon", dropna=False).agg(
        prints=("premium", "size"),
        contracts=("size", "sum"),
        premium=("premium", "sum"),
        net_contracts=("signed_size", "sum"),
        opening=("_open_sz", "sum"),
        _wk=("_wk", "sum"),
    ).reset_index()

    # open interest: max per contract inside each bucket, then summed
    ckeys = ["horizon"] + [c for c in ("ticker", "expiry", "strike", "type")
                           if c in b]
    per = b.groupby(ckeys, dropna=False)["open_interest"].max().reset_index()
    g = g.merge(per.groupby("horizon")["open_interest"].sum().reset_index(),
                on="horizon", how="left")

    total = float(g["premium"].sum())
    g["premium_share"] = g["premium"] / total if total > 0 else 0.0
    with np.errstate(invalid="ignore", divide="ignore"):
        g["add_ratio"] = np.where(g["open_interest"] > 0,
                                  g["contracts"] / g["open_interest"], np.nan)
        g["opening_share"] = np.where(g["contracts"] > 0,
                                      g["opening"] / g["contracts"], np.nan)
        g["level"] = np.where(g["premium"] > 0, g["_wk"] / g["premium"], np.nan)
    ref = float(spot) if spot else np.nan
    g["distance_pct"] = ((g["level"] / ref - 1.0) * 100.0
                         if np.isfinite(ref) else np.nan)
    tilt = np.where(g["contracts"] > 0, g["net_contracts"] / g["contracts"], 0.0)
    g["direction"] = np.where(tilt > 0.15, "bought",
                       np.where(tilt < -0.15, "sold", "two-way"))

    # who owns each bucket
    if "flow_type" in b:
        owner = b.groupby(["horizon", "flow_type"])["premium"].sum().reset_index()
        idx = owner.groupby("horizon")["premium"].idxmax()
        top = owner.loc[idx].set_index("horizon")
        g["top_type"] = g["horizon"].map(top["flow_type"])
        g["top_share"] = (g["horizon"].map(top["premium"]) / g["premium"]).where(
            g["premium"] > 0)
    else:
        g["top_type"], g["top_share"] = "", np.nan

    g["dte_range"] = g["horizon"].map(HORIZON_RANGE).fillna("")
    g["_rank"] = g["horizon"].map({h: i for i, h in enumerate(HORIZON_ORDER)})
    return g.sort_values("_rank")[cols].reset_index(drop=True)


def block_strike_ladder(prints: pd.DataFrame, spot: float | None = None,
                        top_n: int = 15, rank: str = "premium",
                        near_pct: float | None = None) -> pd.DataFrame:
    """**Where** the block money and size actually sit: one row per strike.

    Every other block table asks *what kind* of flow this is — by type, by
    tenor, by moneyness. This one asks the question a levels dashboard
    exists for: which strikes. No kernel smoothing and no top-N peak
    finding, just the raw ladder, so a strike either has the money or it
    does not.

    ``rank`` picks what "biggest" means — ``premium`` (dollars committed)
    or ``contracts`` (size). They disagree: cheap far-dated strikes can
    carry huge quantity for little premium, and a deep-ITM strike the
    reverse. Both columns are always present so the mismatch is visible.

    Open interest is summed over the (expiry, right) contracts at that
    strike, each taken as a max over its prints — the same rule as
    everywhere else, so ``add_ratio`` is traded size against the standing
    book on that strike.

    Columns: strike, distance_pct, premium, premium_share, contracts,
    contracts_share, prints, open_interest, add_ratio, net_contracts,
    direction, call_premium, put_premium, side, dte, top_type.
    """
    cols = ["strike", "distance_pct", "premium", "premium_share", "contracts",
            "contracts_share", "prints", "open_interest", "add_ratio",
            "net_contracts", "direction", "call_premium", "put_premium",
            "side", "dte", "top_type"]
    if prints is None or prints.empty or "is_block" not in prints:
        return pd.DataFrame(columns=cols)
    b = prints[prints["is_block"].fillna(False)].copy()
    if b.empty:
        return pd.DataFrame(columns=cols)
    b["strike"] = pd.to_numeric(b["strike"], errors="coerce")
    b = b[b["strike"].notna()]
    if b.empty:
        return pd.DataFrame(columns=cols)

    ref = float(spot) if spot else float(
        pd.to_numeric(b.get("ref_price", np.nan), errors="coerce").median())
    if near_pct and np.isfinite(ref):
        b = b[(b["strike"] - ref).abs() <= ref * near_pct]
        if b.empty:
            return pd.DataFrame(columns=cols)

    b["premium"] = pd.to_numeric(b.get("premium", 0.0), errors="coerce").fillna(0.0)
    b["size"] = pd.to_numeric(b.get("size", 0.0), errors="coerce").fillna(0.0)
    b["signed_size"] = pd.to_numeric(b.get("signed_size", 0.0),
                                     errors="coerce").fillna(0.0)
    b["_cp"] = b["type"].astype(str).str.strip().str.upper().str[0]
    b["_call_prem"] = b["premium"].where(b["_cp"] == "C", 0.0)
    b["_put_prem"] = b["premium"].where(b["_cp"] == "P", 0.0)
    b["_wd"] = b["premium"] * pd.to_numeric(b.get("dte", np.nan), errors="coerce")

    g = b.groupby("strike", dropna=False).agg(
        premium=("premium", "sum"),
        contracts=("size", "sum"),
        prints=("premium", "size"),
        net_contracts=("signed_size", "sum"),
        call_premium=("_call_prem", "sum"),
        put_premium=("_put_prem", "sum"),
        _wd=("_wd", "sum"),
    ).reset_index()

    # open interest belongs to the contract: max per (strike, expiry, right)
    okeys = ["strike"] + [c for c in ("ticker", "expiry", "type") if c in b]
    per = b.groupby(okeys, dropna=False)["open_interest"].max().reset_index()
    g = g.merge(per.groupby("strike")["open_interest"].sum().reset_index(),
                on="strike", how="left")

    tot_p, tot_c = float(g["premium"].sum()), float(g["contracts"].sum())
    g["premium_share"] = g["premium"] / tot_p if tot_p > 0 else np.nan
    g["contracts_share"] = g["contracts"] / tot_c if tot_c > 0 else np.nan
    with np.errstate(invalid="ignore", divide="ignore"):
        g["add_ratio"] = np.where(g["open_interest"] > 0,
                                  g["contracts"] / g["open_interest"], np.nan)
        g["dte"] = np.where(g["premium"] > 0, g["_wd"] / g["premium"], np.nan)
    g["distance_pct"] = ((g["strike"] / ref - 1.0) * 100.0
                         if np.isfinite(ref) else np.nan)
    tilt = np.where(g["contracts"] > 0, g["net_contracts"] / g["contracts"], 0.0)
    g["direction"] = np.where(tilt > 0.15, "bought",
                       np.where(tilt < -0.15, "sold", "two-way"))
    share_c = np.where(g["premium"] > 0, g["call_premium"] / g["premium"], 0.5)
    g["side"] = np.where(share_c > 0.65, "calls",
                  np.where(share_c < 0.35, "puts", "mixed"))

    if "flow_type" in b:
        owner = b.groupby(["strike", "flow_type"])["premium"].sum().reset_index()
        idx = owner.groupby("strike")["premium"].idxmax()
        g["top_type"] = g["strike"].map(owner.loc[idx].set_index("strike")["flow_type"])
    else:
        g["top_type"] = ""

    key = "contracts" if rank == "contracts" else "premium"
    return g.nlargest(top_n, key)[cols].reset_index(drop=True)


#: Moneyness buckets, in the order a strike ladder reads them.
MONEYNESS_ORDER = ["ITM", "ATM", "OTM"]
#: Half-width of the at-the-money band, as a fraction of spot. A file's own
#: "at the money" flag is usually strict equality — on a real export only 32
#: of 19,475 prints qualify — which is true and useless. A quarter of a
#: percent is roughly the nearest strike or two on a liquid name.
ATM_BAND_PCT = 0.0025


def moneyness_band(spot: float, spacing: float | None = None,
                   band_pct: float = ATM_BAND_PCT) -> float:
    """Half-width of the ATM band in price, floored at half a strike step.

    Without the floor a cheap underlying with wide strikes gets an empty
    ATM bucket: 0.25% of $30 is 7 cents, and no strike is ever that close.
    """
    band = float(spot) * float(band_pct)
    if spacing and np.isfinite(spacing) and spacing > 0:
        band = max(band, spacing * 0.5)
    return band


def classify_moneyness(otm_pct, band_pct: float = ATM_BAND_PCT * 100):
    """ITM / ATM / OTM from a signed out-of-the-money percentage.

    ``otm_pct`` is positive out of the money and negative in the money on
    both sides (see ``parsing``), so one comparison covers calls and puts.
    """
    v = pd.to_numeric(pd.Series(otm_pct), errors="coerce")
    out = pd.Series(np.where(v.abs() <= band_pct, "ATM",
                    np.where(v > 0, "OTM", "ITM")), index=v.index, dtype=object)
    return out.mask(v.isna(), "")


def block_moneyness_breakdown(prints: pd.DataFrame, spot: float | None = None,
                              band_pct: float | None = None) -> pd.DataFrame:
    """Block flow by **where it struck relative to spot**, ITM to OTM.

    The counterpart to the expiration table: that one splits the book
    across time, this one splits it across the strike ladder. Deep OTM
    size bought is a lottery ticket or a hedge; ITM size is closer to
    stock and usually carries delta someone wanted; ATM size is where
    gamma actually lives, so it is the part that moves dealer hedging now.

    Each print is classified against **its own reference price** — what
    spot was when it traded — not against the latest spot, so a print is
    not retroactively re-labelled by a move that happened after it.

    Columns: moneyness, band, prints, contracts, premium, premium_share,
    open_interest, add_ratio, opening_share, net_contracts, direction,
    avg_otm_pct, level, distance_pct, top_type, top_share.
    """
    cols = ["moneyness", "band", "prints", "contracts", "premium",
            "premium_share", "open_interest", "add_ratio", "opening_share",
            "net_contracts", "direction", "avg_otm_pct", "level",
            "distance_pct", "top_type", "top_share"]
    if prints is None or prints.empty or "is_block" not in prints:
        return pd.DataFrame(columns=cols)
    b = prints[prints["is_block"].fillna(False)].copy()
    if b.empty or "otm_pct" not in b:
        return pd.DataFrame(columns=cols)

    b["_otm"] = pd.to_numeric(b["otm_pct"], errors="coerce")
    b = b[b["_otm"].notna()]
    if b.empty:
        return pd.DataFrame(columns=cols)

    ks = np.sort(pd.to_numeric(b["strike"], errors="coerce").dropna().unique())
    spacing = float(np.median(np.diff(ks))) if len(ks) > 1 else None
    ref = float(spot) if spot else float(
        pd.to_numeric(b["ref_price"], errors="coerce").median())
    if band_pct is None:
        band_pct = moneyness_band(ref, spacing) / ref * 100.0 if ref else \
            ATM_BAND_PCT * 100.0
    b["moneyness"] = classify_moneyness(b["_otm"], band_pct).to_numpy()

    b["premium"] = pd.to_numeric(b.get("premium", 0.0), errors="coerce").fillna(0.0)
    b["size"] = pd.to_numeric(b.get("size", 0.0), errors="coerce").fillna(0.0)
    b["signed_size"] = pd.to_numeric(b.get("signed_size", 0.0),
                                     errors="coerce").fillna(0.0)
    b["open_interest"] = pd.to_numeric(b.get("open_interest", 0.0),
                                       errors="coerce").fillna(0.0)
    b["_open_sz"] = b["size"] * b.get("is_opening", False).astype(float)
    b["_wk"] = b["premium"] * pd.to_numeric(b["strike"], errors="coerce")
    b["_wo"] = b["premium"] * b["_otm"]

    g = b.groupby("moneyness", dropna=False).agg(
        prints=("premium", "size"),
        contracts=("size", "sum"),
        premium=("premium", "sum"),
        net_contracts=("signed_size", "sum"),
        opening=("_open_sz", "sum"),
        _wk=("_wk", "sum"), _wo=("_wo", "sum"),
    ).reset_index()

    ckeys = ["moneyness"] + [c for c in ("ticker", "expiry", "strike", "type")
                             if c in b]
    per = b.groupby(ckeys, dropna=False)["open_interest"].max().reset_index()
    g = g.merge(per.groupby("moneyness")["open_interest"].sum().reset_index(),
                on="moneyness", how="left")

    total = float(g["premium"].sum())
    g["premium_share"] = g["premium"] / total if total > 0 else 0.0
    with np.errstate(invalid="ignore", divide="ignore"):
        g["add_ratio"] = np.where(g["open_interest"] > 0,
                                  g["contracts"] / g["open_interest"], np.nan)
        g["opening_share"] = np.where(g["contracts"] > 0,
                                      g["opening"] / g["contracts"], np.nan)
        g["level"] = np.where(g["premium"] > 0, g["_wk"] / g["premium"], np.nan)
        g["avg_otm_pct"] = np.where(g["premium"] > 0, g["_wo"] / g["premium"], np.nan)
    g["distance_pct"] = ((g["level"] / ref - 1.0) * 100.0 if ref else np.nan)
    tilt = np.where(g["contracts"] > 0, g["net_contracts"] / g["contracts"], 0.0)
    g["direction"] = np.where(tilt > 0.15, "bought",
                       np.where(tilt < -0.15, "sold", "two-way"))

    if "flow_type" in b:
        owner = b.groupby(["moneyness", "flow_type"])["premium"].sum().reset_index()
        idx = owner.groupby("moneyness")["premium"].idxmax()
        top = owner.loc[idx].set_index("moneyness")
        g["top_type"] = g["moneyness"].map(top["flow_type"])
        g["top_share"] = (g["moneyness"].map(top["premium"]) / g["premium"]).where(
            g["premium"] > 0)
    else:
        g["top_type"], g["top_share"] = "", np.nan

    g["band"] = f"±{band_pct:.2f}%"
    g["_rank"] = g["moneyness"].map({m: i for i, m in enumerate(MONEYNESS_ORDER)})
    return g.sort_values("_rank")[cols].reset_index(drop=True)


#: Tenor windows for the focused block view. Cumulative, not exclusive:
#: "weekly" is everything expiring within ten days, 0DTE included, because
#: what is in play this week is the question — not what is in play only in
#: the second half of it.
BLOCK_WINDOWS = {"0DTE": 0, "Weekly": 10, "Monthly": 45}


def block_window_breakdown(prints: pd.DataFrame, max_dte: int,
                           spot: float | None = None) -> pd.DataFrame:
    """Block types expiring within ``max_dte`` days, premium and book in one
    table.

    The term-structure table splits the whole book across every tenor; this
    is the opposite move — pick the window you are trading and see only the
    types living in it, with the premium and open-interest columns side by
    side rather than in two places.

    ``share`` is the type's share of premium *within the window*, and
    ``share_of_book`` its share of all block premium, so a window that is
    a rounding error against the whole book cannot look dominant.

    Columns: block_type, code, tier, prints, contracts, premium, share,
    share_of_book, open_interest, add_ratio, opening_share, dte, level,
    distance_pct, net_contracts, direction.
    """
    cols = ["block_type", "code", "tier", "prints", "contracts", "premium",
            "share", "share_of_book", "open_interest", "add_ratio",
            "opening_share", "dte", "level", "distance_pct", "net_contracts",
            "direction"]
    if prints is None or prints.empty or "is_block" not in prints:
        return pd.DataFrame(columns=cols)
    blocks = prints[prints["is_block"].fillna(False)]
    if blocks.empty or "dte" not in blocks:
        return pd.DataFrame(columns=cols)

    dte = pd.to_numeric(blocks["dte"], errors="coerce")
    window = blocks[dte.notna() & (dte >= 0) & (dte <= float(max_dte))]
    if window.empty:
        return pd.DataFrame(columns=cols)

    prem = block_type_breakdown(window, spot)
    oi = block_oi_breakdown(window, spot)
    if prem.empty:
        return pd.DataFrame(columns=cols)

    out = prem[["block_type", "code", "tier", "prints", "contracts", "premium",
                "premium_share", "dte", "level", "distance_pct",
                "net_contracts", "direction"]].rename(
        columns={"premium_share": "share"})
    if not oi.empty:
        out = out.merge(
            oi[["block_type", "open_interest", "add_ratio", "opening_share"]],
            on="block_type", how="left")
    else:
        for c in ("open_interest", "add_ratio", "opening_share"):
            out[c] = np.nan

    book = pd.to_numeric(blocks.get("premium", 0.0), errors="coerce").fillna(0.0).sum()
    out["share_of_book"] = out["premium"] / book if book > 0 else np.nan
    return out[cols].sort_values("premium", ascending=False).reset_index(drop=True)


def block_window_summary(prints: pd.DataFrame,
                         windows: dict | None = None) -> pd.DataFrame:
    """One row per tenor window: how much of the block book is in play
    inside it, and which type owns it.

    Windows are cumulative, so the rows nest rather than partition — the
    monthly row contains the weekly row contains 0DTE. That is what a
    trader means by "what is in play this week", and the column names say
    so rather than implying a split.
    """
    cols = ["window", "max_dte", "prints", "contracts", "premium",
            "share_of_book", "top_type", "top_share"]
    windows = windows or BLOCK_WINDOWS
    if prints is None or prints.empty or "is_block" not in prints:
        return pd.DataFrame(columns=cols)
    blocks = prints[prints["is_block"].fillna(False)]
    if blocks.empty or "dte" not in blocks:
        return pd.DataFrame(columns=cols)
    book = pd.to_numeric(blocks.get("premium", 0.0), errors="coerce").fillna(0.0).sum()

    rows = []
    for name, max_dte in windows.items():
        w = block_window_breakdown(blocks, max_dte)
        if w.empty:
            rows.append({"window": name, "max_dte": max_dte, "prints": 0,
                         "contracts": 0.0, "premium": 0.0,
                         "share_of_book": 0.0, "top_type": "", "top_share": np.nan})
            continue
        prem = float(w["premium"].sum())
        top = w.iloc[0]
        rows.append({
            "window": name, "max_dte": max_dte,
            "prints": int(w["prints"].sum()),
            "contracts": float(w["contracts"].sum()), "premium": prem,
            "share_of_book": prem / book if book > 0 else np.nan,
            "top_type": str(top["block_type"]),
            "top_share": float(top["premium"] / prem) if prem > 0 else np.nan,
        })
    return pd.DataFrame(rows, columns=cols)


def block_dominance(prints: pd.DataFrame, spot: float | None = None,
                    min_share: float = 0.05) -> dict:
    """Which block type is actually running this book.

    Judged on three independent lenses, because they disagree and the
    disagreement is the information:

    * **money** — share of block premium;
    * **book** — share of block open interest;
    * **impact** — traded size over the open interest it landed on, i.e.
      how much the type moved the book it touched.

    The impact lens only considers types holding at least ``min_share`` of
    premium or open interest, so a two-print type with a 200% add ratio
    cannot win on a rounding error. A type leading two of three lenses is
    the leader; otherwise the book is called **split** and each lens
    reports its own winner, because forcing a single name on a genuinely
    divided book would be a made-up answer.

    Returns ``{}`` when there are no blocks to rank.
    """
    prem = block_type_breakdown(prints, spot)
    if prem.empty:
        return {}
    oi = block_oi_breakdown(prints, spot)

    p_idx = prem.set_index("block_type")
    o_idx = oi.set_index("block_type") if not oi.empty else pd.DataFrame()

    def _top(frame, col):
        if frame is None or frame.empty or col not in frame:
            return None
        s = frame[col].dropna()
        if s.empty or s.max() <= 0:
            return None
        name = s.idxmax()
        return {"block_type": str(name), "value": float(s.max()),
                "tier": str(frame.loc[name, "tier"]) if "tier" in frame else ""}

    by_money = _top(p_idx, "premium_share")
    by_book = _top(o_idx, "oi_share")

    # impact: only types that are materially present on either lens
    big = None
    if not o_idx.empty:
        share = o_idx["oi_share"].reindex(p_idx.index).fillna(0.0)
        keep = (p_idx["premium_share"] >= min_share) | (share >= min_share)
        names = [n for n in p_idx.index[keep] if n in o_idx.index]
        big = o_idx.loc[names] if names else None
    by_impact = _top(big, "add_ratio")

    lenses = {"money": by_money, "book": by_book, "impact": by_impact}
    present = {k: v for k, v in lenses.items() if v}
    if not present:
        return {}
    tally: dict[str, int] = {}
    for v in present.values():
        tally[v["block_type"]] = tally.get(v["block_type"], 0) + 1
    leader, won = max(tally.items(), key=lambda kv: kv[1])
    clear = won >= 2 and won > max(
        [n for t, n in tally.items() if t != leader] or [0])

    tiers = block_tier_summary(prem)
    tier_leader = str(tiers.iloc[0]["tier"]) if not tiers.empty else ""
    if not tiers.empty:
        top_tier = tiers.loc[tiers["premium"].idxmax()]
        tier_leader = str(top_tier["tier"])
        tier_share = float(top_tier["premium_share"])
    else:
        tier_share = float("nan")

    lead_tier = ""
    if leader in p_idx.index and "tier" in p_idx:
        lead_tier = str(p_idx.loc[leader, "tier"])

    return {
        "leader": leader if clear else None,
        "leader_tier": lead_tier if clear else "",
        "verdict": "clear" if clear else "split",
        "lenses_won": won, "lenses": len(present),
        "by_money": by_money, "by_book": by_book, "by_impact": by_impact,
        "tier_leader": tier_leader, "tier_share": tier_share,
        "label": _dominance_label(leader if clear else None, present,
                                  tier_leader, tier_share),
    }


_LENS_WORD = {"money": "premium", "book": "open interest", "impact": "book impact"}


def _dominance_label(leader: str | None, present: dict, tier_leader: str,
                     tier_share: float) -> str:
    """One line naming who runs the block book."""
    def _fmt(lens, v):
        val = (f"{v['value']:.0%}" if lens != "impact"
               else f"{v['value']:.0%} add")
        return f"{_LENS_WORD[lens]} ({val})"

    if leader:
        won = [_fmt(k, v) for k, v in present.items()
               if v["block_type"] == leader]
        lost = [f"{v['block_type']} on {_LENS_WORD[k]}"
                for k, v in present.items() if v["block_type"] != leader]
        text = f"**{leader}** dominates — leads on " + " and ".join(won)
        if lost:
            text += "; " + ", ".join(lost) + " leads the rest"
    else:
        parts = [f"**{v['block_type']}** on {_LENS_WORD[k]}"
                 for k, v in present.items()]
        text = "Split book — " + ", ".join(parts)
    if tier_leader and np.isfinite(tier_share):
        text += (f". {tier_leader.title()} size is {tier_share:.0%} of block "
                 "premium")
    return text + "."


def institutional_mask(prints: pd.DataFrame, types: list[str] | None = None,
                       drop_fragments: bool = True) -> pd.Series:
    """Which prints count as institutional size.

    Defaults to the parser's ``is_institutional`` (block-shaped, floor, or
    cross prints) minus spread legs: a leg of a package is not an outright
    position, and in a real export legs can be most of the block prints
    while carrying a few percent of block premium. Pass
    ``drop_fragments=False`` to keep them, or ``types`` — display labels
    from the breakdowns — to select explicitly, which is what the sidebar
    picker does.
    """
    if prints is None or prints.empty:
        return pd.Series(dtype=bool)
    if types is not None:
        if "flow_type" not in prints:
            return pd.Series(False, index=prints.index)
        return prints["flow_type"].isin(types)
    if "is_institutional" in prints:
        m = prints["is_institutional"].fillna(False)
    elif "is_block" in prints:
        m = prints["is_block"].fillna(False)
    else:
        return pd.Series(False, index=prints.index)
    if drop_fragments and "is_spread_leg" in prints:
        m = m & ~prints["is_spread_leg"].fillna(False)
    return m


def flow_books(prints: pd.DataFrame, spot: float, asof: date, rate: float = 0.045,
               multiplier: float = DEFAULT_MULTIPLIER) -> dict[str, "Analysis"]:
    """Analyze the blocks-only and sweeps-only books separately (signed-flow
    weighting) — patient institutional size vs urgent aggressive flow."""
    from dealer_gex.parsing import ChainParseError, aggregate_prints

    defs = [("blocks", institutional_mask(prints)), ("sweeps", prints["is_sweep"])]
    # floor prints get their own book when the file distinguishes them: an
    # upstairs-negotiated book reads differently from electronic size
    if "is_floor" in prints and prints["is_floor"].any():
        defs.append(("floor", prints["is_floor"]))

    books: dict[str, Analysis] = {}
    for name, mask in defs:
        sub = prints[mask]
        if sub.empty or sub["signed_size"].abs().sum() == 0:
            continue
        try:
            books[name] = analyze(aggregate_prints(sub), spot, asof, rate,
                                  weight="flow", multiplier=multiplier)
        except (ValueError, ChainParseError):  # nothing analyzable in this slice
            continue
    return books


def intraday_flow(prints: pd.DataFrame, freq: str = "5min") -> pd.DataFrame:
    """Cumulative signed order flow through the session, binned in time.
    Positive = net customer buying (dealers pushed short). Separate running
    totals for all prints, blocks, and sweeps so you can see *when* — and
    *who* — the flow landed. Columns: time, cum_all, cum_block, cum_sweep."""
    cols = ["time", "cum_all", "cum_block", "cum_sweep"]
    if prints is None or "trade_time" not in prints or "signed_size" not in prints:
        return pd.DataFrame(columns=cols)
    p = prints.dropna(subset=["trade_time"])
    if p.empty:
        return pd.DataFrame(columns=cols)
    idx = p.set_index("trade_time").sort_index()
    ss = idx["signed_size"]
    allc = ss.resample(freq).sum().cumsum()
    blk = ss.where(idx.get("is_block", False), 0.0).resample(freq).sum().cumsum()
    swp = ss.where(idx.get("is_sweep", False), 0.0).resample(freq).sum().cumsum()
    return pd.DataFrame({"time": allc.index, "cum_all": allc.to_numpy(),
                         "cum_block": blk.reindex(allc.index).to_numpy(),
                         "cum_sweep": swp.reindex(allc.index).to_numpy()})


def session_range(prints: pd.DataFrame) -> tuple[float, float, float, float] | None:
    """Reconstruct a session's (low, high, close, open) from the reference
    price stamped on each print — the traded-through range, a stand-in for
    the day's OHLC using only the data in the flow file."""
    if prints is None or "ref_price" not in prints:
        return None
    sub = prints.dropna(subset=["ref_price"])
    if sub.empty:
        return None
    lo, hi = float(sub["ref_price"].min()), float(sub["ref_price"].max())
    if "trade_time" in sub and sub["trade_time"].notna().any():
        s = sub.dropna(subset=["trade_time"]).sort_values("trade_time")
        return lo, hi, float(s["ref_price"].iloc[-1]), float(s["ref_price"].iloc[0])
    return lo, hi, float(sub["ref_price"].iloc[-1]), float(sub["ref_price"].iloc[0])


def session_ohlc(candles: pd.DataFrame,
                 session_tz: str = "America/New_York") -> pd.DataFrame:
    """Collapse candles into one true OHLC row per trading session.

    Sessions are cut on the **exchange** date, not the file's local date:
    a bar stamped 19:15 IST belongs to that day's New York session at
    09:45, and grouping on the IST calendar day would file the US
    afternoon under tomorrow.

    Aggregation is by time, not row order — open is the first bar's open
    and close the last bar's close after sorting, so an unsorted export
    cannot invert them.

    Columns: date, open, high, low, close, volume, bars.
    """
    cols = ["date", "open", "high", "low", "close", "volume", "bars"]
    if candles is None or candles.empty or "time" not in candles:
        return pd.DataFrame(columns=cols)
    c = candles.dropna(subset=["time"]).sort_values("time")
    if c.empty:
        return pd.DataFrame(columns=cols)
    local = pd.to_datetime(c["time"], utc=True).dt.tz_convert(session_tz)
    c = c.assign(_d=local.dt.date)
    g = c.groupby("_d").agg(
        open=("open", "first"), high=("high", "max"),
        low=("low", "min"), close=("close", "last"),
        volume=("volume", "sum"), bars=("close", "size"),
    ).reset_index().rename(columns={"_d": "date"})
    return g[cols].sort_values("date").reset_index(drop=True)


def range_from_ohlc(candles: pd.DataFrame, day,
                    session_tz: str = "America/New_York"):
    """``(low, high, close, open)`` for one session from real candles —
    the same shape ``session_range`` reconstructs from print reference
    prices, but measured rather than inferred. ``None`` when that session
    is not in the file."""
    sess = session_ohlc(candles, session_tz)
    if sess.empty:
        return None
    row = sess[sess["date"] == day]
    if row.empty:
        return None
    r = row.iloc[0]
    return float(r["low"]), float(r["high"]), float(r["close"]), float(r["open"])


def _role_kind(role: str) -> str:
    if "pin" in role or role in ("Regime pivot", "At spot"):
        return "pin"
    if role.startswith("Support"):
        return "support"
    if role.startswith("Resistance"):
        return "resistance"
    return "pin"


def level_hit_rate(days: list[dict], touch: float = 0.0015,
                   brk: float = 0.003) -> tuple[pd.DataFrame, dict]:
    """Did prior-day levels hold the next day? For each consecutive day
    pair, test the earlier day's confluence levels against the later day's
    reconstructed range.

    ``days``: sorted list of {date, master (DataFrame), low, high, close}.
    Support holds if price reached it (low ≤ level·(1+touch)) without
    breaking through (low ≥ level·(1-brk)); resistance is the mirror; pins
    hold if the range straddled the level and price closed within 2·touch.

    Returns (detail_df, summary_dict). The summary carries overall and
    per-confidence hit rates. Honest caveat: the range comes from print
    reference prices, not exchange OHLC, and it is a small sample.
    """
    cols = ["from_date", "to_date", "level", "role", "confidence",
            "type", "tested", "held", "outcome", "families"]
    rows = []
    for prev, nxt in zip(days, days[1:]):
        lo, hi, close = nxt["low"], nxt["high"], nxt["close"]
        if prev.get("master") is None or prev["master"].empty:
            continue
        has_fam = "families" in prev["master"].columns
        for _, lv in prev["master"].iterrows():
            level, role, conf = float(lv["level"]), lv["role"], lv["confidence"]
            fams = list(lv["families"]) if has_fam else []
            kind = _role_kind(role)
            tested = held = False
            if kind == "support":
                tested = lo <= level * (1 + touch)
                held = tested and lo >= level * (1 - brk)
            elif kind == "resistance":
                tested = hi >= level * (1 - touch)
                held = tested and hi <= level * (1 + brk)
            else:  # pin
                tested = lo <= level <= hi
                held = tested and abs(close - level) <= level * touch * 2
            outcome = "not tested" if not tested else ("held" if held else "broke")
            rows.append((prev["date"], nxt["date"], level, role, conf,
                         kind, tested, held, outcome, fams))

    detail = pd.DataFrame(rows, columns=cols)
    tested = detail[detail["tested"]]
    n_t = len(tested)
    summary = {
        "pairs": max(len(days) - 1, 0),
        "tested": n_t,
        "held": int(tested["held"].sum()) if n_t else 0,
        "rate": float(tested["held"].mean()) if n_t else None,
        "by_confidence": {
            c: {"tested": int((tested["confidence"] == c).sum()),
                "held": int(tested.loc[tested["confidence"] == c, "held"].sum())}
            for c in ("high", "medium", "low")
        },
        "by_type": {
            k: {"tested": int((tested["type"] == k).sum()),
                "held": int(tested.loc[tested["type"] == k, "held"].sum())}
            for k in ("support", "resistance", "pin")
        },
    }
    return detail, summary


def block_campaigns(day_prints: list[tuple[date, pd.DataFrame]],
                    min_days: int = 2, top_n: int = 10) -> pd.DataFrame:
    """Detect multi-day block campaigns: option contracts hit by block
    prints on two or more distinct days — the footprint of an institution
    building (or unwinding) a position over time, not a one-off trade.

    ``day_prints`` is a list of (date, prints_df) — one entry per uploaded
    day. Returns one row per (expiry, strike, type) touched by blocks on at
    least ``min_days`` days, ranked by total block premium.

    Columns: expiry, strike, type, days, first_day, last_day, premium,
    net_size (signed; + = customers net bought, dealers short), direction,
    daily (list of per-day signed sizes for a sparkline).
    """
    cols = ["expiry", "strike", "type", "days", "first_day", "last_day",
            "premium", "net_size", "direction", "daily"]
    frames = []
    for d, p in day_prints:
        if p is None or p.empty:
            continue
        b = p[institutional_mask(p) & p["strike"].notna()].copy()
        if b.empty:
            continue
        b["day"] = d
        b["cp"] = b["type"].astype(str).str.strip().str.upper().str[0]
        frames.append(b)
    if not frames:
        return pd.DataFrame(columns=cols)
    allb = pd.concat(frames, ignore_index=True)

    rows = []
    for (exp, strike, cp), g in allb.groupby(["expiry", "strike", "cp"], dropna=False):
        days = sorted(g["day"].unique())
        if len(days) < min_days:
            continue
        net = float(g["signed_size"].sum())
        traded = float(g["size"].sum())
        direction = ("buy" if net > 0.1 * traded
                     else "sell" if net < -0.1 * traded else "mixed")
        daily = [float(g.loc[g["day"] == d, "signed_size"].sum()) for d in days]
        rows.append({
            "expiry": pd.to_datetime(exp, errors="coerce"),
            "strike": float(strike), "type": cp, "days": len(days),
            "first_day": days[0], "last_day": days[-1],
            "premium": float(g["premium"].sum()), "net_size": net,
            "direction": direction, "daily": daily,
        })
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows).sort_values(
        ["days", "premium"], ascending=[False, False]
    ).head(top_n).reset_index(drop=True)
    return df


def data_quality(a: "Analysis", prints: pd.DataFrame | None = None) -> dict:
    """Honest read on how much to trust today's levels. Returns
    {'level': high|medium|low, 'notes': [...], ...}. Thin chains, light OI,
    and a high undirected-print ratio all soften the signal."""
    order = {"high": 2, "medium": 1, "low": 0}
    level, notes = "high", []

    def cap(l: str) -> None:
        nonlocal level
        if order[l] < order[level]:
            level = l

    total_oi = float(a.by_strike[["call_oi", "put_oi"]].to_numpy().sum())
    if a.n_contracts < 25:
        notes.append(f"Only {a.n_contracts} contracts — levels are sparse and jumpy.")
        cap("low")
    elif a.n_contracts < 80:
        notes.append(f"{a.n_contracts} contracts — moderate coverage.")
        cap("medium")
    if total_oi and total_oi < 20000:
        notes.append("Light total open interest — walls may not hold.")
        cap("medium")

    signed_ratio = None
    if prints is not None and len(prints) and "signed_size" in prints:
        undirected = float((prints["signed_size"] == 0).mean())
        signed_ratio = 1 - undirected
        if a.weight_mode == "flow" and undirected > 0.5:
            notes.append(f"{undirected:.0%} of prints are mid/undirected — "
                         "signed-flow direction is soft.")
            cap("medium")

    return {"level": level, "notes": notes, "n_contracts": a.n_contracts,
            "total_oi": total_oi, "signed_ratio": signed_ratio}


def darkpool_levels(dark: pd.DataFrame, spot: float, top_n: int = 5) -> pd.DataFrame:
    """Price levels where dark-pool (equity block) size concentrated —
    institutional interest zones that act as support/resistance. Kernel
    density of traded size over price, weighted by dollar value.

    Columns: level, strength (0-100), premium, size, distance_pct.
    """
    cols = ["level", "strength", "premium", "size", "distance_pct"]
    if dark is None or dark.empty:
        return pd.DataFrame(columns=cols)
    d = dark[dark["price"].between(spot * 0.90, spot * 1.10)].copy()
    if d.empty:
        return pd.DataFrame(columns=cols)
    # bin to a cent-ish grid, weight by premium (falls back to size)
    per = d.groupby(d["price"].round(2)).agg(
        premium=("premium", "sum"), size=("size", "sum")).reset_index()
    px = per["price"].to_numpy()
    w = per["premium"].to_numpy(dtype=float)
    if w.sum() <= 0:
        w = per["size"].to_numpy(dtype=float)
    bw = max(spot * 0.001, 0.05)
    lo, hi = spot * 0.90, spot * 1.10
    grid = np.arange(lo, hi, bw)
    dens = np.array([_kernel_density(px, w, x, bw) for x in grid])
    if dens.max() <= 0:
        return pd.DataFrame(columns=cols)

    floor = 0.05 * dens.max()
    rows = []
    for i in range(1, len(grid) - 1):
        if dens[i] > floor and dens[i] > dens[i - 1] and dens[i] >= dens[i + 1]:
            level = _golden_max(lambda x: _kernel_density(px, w, x, bw),
                                grid[i] - bw * 3, grid[i] + bw * 3)
            near = per[np.abs(per["price"] - level) <= bw * 3]
            rows.append((level, _kernel_density(px, w, level, bw),
                         float(near["premium"].sum()), float(near["size"].sum()),
                         (level / spot - 1) * 100))
    if not rows:
        return pd.DataFrame(columns=cols)
    df = pd.DataFrame(rows, columns=cols)
    df = df.sort_values("strength", ascending=False)
    kept: list[int] = []
    for idx, row in df.iterrows():
        if all(abs(row["level"] - df.loc[k, "level"]) > bw * 3 for k in kept):
            kept.append(idx)
    df = df.loc[kept].head(top_n).reset_index(drop=True)
    df["strength"] = (100 * df["strength"] / df["strength"].max()).round(0)
    return df


def _confluence_role(families: set, center: float, spot: float) -> str:
    if "flip" in families:
        return "Regime pivot"
    base = "Resistance" if center > spot else "Support"
    if "accel" in families and "magnet" not in families:
        return f"{base} · accelerator"
    if "magnet" in families or "max_pain" in families:
        return f"{base} · pin"
    return base


def confluence_levels(a: "Analysis", magnets: pd.DataFrame | None = None,
                      oi_lvls: pd.DataFrame | None = None,
                      block_lvls: pd.DataFrame | None = None,
                      dark_lvls: pd.DataFrame | None = None,
                      band: float | None = None, top_n: int = 6,
                      weights: dict | None = None) -> pd.DataFrame:
    """Fuse every level system into one ranked master table. Each source
    contributes a weighted vote to a price; nearby votes cluster, and the
    cluster's total weight becomes a 0-100 confluence score. A level
    confirmed by many independent layers is the one to trade.

    ``weights`` optionally multiplies each layer family's base weight — used
    to feed empirical hit-rate back into the scoring (see tuned_layer_weights).

    Columns: level, score, n_layers, layers (list), families (list), role,
    distance_pct, confidence (high/medium/low — from the count of distinct
    layer families).
    """
    weights = weights or {}
    if magnets is None:
        magnets = magnet_levels(a)
    if oi_lvls is None:
        oi_lvls = oi_levels(a)
    ow = oi_walls(a)
    ks = a.by_strike["strike"].to_numpy()
    spacing = float(np.median(np.diff(np.sort(np.unique(ks))))) if len(ks) > 1 else a.spot * 0.005
    if band is None:
        band = max(spacing * 0.5, a.spot * 0.0015)

    # (price, weight, label, family)
    src: list[tuple] = [
        (a.call_wall, 1.0, "Gamma call wall", "gamma_wall"),
        (a.put_wall, 1.0, "Gamma put wall", "gamma_wall"),
        (a.max_pain, 0.6, "Max pain", "max_pain"),
    ]
    if a.gamma_flip is not None:
        src.append((a.gamma_flip, 0.9, "Gamma flip", "flip"))
    if ow.call is not None:
        src.append((ow.call, 1.0, "Call OI wall", "oi_wall"))
    if ow.put is not None:
        src.append((ow.put, 1.0, "Put OI wall", "oi_wall"))
    for _, r in magnets.iterrows():
        if r["kind"] == "magnet":
            src.append((r["level"], r["strength"] / 100, "Magnet", "magnet"))
        else:
            src.append((r["level"], 0.8 * r["strength"] / 100, "Accelerator", "accel"))
    for _, r in oi_lvls.iterrows():
        src.append((r["level"], 0.8 * r["strength"] / 100,
                    f"OI cluster ({r['side']})", "oi_cluster"))
    if block_lvls is not None:
        for _, r in block_lvls.iterrows():
            src.append((r["level"], 1.2 * r["strength"] / 100,
                        f"Block {r['direction']}", "block"))
    if dark_lvls is not None:
        for _, r in dark_lvls.iterrows():
            src.append((r["level"], 0.8 * r["strength"] / 100,
                        "Dark pool", "dark_pool"))

    # apply per-family weight multipliers (empirical tuning), then filter
    src = [(p, w * weights.get(fam, 1.0), lbl, fam) for (p, w, lbl, fam) in src]
    src = [s for s in src if s[0] is not None and abs(s[0] / a.spot - 1) <= 0.12 and s[1] > 0]
    cols = ["level", "score", "n_layers", "layers", "families", "role",
            "distance_pct", "confidence"]
    if not src:
        return pd.DataFrame(columns=cols)

    src.sort(key=lambda s: s[0])
    clusters, cur = [], [src[0]]
    for s in src[1:]:
        if s[0] - cur[-1][0] <= band:
            cur.append(s)
        else:
            clusters.append(cur)
            cur = [s]
    clusters.append(cur)

    rows = []
    for cl in clusters:
        w = sum(x[1] for x in cl)
        center = sum(x[0] * x[1] for x in cl) / w
        fams = {x[3] for x in cl}
        seen, labels = set(), []
        for x in sorted(cl, key=lambda y: -y[1]):
            if x[2] not in seen:
                seen.add(x[2])
                labels.append(x[2])
        rows.append((center, w, len(fams), labels, sorted(fams),
                     _confluence_role(fams, center, a.spot),
                     (center / a.spot - 1) * 100))
    df = pd.DataFrame(rows, columns=["level", "weight", "n_layers", "layers",
                                     "families", "role", "distance_pct"])
    df = df.sort_values("weight", ascending=False).head(top_n).reset_index(drop=True)
    df["score"] = (100 * df["weight"] / df["weight"].max()).round(0)
    df["confidence"] = df["n_layers"].map(
        lambda n: "high" if n >= 3 else "medium" if n == 2 else "low")
    return df[cols]


_FAMILY_LABEL = {
    "gamma_wall": "gamma walls", "flip": "gamma flip", "oi_wall": "OI walls",
    "magnet": "magnets", "accel": "accelerators", "oi_cluster": "OI clusters",
    "block": "block levels", "dark_pool": "dark-pool", "max_pain": "max pain",
}


def tuned_layer_weights(detail: pd.DataFrame, min_n: int = 4) -> dict:
    """Turn historical hit-rate into per-family confluence weight multipliers.

    Among tested levels, each layer family's held-rate becomes a multiplier
    in [0.5, 1.5] (0.5 + rate): families whose levels held more get more say
    in future scoring, and vice versa. Families with fewer than ``min_n``
    tested instances keep their default weight (1.0) — no tuning on noise.

    Returns {family: multiplier} (only families that met the sample gate).
    """
    if detail is None or detail.empty or "families" not in detail:
        return {}
    tested = detail[detail["tested"]]
    if tested.empty:
        return {}
    tally: dict[str, list[int]] = {}
    for _, r in tested.iterrows():
        for fam in (r["families"] or []):
            t = tally.setdefault(fam, [0, 0])
            t[0] += 1
            t[1] += int(bool(r["held"]))
    out = {}
    for fam, (n, held) in tally.items():
        if n >= min_n:
            out[fam] = round(float(np.clip(0.5 + held / n, 0.5, 1.5)), 3)
    return out


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
    if weight == "flow":
        if "net_customer_size" not in chain.columns:
            raise ValueError(
                "Signed-flow weighting needs trade-level data with side codes "
                "(e.g. a QuantData order-flow export)."
            )
        weight_col = "net_customer_size"
    else:
        weight_col = weight
    df = _fill_gamma(chain, spot, asof, rate, weight_col=weight_col)

    strikes = gex_by_strike(df, spot, multiplier)
    curve = gex_curve(df, spot, rate, multiplier=multiplier)
    total = total_gex_at(df, spot, rate, multiplier)
    flips = flip_levels(df, curve, spot, rate)  # crossings are scale-invariant
    flip = flips[0] if flips else None

    uniq = np.sort(strikes["strike"].unique())
    spacing = float(np.median(np.diff(uniq))) if len(uniq) > 1 else spot * 0.01

    if weight == "flow":
        # signed flow has no fixed call/put polarity: walls are the peaks of
        # positive (pin/resistance) and negative (acceleration) net dealer gamma
        pos = strikes[strikes["net_gex"] > 0]
        neg = strikes[strikes["net_gex"] < 0]
        call_seed = float(pos.loc[pos["net_gex"].idxmax(), "strike"]) if not pos.empty else spot
        put_seed = float(neg.loc[neg["net_gex"].idxmin(), "strike"]) if not neg.empty else spot
        call_wall = wall_level(strikes, call_seed, spacing, "POS") if not pos.empty else spot
        put_wall = wall_level(strikes, put_seed, spacing, "NEG") if not neg.empty else spot
    else:
        pos = strikes[strikes["call_gex"] > 0]
        neg = strikes[strikes["put_gex"] < 0]
        call_seed = float(pos.loc[pos["call_gex"].idxmax(), "strike"]) if not pos.empty else spot
        put_seed = float(neg.loc[neg["put_gex"].idxmin(), "strike"]) if not neg.empty else spot
        call_wall = wall_level(strikes, call_seed, spacing, "C") if not pos.empty else spot
        put_wall = wall_level(strikes, put_seed, spacing, "P") if not neg.empty else spot

    dex, vanna_flow, charm_flow = hedge_flows(df, spot, rate, multiplier)
    em, nearest = expected_move(df, spot)

    df = df.assign(net_gex=_gsign(df) * _dollar_gex(df["gamma"], _weights(df), spot, multiplier))
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
        flip_levels=flips,
    )


def fmt_dollars(x: float) -> str:
    """$1.23B-style formatting for GEX magnitudes."""
    sign = "-" if x < 0 else ""
    x = abs(x)
    for div, suffix in ((1e12, "T"), (1e9, "B"), (1e6, "M"), (1e3, "K")):
        if x >= div:
            return f"{sign}${x / div:.2f}{suffix}"
    return f"{sign}${x:.0f}"
