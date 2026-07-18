"""Magnitude detectors: pure pandas/numpy functions over a canonical chain.

Every function takes DataFrames with canonical column names (see schema.py)
and returns a DataFrame (empty on insufficient input) — no Streamlit here.
The UI layer decides what to say about empty results.
"""

import numpy as np
import pandas as pd

# Canonical columns each detector needs (checked by the UI for messaging).
IMPLIED_MOVE_REQUIRED = ['type', 'strike', 'expiration']
UNUSUAL_REQUIRED = ['volume']
PRICE_MOVES_REQUIRED = ['close']  # on the price-history frame
IV_SNAPSHOT_REQUIRED = ['implied_volatility', 'strike', 'expiration']


def _straddle_mids(chain):
    """Per (expiration, strike): call_mid and put_mid columns, both > 0."""
    if 'mid' not in chain:
        return pd.DataFrame()
    sides = (chain
             .dropna(subset=['type', 'strike', 'expiration', 'mid'])
             .pivot_table(index=['expiration', 'strike'], columns='type',
                          values='mid', aggfunc='mean')
             .reset_index())
    if 'call' not in sides or 'put' not in sides:
        return pd.DataFrame()
    sides = sides[(sides['call'] > 0) & (sides['put'] > 0)]
    return sides.rename(columns={'call': 'call_mid', 'put': 'put_mid'})


def infer_spot(chain):
    """Infer the underlying price. Returns (spot, method) or (None, None)."""
    if 'underlying_price' in chain and chain['underlying_price'].notna().any():
        return float(chain['underlying_price'].median()), 'underlying price column'

    sides = _straddle_mids(chain)
    if not sides.empty:
        nearest = sides[sides['expiration'] == sides['expiration'].min()]
        parity = (nearest['call_mid'] - nearest['put_mid'] + nearest['strike'])
        if len(parity) >= 3:
            return float(parity.median()), 'put-call parity'
        row = nearest.loc[(nearest['call_mid'] - nearest['put_mid']).abs().idxmin()]
        return float(row['strike']), 'strike with call mid ≈ put mid'

    return None, None


def implied_move(chain, spot):
    """Expected move per expiry from ATM straddles and/or ATM IV.

    Returns one row per expiry: dte, atm_strike, straddle price, move_$ and
    move_% for each available method, and the implied price range.
    """
    if spot is None or not spot > 0:
        return pd.DataFrame()
    sides = _straddle_mids(chain)
    if sides.empty:
        return pd.DataFrame()

    has_iv = ('implied_volatility' in chain
              and chain['implied_volatility'].notna().any())
    as_of = chain.attrs.get('as_of')
    as_of = (pd.Timestamp(as_of) if as_of is not None
             else pd.Timestamp.today().normalize())

    rows = []
    for expiration, grp in sides.groupby('expiration'):
        atm = grp.loc[(grp['strike'] - spot).abs().idxmin()]
        straddle = atm['call_mid'] + atm['put_mid']
        dte = max((expiration - as_of).days, 1)
        row = {
            'expiration': expiration,
            'dte': dte,
            'atm_strike': atm['strike'],
            'straddle': straddle,
            'straddle_move': 0.85 * straddle,
            'iv_move': np.nan,
            'atm_iv': np.nan,
        }
        if has_iv:
            atm_iv = chain.loc[
                (chain['expiration'] == expiration)
                & (chain['strike'] == atm['strike']),
                'implied_volatility'].mean()
            if pd.notna(atm_iv) and atm_iv > 0:
                row['atm_iv'] = atm_iv
                row['iv_move'] = spot * atm_iv * np.sqrt(dte / 365.0)
        rows.append(row)

    out = pd.DataFrame(rows).sort_values('expiration').reset_index(drop=True)
    out['move'] = out['straddle_move'].fillna(out['iv_move'])
    out['move_pct'] = out['move'] / spot
    out['range_low'] = spot - out['move']
    out['range_high'] = spot + out['move']
    return out


def unusual_activity(chain, min_volume=100, ratio_thr=2.0, premium_floor=100_000):
    """Score and flag contracts with outsized volume vs open interest.

    Returns contracts with volume >= min_volume, ranked by a composite
    percentile score; `flagged` marks rows where vol/OI exceeds ratio_thr
    and the traded premium clears premium_floor.
    """
    if 'volume' not in chain or chain['volume'].isna().all():
        return pd.DataFrame()
    df = chain[chain['volume'].fillna(0) >= min_volume].copy()
    if df.empty:
        return df

    has_oi = 'open_interest' in df and df['open_interest'].notna().any()
    has_mid = 'mid' in df and df['mid'].notna().any()

    df['vol_oi'] = (df['volume'] / df['open_interest'].clip(lower=1)
                    if has_oi else np.nan)
    df['premium'] = df['volume'] * df['mid'] * 100 if has_mid else np.nan

    parts = []
    if has_oi:
        parts.append((0.6, df['vol_oi'].rank(pct=True)))
    if has_mid:
        parts.append((0.4, df['premium'].rank(pct=True)))
    if not parts:
        return pd.DataFrame()
    total_w = sum(w for w, _ in parts)
    df['score'] = 100 * sum(w * r for w, r in parts) / total_w

    # Volume outsized vs OI is the signal; premium is a size filter on top
    # of it (big premium alone is normal in liquid names). With no OI the
    # premium floor is all we have.
    if has_oi and has_mid:
        df['flagged'] = (df['vol_oi'] >= ratio_thr) & (df['premium'] >= premium_floor)
    elif has_oi:
        df['flagged'] = df['vol_oi'] >= ratio_thr
    else:
        df['flagged'] = df['premium'] >= premium_floor

    keep = [c for c in ['type', 'strike', 'expiration', 'dte', 'mid', 'volume',
                        'open_interest', 'vol_oi', 'premium', 'score', 'flagged']
            if c in df]
    return df[keep].sort_values('score', ascending=False).reset_index(drop=True)


def big_price_moves(prices, window=20, z_thr=2.5, burst_ratio=1.5):
    """Flag outsized daily moves in a price series.

    prices: DataFrame with 'date' and 'close'. Returns the series with
    log return, trailing sigma (excluding the current day), z-score,
    vol-burst ratio, and boolean flags.
    """
    if prices is None or 'close' not in prices:
        return pd.DataFrame()
    df = prices.dropna(subset=['close']).sort_values('date').copy()
    if len(df) < window + 5:
        return pd.DataFrame()

    df['log_ret'] = np.log(df['close']).diff()
    df['sigma'] = df['log_ret'].rolling(window).std().shift(1)
    df['z'] = df['log_ret'] / df['sigma']
    df['big_move'] = df['z'].abs() >= z_thr

    rv_short = df['log_ret'].rolling(5).std()
    rv_long = df['log_ret'].rolling(window).std()
    df['vol_burst_ratio'] = rv_short / rv_long
    df['vol_burst'] = df['vol_burst_ratio'] >= burst_ratio

    return df.reset_index(drop=True)


def iv_snapshot_outliers(chain, spot, moneyness=0.20, z_thr=3.0,
                         include_crushes=False):
    """Robust-z IV outliers within each expiry, near the money.

    IV varies structurally with strike (skew/smile), so raw deviation from
    the expiry median would flag the whole skew. Instead each expiry gets a
    quadratic fit of IV vs log-moneyness and the robust z is computed on
    the residuals. Returns contracts with iv_fit, robust_z and flagged.
    """
    if ('implied_volatility' not in chain
            or chain['implied_volatility'].isna().all()
            or spot is None or not spot > 0):
        return pd.DataFrame()
    df = chain.dropna(subset=['implied_volatility', 'strike', 'expiration']).copy()
    df = df[(df['strike'] / spot - 1).abs() <= moneyness]
    if df.empty:
        return df

    df['_m'] = np.log(df['strike'] / spot)

    def smile_residual(grp):
        if len(grp) < 5:
            return pd.Series(grp['implied_volatility'].median(), index=grp.index)
        # Two-pass fit: gross outliers drag the first fit, so refit without
        # them before scoring residuals.
        m, iv = grp['_m'].to_numpy(), grp['implied_volatility'].to_numpy()
        fit = np.polyval(np.polyfit(m, iv, 2), m)
        resid = iv - fit
        mad = np.median(np.abs(resid - np.median(resid)))
        inlier = np.abs(resid - np.median(resid)) <= 5 * max(mad, 1e-6) / 0.6745
        if inlier.sum() >= 5 and not inlier.all():
            fit = np.polyval(np.polyfit(m[inlier], iv[inlier], 2), m)
        return pd.Series(fit, index=grp.index)

    df['iv_fit'] = (df.groupby('expiration', group_keys=False)
                    .apply(smile_residual))
    resid = df['implied_volatility'] - df['iv_fit']
    by_expiry = resid.groupby(df['expiration'])
    mad = by_expiry.transform(lambda s: (s - s.median()).abs().median())
    df['robust_z'] = (0.6745 * (resid - by_expiry.transform('median'))
                      / mad.clip(lower=1e-6))
    df = df.drop(columns=['_m'])
    df['flagged'] = (df['robust_z'].abs() >= z_thr if include_crushes
                     else df['robust_z'] >= z_thr)

    keep = [c for c in ['type', 'strike', 'expiration', 'dte', 'mid', 'volume',
                        'implied_volatility', 'iv_fit', 'robust_z', 'flagged']
            if c in df]
    return df[keep].sort_values('robust_z', ascending=False).reset_index(drop=True)


def iv_term_structure(chain, spot):
    """ATM IV per expiry, for the term-structure chart."""
    moves = implied_move(chain, spot)
    if moves.empty or moves['atm_iv'].isna().all():
        return pd.DataFrame()
    return moves[['expiration', 'dte', 'atm_iv']].dropna(subset=['atm_iv'])


def iv_time_series_spikes(iv_series, z_thr=2.5, abs_thr=0.05):
    """Flag one-day IV jumps in a time series.

    iv_series: DataFrame with 'date' and 'iv'. Returns the series with the
    one-day change, its z-score and flags.
    """
    if iv_series is None or 'iv' not in iv_series:
        return pd.DataFrame()
    df = iv_series.dropna(subset=['iv']).sort_values('date').copy()
    if len(df) < 10:
        return pd.DataFrame()

    df['iv_change'] = df['iv'].diff()
    win = min(20, len(df) // 2)
    df['z'] = df['iv_change'] / df['iv_change'].rolling(win).std().shift(1)
    df['flagged'] = (df['z'].abs() >= z_thr) | (df['iv_change'].abs() >= abs_thr)
    return df.reset_index(drop=True)


def iv_series_from_chain(chain, moneyness=0.05):
    """Build a near-ATM mean-IV time series from a multi-date chain.

    Needs timestamp, underlying_price and implied_volatility; returns a
    DataFrame with 'date' and 'iv' (empty if fewer than 10 timestamps).
    """
    needed = ['timestamp', 'underlying_price', 'implied_volatility']
    if any(c not in chain or chain[c].isna().all() for c in needed):
        return pd.DataFrame()
    df = chain.dropna(subset=needed).copy()
    df = df[(df['strike'] / df['underlying_price'] - 1).abs() <= moneyness]
    out = (df.groupby('timestamp')['implied_volatility'].mean()
           .rename('iv').reset_index().rename(columns={'timestamp': 'date'}))
    return out if len(out) >= 10 else pd.DataFrame()
