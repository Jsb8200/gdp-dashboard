"""Pre-move confluence features.

Every feature uses only information available at bar t (backward-looking),
so the model can be scored live on the most recent closed bar.
"""

import numpy as np
import pandas as pd

from .config import Config


def atr(df: pd.DataFrame, period: int) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - prev_close).abs(),
            (df["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1.0 / period, min_periods=period).mean()


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1.0 / period, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1.0 / period, min_periods=period).mean()
    rs = gain / loss.replace(0.0, np.nan)
    return 100 - 100 / (1 + rs)


def _adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    tr = atr(df, period)
    plus_di = 100 * plus_dm.ewm(alpha=1.0 / period, min_periods=period).mean() / tr
    minus_di = 100 * minus_dm.ewm(alpha=1.0 / period, min_periods=period).mean() / tr
    denom = (plus_di + minus_di).replace(0.0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / denom
    return dx.ewm(alpha=1.0 / period, min_periods=period).mean()


def _rolling_pct_rank(s: pd.Series, window: int) -> pd.Series:
    """Percentile rank of the latest value within its trailing window."""
    return s.rolling(window, min_periods=window // 2).rank(pct=True)


def build_features(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Return a feature frame aligned to df.index."""
    f = pd.DataFrame(index=df.index)
    close, high, low, volume = df["close"], df["high"], df["low"], df["volume"]

    atr_s = atr(df, cfg.atr_period)
    ret = close.pct_change()

    # --- Volatility compression / squeeze ----------------------------------
    ma = close.rolling(cfg.bb_period).mean()
    sd = close.rolling(cfg.bb_period).std()
    bb_width = (4 * sd) / ma  # (upper-lower)/mid with 2-sigma bands
    f["bb_width"] = bb_width
    f["bb_width_pctile"] = _rolling_pct_rank(bb_width, cfg.squeeze_lookback)

    atr_fast = atr(df, 5)
    f["atr_ratio_fast_slow"] = atr_fast / atr_s
    f["atr_pctile"] = _rolling_pct_rank(atr_s / close, cfg.squeeze_lookback)

    # Keltner squeeze: Bollinger bands inside Keltner channel.
    keltner_half = 1.5 * atr_s
    f["squeeze_on"] = ((2 * sd) < keltner_half).astype(float)
    f["squeeze_bars"] = (
        f["squeeze_on"].groupby((f["squeeze_on"] != f["squeeze_on"].shift()).cumsum())
        .cumsum()
    )

    # Range contraction: today's range vs trailing mean range (NR-style).
    bar_range = (high - low)
    f["range_vs_mean"] = bar_range / bar_range.rolling(cfg.vol_lookback).mean()
    f["realized_vol"] = ret.rolling(cfg.vol_lookback).std()
    f["vol_slope"] = f["realized_vol"].pct_change(5)

    # --- Trend / EMA confluence --------------------------------------------
    emas = {p: close.ewm(span=p, min_periods=p).mean() for p in cfg.ema_periods}
    periods = sorted(cfg.ema_periods)
    # Fraction of adjacent EMA pairs correctly stacked (either direction).
    stack_up = sum(
        (emas[a] > emas[b]).astype(float) for a, b in zip(periods, periods[1:])
    ) / (len(periods) - 1)
    f["ema_stack"] = (stack_up - 0.5).abs() * 2  # 1 = fully aligned either way
    f["dist_ema_slow"] = (close - emas[periods[-1]]) / atr_s
    f["adx"] = _adx(df)
    f["rsi"] = _rsi(close)
    macd = close.ewm(span=12).mean() - close.ewm(span=26).mean()
    f["macd_hist"] = (macd - macd.ewm(span=9).mean()) / atr_s

    # --- Range position / zone behaviour -----------------------------------
    dc_high = high.rolling(cfg.donchian_period).max()
    dc_low = low.rolling(cfg.donchian_period).min()
    dc_range = (dc_high - dc_low).replace(0.0, np.nan)
    f["donchian_pos"] = (close - dc_low) / dc_range
    f["donchian_width"] = dc_range / atr_s
    f["dist_to_high"] = (dc_high - close) / atr_s
    f["dist_to_low"] = (close - dc_low) / atr_s

    # How many times the zone edges were tested recently (touch = within
    # 0.25 ATR of the rolling extreme). More tests -> weaker level.
    f["tests_high"] = ((dc_high - high) < 0.25 * atr_s).rolling(cfg.donchian_period).sum()
    f["tests_low"] = ((low - dc_low) < 0.25 * atr_s).rolling(cfg.donchian_period).sum()

    # --- Volume -------------------------------------------------------------
    vol_ma = volume.rolling(cfg.vol_lookback).mean()
    f["volume_z"] = (volume - vol_ma) / volume.rolling(cfg.vol_lookback).std()
    f["volume_dryup"] = volume.rolling(5).mean() / vol_ma
    obv = (np.sign(close.diff()).fillna(0.0) * volume).cumsum()
    f["obv_slope"] = obv.diff(10) / vol_ma.replace(0.0, np.nan) / 10

    # --- Candle structure ----------------------------------------------------
    body = (close - df["open"]).abs()
    rng_safe = bar_range.replace(0.0, np.nan)
    f["body_ratio_mean"] = (body / rng_safe).rolling(10).mean()
    inside = ((high < high.shift(1)) & (low > low.shift(1))).astype(float)
    f["inside_bar_streak"] = inside.groupby((inside != inside.shift()).cumsum()).cumsum()

    # --- Composite confluence score ------------------------------------------
    signals = pd.concat(
        [
            (f["bb_width_pctile"] < 0.2),           # vol compressed
            (f["squeeze_on"] > 0),                   # squeeze active
            (f["volume_dryup"] < 0.8),               # volume drying up
            (f["range_vs_mean"] < 0.7),              # contracting ranges
            (f["ema_stack"] > 0.9),                  # EMAs aligned
            (f["donchian_width"] < f["donchian_width"].rolling(
                cfg.squeeze_lookback, min_periods=cfg.donchian_period
            ).median()),                             # tight box
            (f[["tests_high", "tests_low"]].max(axis=1) >= 3),  # tested level
        ],
        axis=1,
    )
    f["confluence_score"] = signals.astype(float).sum(axis=1)

    return f
