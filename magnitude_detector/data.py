"""Data loading and synthetic OHLCV generation."""

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["open", "high", "low", "close", "volume"]


def load_ohlcv(path: str) -> pd.DataFrame:
    """Load an OHLCV CSV.

    Expects columns open/high/low/close/volume (case-insensitive) and
    optionally a timestamp/date/time column which becomes the index.
    """
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]

    for ts_col in ("timestamp", "datetime", "date", "time"):
        if ts_col in df.columns:
            df[ts_col] = pd.to_datetime(df[ts_col])
            df = df.set_index(ts_col)
            break

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    df = df[REQUIRED_COLUMNS].astype(float).sort_index()
    if df[["open", "high", "low", "close"]].isna().any().any():
        raise ValueError("OHLC columns contain NaNs")
    return df


def make_synthetic_ohlcv(n_bars: int = 20000, seed: int = 7) -> pd.DataFrame:
    """Regime-switching synthetic series for demos and tests.

    Alternates quiet consolidation regimes with breakout regimes. Breakouts
    tend to follow volatility compression and volume dry-up, so the
    "compression precedes expansion" structure the detector looks for is
    actually present and learnable.
    """
    rng = np.random.default_rng(seed)

    log_price = np.log(100.0)
    prices, vols_used, volumes = [], [], []

    base_vol = 0.004
    i = 0
    while i < n_bars:
        # Consolidation phase: volatility decays toward a floor.
        quiet_len = int(rng.integers(40, 160))
        decay = np.linspace(1.0, rng.uniform(0.3, 0.6), quiet_len)
        quiet_vol = base_vol * decay
        # Volume dries up alongside volatility.
        quiet_volu = rng.lognormal(mean=0.0, sigma=0.3, size=quiet_len) * decay

        for v, vu in zip(quiet_vol, quiet_volu):
            log_price += rng.normal(0.0, v)
            prices.append(log_price)
            vols_used.append(v)
            volumes.append(vu)
            i += 1
            if i >= n_bars:
                break

        if i >= n_bars:
            break

        # With some probability the compression resolves into a breakout.
        if rng.random() < 0.65:
            burst_len = int(rng.integers(10, 40))
            direction = rng.choice([-1.0, 1.0])
            drift = direction * rng.uniform(1.0, 2.5) * base_vol
            burst_vol = base_vol * rng.uniform(1.8, 3.5)
            for _ in range(burst_len):
                log_price += drift + rng.normal(0.0, burst_vol)
                prices.append(log_price)
                vols_used.append(burst_vol)
                volumes.append(rng.lognormal(mean=0.8, sigma=0.4))
                i += 1
                if i >= n_bars:
                    break

    close = np.exp(np.array(prices[:n_bars]))
    vols_used = np.array(vols_used[:n_bars])
    volumes = np.array(volumes[:n_bars]) * 1e6

    # Build OHLC around the close path.
    open_ = np.empty_like(close)
    open_[0] = close[0]
    open_[1:] = close[:-1]
    spread = np.abs(rng.normal(0.0, vols_used)) * close
    high = np.maximum(open_, close) + spread
    low = np.minimum(open_, close) - spread

    idx = pd.date_range("2020-01-01", periods=n_bars, freq="1h")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close,
         "volume": volumes},
        index=idx,
    )
