"""Forward-looking magnitude labels.

For each bar t we measure the maximum excursion (in either direction) that
price achieves over the next `horizon` bars, normalized by ATR at t:

    fwd_mag[t] = max( max(high[t+1..t+H]) - close[t],
                      close[t] - min(low[t+1..t+H]) ) / ATR[t]

The binary label is 1 when fwd_mag >= magnitude_threshold, i.e. a big move
starts right after bar t. Labels use only future bars; features use only
past bars — the split point is the close of bar t.
"""

import numpy as np
import pandas as pd

from .config import Config
from .features import atr


def build_labels(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    h = cfg.horizon
    atr_s = atr(df, cfg.atr_period)

    # Forward-looking extremes over (t+1 .. t+H]: reverse, roll, reverse.
    fwd_high = (
        df["high"][::-1].rolling(h, min_periods=h).max()[::-1].shift(-1)
    )
    fwd_low = (
        df["low"][::-1].rolling(h, min_periods=h).min()[::-1].shift(-1)
    )

    up_excursion = fwd_high - df["close"]
    down_excursion = df["close"] - fwd_low
    fwd_mag = np.maximum(up_excursion, down_excursion) / atr_s

    out = pd.DataFrame(index=df.index)
    out["fwd_mag"] = fwd_mag
    out["big_move"] = (fwd_mag >= cfg.magnitude_threshold).astype(float)
    # Direction of the dominant excursion (context only, not the main target).
    out["fwd_direction"] = np.sign(up_excursion - down_excursion)

    # Last `horizon` bars have no complete future window.
    out.iloc[-h:] = np.nan
    return out
