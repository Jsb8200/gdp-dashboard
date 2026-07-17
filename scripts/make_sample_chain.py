"""Generate data/sample_option_chain.csv — a synthetic SPY-like chain.

Deterministic (seeded); rerun after changing parameters:
    python scripts/make_sample_chain.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

SPOT = 628.50
QUOTE_DATE = "2026-07-17"
EXPIRIES = ["2026-07-24", "2026-07-31", "2026-08-21", "2026-09-18"]
EXPIRY_WEIGHT = {  # near-dated expiries carry most open interest
    "2026-07-24": 1.0, "2026-07-31": 0.8, "2026-08-21": 0.55, "2026-09-18": 0.35,
}
STRIKES = np.arange(540, 705, 5)

rng = np.random.default_rng(42)
rows = []
for expiry in EXPIRIES:
    w = EXPIRY_WEIGHT[expiry]
    t = (pd.Timestamp(expiry) - pd.Timestamp(QUOTE_DATE)).days / 365.0
    for k in STRIKES:
        moneyness = np.log(k / SPOT)
        # put-skewed smile, flattening with maturity
        iv = 0.155 + max(0.0, -moneyness) * 0.55 / (1 + 4 * t) + max(0.0, moneyness) * 0.10
        # OI concentrated near the money and at round strikes
        atm = np.exp(-((k - SPOT) / 40.0) ** 2)
        round_boost = 2.2 if k % 25 == 0 else (1.5 if k % 10 == 0 else 1.0)
        base = 9000 * w * atm * round_boost
        # customers hold downside-protection puts and at/above-spot calls,
        # with hedging flows stacked at the big round strikes (600 / 650)
        call_oi = base * (1.9 if k >= SPOT - 5 else 0.5) * (3.0 if k == 650 else 1.0)
        put_oi = (
            base
            * (2.4 if k <= SPOT * 0.97 else (0.8 if k <= SPOT else 0.4))
            * (3.5 if k == 600 else 1.0)
        )
        for typ, oi in (("C", call_oi), ("P", put_oi)):
            oi = int(oi * rng.uniform(0.8, 1.2))
            if oi < 25:
                continue
            rows.append({
                "Expiration Date": expiry,
                "Type": "Call" if typ == "C" else "Put",
                "Strike": k,
                "Open Interest": oi,
                "Volume": int(oi * rng.uniform(0.05, 0.45)),
                "Implied Volatility": round(iv * 100, 2),  # percent, like broker exports
                "Underlying Price": SPOT,
            })

df = pd.DataFrame(rows)
out = Path(__file__).resolve().parent.parent / "data" / "sample_option_chain.csv"
df.to_csv(out, index=False)
print(f"wrote {out} ({len(df)} rows)")
