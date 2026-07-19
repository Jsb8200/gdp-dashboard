"""Generate the bundled sample CSVs (deterministic; safe to re-run).

Writes data/sample_options_chain.csv and data/sample_price_history.csv.
The chain uses alias headers on purpose so it exercises the column mapper.
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent.parent / 'data'

QUOTE_DATE = pd.Timestamp('2026-07-17')  # fixed so committed CSVs stay stable
SPOT = 187.50
RATE = 0.04
EXPIRY_DAYS = [7, 14, 45, 90]
STRIKES = np.arange(150.0, 225.1, 2.5)

# (expiry_days, strike, extra_iv) — planted IV outliers for the detector demo
IV_OUTLIERS = [(14, 175.0, 0.20), (45, 200.0, 0.25), (90, 182.5, 0.15)]
# (expiry_days, strike, type, volume, oi) — planted unusual-activity contracts
UNUSUAL = [(7, 190.0, 'C', 12500, 1600), (14, 180.0, 'P', 9800, 2100),
           (45, 195.0, 'C', 15200, 4900), (45, 170.0, 'P', 7400, 1200)]


def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_price(spot, strike, t, sigma, rate, is_call):
    if t <= 0 or sigma <= 0:
        intrinsic = spot - strike if is_call else strike - spot
        return max(intrinsic, 0.0)
    d1 = (math.log(spot / strike) + (rate + sigma ** 2 / 2) * t) / (sigma * math.sqrt(t))
    d2 = d1 - sigma * math.sqrt(t)
    if is_call:
        return spot * norm_cdf(d1) - strike * math.exp(-rate * t) * norm_cdf(d2)
    return strike * math.exp(-rate * t) * norm_cdf(-d2) - spot * norm_cdf(-d1)


def surface_iv(days, strike):
    """Base IV surface: term structure + put skew + smile."""
    term = 0.25 + 0.08 * (days - 7) / 83  # 0.25 at 7d -> 0.33 at 90d
    m = math.log(strike / SPOT)
    skew = -0.25 * m       # higher IV below spot
    smile = 1.2 * m ** 2
    return term + skew + smile


def generate_chain(rng):
    outliers = {(d, k): extra for d, k, extra in IV_OUTLIERS}
    unusual = {(d, k, t): (vol, oi) for d, k, t, vol, oi in UNUSUAL}

    rows = []
    for days in EXPIRY_DAYS:
        expiry = QUOTE_DATE + pd.Timedelta(days=days)
        t = days / 365.0
        for strike in STRIKES:
            for cp in ('C', 'P'):
                iv = surface_iv(days, strike) + rng.normal(0, 0.004)
                iv += outliers.get((days, strike), 0.0)
                iv = max(iv, 0.05)

                fair = bs_price(SPOT, strike, t, iv, RATE, cp == 'C')
                spread = max(0.05, fair * rng.uniform(0.02, 0.06))
                bid = max(fair - spread / 2, 0.0)
                ask = fair + spread / 2
                last = max(fair + rng.normal(0, spread / 4), 0.01)

                if (days, strike, cp) in unusual:
                    volume, oi = unusual[(days, strike, cp)]
                else:
                    # Liquidity concentrated near the money
                    atm_weight = math.exp(-((strike - SPOT) / 12.0) ** 2)
                    oi = int(rng.gamma(2.0, 900 * atm_weight + 40))
                    volume = int(oi * rng.uniform(0.05, 0.6))

                rows.append({
                    'Type': cp,
                    'Strike': strike,
                    'Exp Date': expiry.date().isoformat(),
                    'Bid': round(bid, 2),
                    'Ask': round(ask, 2),
                    'Last': round(last, 2),
                    'Volume': volume,
                    'OI': oi,
                    'IV': round(iv, 4),
                    'Underlying Price': SPOT,
                    'Quote Date': QUOTE_DATE.date().isoformat(),
                })
    return pd.DataFrame(rows)


def generate_price_history(rng):
    n = 120
    dates = pd.bdate_range(end=QUOTE_DATE, periods=n)
    daily_vol = 0.28 / math.sqrt(252)
    rets = rng.normal(0.0002, daily_vol, n)
    jumps = {n - 60: -0.07, n - 35: 0.06, n - 12: -0.05}
    for i, jump in jumps.items():
        rets[i] = jump

    log_price = np.cumsum(rets)
    close = SPOT * np.exp(log_price - log_price[-1])  # ends exactly at SPOT

    iv = 0.25 + np.cumsum(rng.normal(0, 0.004, n))
    iv = np.clip(iv - (iv[-1] - 0.25), 0.12, 0.60)  # ends near 0.25
    iv[n - 25] += 0.09
    iv[n - 8] += 0.07

    return pd.DataFrame({
        'date': dates.date,
        'close': np.round(close, 2),
        'iv': np.round(iv, 4),
    })


def generate_order_flow(rng):
    """Per-trade rows in a QuantData-like export format ($, commas, %).

    Two tickers, several trades per contract, intraday timestamps — used by
    the sanity suite to exercise the order-flow path (symbol filtering,
    trade collapsing, tz-aware timestamps, intraday series derivation).
    """
    rows = []
    for ticker, spot0, n_trades in (('DEMO', 187.50, 260), ('AUX', 52.00, 40)):
        expiries = [QUOTE_DATE + pd.Timedelta(days=d) for d in (7, 30)]
        strikes = np.round(np.arange(0.9, 1.11, 0.025) * spot0 / 2.5) * 2.5
        start = QUOTE_DATE - pd.Timedelta(days=1) + pd.Timedelta(hours=13, minutes=30)
        seconds = np.sort(rng.uniform(0, 2 * 24 * 3600, n_trades))
        drift = np.cumsum(rng.normal(0, spot0 * 0.0004, n_trades))
        for i in range(n_trades):
            ts = start + pd.Timedelta(seconds=float(seconds[i]))
            spot = spot0 + drift[i]
            strike = float(rng.choice(strikes))
            expiry = expiries[int(rng.integers(0, 2))]
            cp = 'CALL' if rng.random() < 0.5 else 'PUT'
            iv = max(0.18 + 0.2 * abs(np.log(strike / spot)) + rng.normal(0, 0.01), 0.05)
            price = max(0.05, abs(spot - strike) * 0.4 + spot * iv * 0.05
                        + rng.normal(0, 0.05))
            size = int(rng.integers(1, 400))
            volume = int(rng.integers(200, 12000))
            oi = int(rng.integers(100, 8000))
            rows.append({
                'Trade Time': ts.strftime('%Y-%m-%dT%H:%M:%S.000Z'),
                'Ticker': ticker,
                'Expiration Date': expiry.date().isoformat(),
                'Strike Price': f'${strike:.2f}',
                'Contract Type': cp,
                'Reference Price': f'${spot:.2f}',
                'Size': size,
                'Option Price': f'${price:.2f}',
                'Bid Price': f'${max(price - 0.05, 0):.2f}',
                'Ask Price': f'${price + 0.05:.2f}',
                'Volume': f'{volume:,}',
                'Open Interest': f'{oi:,}',
                'Implied Volatility': f'{iv * 100:.2f}%',
            })
    return pd.DataFrame(rows)


def main():
    rng = np.random.default_rng(42)
    DATA_DIR.mkdir(exist_ok=True)

    chain = generate_chain(rng)
    chain.to_csv(DATA_DIR / 'sample_options_chain.csv', index=False)
    print(f'wrote sample_options_chain.csv ({len(chain)} rows)')

    prices = generate_price_history(rng)
    prices.to_csv(DATA_DIR / 'sample_price_history.csv', index=False)
    print(f'wrote sample_price_history.csv ({len(prices)} rows)')

    flow = generate_order_flow(rng)
    flow.to_csv(DATA_DIR / 'sample_order_flow.csv', index=False)
    print(f'wrote sample_order_flow.csv ({len(flow)} rows)')


if __name__ == '__main__':
    main()
