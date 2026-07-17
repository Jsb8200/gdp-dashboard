from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dealer_gex.analytics import analyze, bs_gamma, fmt_dollars, gamma_flip, gex_by_strike
from dealer_gex.parsing import ChainParseError, read_chain

REPO = Path(__file__).resolve().parent.parent
ASOF = date(2026, 7, 17)

LONG_CSV = """Expiration Date,Type,Strike,Open Interest,Volume,Implied Volatility,Underlying Price
2026-08-21,Call,100,500,50,20.0,102.5
2026-08-21,Put,100,800,80,22.0,102.5
2026-08-21,Call,105,300,30,19.0,102.5
2026-08-21,Put,95,900,90,25.0,102.5
"""

CBOE_CSV = """SPY (SPDR S&P 500 ETF Trust),601.23,+1.10
"Jul 17 2026 @ 15:45 ET,Last: 601.23,Change: +1.10"
Expiration Date,Calls,Last Sale,Net,Bid,Ask,Vol,IV,Delta,Gamma,Open Int,Strike,Puts,Last Sale,Net,Bid,Ask,Vol,IV,Delta,Gamma,Open Int
2026-08-21,SPY C600,12.1,0.5,12.0,12.2,150,0.18,0.55,0.012,4000,600,SPY P600,10.2,-0.3,10.1,10.3,220,0.20,-0.45,0.011,6500
2026-08-21,SPY C610,7.0,0.2,6.9,7.1,90,0.17,0.40,0.011,3000,610,SPY P610,15.1,-0.5,15.0,15.2,60,0.21,-0.60,0.010,1500
"""


def test_parse_long_format():
    chain, spot = read_chain(LONG_CSV)
    assert spot == 102.5
    assert len(chain) == 4
    assert set(chain["type"]) == {"C", "P"}
    # percent IV normalized to decimals
    assert chain["iv"].between(0.15, 0.30).all()
    assert chain["expiry"].dt.year.eq(2026).all()


def test_parse_cboe_side_by_side():
    chain, spot = read_chain(CBOE_CSV)
    assert spot == 601.23
    assert len(chain) == 4  # 2 strikes x calls+puts
    calls = chain[chain["type"] == "C"]
    puts = chain[chain["type"] == "P"]
    assert calls.set_index("strike")["open_interest"].to_dict() == {600: 4000, 610: 3000}
    assert puts.set_index("strike")["open_interest"].to_dict() == {600: 6500, 610: 1500}
    # file-supplied gamma preserved
    assert calls["gamma"].notna().all()


def test_parse_garbage_raises():
    with pytest.raises(ChainParseError):
        read_chain("just,some,random\n1,2,3\n")


def test_bs_gamma_peaks_atm_and_positive():
    strikes = np.array([80.0, 100.0, 120.0])
    g = bs_gamma(100.0, strikes, 30 / 365, 0.2)
    assert (g > 0).all()
    assert g[1] == max(g)  # ATM gamma is the largest
    assert bs_gamma(100.0, 100.0, 0.0, 0.2) == 0.0  # expired -> no gamma


def test_gex_sign_convention():
    df = pd.DataFrame({
        "strike": [100.0, 100.0],
        "type": ["C", "P"],
        "open_interest": [10.0, 10.0],
        "gamma": [0.01, 0.01],
        "expiry": pd.to_datetime(["2026-08-21"] * 2),
        "iv": [0.2, 0.2],
        "volume": [0, 0],
    })
    out = gex_by_strike(df, spot=100.0)
    assert out.loc[0, "call_gex"] > 0
    assert out.loc[0, "put_gex"] < 0
    assert out.loc[0, "net_gex"] == pytest.approx(0.0)  # equal OI cancels


def test_gamma_flip_interpolation():
    curve = pd.DataFrame({"spot_level": [90.0, 100.0, 110.0],
                          "total_gex": [-1e9, 1e9, 3e9]})
    assert gamma_flip(curve, spot=100.0) == pytest.approx(95.0)
    no_cross = pd.DataFrame({"spot_level": [90.0, 110.0], "total_gex": [1e9, 2e9]})
    assert gamma_flip(no_cross, spot=100.0) is None


def test_analyze_put_heavy_chain_is_short_gamma():
    rows = []
    for strike, typ, oi in [(95, "P", 20000), (100, "P", 15000), (105, "C", 2000)]:
        rows.append({"expiry": pd.Timestamp("2026-08-21"), "strike": float(strike),
                     "type": typ, "open_interest": float(oi), "volume": 0.0,
                     "iv": 0.25, "gamma": np.nan})
    a = analyze(pd.DataFrame(rows), spot=100.0, asof=ASOF)
    assert a.regime == "short_gamma"
    assert a.total_gex < 0
    assert a.call_wall == 105.0
    assert a.put_wall in (95.0, 100.0)


def test_analyze_sample_chain():
    chain, spot = read_chain((REPO / "data" / "sample_option_chain.csv").read_bytes())
    a = analyze(chain, spot, ASOF)
    assert a.regime == "long_gamma"
    assert a.gamma_flip is not None and a.gamma_flip < spot
    assert a.call_wall > spot > a.put_wall
    assert len(a.by_expiry) == 4


def test_analyze_drops_expired_contracts():
    chain, spot = read_chain(LONG_CSV)
    with pytest.raises(ValueError):
        analyze(chain, spot, date(2027, 1, 1))  # everything expired


def test_report_contents():
    from dealer_gex.report import build_markdown, markdown_to_html

    chain, spot = read_chain((REPO / "data" / "sample_option_chain.csv").read_bytes())
    a = analyze(chain, spot, ASOF)
    md = build_markdown(a, ticker="SPY")
    assert "SPY Dealer Positioning Report" in md
    assert "LONG GAMMA" in md
    assert "Gamma flip" in md and "Call wall" in md and "Put wall" in md
    assert "not trading advice" in md
    html = markdown_to_html(md)
    assert "<table>" in html and "LONG GAMMA" in html


def test_fmt_dollars():
    assert fmt_dollars(1_460_000_000) == "$1.46B"
    assert fmt_dollars(-441_430_000) == "-$441.43M"
    assert fmt_dollars(950) == "$950"
