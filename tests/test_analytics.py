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
    assert a.call_wall_strike == 105.0
    assert abs(a.call_wall - 105.0) <= 5.0  # pinpoint, anchored near the strike
    assert a.put_wall_strike in (95.0, 100.0)
    assert abs(a.put_wall - a.put_wall_strike) <= 5.0


def test_analyze_sample_chain():
    chain, spot = read_chain((REPO / "data" / "sample_option_chain.csv").read_bytes())
    a = analyze(chain, spot, ASOF)
    assert a.regime == "long_gamma"
    assert a.gamma_flip is not None and a.gamma_flip < spot
    assert a.call_wall > spot > a.put_wall
    assert len(a.by_expiry) == 4


def test_levels_are_pinpoint():
    from dealer_gex.analytics import side_gamma_density, total_gex_at, _fill_gamma

    chain, spot = read_chain((REPO / "data" / "sample_option_chain.csv").read_bytes())
    a = analyze(chain, spot, ASOF)
    df = _fill_gamma(chain, spot, ASOF, a.rate)

    # flip is a true zero of the GEX function, to sub-cent scale
    scale = abs(total_gex_at(df, spot, a.rate))
    assert abs(total_gex_at(df, a.gamma_flip, a.rate)) < scale * 1e-3

    # walls are local maxima of each side's smoothed gamma density
    for wall, strike, side in ((a.call_wall, a.call_wall_strike, "C"),
                               (a.put_wall, a.put_wall_strike, "P")):
        assert abs(wall - strike) <= 5.0  # anchored within one strike spacing
        peak = side_gamma_density(a.by_strike, wall, side, 5.0)
        assert peak >= side_gamma_density(a.by_strike, wall - 0.25, side, 5.0)
        assert peak >= side_gamma_density(a.by_strike, wall + 0.25, side, 5.0)

    # max pain interpolates between strikes but stays near the discrete min
    assert 620.0 <= a.max_pain <= 630.0


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


def test_volume_weighting_changes_results():
    chain, spot = read_chain((REPO / "data" / "sample_option_chain.csv").read_bytes())
    a_oi = analyze(chain, spot, ASOF)
    a_vol = analyze(chain, spot, ASOF, weight="volume")
    assert a_oi.weight_mode == "open_interest" and a_vol.weight_mode == "volume"
    assert a_vol.total_gex != a_oi.total_gex
    # volume is a fraction of OI in the sample, so magnitudes shrink
    assert abs(a_vol.total_gex) < abs(a_oi.total_gex)


def test_hedge_flows_and_expected_move():
    chain, spot = read_chain((REPO / "data" / "sample_option_chain.csv").read_bytes())
    a = analyze(chain, spot, ASOF)
    # dealers long calls / short puts => positive delta inventory
    assert a.dex > 0
    assert np.isfinite(a.vanna_flow) and np.isfinite(a.charm_flow)
    # 1-sigma move to the 7-day expiry: roughly spot * iv * sqrt(t)
    assert a.nearest_expiry == date(2026, 7, 24)
    approx = spot * 0.16 * np.sqrt(7 / 365)
    assert 0.5 * approx < a.expected_move < 2 * approx


def test_norm_cdf_accuracy():
    from dealer_gex.analytics import _norm_cdf

    assert _norm_cdf(0.0) == pytest.approx(0.5, abs=1e-7)
    assert _norm_cdf(1.96) == pytest.approx(0.9750021, abs=1e-6)
    assert _norm_cdf(-1.96) == pytest.approx(0.0249979, abs=1e-6)


def test_playbook_and_ladder():
    from dealer_gex.report import build_playbook, key_ladder

    chain, spot = read_chain((REPO / "data" / "sample_option_chain.csv").read_bytes())
    a = analyze(chain, spot, ASOF)
    text = "\n".join(build_playbook(a))
    assert "long gamma" in text
    assert f"{a.gamma_flip:,.2f}" in text
    assert "vanna" in text and "charm" in text
    ladder = key_ladder(a)
    assert list(ladder["Price"]) == sorted(ladder["Price"], reverse=True)
    assert {"Call wall", "Put wall", "Spot", "Gamma flip"} <= set(ladder["Level"])


def test_fmt_dollars():
    assert fmt_dollars(1_460_000_000) == "$1.46B"
    assert fmt_dollars(-441_430_000) == "-$441.43M"
    assert fmt_dollars(950) == "$950"
