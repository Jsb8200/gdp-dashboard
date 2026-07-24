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


def test_multiplier_scales_dollars_not_levels():
    chain, spot = read_chain((REPO / "data" / "sample_option_chain.csv").read_bytes())
    a100 = analyze(chain, spot, ASOF)
    a50 = analyze(chain, spot, ASOF, multiplier=50.0)
    for field in ("total_gex", "dex", "vanna_flow", "charm_flow"):
        assert getattr(a50, field) == pytest.approx(getattr(a100, field) / 2, rel=1e-9)
    # levels are scale-invariant
    assert a50.gamma_flip == pytest.approx(a100.gamma_flip, abs=0.02)
    assert a50.call_wall == pytest.approx(a100.call_wall, abs=0.02)
    assert a50.put_wall == pytest.approx(a100.put_wall, abs=0.02)
    assert a50.regime == a100.regime


QUANTDATA_CSV = """Trade ID,Trade Time,Ticker,Expiration Date,Strike Price,Contract Type,Reference Price,Size,Volume,Open Interest,Side Code,Implied Volatility,Gamma,Premium Price,Consolidation Type,Is Golden Sweep,Is Unusual,Is Opening Position
1,2026-07-14T13:30:00.100Z,QQQ,2026-08-21,$720.00,CALL,$719.50,100,"1,000","5,000",A,18.5%,0.012,"$50,000.00",SWEEP,No,No,Yes
2,2026-07-14T14:00:00.200Z,QQQ,2026-08-21,$720.00,CALL,$720.10,50,"1,500","5,000",AA,18.7%,0.012,"$25,000.00",BLOCK,Yes,No,No
3,2026-07-14T14:30:00.300Z,QQQ,2026-08-21,$720.00,CALL,$720.30,30,"1,800","5,000",B,18.6%,0.012,"$15,000.00",AUTO,No,Yes,No
4,2026-07-14T15:00:00Z,QQQ,2026-08-21,$720.00,CALL,$720.40,20,"2,000","5,000",M,18.4%,0.012,"$10,000.00",AUTO,No,No,No
5,2026-07-14T15:10:00.500Z,QQQ,2026-08-21,$700.00,PUT,$720.60,80,900,"8,000",BB,22.1%,0.010,"$80,000.00",SWEEP,No,No,No
6,2026-07-14T17:42:16.130Z,QQQ,2026-08-21,$700.00,PUT,$721.10,40,950,"8,000",A,22.3%,0.010,"$40,000.00",AUTO,No,No,No
7,2026-07-13T14:00:00.100Z,GLD,2026-08-21,$370.00,CALL,$372.64,10,200,"1,200",A,15.0%,0.020,"$5,000.00",AUTO,No,No,No
"""


def test_quantdata_trade_flow_parsing():
    from dealer_gex.parsing import parse_file

    pf = parse_file(QUANTDATA_CSV)
    assert pf.tickers == ["QQQ", "GLD"]
    assert pf.asof == date(2026, 7, 14)
    # spot = ref price of the LAST trade per ticker (incl. mixed-format timestamps)
    assert pf.spots["QQQ"] == pytest.approx(721.10)
    assert pf.spots["GLD"] == pytest.approx(372.64)

    chain = pf.chain
    assert len(chain) == 3  # prints collapsed per contract
    c720 = chain[(chain["strike"] == 720) & (chain["type"] == "C")].iloc[0]
    # OI is a contract property: max, never summed across prints
    assert c720["open_interest"] == 5000
    # volume = max of the cumulative column, not a sum
    assert c720["volume"] == 2000
    # signed customer flow: +100 (A) +50 (AA) -30 (B) + 0 (M) = +120
    assert c720["net_customer_size"] == 120
    p700 = chain[(chain["strike"] == 700) & (chain["type"] == "P")].iloc[0]
    assert p700["net_customer_size"] == -80 + 40
    assert 0.1 < c720["iv"] < 0.3  # percent normalized to decimal


def test_flow_mode_signs_from_side_codes():
    from dealer_gex.parsing import parse_file

    pf = parse_file(QUANTDATA_CSV)
    qqq = pf.chain[pf.chain["ticker"] == "QQQ"]
    # customers net-bought both the calls (+120) and puts (-40 => net sold);
    # dealer gamma = -(customer): short 120 calls, long 40 puts
    a = analyze(qqq, pf.spots["QQQ"], pf.asof, weight="flow")
    # dealer: -120 call gamma + 40 put gamma, call gamma dominates => short
    assert a.total_gex < 0
    assert a.regime == "short_gamma"
    assert a.weight_mode == "flow"

    # convention (OI) mode on the same chain is unaffected by side codes
    a_oi = analyze(qqq, pf.spots["QQQ"], pf.asof)
    assert a_oi.total_gex != a.total_gex


def test_flow_mode_requires_flow_data():
    chain, spot = read_chain(LONG_CSV)
    with pytest.raises(ValueError):
        analyze(chain, spot, ASOF, weight="flow")


def test_magnet_levels_sample_chain():
    from dealer_gex.analytics import magnet_levels

    chain, spot = read_chain((REPO / "data" / "sample_option_chain.csv").read_bytes())
    a = analyze(chain, spot, ASOF)
    m = magnet_levels(a)
    assert not m.empty
    # strengths sorted desc, normalized to 100
    assert list(m["strength"]) == sorted(m["strength"], reverse=True)
    assert m["strength"].iloc[0] == 100
    # the call cluster at 650 and ATM cluster near 625-630 produce magnets
    mags = m[m["kind"] == "magnet"]["level"]
    assert any(abs(x - 650) < 5 for x in mags)
    # the put cluster at 600 is an accelerator (negative dealer gamma)
    accs = m[m["kind"] == "accelerator"]["level"]
    assert any(abs(x - 600) < 5 for x in accs)
    # every level sits within one strike spacing of its anchor
    assert (abs(m["level"] - m["anchor_strike"]) <= 5.0).all()


def test_magnets_in_flow_mode():
    from dealer_gex.parsing import parse_file
    from dealer_gex.analytics import magnet_levels

    pf = parse_file(QUANTDATA_CSV)
    qqq = pf.chain[pf.chain["ticker"] == "QQQ"]
    a = analyze(qqq, pf.spots["QQQ"], pf.asof, weight="flow")
    m = magnet_levels(a)
    # customers bought the 720 calls (dealer short there) => accelerator at 720
    accs = m[m["kind"] == "accelerator"]["level"]
    assert any(abs(x - 720) < 5 for x in accs)
    # customers net-sold the 700 puts (dealer long) => magnet at 700
    mags = m[m["kind"] == "magnet"]["level"]
    assert any(abs(x - 700) < 5 for x in mags)


def test_report_includes_magnets():
    from dealer_gex.report import build_markdown

    chain, spot = read_chain((REPO / "data" / "sample_option_chain.csv").read_bytes())
    a = analyze(chain, spot, ASOF)
    md = build_markdown(a, ticker="SPY")
    assert "## Magnet levels" in md
    assert "Nearest magnets" in md


def test_oi_levels_sample_chain():
    from dealer_gex.analytics import oi_levels

    chain, spot = read_chain((REPO / "data" / "sample_option_chain.csv").read_bytes())
    a = analyze(chain, spot, ASOF)
    lv = oi_levels(a)
    assert not lv.empty
    assert list(lv["strength"]) == sorted(lv["strength"], reverse=True)
    assert lv["strength"].iloc[0] == 100
    # put cluster at 600 below spot, call cluster at 650 above
    floors = lv[(lv["side"] == "put") & (lv["level"] < spot)]
    caps = lv[(lv["side"] == "call") & (lv["level"] > spot)]
    assert any(abs(x - 600) < 5 for x in floors["level"])
    assert any(abs(x - 650) < 5 for x in caps["level"])
    # side labels reflect the dominant OI at the anchor
    r600 = floors.loc[floors["level"].sub(600).abs().idxmin()]
    assert r600["put_oi"] > r600["call_oi"]


def test_report_includes_oi_levels():
    from dealer_gex.report import build_markdown

    chain, spot = read_chain((REPO / "data" / "sample_option_chain.csv").read_bytes())
    a = analyze(chain, spot, ASOF)
    md = build_markdown(a, ticker="SPY")
    assert "## OI levels (raw open interest)" in md
    assert "Raw OI structure" in md


def test_oi_walls():
    from dealer_gex.analytics import _kernel_density, oi_walls
    from dealer_gex.report import build_markdown, key_ladder

    chain, spot = read_chain((REPO / "data" / "sample_option_chain.csv").read_bytes())
    a = analyze(chain, spot, ASOF)
    ow = oi_walls(a)
    # anchors: the raw max-OI spikes from the sample generator
    assert ow.call_strike == 650.0
    assert ow.put_strike == 600.0
    # levels are pinpoint: within one spacing of the anchor, and a true local
    # max of the smoothed OI density
    for level, strike, col in ((ow.call, 650.0, "call_oi"), (ow.put, 600.0, "put_oi")):
        assert abs(level - strike) <= 5.0
        ks = a.by_strike["strike"].to_numpy()
        w = a.by_strike[col].to_numpy(dtype=float)
        peak = _kernel_density(ks, w, level, 5.0)
        assert peak >= _kernel_density(ks, w, level - 0.25, 5.0)
        assert peak >= _kernel_density(ks, w, level + 0.25, 5.0)
    ladder = key_ladder(a)
    assert {"Call OI wall", "Put OI wall"} <= set(ladder["Level"])
    md = build_markdown(a, ticker="SPY")
    assert "Call OI wall" in md and "Put OI wall" in md


def test_prints_retained_with_flags_and_premium():
    from dealer_gex.parsing import parse_file

    pf = parse_file(QUANTDATA_CSV)
    p = pf.prints
    assert p is not None and len(p) == 7
    assert p["is_sweep"].sum() == 2          # rows 1 and 5 (Consolidation SWEEP)
    assert p["is_block"].sum() == 1          # row 2 (Consolidation BLOCK)
    assert p["is_golden"].sum() == 1
    assert p["is_unusual"].sum() == 1
    assert p["is_opening"].sum() == 1
    assert p["premium"].max() == 80000.0     # $-and-comma formatted


def test_conviction_aggregation():
    from dealer_gex.parsing import parse_file, aggregate_prints

    pf = parse_file(QUANTDATA_CSV)
    p = pf.prints
    conv = aggregate_prints(p[p["is_sweep"]])
    # sweep prints only: the 720C (100 @ ask) and 700P (80 @ bid)
    assert len(conv) == 2
    c720 = conv[(conv["strike"] == 720) & (conv["type"] == "C")].iloc[0]
    assert c720["net_customer_size"] == 100  # only the ask-side sweep remains
    assert c720["open_interest"] == 5000     # OI still a contract property


def test_report_level_migration_section():
    from dealer_gex.report import build_markdown

    chain, spot = read_chain((REPO / "data" / "sample_option_chain.csv").read_bytes())
    a = analyze(chain, spot, ASOF)
    hist = pd.DataFrame([
        {"date": date(2026, 7, 16), "spot": 625.0, "flip": 614.2, "call_wall": 648.0,
         "put_wall": 599.5, "call_oi_wall": 650.0, "put_oi_wall": 600.0,
         "max_pain": 622.0, "net_gex": 1.2e9, "regime": "long_gamma"},
        {"date": date(2026, 7, 17), "spot": 628.5, "flip": 616.5, "call_wall": 648.7,
         "put_wall": 600.6, "call_oi_wall": 650.0, "put_oi_wall": 600.0,
         "max_pain": 623.7, "net_gex": 1.8e9, "regime": "long_gamma"},
    ])
    md = build_markdown(a, ticker="SPY", history=hist)
    assert "## Level migration" in md
    assert "2026-07-16" in md and "2026-07-17" in md
    # without history the section is absent
    assert "## Level migration" not in build_markdown(a, ticker="SPY")


def test_scenario_repricing_premises():
    """The scenario simulator's core claims: regime flips below the flip
    level, IV shifts move the numbers, and hedge flow is the dex delta."""
    chain, spot = read_chain((REPO / "data" / "sample_option_chain.csv").read_bytes())
    a = analyze(chain, spot, ASOF)
    assert a.regime == "long_gamma"

    below = analyze(chain, a.gamma_flip - 8.0, ASOF)
    assert below.regime == "short_gamma"

    crushed = analyze(chain.assign(iv=(chain["iv"] - 0.05).clip(lower=0.005)), spot, ASOF)
    assert crushed.total_gex != a.total_gex
    assert crushed.dex != a.dex  # hedge flow = -(dex' - dex) is nonzero


def test_zero_dte_analysis():
    rows = []
    for strike, typ, oi in [(98, "P", 5000), (100, "C", 4000), (100, "P", 4000),
                            (102, "C", 5000)]:
        rows.append({"expiry": pd.Timestamp(ASOF), "strike": float(strike),
                     "type": typ, "open_interest": float(oi), "volume": 100.0,
                     "iv": 0.30, "gamma": np.nan})
    a = analyze(pd.DataFrame(rows), spot=100.0, asof=ASOF)
    # 0DTE contracts get the intraday time floor, not dropped and not exploded
    assert a.n_contracts == 4
    assert np.isfinite(a.total_gex) and a.total_gex != 0
    # expected move ~ spot * iv * sqrt(MIN_T): a fraction of a full day's move
    from dealer_gex.analytics import MIN_T
    approx = 100.0 * 0.30 * np.sqrt(MIN_T)
    assert a.expected_move == pytest.approx(approx, rel=0.01)
    assert 0 < a.expected_move < 100.0 * 0.30 * np.sqrt(1 / 252)


def test_block_levels_and_flow_books():
    from dealer_gex.parsing import parse_file
    from dealer_gex.analytics import block_levels, flow_books

    pf = parse_file(QUANTDATA_CSV)
    qqq = pf.prints[pf.prints["ticker"] == "QQQ"]

    lvls = block_levels(qqq, pf.spots["QQQ"])
    # the single block print: 50x 720C @ ask, $25k premium
    assert len(lvls) == 1
    assert abs(lvls.iloc[0]["level"] - 720) < 5
    assert lvls.iloc[0]["side"] == "call"
    assert lvls.iloc[0]["direction"] == "buy"
    assert lvls.iloc[0]["strength"] == 100
    assert lvls.iloc[0]["premium"] == 25000.0

    books = flow_books(qqq, pf.spots["QQQ"], pf.asof)
    assert set(books) == {"blocks", "sweeps"}
    # block book: customer bought 50 calls at ask -> dealer short gamma,
    # customer delta positive
    assert books["blocks"].total_gex < 0
    assert -books["blocks"].dex > 0
    # sweep book: +100 calls bought, -80 puts sold -> customer long delta too
    assert -books["sweeps"].dex > 0


def test_report_block_intelligence():
    from dealer_gex.parsing import parse_file
    from dealer_gex.analytics import block_levels, flow_books
    from dealer_gex.report import build_markdown

    pf = parse_file(QUANTDATA_CSV)
    qqq_prints = pf.prints[pf.prints["ticker"] == "QQQ"]
    qqq_chain = pf.chain[pf.chain["ticker"] == "QQQ"]
    a = analyze(qqq_chain, pf.spots["QQQ"], pf.asof)
    books = flow_books(qqq_prints, pf.spots["QQQ"], pf.asof)
    lvls = block_levels(qqq_prints, pf.spots["QQQ"])
    md = build_markdown(a, ticker="QQQ", block_books=books, block_lvls=lvls)
    assert "## Block intelligence" in md
    assert "## Block commitment levels" in md
    assert "Blocks (institutional)" in md and "Sweeps (urgent)" in md


def test_split_flag_parsed():
    from dealer_gex.parsing import parse_file

    csv = QUANTDATA_CSV.replace("4,2026-07-14T15:00:00Z,QQQ,2026-08-21,$720.00,CALL,$720.40,20,\"2,000\",\"5,000\",M,18.4%,0.012,\"$10,000.00\",AUTO,No,No,No",
                                "4,2026-07-14T15:00:00Z,QQQ,2026-08-21,$720.00,CALL,$720.40,20,\"2,000\",\"5,000\",A,18.4%,0.012,\"$10,000.00\",SPLIT,No,No,No")
    pf = parse_file(csv)
    assert pf.prints["is_split"].sum() == 1


def test_block_campaigns_multi_day():
    from datetime import date as _date
    from dealer_gex.parsing import parse_file
    from dealer_gex.analytics import block_campaigns

    pf = parse_file(QUANTDATA_CSV)
    p = pf.prints[pf.prints["ticker"] == "QQQ"].copy()
    # only one block in the fixture (720C); replay it across two days as a
    # build, plus a second day-2-only block that must NOT qualify
    day1 = p[p["is_block"]].copy()
    day2 = p[p["is_block"]].copy()
    day2b = p[p["is_block"]].copy()
    day2b["strike"] = 730.0
    day2 = pd.concat([day2, day2b], ignore_index=True)
    camps = block_campaigns([(_date(2026, 7, 13), day1),
                             (_date(2026, 7, 14), day2)])
    # 720C hit both days -> campaign; 730C only day 2 -> excluded
    assert len(camps) == 1
    r = camps.iloc[0]
    assert r["strike"] == 720.0 and r["type"] == "C"
    assert r["days"] == 2
    assert r["direction"] == "buy"        # ask-side both days
    assert len(r["daily"]) == 2
    # single-day data yields no campaigns
    assert block_campaigns([(_date(2026, 7, 14), day1)]).empty


def test_confluence_levels_fuses_layers():
    from dealer_gex.analytics import confluence_levels

    chain, spot = read_chain((REPO / "data" / "sample_option_chain.csv").read_bytes())
    a = analyze(chain, spot, ASOF)
    m = confluence_levels(a)
    assert not m.empty
    # ranked by score, normalized to 100
    assert list(m["score"]) == sorted(m["score"], reverse=True)
    assert m["score"].iloc[0] == 100
    # the top level is confirmed by multiple independent layers
    assert m["n_layers"].iloc[0] >= 3
    assert m["confidence"].iloc[0] == "high"
    # confidence tracks layer count
    for _, r in m.iterrows():
        expected = "high" if r["n_layers"] >= 3 else "medium" if r["n_layers"] == 2 else "low"
        assert r["confidence"] == expected
    # 600 (put wall + OI fortress + accelerator cluster) should surface strongly
    assert any(abs(x - 600) < 5 for x in m["level"])


def test_confidence_vs_data_quality_are_independent():
    """Layer-confidence measures how many systems agree; data_quality
    measures whether the underlying data is trustworthy. A lone strike can
    show high layer-confluence yet fail the data-quality gate — that split
    is the design."""
    from dealer_gex.analytics import confluence_levels, data_quality

    rows = [{"expiry": pd.Timestamp("2026-08-21"), "strike": 100.0, "type": t,
             "open_interest": 500.0, "volume": 0.0, "iv": 0.25, "gamma": np.nan}
            for t in ("C", "P")]
    a = analyze(pd.DataFrame(rows), spot=100.0, asof=ASOF)
    m = confluence_levels(a)
    # multiple systems pile onto the single strike -> layer-confidence can be high
    assert not m.empty
    # but the data-quality gate catches the thinness (2 contracts)
    assert data_quality(a)["level"] == "low"


def test_data_quality_flags():
    from dealer_gex.analytics import data_quality
    from dealer_gex.parsing import parse_file

    chain, spot = read_chain((REPO / "data" / "sample_option_chain.csv").read_bytes())
    a = analyze(chain, spot, ASOF)
    dq = data_quality(a)
    assert dq["level"] == "high" and dq["n_contracts"] > 80

    thin = analyze(pd.DataFrame([
        {"expiry": pd.Timestamp("2026-08-21"), "strike": 100.0, "type": "C",
         "open_interest": 50.0, "volume": 0.0, "iv": 0.25, "gamma": np.nan},
        {"expiry": pd.Timestamp("2026-08-21"), "strike": 100.0, "type": "P",
         "open_interest": 50.0, "volume": 0.0, "iv": 0.25, "gamma": np.nan},
    ]), spot=100.0, asof=ASOF)
    dq_thin = data_quality(thin)
    assert dq_thin["level"] == "low" and dq_thin["notes"]

    # flow file with mostly mid prints -> soft signed-flow note
    pf = parse_file(QUANTDATA_CSV)
    qqq_ch = pf.chain[pf.chain["ticker"] == "QQQ"]
    a_flow = analyze(qqq_ch, pf.spots["QQQ"], pf.asof, weight="flow")
    dq_flow = data_quality(a_flow, pf.prints[pf.prints["ticker"] == "QQQ"])
    assert dq_flow["signed_ratio"] is not None


def test_report_dedups_ladder_when_master_present():
    from dealer_gex.analytics import confluence_levels
    from dealer_gex.report import build_markdown

    chain, spot = read_chain((REPO / "data" / "sample_option_chain.csv").read_bytes())
    a = analyze(chain, spot, ASOF)
    m = confluence_levels(a)
    # with the confluence master table, the flat ladder is dropped
    with_master = build_markdown(a, ticker="SPY", master=m)
    assert "## Master levels (confluence)" in with_master
    assert "## Level ladder" not in with_master
    # without it, the ladder is the fallback
    without = build_markdown(a, ticker="SPY")
    assert "## Level ladder" in without


def test_report_master_levels_section():
    from dealer_gex.analytics import confluence_levels, data_quality
    from dealer_gex.report import build_markdown

    chain, spot = read_chain((REPO / "data" / "sample_option_chain.csv").read_bytes())
    a = analyze(chain, spot, ASOF)
    m = confluence_levels(a)
    md = build_markdown(a, ticker="SPY", master=m, dq=data_quality(a))
    assert "## Master levels (confluence)" in md
    assert "Confirmed by" in md


def test_directional_lean_bullish_when_customers_buy():
    from dealer_gex.parsing import parse_file
    from dealer_gex.analytics import directional_lean

    # every print is an ask-side buy of calls -> strongly bullish flow
    csv_lines = ["Trade ID,Trade Time,Ticker,Expiration Date,Strike Price,"
                 "Contract Type,Reference Price,Size,Volume,Open Interest,Side Code,"
                 "Implied Volatility,Gamma,Premium Price,Consolidation Type,"
                 "Is Golden Sweep,Is Unusual,Is Opening Position"]
    for i, k in enumerate((615, 620, 625, 630)):
        csv_lines.append(f"{i},2026-07-14T14:00:0{i}Z,QQQ,2026-08-21,${k}.00,CALL,"
                         f"$620.00,200,500,3000,A,20%,0.012,\"$40,000.00\",SWEEP,No,No,Yes")
    pf = parse_file("\n".join(csv_lines) + "\n")
    p = pf.prints
    a = analyze(pf.chain, pf.spots["QQQ"], pf.asof, weight="flow")
    L = directional_lean(a, p)
    assert L["score"] > 15          # bullish tilt
    assert "bullish" in L["label"].lower()
    # order-flow aggressor component is fully bullish (all ask-side)
    of = [v for n, v, _ in L["components"] if n.startswith("Order-flow")][0]
    assert of == pytest.approx(100.0)


def test_directional_lean_neutral_without_flow():
    from dealer_gex.analytics import directional_lean

    chain, spot = read_chain((REPO / "data" / "sample_option_chain.csv").read_bytes())
    a = analyze(chain, spot, ASOF)   # OI mode, no prints
    L = directional_lean(a, None)
    # only max-pain pull is available -> low confidence, single component
    assert L["confidence"] == "low"
    assert L["has_flow"] is False
    assert len(L["components"]) == 1


def test_executive_summary_and_confluence_playbook():
    from dealer_gex.analytics import confluence_levels, directional_lean
    from dealer_gex.report import build_playbook, executive_summary

    chain, spot = read_chain((REPO / "data" / "sample_option_chain.csv").read_bytes())
    a = analyze(chain, spot, ASOF)
    m = confluence_levels(a)
    lean = directional_lean(a)

    tldr = executive_summary(a, m, lean)
    assert tldr[0].isupper() and tldr.endswith(".")
    assert ". o" not in tldr and ". t" not in tldr  # every sentence capitalized
    assert f"{m.iloc[0]['level']:,.2f}" in tldr     # names the top confluence level

    pb = build_playbook(a, master=m, lean=lean)
    # the playbook now leads with the confluence ranking, not a lone wall
    assert "confluence" in pb[0].lower()
    assert f"{m.iloc[0]['level']:,.2f}" in pb[0]
    # and folds in the lean as a tiebreaker
    assert any("Positioning lean" in b for b in pb) == lean["has_flow"]


def test_report_directional_lean_section():
    from dealer_gex.analytics import directional_lean
    from dealer_gex.report import build_markdown

    chain, spot = read_chain((REPO / "data" / "sample_option_chain.csv").read_bytes())
    a = analyze(chain, spot, ASOF)
    md = build_markdown(a, ticker="SPY", lean=directional_lean(a))
    assert "## Directional lean" in md
    assert "lean*, not a signal" in md


def test_session_range_from_ref_prices():
    from dealer_gex.parsing import parse_file
    from dealer_gex.analytics import session_range

    pf = parse_file(QUANTDATA_CSV)
    qqq = pf.prints[pf.prints["ticker"] == "QQQ"]
    rng = session_range(qqq)
    assert rng is not None
    lo, hi, close, open_ = rng
    assert lo <= open_ <= hi and lo <= close <= hi
    # last QQQ print in the fixture stamps ref 721.10
    assert close == pytest.approx(721.10)


def test_level_hit_rate_classification():
    from dealer_gex.analytics import level_hit_rate

    # day-1 levels; day-2 range [598, 652] closing 650
    master = pd.DataFrame({
        "level": [600.0, 650.0, 625.0, 700.0],
        "role": ["Support", "Resistance", "Support · pin", "Resistance"],
        "confidence": ["high", "high", "medium", "low"],
    })
    # day-2 range [599, 652] closing 650: reaches 600 (holds), pushes above 650
    days = [
        {"date": date(2026, 7, 16), "master": master, "low": 620, "high": 630, "close": 625},
        {"date": date(2026, 7, 17), "master": master, "low": 599.0, "high": 652.0, "close": 650.0},
    ]
    detail, s = level_hit_rate(days)
    by = {r["level"]: r["outcome"] for _, r in detail.iterrows()}
    assert by[600.0] == "held"      # low 599 >= 600*(1-0.003)=598.2 -> held support
    assert by[650.0] == "broke"     # high 652 > 650*(1+0.003)=651.95 -> broke resistance
    assert by[625.0] == "broke"     # range straddles but close 650 far from 625 -> not pinned
    assert by[700.0] == "not tested"  # high 652 never reached 700
    assert s["tested"] == 3 and s["pairs"] == 1
    assert s["held"] == 1
    assert s["by_confidence"]["high"]["tested"] == 2


def test_level_hit_rate_pin_holds():
    from dealer_gex.analytics import level_hit_rate

    master = pd.DataFrame({"level": [625.0], "role": ["Support · pin"],
                           "confidence": ["high"]})
    days = [
        {"date": date(2026, 7, 16), "master": master, "low": 620, "high": 630, "close": 625},
        {"date": date(2026, 7, 17), "master": master, "low": 620.0, "high": 630.0, "close": 625.3},
    ]
    detail, s = level_hit_rate(days)
    assert detail.iloc[0]["outcome"] == "held"  # closed within 2*touch of the pin
    assert s["rate"] == 1.0


DARKPOOL_CSV = """Time,Symbol,Price,Size,Value
2026-07-14T14:00:00Z,QQQ,700.10,40000,"28,004,000"
2026-07-14T14:01:00Z,QQQ,700.05,35000,"24,501,750"
2026-07-14T14:02:00Z,QQQ,699.95,50000,"34,997,500"
2026-07-14T14:03:00Z,QQQ,720.20,20000,"14,404,000"
2026-07-14T14:04:00Z,QQQ,720.10,18000,"12,961,800"
2026-07-14T15:00:00Z,QQQ,711.50,5000,"3,557,500"
"""


def test_darkpool_detection_and_levels():
    from dealer_gex.parsing import parse_file
    from dealer_gex.analytics import darkpool_levels

    pf = parse_file(DARKPOOL_CSV)
    assert pf.dark is not None and pf.chain.empty
    assert pf.spots["QQQ"] == pytest.approx(711.50)  # last by time
    assert len(pf.dark) == 6

    lv = darkpool_levels(pf.dark, spot=710.0)
    assert not lv.empty
    assert list(lv["strength"]) == sorted(lv["strength"], reverse=True)
    assert lv["strength"].iloc[0] == 100
    # heaviest concentration near 700 (125k shares), secondary near 720
    assert abs(lv.iloc[0]["level"] - 700) < 3


def test_darkpool_feeds_confluence():
    from dealer_gex.parsing import parse_file
    from dealer_gex.analytics import confluence_levels, darkpool_levels

    chain, spot = read_chain((REPO / "data" / "sample_option_chain.csv").read_bytes())
    a = analyze(chain, spot, ASOF)
    # a dark level right at the 600 put fortress should reinforce it
    dark = pd.DataFrame({"ticker": ["SPY"] * 3, "price": [600.0, 600.1, 599.9],
                         "size": [50000.0, 40000.0, 45000.0],
                         "premium": [3e7, 2.4e7, 2.7e7], "time": pd.NaT})
    dlv = darkpool_levels(dark, spot)
    m = confluence_levels(a, dark_lvls=dlv)
    row600 = m[m["level"].sub(600).abs() < 3]
    assert not row600.empty
    assert any("Dark pool" in ls for ls in row600.iloc[0]["layers"])


def test_option_file_not_misdetected_as_darkpool():
    from dealer_gex.parsing import parse_file

    # the QuantData flow file has strike+type -> must NOT be read as dark-pool
    pf = parse_file(QUANTDATA_CSV)
    assert pf.dark is None and not pf.chain.empty


def test_intraday_flow_cumulative():
    from dealer_gex.parsing import parse_file
    from dealer_gex.analytics import intraday_flow

    pf = parse_file(QUANTDATA_CSV)
    qqq = pf.prints[pf.prints["ticker"] == "QQQ"]
    tl = intraday_flow(qqq, freq="1min")
    assert not tl.empty
    assert list(tl.columns) == ["time", "cum_all", "cum_block", "cum_sweep"]
    # cumulative and monotone in time; final all-flow = net signed of all prints
    assert tl["cum_all"].iloc[-1] == pytest.approx(qqq["signed_size"].sum())
    # block line only accumulates block prints
    assert tl["cum_block"].iloc[-1] == pytest.approx(
        qqq.loc[qqq["is_block"], "signed_size"].sum())


def test_tuned_layer_weights_gating_and_direction():
    from dealer_gex.analytics import tuned_layer_weights

    # 6 tested levels carrying family "gamma_wall": 5 held, 1 broke -> up-weight
    rows = []
    for i in range(6):
        rows.append({"tested": True, "held": i < 5, "families": ["gamma_wall"]})
    # a family seen only twice must be gated out (min_n=4)
    rows.append({"tested": True, "held": False, "families": ["dark_pool"]})
    rows.append({"tested": True, "held": False, "families": ["dark_pool"]})
    detail = pd.DataFrame(rows)
    w = tuned_layer_weights(detail, min_n=4)
    assert "gamma_wall" in w and w["gamma_wall"] == pytest.approx(0.5 + 5 / 6, abs=1e-3)
    assert "dark_pool" not in w          # gated: too few samples
    assert 0.5 <= w["gamma_wall"] <= 1.5  # bounded


def test_tuned_weights_reweight_confluence():
    from dealer_gex.analytics import confluence_levels

    chain, spot = read_chain((REPO / "data" / "sample_option_chain.csv").read_bytes())
    a = analyze(chain, spot, ASOF)
    base = confluence_levels(a, top_n=20)
    assert "families" in base.columns and base["families"].map(len).max() >= 1
    # the OI-wall family is present in the default fusion
    assert base["families"].map(lambda f: "oi_wall" in f).any()
    # zeroing that family removes its vote entirely -> it vanishes from fusion
    zeroed = confluence_levels(a, top_n=20, weights={"oi_wall": 0.0})
    assert not zeroed["families"].map(lambda f: "oi_wall" in f).any()


def test_scan_ticker_summary():
    from dealer_gex.parsing import parse_file
    from dealer_gex.analytics import scan_ticker

    pf = parse_file(QUANTDATA_CSV)
    qqq_ch = pf.chain[pf.chain["ticker"] == "QQQ"]
    qqq_pr = pf.prints[pf.prints["ticker"] == "QQQ"]
    r = scan_ticker(qqq_ch, pf.spots["QQQ"], pf.asof, 0.045, "open_interest",
                    100.0, qqq_pr)
    assert r is not None
    assert r["regime"] in ("long_gamma", "short_gamma")
    assert set(r) >= {"spot", "flip", "flip_dist", "net_gex", "lean",
                      "lean_label", "divergence"}
    assert -100 <= r["lean"] <= 100
    # a ticker with no unexpired contracts returns None, not a crash
    from datetime import date as _date
    assert scan_ticker(qqq_ch, pf.spots["QQQ"], _date(2027, 1, 1), 0.045,
                       "open_interest", 100.0, qqq_pr) is None


def test_fmt_dollars():
    assert fmt_dollars(1_460_000_000) == "$1.46B"
    assert fmt_dollars(-441_430_000) == "-$441.43M"
    assert fmt_dollars(950) == "$950"
