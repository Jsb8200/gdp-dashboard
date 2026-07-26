"""ITM / ATM / OTM classification and the moneyness breakdown."""

import numpy as np
import pandas as pd
import pytest

from dealer_gex.analytics import (
    ATM_BAND_PCT, analyze, block_moneyness_breakdown, block_type_breakdown,
    classify_moneyness, moneyness_band,
)
from dealer_gex.parsing import parse_file

from test_flow_types import REAL_CSV, _behaviour_csv


def test_otm_is_signed_the_same_way_on_both_sides():
    """A call 2% above spot and a put 2% below it are both 2% out of the
    money. One comparison then covers the whole book."""
    rows = [
        (0, 714, "CALL", 700.0, 10, "A", 1_000, "FLR", "BLOCK"),   # +2% call
        (1, 686, "PUT", 700.0, 10, "A", 1_000, "FLR", "BLOCK"),    # -2% put
        (2, 686, "CALL", 700.0, 10, "A", 1_000, "FLR", "BLOCK"),   # ITM call
        (3, 714, "PUT", 700.0, 10, "A", 1_000, "FLR", "BLOCK"),    # ITM put
    ]
    p = parse_file(_behaviour_csv(rows)).prints.sort_values("trade_time")
    otm = list(p["otm_pct"].round(3))
    assert otm[0] == pytest.approx(2.0)
    assert otm[1] == pytest.approx(2.041, abs=0.01)     # 700/686 - 1
    assert otm[2] < 0 and otm[3] < 0                    # both in the money


def test_classification_buckets_on_the_band():
    got = classify_moneyness([5.0, 0.2, -0.2, -5.0, np.nan], band_pct=0.25)
    assert list(got) == ["OTM", "ATM", "ATM", "ITM", ""]


def test_band_is_floored_at_half_a_strike_step():
    """0.25% of a $30 underlying is 7 cents — no strike is ever that close,
    so a percentage alone leaves the ATM bucket empty."""
    assert moneyness_band(700.0, 1.0) == pytest.approx(1.75)     # % wins
    assert moneyness_band(30.0, 1.0) == pytest.approx(0.5)       # floor wins
    assert moneyness_band(30.0, None) == pytest.approx(30 * ATM_BAND_PCT)


def test_breakdown_splits_the_ladder(real):
    m = block_moneyness_breakdown(real.prints, real.spots["QQQ"])
    assert list(m["moneyness"]) == [x for x in ("ITM", "ATM", "OTM")
                                    if x in set(m["moneyness"])]
    assert m["premium_share"].sum() == pytest.approx(1.0)
    assert (m["band"] == m["band"].iloc[0]).all()
    # the signed average has to agree with the bucket it labels
    for _, r in m.iterrows():
        if r["moneyness"] == "OTM":
            assert r["avg_otm_pct"] > 0
        elif r["moneyness"] == "ITM":
            assert r["avg_otm_pct"] < 0


def test_prints_are_classified_against_their_own_reference_price():
    """Spot moving later must not retroactively relabel an earlier print."""
    rows = [
        (0, 700, "CALL", 700.0, 10, "A", 1_000, "FLR", "BLOCK"),   # ATM then
        (90, 700, "CALL", 760.0, 10, "A", 1_000, "AUTO", "BLOCK"),  # ITM then
    ]
    p = parse_file(_behaviour_csv(rows)).prints
    m = block_moneyness_breakdown(p, spot=760.0).set_index("moneyness")
    assert "ATM" in m.index and "ITM" in m.index
    assert m.loc["ATM", "prints"] == 1      # not re-labelled by the later spot
    assert m.loc["ITM", "prints"] == 1


def test_open_interest_is_counted_once_per_contract_in_each_bucket():
    rows = [
        (0, 700, "CALL", 700.0, 100, "A", 1_000, "FLR", "BLOCK"),
        (1, 700, "CALL", 700.0, 100, "A", 1_000, "AUTO", "BLOCK"),
    ]
    m = block_moneyness_breakdown(parse_file(_behaviour_csv(rows)).prints)
    row = m.set_index("moneyness").loc["ATM"]
    assert row["prints"] == 2
    assert row["open_interest"] == 20_000        # one contract, not two
    assert row["contracts"] == 200


def test_per_type_table_carries_a_moneyness_read(real):
    b = block_type_breakdown(real.prints, real.spots["QQQ"]).set_index("block_type")
    assert set(b["moneyness"]) <= {"ITM", "ATM", "OTM", ""}
    for _, r in b.iterrows():
        if r["moneyness"] == "OTM":
            assert r["otm_pct"] > 0
        elif r["moneyness"] == "ITM":
            assert r["otm_pct"] < 0


def test_breakdown_handles_missing_inputs():
    assert block_moneyness_breakdown(None).empty
    assert block_moneyness_breakdown(pd.DataFrame()).empty
    p = parse_file(REAL_CSV.encode()).prints.drop(columns=["otm_pct"])
    assert block_moneyness_breakdown(p).empty


def test_report_carries_the_moneyness_table(real):
    from dealer_gex.report import build_markdown

    a = analyze(real.chain, real.spots["QQQ"], real.asof)
    md = build_markdown(a, ticker="QQQ",
                        block_types=block_type_breakdown(real.prints, a.spot),
                        block_prints=real.prints)
    assert "### By moneyness — where they struck relative to spot" in md
    assert "| Moneyness | Premium | % | Prints |" in md
    assert "its own reference price" in md
