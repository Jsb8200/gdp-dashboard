"""Precision of the consolidated-flow type: FLR / AUTO / CROSS / BLOCK …"""

import numpy as np
import pandas as pd
import pytest

from dealer_gex.analytics import (
    analyze, block_levels, flow_books, flow_type_breakdown, institutional_mask,
)
from dealer_gex.parsing import classify_flow, parse_file

# A QuantData-shaped export whose Consolidation Type uses the venue codes
# (FLR / AUTO / CROSS) alongside the shape codes, including combinations and
# one code deliberately outside the mapping.
FLOW_CSV = """Trade ID,Trade Time,Ticker,Expiration Date,Strike Price,Contract Type,Reference Price,Size,Volume,Open Interest,Side Code,Implied Volatility,Gamma,Premium Price,Consolidation Type,Is Golden Sweep,Is Unusual,Is Opening Position
1,2026-07-14T13:30:00.100Z,XSP,2026-08-21,$700.00,CALL,$699.50,200,"2,000","9,000",A,18.5%,0.012,"$200,000.00",FLR BLOCK,No,Yes,Yes
2,2026-07-14T13:45:00.200Z,XSP,2026-08-21,$705.00,CALL,$700.10,20,"1,000","4,000",A,18.0%,0.011,"$5,000.00",AUTO,No,No,No
3,2026-07-14T14:05:00.300Z,XSP,2026-08-21,$705.00,CALL,$700.40,60,"1,000","4,000",AA,18.2%,0.011,"$30,000.00",AUTO-SWEEP,No,No,No
4,2026-07-14T14:30:00.400Z,XSP,2026-08-21,$680.00,PUT,$700.80,150,"1,800","7,500",B,22.0%,0.009,"$150,000.00",flr,No,Yes,No
5,2026-07-14T15:00:00.500Z,XSP,2026-08-21,$690.00,PUT,$700.20,100,"1,200","6,000",A,21.0%,0.010,"$60,000.00",Cross,No,No,Yes
6,2026-07-14T15:20:00.600Z,XSP,2026-08-21,$710.00,CALL,$700.90,40,900,"3,000",B,17.5%,0.010,"$12,000.00",SWEEP,No,No,No
7,2026-07-14T15:40:00.700Z,XSP,2026-08-21,$710.00,CALL,$700.50,10,900,"3,000",A,17.6%,0.010,"$2,000.00",PHLX SPECIAL,No,No,No
"""


@pytest.fixture(scope="module")
def flow():
    pf = parse_file(FLOW_CSV.encode())
    assert pf.prints is not None
    return pf


# --- classification ----------------------------------------------------------

def test_venue_and_shape_are_separate_axes():
    out = classify_flow(pd.Series(["FLR BLOCK", "AUTO", "AUTO-SWEEP", "flr",
                                   "Cross", "SWEEP", "BLOCK"]))
    assert list(out["flow_venue"]) == ["floor", "auto", "auto", "floor",
                                       "cross", "", ""]
    assert list(out["flow_shape"]) == ["block", "single", "sweep", "single",
                                       "single", "sweep", "block"]
    assert list(out["flow_type"]) == ["floor block", "auto", "auto sweep",
                                      "floor", "cross", "sweep", "block"]


def test_separators_and_casing_do_not_matter():
    out = classify_flow(pd.Series(["flr_block", "Auto/Sweep", " FLR-BLOCK ",
                                   "auto sweep"]))
    assert list(out["flow_type"]) == ["floor block", "auto sweep",
                                      "floor block", "auto sweep"]


def test_unmapped_codes_keep_their_raw_text():
    """An unknown vocabulary must stay visible, not get bucketed as 'single'
    where nobody would notice it."""
    out = classify_flow(pd.Series(["PHLX SPECIAL", "ISE-FACILITATION"]))
    assert list(out["flow_type"]) == ["PHLX SPECIAL", "ISE FACILITATION"]
    assert (out["flow_shape"] == "single").all()
    assert (out["flow_venue"] == "").all()


def test_blank_and_missing_are_single():
    out = classify_flow(pd.Series(["", " ", None, np.nan]))
    assert list(out["flow_type"]) == ["single"] * 4
    assert list(out["consolidation"]) == [""] * 4


# --- parsing -----------------------------------------------------------------

def test_prints_carry_the_precise_type(flow):
    p = flow.prints
    assert list(p["flow_type"]) == [
        "floor block", "auto", "auto sweep", "floor", "cross", "sweep",
        "PHLX SPECIAL",
    ]
    assert p["consolidation"].iloc[0] == "FLR BLOCK"     # raw value preserved
    assert list(p["is_floor"]) == [True, False, False, True, False, False, False]
    assert list(p["is_auto"]) == [False, True, True, False, False, False, False]
    assert list(p["is_cross"]) == [False] * 4 + [True] + [False] * 2
    assert list(p["is_sweep"]) == [False, False, True, False, False, True, False]
    assert list(p["is_block"]) == [True] + [False] * 6


def test_auto_alone_is_not_institutional_but_floor_and_cross_are(flow):
    p = flow.prints
    inst = institutional_mask(p)
    by_type = dict(zip(p["flow_type"], inst))
    assert by_type["floor block"] and by_type["floor"] and by_type["cross"]
    assert not by_type["auto"]          # electronic route, not a size signal
    assert not by_type["auto sweep"]    # ...even when it sweeps
    assert not by_type["sweep"]
    assert inst.sum() == 3


def test_explicit_type_selection_overrides_the_default(flow):
    picked = institutional_mask(flow.prints, types=["auto sweep", "sweep"])
    assert picked.sum() == 2
    assert set(flow.prints.loc[picked, "flow_type"]) == {"auto sweep", "sweep"}


# --- consolidated breakdown --------------------------------------------------

def test_breakdown_totals_premium_and_quantity_per_type(flow):
    br = flow_type_breakdown(flow.prints)
    assert list(br["flow_type"]) == [
        "floor block", "floor", "cross", "auto sweep", "sweep", "auto",
        "PHLX SPECIAL",
    ]                                            # ranked by premium
    row = br.set_index("flow_type").loc["floor block"]
    assert row["premium"] == 200_000.0
    assert row["contracts"] == 200
    assert row["prints"] == 1
    assert row["avg_premium"] == 200_000.0
    assert row["institutional"]
    assert row["venue"] == "floor" and row["shape"] == "block"

    assert br["premium"].sum() == 459_000.0
    assert br["contracts"].sum() == 580
    assert br["prints"].sum() == 7
    assert br["premium_share"].sum() == pytest.approx(1.0)
    assert br.set_index("flow_type").loc["floor block", "premium_share"] == \
        pytest.approx(200_000 / 459_000)


def test_breakdown_direction_comes_from_the_side_code(flow):
    br = flow_type_breakdown(flow.prints).set_index("flow_type")
    assert br.loc["floor block", "direction"] == "bought"    # side A
    assert br.loc["floor block", "net_contracts"] == 200
    assert br.loc["floor", "direction"] == "sold"            # side B
    assert br.loc["floor", "net_contracts"] == -150
    assert br.loc["sweep", "direction"] == "sold"


def test_breakdown_multiple_prints_of_one_type_consolidate():
    two = pd.concat([parse_file(FLOW_CSV.encode()).prints] * 2, ignore_index=True)
    br = flow_type_breakdown(two).set_index("flow_type")
    assert br.loc["floor block", "prints"] == 2
    assert br.loc["floor block", "contracts"] == 400
    assert br.loc["floor block", "premium"] == 400_000.0
    assert br.loc["floor block", "avg_premium"] == 200_000.0


def test_breakdown_is_empty_without_flow_types():
    assert flow_type_breakdown(None).empty
    assert flow_type_breakdown(pd.DataFrame()).empty
    assert flow_type_breakdown(pd.DataFrame({"premium": [1.0]})).empty


def test_min_share_filter():
    br = flow_type_breakdown(parse_file(FLOW_CSV.encode()).prints, min_share=0.10)
    assert set(br["flow_type"]) == {"floor block", "floor", "cross"}


# --- the rest of the app sees them -------------------------------------------

def test_floor_prints_reach_the_block_machinery(flow):
    """Regression: a file that tags negotiated size FLR (never 'BLOCK') used
    to produce an empty block book and no commitment levels."""
    p = flow.prints
    spot = flow.spots["XSP"]
    floor_only = p[p["flow_venue"] == "floor"]
    assert not floor_only["is_block"].all()          # nothing says BLOCK here

    lvls = block_levels(floor_only, spot)
    assert not lvls.empty
    assert set(lvls["side"]) <= {"call", "put", "mixed"}

    books = flow_books(floor_only, spot, flow.asof)
    assert "blocks" in books and "floor" in books
    assert books["blocks"].n_contracts > 0


def test_flow_books_add_a_floor_book_only_when_floor_prints_exist(flow):
    books = flow_books(flow.prints, flow.spots["XSP"], flow.asof)
    assert {"blocks", "sweeps", "floor"} <= set(books)
    no_floor = flow.prints[~flow.prints["is_floor"]]
    assert "floor" not in flow_books(no_floor, flow.spots["XSP"], flow.asof)


def test_report_lists_every_type_with_its_premium(flow):
    from dealer_gex.report import build_markdown

    a = analyze(flow.chain, flow.spots["XSP"], flow.asof)
    md = build_markdown(a, ticker="XSP",
                        flow_types=flow_type_breakdown(flow.prints))
    assert "## Consolidated flow by type" in md
    assert "| floor block ★ |" in md
    assert "| auto |" in md and "| auto ★ |" not in md   # auto is not starred
    assert "PHLX SPECIAL" in md                          # unmapped, still shown
    assert "$200.00K" in md                              # premium per type
    assert "Consolidated flow by type" not in build_markdown(a, ticker="XSP")
