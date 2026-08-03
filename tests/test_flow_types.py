"""Precision of the consolidated-flow type: FLR / AUTO / CROSS / BLOCK …"""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from dealer_gex.analytics import (
    analyze, block_dominance, block_dte_breakdown, block_levels,
    block_oi_breakdown, block_type_behaviour, block_type_breakdown,
    block_window_breakdown, block_window_summary, flow_books,
    flow_type_breakdown, institutional_mask,
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


# --- QuantData's real two-column vocabulary ----------------------------------
#
# Live exports put the *shape* in Consolidation Type (SWEEP / BLOCK / SPLIT)
# and the *mechanism* in a separate Trade Type column, with SPRD_/SPRD_LEG_
# and TIED_ prefixes modifying it.
REAL_CSV = """Trade ID,Trade Time,Ticker,Expiration Date,Strike Price,Contract Type,Reference Price,Size,Volume,Open Interest,Side Code,Implied Volatility,Gamma,Premium Price,Trade Type,Consolidation Type,Is Golden Sweep,Is Unusual,Is Opening Position
1,2026-07-08T13:30:00.100Z,QQQ,2026-08-21,$710.00,PUT,$708.60,17000,"20,000","30,000",A,22.0%,0.009,"$25,058,000.00",SPRD_FLR,BLOCK,No,Yes,No
2,2026-07-08T13:40:00.200Z,QQQ,2026-09-18,$645.00,CALL,$708.50,3000,"5,000","12,000",B,20.0%,0.006,"$25,350,000.00",SPRD_TIED_CROSS,BLOCK,No,Yes,No
3,2026-07-08T13:50:00.300Z,QQQ,2026-07-17,$775.00,PUT,$708.40,2100,"3,000","9,000",B,25.0%,0.004,"$14,084,700.00",FLR,BLOCK,No,Yes,No
4,2026-07-08T14:00:00.400Z,QQQ,2026-07-17,$700.00,CALL,$708.30,500,"9,000","20,000",A,19.0%,0.012,"$1,000,000.00",COB,BLOCK,No,No,No
5,2026-07-08T14:10:00.500Z,QQQ,2026-07-17,$700.00,CALL,$708.20,300,"9,000","20,000",A,19.0%,0.012,"$600,000.00",COB_AUCT,BLOCK,No,No,No
6,2026-07-08T14:20:00.600Z,QQQ,2026-07-17,$715.00,CALL,$708.10,50,"4,000","11,000",A,18.0%,0.011,"$60,000.00",SPRD_LEG_AUTO,BLOCK,No,No,No
7,2026-07-08T14:30:00.700Z,QQQ,2026-07-17,$715.00,CALL,$708.00,10,"4,000","11,000",B,18.0%,0.011,"$435.00",SPRD_LEG_AUTO,BLOCK,No,No,No
8,2026-07-08T14:40:00.800Z,QQQ,2026-07-17,$690.00,PUT,$707.90,224,"139,409","2,393",BB,27.5%,0.096,"$32,916.00",AUTO,SWEEP,No,No,No
9,2026-07-08T14:50:00.900Z,QQQ,2026-07-17,$690.00,PUT,$707.80,100,"139,409","2,393",A,27.5%,0.096,"$15,000.00",ISO,SWEEP,No,No,No
10,2026-07-08T15:00:00.000Z,QQQ,2026-07-17,$705.00,CALL,$707.70,900,"6,000","14,000",A,18.5%,0.012,"$4,704,000.00",CANCEL,BLOCK,No,Yes,No
"""


@pytest.fixture(scope="module")
def real():
    pf = parse_file(REAL_CSV.encode())
    assert pf.prints is not None
    return pf


def test_mechanism_is_read_from_its_own_column(real):
    """FLR/AUTO/CROSS live in Trade Type, not Consolidation Type — reading
    only the latter left every one of these prints untyped."""
    p = real.prints.sort_values("trade_time").reset_index(drop=True)
    assert list(p["mechanism"]) == [
        "floor", "cross", "floor", "cob", "cob auction", "auto", "auto",
        "auto", "iso",
    ]
    assert list(p["flow_shape"]) == ["block"] * 7 + ["sweep", "sweep"]
    assert list(p["flow_type"])[:5] == [
        "M2M FLR block", "tied multi cross block", "FLR single leg block",
        "COB block", "COB auction block",
    ]


def test_spread_legs_tied_and_packages_are_flagged(real):
    p = real.prints.set_index("trade_id" if "trade_id" in real.prints else
                              real.prints.index)
    p = real.prints.sort_values("trade_time").reset_index(drop=True)
    assert list(p["is_spread"]) == [True, True, False, False, False, True,
                                    True, False, False]
    assert list(p["is_spread_leg"]) == [False, False, False, False, False,
                                        True, True, False, False]
    assert list(p["is_tied"]) == [False, True, False, False, False, False,
                                  False, False, False]
    assert p.loc[1, "flow_type"] == "tied multi cross block"   # package, tied
    assert p.loc[5, "flow_type"] == "auto multi leg block"           # one leg only


def test_cancelled_prints_are_dropped(real):
    """A busted $4.7M block would otherwise top the notable-flow table."""
    assert len(real.prints) == 9                      # 10 rows, one CANCEL
    assert "CANCEL" not in set(real.prints["trade_type"])
    assert real.prints["premium"].max() < 25_400_000  # the cancel is gone


def test_block_tiers_rank_by_what_the_print_means(real):
    from dealer_gex.analytics import block_tier_summary, block_type_breakdown

    br = block_type_breakdown(real.prints)
    assert list(br["block_type"]) == [
        "tied multi cross block", "M2M FLR block", "FLR single leg block",
        "COB block", "COB auction block", "auto multi leg block",
    ]                                                  # ranked by premium
    tied = br.set_index("block_type").loc["tied multi cross block"]
    assert tied["tied"] and tied["tier"] == "negotiated"
    assert br.set_index("block_type").loc["auto multi leg block", "tier"] == "fragment"

    tiers = block_tier_summary(br)
    assert list(tiers["tier"]) == ["negotiated", "facilitated", "fragment"]
    assert tiers.iloc[0]["premium_share"] > 0.95       # negotiated dominates
    # the fragments are 2 of 7 block prints but a rounding error of premium
    frag = tiers.set_index("tier").loc["fragment"]
    assert frag["prints"] == 2 and frag["premium_share"] < 0.001


def test_spread_legs_are_kept_out_of_the_block_book_by_default(real):
    p = real.prints
    default = institutional_mask(p)
    kept = institutional_mask(p, drop_fragments=False)
    assert int(kept.sum() - default.sum()) == 2        # the two SPRD_LEG rows
    assert not default[p["is_spread_leg"]].any()


def test_report_ranks_block_types_by_tier(real):
    from dealer_gex.analytics import block_type_breakdown
    from dealer_gex.report import build_markdown

    a = analyze(real.chain, real.spots["QQQ"], real.asof)
    md = build_markdown(a, ticker="QQQ",
                        block_types=block_type_breakdown(real.prints))
    assert "## Block types (how the size printed)" in md
    assert "| **negotiated** |" in md and "| **fragment** |" in md
    assert "tied multi cross block (stock-tied)" in md
    assert "Read premium, not print count" in md


def test_block_types_carry_where_the_money_sat(real):
    """Premium-weighted strike per type, placed against spot — a $25M block
    5% out is a different trade from the same size at the money."""
    from dealer_gex.analytics import block_type_breakdown

    spot = real.spots["QQQ"]
    br = block_type_breakdown(real.prints, spot).set_index("block_type")
    # the two floor prints: 17,000x 710P at $25.058M and 2,100x 775P at
    # $14.0847M are separate types, so each level is its own strike
    assert br.loc["M2M FLR block", "level"] == pytest.approx(710.0)
    assert br.loc["FLR single leg block", "level"] == pytest.approx(775.0)
    assert br.loc["FLR single leg block", "distance_pct"] == pytest.approx(
        (775.0 / spot - 1) * 100)
    # ref_price is the underlying when those prints hit, not the current spot
    assert br.loc["FLR single leg block", "ref_price"] == pytest.approx(708.40)

    # a type spanning two strikes gets the premium-weighted blend
    legs = br.loc["auto multi leg block"]
    assert legs["level"] == pytest.approx(715.0)     # both legs at 715


def test_block_level_is_premium_weighted_not_a_plain_mean():
    from dealer_gex.analytics import block_type_breakdown

    csv = REAL_CSV.replace(
        "7,2026-07-08T14:30:00.700Z,QQQ,2026-07-17,$715.00,CALL,$708.00,10,"
        '"4,000","11,000",B,18.0%,0.011,"$435.00",SPRD_LEG_AUTO,BLOCK,No,No,No',
        "7,2026-07-08T14:30:00.700Z,QQQ,2026-07-17,$600.00,CALL,$708.00,10,"
        '"4,000","11,000",B,18.0%,0.011,"$435.00",SPRD_LEG_AUTO,BLOCK,No,No,No')
    br = block_type_breakdown(parse_file(csv.encode()).prints).set_index("block_type")
    lvl = br.loc["auto multi leg block", "level"]
    assert lvl == pytest.approx((715 * 60000 + 600 * 435) / 60435)
    assert lvl > 714                       # the $435 leg barely moves it
    assert lvl != pytest.approx(657.5)     # not the unweighted mean


def test_block_level_without_a_spot_falls_back_to_reference_price():
    from dealer_gex.analytics import block_type_breakdown

    pf = parse_file(REAL_CSV.encode())
    br = block_type_breakdown(pf.prints)          # no spot passed
    assert br["level"].notna().all()
    assert br["distance_pct"].notna().all()       # measured off the ref prices


# --- behaviour ---------------------------------------------------------------

BEHAVIOUR_CSV_HEADER = ("Trade ID,Trade Time,Ticker,Expiration Date,Strike Price,"
                        "Contract Type,Reference Price,Size,Volume,Open Interest,"
                        "Side Code,Implied Volatility,Gamma,Premium Price,"
                        "Trade Type,Consolidation Type\n")


def _behaviour_csv(rows) -> bytes:
    """rows: (minute, strike, cp, ref, size, side, premium, trade_type, cons)"""
    out = [BEHAVIOUR_CSV_HEADER]
    for i, (m, k, cp, ref, size, side, prem, tt, cons) in enumerate(rows, 1):
        out.append(
            f"{i},2026-07-08T{13 + m // 60:02d}:{m % 60:02d}:00.000Z,QQQ,"
            f"2026-08-21,${k:.2f},{cp},${ref:.2f},{size},\"9,000\",\"20,000\","
            f"{side},20.0%,0.010,\"${prem:,.2f}\",{tt},{cons}\n")
    return "".join(out).encode()


def test_behaviour_signs_bought_calls_and_sold_puts_as_bullish():
    """Price rises after both a bought call and a sold put: both leaned
    bullish, so both must score positive."""
    rows = [
        (0, 700, "CALL", 700.0, 100, "A", 1_000_000, "FLR", "BLOCK"),
        (1, 700, "PUT", 700.0, 100, "B", 1_000_000, "FLR", "BLOCK"),
        (2, 700, "CALL", 700.0, 100, "A", 1_000_000, "FLR", "BLOCK"),
        (90, 700, "CALL", 707.0, 1, "A", 1, "AUTO", "SWEEP"),   # price +1%
    ]
    bh = block_type_behaviour(parse_file(_behaviour_csv(rows)).prints)
    r = bh.set_index("block_type").loc["FLR single leg block"]
    assert r["scored"] == 3
    assert r["followed_close"] == pytest.approx(1.0, abs=0.01)
    assert r["hit_rate"] == 1.0


def test_behaviour_is_negative_when_the_tape_fades_the_block():
    rows = [
        (0, 700, "CALL", 700.0, 100, "A", 1_000_000, "FLR", "BLOCK"),
        (1, 700, "CALL", 700.0, 100, "A", 1_000_000, "FLR", "BLOCK"),
        (2, 700, "CALL", 700.0, 100, "A", 1_000_000, "FLR", "BLOCK"),
        (90, 700, "CALL", 693.0, 1, "A", 1, "AUTO", "SWEEP"),   # price -1%
    ]
    bh = block_type_behaviour(parse_file(_behaviour_csv(rows)).prints)
    r = bh.set_index("block_type").loc["FLR single leg block"]
    assert r["followed_close"] == pytest.approx(-1.0, abs=0.01)
    assert r["hit_rate"] == 0.0


def test_behaviour_is_premium_weighted_across_prints():
    """A $10M block that worked outweighs three $10K ones that didn't."""
    rows = [
        (0, 700, "CALL", 700.0, 1000, "A", 10_000_000, "FLR", "BLOCK"),
        (1, 700, "CALL", 710.0, 1, "A", 10_000, "FLR", "BLOCK"),
        (2, 700, "CALL", 710.0, 1, "A", 10_000, "FLR", "BLOCK"),
        (3, 700, "CALL", 710.0, 1, "A", 10_000, "FLR", "BLOCK"),
        (90, 700, "CALL", 707.0, 1, "A", 1, "AUTO", "SWEEP"),
    ]
    bh = block_type_behaviour(parse_file(_behaviour_csv(rows)).prints)
    r = bh.set_index("block_type").loc["FLR single leg block"]
    assert r["hit_rate"] == pytest.approx(0.25)     # 1 of 4 prints was right
    assert r["followed_close"] > 0.9               # ...but it was ~all the money


def test_late_prints_have_no_horizon_and_are_not_scored_as_zero():
    """A block printed at the bell has no 30 minutes left to be judged on.
    Scoring it zero would silently dilute the type it belongs to."""
    rows = [
        (0, 700, "CALL", 700.0, 100, "A", 1_000_000, "FLR", "BLOCK"),
        (20, 700, "CALL", 707.0, 1, "A", 1, "AUTO", "SWEEP"),     # +1%
        (100, 700, "CALL", 707.0, 100, "A", 1_000_000, "FLR", "BLOCK"),  # last print
    ]
    p = parse_file(_behaviour_csv(rows)).prints
    bh = block_type_behaviour(p, horizon_min=30, min_prints=1).set_index("block_type")
    r = bh.loc["FLR single leg block"]
    assert r["scored"] == 2                                    # both reach a close
    # to the close: +1% for the early print, 0% for the late one -> +0.5%
    assert r["followed_close"] == pytest.approx(0.5, abs=0.01)
    # at +30 min only the early print is measurable, and it is the full +1%
    assert r["followed_horizon"] == pytest.approx(1.0, abs=0.01)


def test_thin_types_are_flagged_and_left_unscored():
    rows = [
        (0, 700, "CALL", 700.0, 100, "A", 1_000_000, "FLR", "BLOCK"),
        (1, 700, "CALL", 700.0, 100, "A", 1_000_000, "CROSS", "BLOCK"),
        (90, 700, "CALL", 707.0, 1, "A", 1, "AUTO", "SWEEP"),
    ]
    bh = block_type_behaviour(parse_file(_behaviour_csv(rows)).prints,
                              min_prints=3).set_index("block_type")
    assert bh.loc["FLR single leg block", "sample"] == "thin"
    assert pd.isna(bh.loc["FLR single leg block", "followed_close"])
    assert bh.loc["FLR single leg block", "scored"] == 1        # counted, just not read


def test_every_block_type_carries_a_behavioural_read(real):
    bh = block_type_behaviour(real.prints).set_index("block_type")
    assert (bh["behaviour"].str.len() > 40).all()
    assert "find the other side" in bh.loc["FLR single leg block", "behaviour"]
    assert "arranged before the print" in bh.loc["tied multi cross block", "behaviour"]
    assert "delta was hedged" in bh.loc["tied multi cross block", "behaviour"]
    # a leg gets the fragment warning instead of a mechanism read
    leg = bh.loc["auto multi leg block", "behaviour"]
    assert "Do not read it directionally" in leg
    assert "default electronic route" not in leg


def test_behaviour_without_timestamps_still_explains_the_types():
    p = parse_file(FLOW_CSV.encode()).prints.drop(columns=["trade_time"])
    bh = block_type_behaviour(p)
    assert (bh["sample"] == "no timestamps").all()
    assert bh["scored"].sum() == 0
    assert (bh["behaviour"].str.len() > 40).all()


def test_report_includes_behaviour(real):
    from dealer_gex.analytics import block_type_breakdown
    from dealer_gex.report import build_markdown

    a = analyze(real.chain, real.spots["QQQ"], real.asof)
    md = build_markdown(a, ticker="QQQ",
                        block_types=block_type_breakdown(real.prints),
                        block_prints=real.prints)
    assert "### Behaviour — did the tape follow?" in md
    assert "**What each type means**" in md
    assert "find the other side" in md


# --- the open-interest view --------------------------------------------------

def test_open_interest_is_max_per_contract_never_summed_over_prints():
    """Three blocks on one 5,000-OI contract is 5,000 of standing book, not
    15,000 — OI belongs to the contract, not the print."""
    rows = [
        (0, 700, "CALL", 700.0, 100, "A", 100_000, "FLR", "BLOCK"),
        (1, 700, "CALL", 700.0, 200, "A", 200_000, "FLR", "BLOCK"),
        (2, 700, "CALL", 700.0, 300, "A", 300_000, "FLR", "BLOCK"),
    ]
    p = parse_file(_behaviour_csv(rows)).prints
    assert p["open_interest"].sum() == 60_000        # the naive number
    oi = block_oi_breakdown(p).set_index("block_type")
    assert oi.loc["FLR single leg block", "open_interest"] == 20_000    # one contract
    assert oi.loc["FLR single leg block", "contracts_touched"] == 1
    assert oi.loc["FLR single leg block", "traded"] == 600
    assert oi.loc["FLR single leg block", "add_ratio"] == pytest.approx(600 / 20_000)


def test_open_interest_sums_across_distinct_contracts():
    rows = [
        (0, 700, "CALL", 700.0, 100, "A", 100_000, "FLR", "BLOCK"),
        (1, 710, "CALL", 700.0, 100, "A", 100_000, "FLR", "BLOCK"),
        (2, 700, "PUT", 700.0, 100, "A", 100_000, "FLR", "BLOCK"),
    ]
    oi = block_oi_breakdown(parse_file(_behaviour_csv(rows)).prints)
    row = oi.set_index("block_type").loc["FLR single leg block"]
    assert row["contracts_touched"] == 3            # 700C, 710C, 700P
    assert row["open_interest"] == 60_000


def test_add_ratio_flags_a_position_being_built():
    """Same premium, opposite meaning: heavy size on a thin strike is a new
    position; the same size inside a crowded one is noise."""
    rows = [
        (0, 700, "CALL", 700.0, 5000, "A", 1_000_000, "FLR", "BLOCK"),
        (1, 710, "CALL", 700.0, 5000, "A", 1_000_000, "AUTO", "BLOCK"),
    ]
    csv = _behaviour_csv(rows).decode()
    csv = csv.replace('5000,"9,000","20,000",A,20.0%,0.010,"$1,000,000.00",FLR',
                      '5000,"9,000","5,000",A,20.0%,0.010,"$1,000,000.00",FLR')
    csv = csv.replace('5000,"9,000","20,000",A,20.0%,0.010,"$1,000,000.00",AUTO',
                      '5000,"9,000","500,000",A,20.0%,0.010,"$1,000,000.00",AUTO')
    oi = block_oi_breakdown(parse_file(csv.encode()).prints).set_index("block_type")
    assert oi.loc["FLR single leg block", "add_ratio"] == pytest.approx(1.0)    # doubled it
    assert oi.loc["auto single leg block", "add_ratio"] == pytest.approx(0.01)    # noise
    # ...and the OI ranking is the reverse of the premium ranking here
    assert list(oi.index) == ["auto single leg block", "FLR single leg block"]


def test_oi_level_is_open_interest_weighted_not_premium_weighted():
    """Where the standing book sits is not always where the premium went."""
    rows = [
        (0, 600, "CALL", 700.0, 10, "A", 5_000_000, "FLR", "BLOCK"),
        (1, 800, "CALL", 700.0, 10, "A", 100, "FLR", "BLOCK"),
    ]
    csv = _behaviour_csv(rows).decode()
    csv = csv.replace('10,"9,000","20,000",A,20.0%,0.010,"$5,000,000.00"',
                      '10,"9,000","1,000",A,20.0%,0.010,"$5,000,000.00"')
    csv = csv.replace('10,"9,000","20,000",A,20.0%,0.010,"$100.00"',
                      '10,"9,000","99,000",A,20.0%,0.010,"$100.00"')
    p = parse_file(csv.encode()).prints
    oi = block_oi_breakdown(p, spot=700.0).set_index("block_type")
    prem = block_type_breakdown(p, spot=700.0).set_index("block_type")
    # premium is all at the 600 strike, open interest almost all at 800
    assert prem.loc["FLR single leg block", "level"] == pytest.approx(600.0, abs=1.0)
    assert oi.loc["FLR single leg block", "level"] == pytest.approx(798.0, abs=1.0)
    assert oi.loc["FLR single leg block", "distance_pct"] == pytest.approx(14.0, abs=0.3)


def test_opening_share_is_size_weighted(real):
    oi = block_oi_breakdown(real.prints).set_index("block_type")
    assert 0.0 <= oi["opening_share"].max() <= 1.0
    assert oi["oi_share"].sum() == pytest.approx(1.0)


def test_oi_breakdown_handles_missing_inputs():
    assert block_oi_breakdown(None).empty
    assert block_oi_breakdown(pd.DataFrame()).empty
    assert block_oi_breakdown(pd.DataFrame({"is_block": [True]})).empty


def test_report_carries_the_oi_table(real):
    from dealer_gex.analytics import block_type_breakdown
    from dealer_gex.report import build_markdown

    a = analyze(real.chain, real.spots["QQQ"], real.asof)
    md = build_markdown(a, ticker="QQQ",
                        block_types=block_type_breakdown(real.prints),
                        block_prints=real.prints)
    assert "### By open interest — what the flow landed on" in md
    assert "| Block type | Open interest | Contracts | Traded | Add |" in md
    assert "never summed over" in md


# --- who is dominant ---------------------------------------------------------

def test_dominance_names_a_leader_that_wins_two_lenses():
    """One type with the premium and the book impact, another with the raw
    open interest: two of three lenses is a clear leader."""
    rows = [
        (0, 700, "CALL", 700.0, 5000, "A", 20_000_000, "FLR", "BLOCK"),
        (1, 710, "CALL", 700.0, 100, "A", 100_000, "AUTO", "BLOCK"),
    ]
    csv = _behaviour_csv(rows).decode()
    csv = csv.replace('5000,"9,000","20,000",A,20.0%,0.010,"$20,000,000.00",FLR',
                      '5000,"9,000","5,000",A,20.0%,0.010,"$20,000,000.00",FLR')
    csv = csv.replace('100,"9,000","20,000",A,20.0%,0.010,"$100,000.00",AUTO',
                      '100,"9,000","900,000",A,20.0%,0.010,"$100,000.00",AUTO')
    d = block_dominance(parse_file(csv.encode()).prints, spot=700.0)
    assert d["verdict"] == "clear"
    assert d["leader"] == "FLR single leg block"
    assert d["lenses_won"] == 2
    assert d["by_money"]["block_type"] == "FLR single leg block"
    assert d["by_impact"]["block_type"] == "FLR single leg block"
    assert d["by_book"]["block_type"] == "auto single leg block"     # the loser's lens
    assert "dominates" in d["label"] and "FLR single leg block" in d["label"]
    assert "auto single leg block on open interest" in d["label"]


def test_dominance_calls_a_split_book_split():
    """Three types, one lens each — no leader, and saying otherwise would be
    inventing one."""
    rows = [
        (0, 700, "CALL", 700.0, 100, "A", 9_000_000, "FLR", "BLOCK"),
        (1, 710, "CALL", 700.0, 100, "A", 1_000_000, "AUTO", "BLOCK"),
        (2, 720, "CALL", 700.0, 9000, "A", 1_000_000, "COB", "BLOCK"),
    ]
    csv = _behaviour_csv(rows).decode()
    csv = csv.replace('100,"9,000","20,000",A,20.0%,0.010,"$9,000,000.00",FLR',
                      '100,"9,000","50,000",A,20.0%,0.010,"$9,000,000.00",FLR')
    csv = csv.replace('100,"9,000","20,000",A,20.0%,0.010,"$1,000,000.00",AUTO',
                      '100,"9,000","900,000",A,20.0%,0.010,"$1,000,000.00",AUTO')
    csv = csv.replace('9000,"9,000","20,000",A,20.0%,0.010,"$1,000,000.00",COB',
                      '9000,"9,000","50,000",A,20.0%,0.010,"$1,000,000.00",COB')
    d = block_dominance(parse_file(csv.encode()).prints, spot=700.0)
    assert d["verdict"] == "split"
    assert d["leader"] is None
    assert d["label"].startswith("Split book —")
    assert len({d[k]["block_type"] for k in ("by_money", "by_book", "by_impact")}) == 3


def test_a_tiny_type_cannot_win_impact_on_a_rounding_error():
    """Two prints with a 200% add ratio should not out-rank the book."""
    rows = [
        (0, 700, "CALL", 700.0, 50_000, "A", 50_000_000, "COB", "BLOCK"),
        (1, 710, "CALL", 700.0, 200, "A", 1_000, "FLR", "BLOCK"),
    ]
    csv = _behaviour_csv(rows).decode()
    csv = csv.replace('50000,"9,000","20,000",A,20.0%,0.010,"$50,000,000.00",COB',
                      '50000,"9,000","500,000",A,20.0%,0.010,"$50,000,000.00",COB')
    csv = csv.replace('200,"9,000","20,000",A,20.0%,0.010,"$1,000.00",FLR',
                      '200,"9,000","100",A,20.0%,0.010,"$1,000.00",FLR')
    p = parse_file(csv.encode()).prints
    oi = block_oi_breakdown(p).set_index("block_type")
    assert oi.loc["FLR single leg block", "add_ratio"] == pytest.approx(2.0)   # 200%
    d = block_dominance(p, spot=700.0)
    assert d["by_impact"]["block_type"] == "COB block"   # the floor type is too small
    assert d["leader"] == "COB block"


def test_dominance_label_reports_the_leading_tier(real):
    d = block_dominance(real.prints, real.spots["QQQ"])
    assert d["tier_leader"] == "negotiated"
    assert "Negotiated size is" in d["label"]
    assert d["label"].endswith(".")


def test_dominance_is_empty_without_blocks():
    assert block_dominance(None) == {}
    assert block_dominance(pd.DataFrame()) == {}
    sweeps = parse_file(FLOW_CSV.encode()).prints
    assert block_dominance(sweeps[~sweeps["is_block"]]) == {}


def test_report_names_who_is_dominant(real):
    from dealer_gex.report import build_markdown

    a = analyze(real.chain, real.spots["QQQ"], real.asof)
    md = build_markdown(a, ticker="QQQ",
                        block_types=block_type_breakdown(real.prints, a.spot),
                        block_prints=real.prints)
    assert "**Who is dominant:**" in md
    assert "| Lens | Leader | Reading |" in md
    assert "Most premium" in md and "Biggest book impact" in md
    assert "👑" in md


# --- the platform's own names ------------------------------------------------

def test_export_codes_are_shown_under_the_names_the_platform_uses():
    """The CSV says SPRD_FLR; the trader is looking for "M2M FLR". Rows have
    to be findable by the name on the screen they came from."""
    from dealer_gex.parsing import classify_trade_type

    out = classify_trade_type(pd.Series([
        "FLR", "SPRD_FLR", "SPRD_LEG_FLR", "TIED_FLR",
        "CROSS", "TIED_CROSS", "SPRD_TIED_CROSS",
        "AUTO", "SPRD_LEG_AUTO", "COB", "COB_AUCT", "AUCT", "ISO",
    ]))
    assert list(out["trade_label"]) == [
        "FLR single leg", "M2M FLR", "FLR multi leg", "tied FLR",
        "cross single leg", "tied cross", "tied multi cross",
        "auto single leg", "auto multi leg", "COB", "COB auction",
        "auction", "ISO",
    ]


def test_the_three_names_that_went_missing_are_findable(real):
    """M2M FLR, FLR single leg and multi cross were all in the file — under
    codes that read nothing like them."""
    p = real.prints
    by_code = dict(zip(p["trade_type"], p["flow_type"]))
    assert by_code["SPRD FLR"] == "M2M FLR block"
    assert by_code["FLR"] == "FLR single leg block"
    assert by_code["SPRD TIED CROSS"] == "tied multi cross block"


def test_the_raw_code_travels_with_every_row(real):
    """A renamed row is only trustworthy if it points back at the CSV."""
    br = block_type_breakdown(real.prints).set_index("block_type")
    assert br.loc["M2M FLR block", "code"] == "SPRD FLR"
    assert br.loc["FLR single leg block", "code"] == "FLR"
    oi = block_oi_breakdown(real.prints).set_index("block_type")
    assert oi.loc["tied multi cross block", "code"] == "SPRD TIED CROSS"


def test_an_unknown_code_still_gets_a_readable_label():
    """An export with a vocabulary we have not mapped must stay legible
    rather than falling back to the bare code."""
    from dealer_gex.parsing import classify_trade_type

    out = classify_trade_type(pd.Series(["SPRD_LEG_XYZDESK", "WEIRD_FLR"]))
    assert list(out["trade_label"]) == ["", ""]        # no platform name
    assert list(out["mechanism"]) == ["", "floor"]     # ...but still classified
    assert list(out["is_spread_leg"]) == [True, False]
    assert list(out["block_tier"]) == ["fragment", "negotiated"]

    csv = REAL_CSV.replace("SPRD_FLR,BLOCK", "MYSTERY_FLR,BLOCK")
    p = parse_file(csv.encode())
    labels = dict(zip(p.prints["trade_type"], p.prints["flow_type"]))
    assert labels["MYSTERY FLR"] == "floor block"      # compositional fallback


# --- days to expiration ------------------------------------------------------

def test_dte_is_read_from_the_export_when_it_has_a_column(real):
    assert real.prints["dte"].notna().all()
    by_code = real.prints.set_index("trade_type")["dte"]
    assert by_code.loc["FLR"] == 9        # 2026-07-08 -> 2026-07-17


def test_dte_is_derived_from_the_expiry_when_there_is_no_column(flow):
    """A file without a DTE column still gets one, from expiry minus the
    print's own date — not from today."""
    assert flow.prints["dte"].notna().all()
    assert (flow.prints["dte"] == 38).all()   # 2026-07-14 -> 2026-08-21


def test_premium_weighted_dte_is_not_the_median_print():
    """Ten 0DTE lottery tickets and one big long-dated block: the median
    print says 0, the money says months out. Both get reported."""
    rows = [
        (0, 700, "CALL", 700.0, 5000, "A", 10_000_000, "FLR", "BLOCK"),
        (1, 700, "CALL", 700.0, 1, "A", 1_000, "FLR", "BLOCK"),
        (2, 700, "CALL", 700.0, 1, "A", 1_000, "FLR", "BLOCK"),
    ]
    csv = _behaviour_csv(rows).decode().replace(
        "2026-08-21,$700.00,CALL,$700.00,5000", "2027-01-15,$700.00,CALL,$700.00,5000")
    br = block_type_breakdown(parse_file(csv.encode()).prints).set_index("block_type")
    r = br.loc["FLR single leg block"]
    assert r["dte_median"] == pytest.approx(44)       # most prints are near-dated
    assert r["dte"] > 180                             # the premium is not
    assert r["horizon"] == "LEAP"


def test_horizon_buckets_name_the_tenor():
    from dealer_gex.analytics import horizon_label

    assert horizon_label(0) == "0DTE"
    assert horizon_label(1) == "weekly" and horizon_label(7) == "weekly"
    assert horizon_label(8) == "monthly" and horizon_label(45) == "monthly"
    assert horizon_label(46) == "quarterly" and horizon_label(180) == "quarterly"
    assert horizon_label(181) == "LEAP" and horizon_label(892) == "LEAP"
    assert horizon_label(float("nan")) == "" and horizon_label(None) == ""
    assert horizon_label(-1) == ""


def test_oi_table_weights_dte_by_open_interest_not_premium():
    """The book's horizon and the money's horizon are different questions."""
    rows = [
        (0, 700, "CALL", 700.0, 10, "A", 9_000_000, "FLR", "BLOCK"),
        (1, 800, "CALL", 700.0, 10, "A", 1_000, "FLR", "BLOCK"),
    ]
    csv = _behaviour_csv(rows).decode()
    csv = csv.replace('10,"9,000","20,000",A,20.0%,0.010,"$9,000,000.00"',
                      '10,"9,000","1,000",A,20.0%,0.010,"$9,000,000.00"')
    csv = csv.replace('2026-08-21,$800.00,CALL,$700.00,10,"9,000","20,000"',
                      '2027-01-15,$800.00,CALL,$700.00,10,"9,000","999,000"')
    p = parse_file(csv.encode()).prints
    prem = block_type_breakdown(p).set_index("block_type")
    oi = block_oi_breakdown(p).set_index("block_type")
    # premium sits in the near-dated strike, open interest in the far one
    assert prem.loc["FLR single leg block", "horizon"] == "monthly"
    assert oi.loc["FLR single leg block", "horizon"] == "LEAP"


def test_dte_separates_near_dated_flow_from_structural_positions(real):
    """The point of the column: same tier, opposite horizons."""
    br = block_type_breakdown(real.prints).set_index("block_type")
    assert br.loc["auto multi leg block", "dte"] == 9         # nine days out
    assert br.loc["auto multi leg block", "horizon"] == "monthly"
    assert br.loc["tied multi cross block", "dte"] == 72      # months out
    assert br.loc["tied multi cross block", "horizon"] == "quarterly"


def test_report_shows_dte_in_both_block_tables(real):
    from dealer_gex.report import build_markdown

    a = analyze(real.chain, real.spots["QQQ"], real.asof)
    md = build_markdown(a, ticker="QQQ",
                        block_types=block_type_breakdown(real.prints, a.spot),
                        block_prints=real.prints)
    assert "| Median print | DTE | Level |" in md
    assert "| Add | Opening | DTE | Level |" in md
    assert "open-interest-weighted" in md


# --- the term-structure table ------------------------------------------------

def test_dte_table_is_in_tenor_order_not_size_order():
    """A term structure read biggest-first is not a term structure."""
    rows = [
        (0, 700, "CALL", 700.0, 10, "A", 1_000, "FLR", "BLOCK"),
        (1, 700, "CALL", 700.0, 10, "A", 50_000_000, "FLR", "BLOCK"),
    ]
    csv = _behaviour_csv(rows).decode()
    csv = csv.replace("2026-08-21,$700.00,CALL,$700.00,10,\"9,000\",\"20,000\",A,20.0%,0.010,\"$1,000.00\"",
                      "2026-07-08,$700.00,CALL,$700.00,10,\"9,000\",\"20,000\",A,20.0%,0.010,\"$1,000.00\"")
    d = block_dte_breakdown(parse_file(csv.encode()).prints)
    assert list(d["horizon"]) == ["0DTE", "monthly"]      # not $50M first
    assert d.iloc[0]["premium"] == 1_000.0
    assert list(d["dte_range"]) == ["0", "8–45"]


def test_dte_table_counts_open_interest_once_per_contract():
    """Three prints on one contract inside a bucket contribute its book
    once, same rule as the per-type table."""
    rows = [
        (0, 700, "CALL", 700.0, 100, "A", 100_000, "FLR", "BLOCK"),
        (1, 700, "CALL", 700.0, 100, "A", 100_000, "AUTO", "BLOCK"),
        (2, 710, "CALL", 700.0, 100, "A", 100_000, "COB", "BLOCK"),
    ]
    d = block_dte_breakdown(parse_file(_behaviour_csv(rows)).prints)
    row = d.set_index("horizon").loc["monthly"]
    assert row["prints"] == 3
    assert row["open_interest"] == 40_000          # 700C and 710C, once each
    assert row["contracts"] == 300
    assert row["add_ratio"] == pytest.approx(300 / 40_000)


def test_dte_table_names_who_owns_each_bucket():
    rows = [
        (0, 700, "CALL", 700.0, 10, "A", 9_000_000, "FLR", "BLOCK"),
        (1, 700, "CALL", 700.0, 10, "A", 1_000_000, "AUTO", "BLOCK"),
    ]
    d = block_dte_breakdown(parse_file(_behaviour_csv(rows)).prints)
    row = d.set_index("horizon").loc["monthly"]
    assert row["top_type"] == "FLR single leg block"
    assert row["top_share"] == pytest.approx(0.9)


def test_dte_table_splits_print_count_from_premium(real):
    """The split the table exists to show: near-dated buckets can hold most
    of the prints and least of the money."""
    d = block_dte_breakdown(real.prints).set_index("horizon")
    assert set(d.index) <= {"0DTE", "weekly", "monthly", "quarterly", "LEAP"}
    assert d["premium_share"].sum() == pytest.approx(1.0)
    # every bucket that exists is ranked and labelled
    assert (d["dte_range"] != "").all()


def test_dte_table_direction_and_level_are_per_bucket():
    rows = [
        (0, 800, "CALL", 700.0, 100, "A", 1_000_000, "FLR", "BLOCK"),
        (1, 600, "PUT", 700.0, 100, "A", 1_000_000, "FLR", "BLOCK"),
    ]
    csv = _behaviour_csv(rows).decode().replace(
        "2026-08-21,$600.00,PUT", "2027-01-15,$600.00,PUT")
    d = block_dte_breakdown(parse_file(csv.encode()).prints,
                            spot=700.0).set_index("horizon")
    assert d.loc["monthly", "level"] == pytest.approx(800.0)
    assert d.loc["monthly", "distance_pct"] == pytest.approx(14.3, abs=0.1)
    assert d.loc["LEAP", "level"] == pytest.approx(600.0)
    assert d.loc["LEAP", "distance_pct"] == pytest.approx(-14.3, abs=0.1)
    assert d.loc["monthly", "direction"] == "bought"


def test_dte_table_handles_missing_inputs():
    assert block_dte_breakdown(None).empty
    assert block_dte_breakdown(pd.DataFrame()).empty
    p = parse_file(FLOW_CSV.encode()).prints.drop(columns=["dte"])
    assert block_dte_breakdown(p).empty


def test_report_carries_the_expiration_table(real):
    from dealer_gex.report import build_markdown

    a = analyze(real.chain, real.spots["QQQ"], real.asof)
    md = build_markdown(a, ticker="QQQ",
                        block_types=block_type_breakdown(real.prints, a.spot),
                        block_prints=real.prints)
    assert "### By expiration — where in time the size sits" in md
    assert "| Horizon | Premium | % | Prints |" in md
    assert "In tenor order, not size order" in md


# --- tenor windows -----------------------------------------------------------

def test_windows_are_cumulative_not_exclusive():
    """"What is in play this week" includes today. Weekly contains 0DTE and
    monthly contains both — the rows nest rather than partition."""
    rows = [
        (0, 700, "CALL", 700.0, 10, "A", 1_000_000, "FLR", "BLOCK"),   # 0DTE
        (1, 700, "CALL", 700.0, 10, "A", 2_000_000, "AUTO", "BLOCK"),  # 44d
    ]
    csv = _behaviour_csv(rows).decode().replace(
        '2026-08-21,$700.00,CALL,$700.00,10,"9,000","20,000",A,20.0%,0.010,"$1,000,000.00"',
        '2026-07-08,$700.00,CALL,$700.00,10,"9,000","20,000",A,20.0%,0.010,"$1,000,000.00"')
    p = parse_file(csv.encode()).prints
    w = block_window_summary(p).set_index("window")
    assert w.loc["0DTE", "premium"] == 1_000_000.0
    assert w.loc["Weekly", "premium"] == 1_000_000.0      # the 44d print is out
    assert w.loc["Monthly", "premium"] == 3_000_000.0     # both are in
    assert w.loc["Monthly", "prints"] == 2


def test_weekly_window_is_zero_to_ten_days_inclusive():
    from dealer_gex.analytics import BLOCK_WINDOWS

    assert BLOCK_WINDOWS == {"0DTE": 0, "Weekly": 10, "Monthly": 45}
    rows = [(i, 700, "CALL", 700.0, 10, "A", 1_000, "FLR", "BLOCK")
            for i in range(2)]
    csv = _behaviour_csv(rows).decode()
    # one print at exactly 10 days, one at 11
    csv = csv.replace("2026-08-21,$700.00,CALL,$700.00,10,\"9,000\",\"20,000\",A,20.0%,0.010,\"$1,000.00\",FLR,BLOCK\n"
                      "2,2026-07-08T13:01:00.000Z,QQQ,2026-08-21",
                      "2026-07-18,$700.00,CALL,$700.00,10,\"9,000\",\"20,000\",A,20.0%,0.010,\"$1,000.00\",FLR,BLOCK\n"
                      "2,2026-07-08T13:01:00.000Z,QQQ,2026-07-19")
    p = parse_file(csv.encode()).prints
    assert sorted(p["dte"]) == [10, 11]
    w = block_window_breakdown(p, 10)
    assert w["prints"].sum() == 1            # the 11-day print is excluded


def test_window_share_is_of_the_window_and_of_the_whole_book():
    """A type owning a tiny window must not read as owning the book."""
    rows = [
        (0, 700, "CALL", 700.0, 10, "A", 100_000, "FLR", "BLOCK"),      # 0DTE
        (1, 700, "CALL", 700.0, 10, "A", 9_900_000, "AUTO", "BLOCK"),   # 44d
    ]
    csv = _behaviour_csv(rows).decode().replace(
        '2026-08-21,$700.00,CALL,$700.00,10,"9,000","20,000",A,20.0%,0.010,"$100,000.00"',
        '2026-07-08,$700.00,CALL,$700.00,10,"9,000","20,000",A,20.0%,0.010,"$100,000.00"')
    p = parse_file(csv.encode()).prints
    w = block_window_breakdown(p, 0).set_index("block_type")
    r = w.loc["FLR single leg block"]
    assert r["share"] == pytest.approx(1.0)             # all of the 0DTE window
    assert r["share_of_book"] == pytest.approx(0.01)    # 1% of the block book


def test_window_table_carries_premium_and_book_side_by_side(real):
    w = block_window_breakdown(real.prints, 45, real.spots["QQQ"])
    assert not w.empty
    for c in ("premium", "open_interest", "add_ratio", "dte", "level",
              "distance_pct", "direction", "code"):
        assert c in w.columns
    assert (w["dte"] <= 45).all()
    assert w["premium"].is_monotonic_decreasing


def test_window_summary_names_the_owner_of_each_window(real):
    w = block_window_summary(real.prints).set_index("window")
    assert list(w.index) == ["0DTE", "Weekly", "Monthly"]
    monthly = w.loc["Monthly"]
    assert monthly["top_type"]
    assert 0 < monthly["top_share"] <= 1
    assert 0 < monthly["share_of_book"] <= 1


def test_window_helpers_handle_missing_inputs():
    assert block_window_breakdown(None, 10).empty
    assert block_window_summary(None).empty
    assert block_window_summary(pd.DataFrame()).empty
    p = parse_file(FLOW_CSV.encode()).prints
    assert block_window_breakdown(p, 0).empty        # nothing expires today


def test_report_lists_the_windows(real):
    from dealer_gex.report import build_markdown

    a = analyze(real.chain, real.spots["QQQ"], real.asof)
    md = build_markdown(a, ticker="QQQ",
                        block_types=block_type_breakdown(real.prints, a.spot),
                        block_prints=real.prints)
    assert "### In play by window" in md
    assert "| Window | Premium | % of block book |" in md
    assert "the rows nest" in md


# --- records that are not trades ---------------------------------------------

def test_adjustment_records_are_dropped_like_cancels():
    """ADJ_LAST is a correction to a print already counted, not new size.
    Kept, it lands in the tables as an untyped block with a nonsense tenor."""
    from dealer_gex.parsing import classify_trade_type

    out = classify_trade_type(pd.Series(["ADJ_LAST", "CANCEL", "AUTO"]))
    assert list(out["is_adjustment"]) == [True, False, False]
    assert list(out["is_cancelled"]) == [False, True, False]
    assert list(out["block_tier"]) == ["adjustment", "cancelled", "electronic"]

    csv = REAL_CSV.replace("SPRD_LEG_AUTO,BLOCK", "ADJ_LAST,BLOCK", 1)
    p = parse_file(csv.encode()).prints
    assert "ADJ LAST" not in set(p["trade_type"])
    assert len(p) == 8              # the CANCEL row and the ADJ row both gone


def test_expected_move_survives_signed_flow_weights():
    """Regression: signed weights went straight into np.average, and mixed
    signs summing positive slipped past the "sum > 0" guard — two strikes at
    20% and 60% IV weighted +1000/-900 gave an implied vol of -3.4 and an
    expected move of -105 on a spot of 100."""
    chain = pd.DataFrame({
        "strike": [99.0, 101.0], "type": ["C", "C"],
        "open_interest": [100.0, 100.0], "iv": [0.20, 0.60],
        "net_customer_size": [-1000.0, 900.0],
        "volume": [0.0, 0.0], "gamma": [np.nan, np.nan],
        "expiry": pd.to_datetime(["2026-08-21"] * 2),
    })
    a = analyze(chain, 100.0, date(2026, 7, 17), weight="flow")
    assert a.expected_move is not None
    assert a.expected_move > 0
    # the blend is an activity-weighted mean, so it stays inside the book's IVs
    implied = a.expected_move / (100.0 * np.sqrt(35 / 365))
    assert 0.20 <= implied <= 0.60


def test_expected_move_matches_oi_weighting_when_flow_is_one_sided():
    chain = pd.DataFrame({
        "strike": [100.0, 100.0], "type": ["C", "P"],
        "open_interest": [500.0, 500.0], "iv": [0.30, 0.30],
        "net_customer_size": [-500.0, -500.0],
        "volume": [0.0, 0.0], "gamma": [np.nan, np.nan],
        "expiry": pd.to_datetime(["2026-08-21"] * 2),
    })
    flow = analyze(chain, 100.0, date(2026, 7, 17), weight="flow")
    oi = analyze(chain, 100.0, date(2026, 7, 17))
    assert flow.expected_move == pytest.approx(oi.expected_move)
