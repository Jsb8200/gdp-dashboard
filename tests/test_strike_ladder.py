"""Where the block premium and quantity sit, by strike."""

import numpy as np
import pandas as pd
import pytest

from dealer_gex.analytics import analyze, block_strike_ladder, fmt_dollars
from dealer_gex.parsing import parse_file

from test_flow_types import REAL_CSV, _behaviour_csv


def test_premium_and_quantity_rank_differently():
    """A cheap far strike can carry the size while a rich near strike
    carries the money. One ranking would hide whichever it is not."""
    rows = [
        # strike 800: 50,000 lots for a small premium
        (0, 800, "CALL", 700.0, 50_000, "A", 500_000, "FLR", "BLOCK"),
        # strike 600: 100 lots for a large one
        (1, 600, "CALL", 700.0, 100, "A", 10_000_000, "FLR", "BLOCK"),
    ]
    p = parse_file(_behaviour_csv(rows)).prints
    by_prem = block_strike_ladder(p, 700.0, top_n=1, rank="premium")
    by_qty = block_strike_ladder(p, 700.0, top_n=1, rank="contracts")
    assert by_prem.iloc[0]["strike"] == 600.0
    assert by_qty.iloc[0]["strike"] == 800.0
    # both columns are present either way, so the mismatch is visible
    assert by_prem.iloc[0]["contracts"] == 100
    assert by_qty.iloc[0]["premium"] == 500_000.0


def test_rows_are_raw_strikes_not_smoothed_levels():
    rows = [(i, k, "CALL", 700.0, 10, "A", 1_000, "FLR", "BLOCK")
            for i, k in enumerate((695, 700, 705))]
    L = block_strike_ladder(parse_file(_behaviour_csv(rows)).prints, 700.0)
    assert sorted(L["strike"]) == [695.0, 700.0, 705.0]     # on the grid, exactly
    assert L["distance_pct"].abs().max() < 1.0


def test_open_interest_is_summed_over_contracts_at_the_strike():
    """Two prints on one contract count its book once; a call and a put at
    the same strike are two contracts and count twice."""
    rows = [
        (0, 700, "CALL", 700.0, 10, "A", 1_000, "FLR", "BLOCK"),
        (1, 700, "CALL", 700.0, 10, "A", 1_000, "AUTO", "BLOCK"),
        (2, 700, "PUT", 700.0, 10, "A", 1_000, "COB", "BLOCK"),
    ]
    L = block_strike_ladder(parse_file(_behaviour_csv(rows)).prints, 700.0)
    row = L.set_index("strike").loc[700.0]
    assert row["prints"] == 3
    assert row["open_interest"] == 40_000        # 20k call + 20k put
    assert row["contracts"] == 30
    assert row["add_ratio"] == pytest.approx(30 / 40_000)


def test_side_split_reports_calls_puts_or_mixed():
    def side(rows):
        L = block_strike_ladder(parse_file(_behaviour_csv(rows)).prints, 700.0)
        return L.set_index("strike").loc[700.0, "side"]

    assert side([(0, 700, "CALL", 700.0, 10, "A", 1_000, "FLR", "BLOCK")]) == "calls"
    assert side([(0, 700, "PUT", 700.0, 10, "A", 1_000, "FLR", "BLOCK")]) == "puts"
    assert side([(0, 700, "CALL", 700.0, 10, "A", 1_000, "FLR", "BLOCK"),
                 (1, 700, "PUT", 700.0, 10, "A", 1_000, "FLR", "BLOCK")]) == "mixed"


def test_shares_are_of_the_whole_block_book_not_the_top_n():
    """Truncating to the top rows must not renormalize the shares — a strike
    holding 5% of the book still reads 5% in a ten-row table."""
    rows = [(i, 700 + i, "CALL", 700.0, 10, "A", 1_000, "FLR", "BLOCK")
            for i in range(20)]
    p = parse_file(_behaviour_csv(rows)).prints
    L = block_strike_ladder(p, 700.0, top_n=5)
    assert len(L) == 5
    assert L["premium_share"].sum() == pytest.approx(5 / 20)


def test_near_filter_restricts_to_a_window():
    rows = [
        (0, 700, "CALL", 700.0, 10, "A", 1_000, "FLR", "BLOCK"),
        (1, 900, "CALL", 700.0, 10, "A", 9_000, "FLR", "BLOCK"),
    ]
    p = parse_file(_behaviour_csv(rows)).prints
    assert len(block_strike_ladder(p, 700.0)) == 2
    near = block_strike_ladder(p, 700.0, near_pct=0.05)
    assert list(near["strike"]) == [700.0]


def test_dte_is_premium_weighted_per_strike():
    """One strike can host a 0DTE trade and a LEAP; the column has to say
    where the money in it sits."""
    rows = [
        (0, 700, "CALL", 700.0, 10, "A", 9_000_000, "FLR", "BLOCK"),
        (1, 700, "CALL", 700.0, 10, "A", 1_000, "AUTO", "BLOCK"),
    ]
    csv = _behaviour_csv(rows).decode().replace(
        '2026-08-21,$700.00,CALL,$700.00,10,"9,000","20,000",A,20.0%,0.010,"$9,000,000.00"',
        '2027-07-16,$700.00,CALL,$700.00,10,"9,000","20,000",A,20.0%,0.010,"$9,000,000.00"')
    L = block_strike_ladder(parse_file(csv.encode()).prints, 700.0)
    assert L.iloc[0]["dte"] > 300      # dragged to the LEAP that holds the money


def test_ladder_handles_missing_inputs():
    assert block_strike_ladder(None).empty
    assert block_strike_ladder(pd.DataFrame()).empty
    p = parse_file(REAL_CSV.encode()).prints
    assert block_strike_ladder(p[~p["is_block"]]).empty


def test_report_carries_the_strike_ladder(real):
    from dealer_gex.analytics import block_type_breakdown
    from dealer_gex.report import build_markdown

    a = analyze(real.chain, real.spots["QQQ"], real.asof)
    md = build_markdown(a, ticker="QQQ",
                        block_types=block_type_breakdown(real.prints, a.spot),
                        block_prints=real.prints)
    assert "### Where the blocks are — by strike" in md
    assert "| Strike | vs spot | Premium |" in md
    assert "no smoothing and no peak finding" in md
