"""Quoting the book's levels on the futures contract actually traded."""

from datetime import date

import pandas as pd
import pytest

from dealer_gex.analytics import analyze, magnet_levels
from dealer_gex.futures import (
    MODES, PRESETS, Conversion, candle_basis, convert_analysis, convert_frame,
)
from dealer_gex.parsing import read_chain

ASOF = date(2026, 7, 17)


@pytest.fixture(scope="module")
def sample():
    chain, spot = read_chain(open("data/sample_option_chain.csv", "rb").read())
    return analyze(chain, spot, ASOF)


def test_offset_and_ratio_are_not_interchangeable():
    """Adding 41.765 to QQQ at 688 is a 6% basis, which is nonsense;
    multiplying is right. The mode is part of the conversion."""
    off = Conversion("ES", "offset", 26.98)
    rat = Conversion("NQ", "ratio", 41.765)
    assert off.level(7484.84) == pytest.approx(7511.82)
    assert rat.level(688.59) == pytest.approx(28758.96, rel=1e-6)
    assert off.level(688.59) != rat.level(688.59)


def test_a_distance_takes_the_scale_but_never_the_offset():
    """A one-sigma move of 74 points is 74 on ES too — carry shifts the
    axis, it does not stretch it."""
    assert Conversion("ES", "offset", 26.98).distance(74.24) == pytest.approx(74.24)
    assert Conversion("NQ", "ratio", 41.765).distance(74.24) == pytest.approx(74.24 * 41.765)


def test_the_round_trip_returns_the_underlying():
    for conv in (Conversion("ES", "offset", 26.98),
                 Conversion("GC", "ratio", 10.87)):
        assert conv.back(conv.level(628.5)) == pytest.approx(628.5)


def test_the_identity_conversion_is_inactive():
    """A ratio of 1 or an offset of 0 should not relabel the dashboard."""
    assert not Conversion("ES", "offset", 0.0).active
    assert not Conversion("NQ", "ratio", 1.0).active
    assert not Conversion("", "offset", 26.98).active
    assert Conversion("ES", "offset", 26.98).active


def test_every_price_level_on_the_analysis_converts(sample):
    conv = Conversion("ES", "offset", 26.98)
    got = convert_analysis(sample, conv)
    for attr in ("spot", "gamma_flip", "call_wall", "put_wall",
                 "call_wall_strike", "put_wall_strike", "max_pain"):
        assert getattr(got, attr) == pytest.approx(getattr(sample, attr) + 26.98)
    assert got.flip_levels == pytest.approx([x + 26.98 for x in sample.flip_levels])
    assert got.by_strike["strike"].tolist() == pytest.approx(
        [s + 26.98 for s in sample.by_strike["strike"]])
    assert got.curve["spot_level"].tolist() == pytest.approx(
        [s + 26.98 for s in sample.curve["spot_level"]])


def test_money_and_ratios_do_not_convert(sample):
    """Net GEX is already dollars, and a percentage of spot is unitless —
    converting either would be double-counting the change of units."""
    conv = Conversion("NQ", "ratio", 41.765)
    got = convert_analysis(sample, conv)
    for attr in ("total_gex", "dex", "vanna_flow", "charm_flow",
                 "n_contracts", "multiplier"):
        assert getattr(got, attr) == getattr(sample, attr)
    m = magnet_levels(sample)
    assert convert_frame(m, conv)["distance_pct"].tolist() == \
        pytest.approx(m["distance_pct"].tolist())


def test_a_ratio_preserves_distance_but_an_offset_must_be_recomputed(sample):
    """`L/S` survives a stretch exactly; it does not survive a shift, and
    carrying the old percentage over would quietly misstate every distance."""
    m = magnet_levels(sample)
    ratio = Conversion("NQ", "ratio", 41.765)
    got = convert_frame(m, ratio, convert_analysis(sample, ratio).spot)
    assert got["distance_pct"].tolist() == pytest.approx(m["distance_pct"].tolist())

    off = Conversion("ES", "offset", 26.98)
    spot = convert_analysis(sample, off).spot
    got = convert_frame(m, off, spot)
    assert got["distance_pct"].tolist() == pytest.approx(
        (got["level"] / spot - 1.0).tolist())
    # and it really is different from what was carried over
    assert got["distance_pct"].tolist() != pytest.approx(m["distance_pct"].tolist())


def test_the_regime_survives_conversion(sample):
    """The verdict is a property of the book, not of the ticks it is
    quoted in."""
    for conv in (Conversion("ES", "offset", 26.98),
                 Conversion("NQ", "ratio", 41.765)):
        assert convert_analysis(sample, conv).regime == sample.regime


def test_levels_keep_their_order_and_spacing_ratio(sample):
    """A conversion may shift or stretch the axis; it must not reorder the
    book or move a wall past the flip."""
    conv = Conversion("NQ", "ratio", 41.765)
    got = convert_analysis(sample, conv)
    assert (sample.put_wall < sample.call_wall) == (got.put_wall < got.call_wall)
    before = (sample.call_wall - sample.spot) / (sample.spot - sample.put_wall)
    after = (got.call_wall - got.spot) / (got.spot - got.put_wall)
    assert before == pytest.approx(after)


def test_level_frames_convert_and_others_pass_through(sample):
    conv = Conversion("ES", "offset", 26.98)
    m = magnet_levels(sample)
    got = convert_frame(m, conv)
    assert got["level"].tolist() == pytest.approx([x + 26.98 for x in m["level"]])
    assert got["anchor_strike"].tolist() == pytest.approx(
        [x + 26.98 for x in m["anchor_strike"]])
    assert got["strength"].tolist() == m["strength"].tolist()
    assert convert_frame(None, conv) is None
    assert convert_frame(m, Conversion()) is m          # inactive is a no-op


def test_the_basis_can_be_measured_from_a_candle():
    """So the number can be checked against what was typed in."""
    assert candle_basis(7511.82, 7484.84, "offset") == pytest.approx(26.98)
    assert candle_basis(28758.96, 688.59, "ratio") == pytest.approx(41.765, rel=1e-5)
    assert candle_basis(1.0, 0.0) == 0.0                # no spot, no basis


def test_every_preset_is_usable():
    for target, p in PRESETS.items():
        assert p["mode"] in MODES
        conv = Conversion(target, p["mode"], p["value"])
        assert conv.active
        assert target in conv.label()
