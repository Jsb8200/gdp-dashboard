"""Precision of the pinpoint levels: flip crossings, walls, magnets."""

from datetime import date

import numpy as np
import pandas as pd
import pytest

from dealer_gex.analytics import (
    _fill_gamma, _golden_max, analyze, flip_levels, gex_at_levels, gex_curve,
    magnet_levels, side_gamma_density, total_gex_at, wall_level,
)
from dealer_gex.parsing import read_chain

ASOF = date(2026, 7, 17)
REPO_CHAIN = "data/sample_option_chain.csv"


def _book(rows) -> pd.DataFrame:
    """rows: (strike, cp, oi, iv)"""
    return pd.DataFrame({
        "strike": [r[0] for r in rows],
        "type": [r[1] for r in rows],
        "open_interest": [float(r[2]) for r in rows],
        "iv": [r[3] for r in rows],
        "volume": [0.0] * len(rows),
        "gamma": [np.nan] * len(rows),
        "expiry": pd.to_datetime(["2026-08-21"] * len(rows)),
    })


# --- the flip ----------------------------------------------------------------

def test_flip_lands_on_an_actual_zero():
    """The reported flip has to be a root of the true GEX function, not of
    the display curve's linear interpolation between samples."""
    chain = _book([(90, "P", 5000, 0.30), (100, "C", 4000, 0.20),
                   (110, "C", 3000, 0.18)])
    a = analyze(chain, 100.0, ASOF)
    assert a.gamma_flip is not None
    df = _fill_gamma(chain, 100.0, ASOF, 0.045, "open_interest")
    # a hundredth of a cent of bisection => GEX vanishes to rounding
    gex = abs(total_gex_at(df, a.gamma_flip, 0.045))
    scale = abs(total_gex_at(df, 100.0, 0.045))
    assert gex < scale * 1e-4


def test_every_crossing_is_found_not_just_the_first():
    """A book that flips back leaves spot in a pocket. Reporting one
    crossing hides the edge where the regime changes back."""
    chain = _book([
        (95, "C", 9000, 0.20),     # long-gamma island below
        (100, "P", 12000, 0.22),   # short-gamma pocket around spot
        (108, "C", 9000, 0.20),    # long gamma again above
    ])
    a = analyze(chain, 100.0, ASOF)
    assert len(a.flip_levels) >= 2, a.flip_levels
    assert a.gamma_flip == a.flip_levels[0]
    # nearest-first ordering
    d = [abs(x - a.spot) for x in a.flip_levels]
    assert d == sorted(d)
    # spot really is bracketed by two crossings
    assert min(a.flip_levels) < a.spot < max(a.flip_levels)
    df = _fill_gamma(chain, 100.0, ASOF, 0.045, "open_interest")
    scale = abs(total_gex_at(df, 100.0, 0.045))
    for lv in a.flip_levels:
        assert abs(total_gex_at(df, lv, 0.045)) < scale * 1e-3


def test_a_finer_grid_does_not_move_the_flip():
    """If the answer depended on the grid it would not be a level."""
    chain, spot = read_chain(open(REPO_CHAIN, "rb").read())
    df = _fill_gamma(chain, spot, ASOF, 0.045, "open_interest")
    got = []
    for n in (601, 1801, 4001):
        curve = gex_curve(df, spot, 0.045, n=n)
        got.append(flip_levels(df, curve, spot, 0.045)[0])
    assert max(got) - min(got) < 0.01      # a cent across a 7x grid change


def test_flip_levels_deduplicates_a_shared_bracket_edge():
    chain, spot = read_chain(open(REPO_CHAIN, "rb").read())
    df = _fill_gamma(chain, spot, ASOF, 0.045, "open_interest")
    roots = flip_levels(df, gex_curve(df, spot, 0.045), spot, 0.045)
    assert len(roots) == len(set(round(r, 6) for r in roots))
    assert all(abs(a - b) > 1e-3 for i, a in enumerate(roots)
               for b in roots[i + 1:])


# --- the vectorized evaluator ------------------------------------------------

def test_vectorized_levels_match_the_scalar_function():
    """The fine grid is only worth having if it computes the same thing."""
    chain, spot = read_chain(open(REPO_CHAIN, "rb").read())
    df = _fill_gamma(chain, spot, ASOF, 0.045, "open_interest")
    lv = np.linspace(spot * 0.85, spot * 1.15, 97)
    fast = gex_at_levels(df, lv, 0.045)
    slow = np.array([total_gex_at(df, x, 0.045) for x in lv])
    assert np.allclose(fast, slow, rtol=1e-12)


def test_vectorized_levels_chunk_boundary_is_seamless():
    chain, spot = read_chain(open(REPO_CHAIN, "rb").read())
    df = _fill_gamma(chain, spot, ASOF, 0.045, "open_interest")
    lv = np.linspace(spot * 0.9, spot * 1.1, 5000)   # forces several chunks
    fast = gex_at_levels(df, lv, 0.045)
    for i in (0, 1234, 2500, 4999):
        assert fast[i] == pytest.approx(total_gex_at(df, lv[i], 0.045), rel=1e-12)
    assert gex_at_levels(df, spot, 0.045).shape == (1,)


# --- walls and magnets -------------------------------------------------------

def test_wall_is_the_true_density_peak_not_a_clamped_bracket():
    """A heavy pair of neighbours can carry the peak past the next strike;
    a one-strike search bracket would clamp it back onto the seed."""
    chain = _book([(100, "C", 5000, 0.20), (102, "C", 4800, 0.20),
                   (104, "C", 4600, 0.20), (80, "P", 100, 0.30)])
    a = analyze(chain, 100.0, ASOF)
    bs, spacing = a.by_strike, 2.0
    dense = np.linspace(96, 108, 24001)
    d = np.array([side_gamma_density(bs, x, "C", spacing) for x in dense])
    assert a.call_wall == pytest.approx(dense[d.argmax()], abs=0.01)
    assert a.call_wall > 100.5      # pulled off the seed strike by neighbours


def test_levels_are_resolved_to_sub_cent():
    chain, spot = read_chain(open(REPO_CHAIN, "rb").read())
    a = analyze(chain, spot, ASOF)
    bs = a.by_strike
    ks = np.sort(bs["strike"].unique())
    spacing = float(np.median(np.diff(ks)))
    for side, level in (("C", a.call_wall), ("P", a.put_wall)):
        f = lambda x: side_gamma_density(bs, x, side, spacing)
        # no nearby point beats the reported level
        probe = np.linspace(level - 0.05, level + 0.05, 501)
        assert f(level) >= max(f(x) for x in probe) * (1 - 1e-9)


def test_golden_max_tolerance_is_sub_cent():
    peak = 100.123456
    got = _golden_max(lambda x: -(x - peak) ** 2, 95.0, 105.0)
    assert abs(got - peak) < 0.001


def test_magnet_levels_sit_on_real_density_peaks():
    chain, spot = read_chain(open(REPO_CHAIN, "rb").read())
    a = analyze(chain, spot, ASOF)
    m = magnet_levels(a)
    if m.empty:
        pytest.skip("no magnets in the sample chain")
    bs = a.by_strike
    ks = np.sort(bs["strike"].unique())
    spacing = float(np.median(np.diff(ks)))
    for _, r in m.iterrows():
        side = "POS" if r["kind"] == "magnet" else "NEG"
        f = lambda x: side_gamma_density(bs, x, side, spacing)
        here = f(r["level"])
        assert here >= f(r["level"] - 0.02) and here >= f(r["level"] + 0.02)
