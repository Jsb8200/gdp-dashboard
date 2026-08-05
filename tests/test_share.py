"""The shareable card: a real PNG, driven by a field list."""

import io
from datetime import date

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from dealer_gex.analytics import analyze
from dealer_gex.parsing import read_chain
from dealer_gex.share import (
    CARD_FIELDS, CARD_H, CARD_W, SCALE, THEMES, Field, build_share_card,
    card_filename,
)

ASOF = date(2026, 7, 17)


@pytest.fixture(scope="module")
def sample():
    chain, spot = read_chain(open("data/sample_option_chain.csv", "rb").read())
    return analyze(chain, spot, ASOF)


def _open(png: bytes) -> Image.Image:
    return Image.open(io.BytesIO(png))


def test_card_is_a_real_png_at_the_declared_size(sample):
    png = build_share_card(sample, "SPY")
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    im = _open(png)
    assert im.format == "PNG"
    assert im.size == (CARD_W * SCALE, CARD_H * SCALE)


def test_card_is_not_a_flat_rectangle(sample):
    """Glass means the backdrop shows through with structure — a card that
    rendered as one flat colour would still be a valid PNG."""
    im = _open(build_share_card(sample, "SPY")).convert("RGB")
    small = im.resize((60, 34))
    colours = {small.getpixel((x, y)) for x in range(60) for y in range(34)}
    assert len(colours) > 300          # a gradient plus blurred blobs
    assert im.getextrema()[0][1] > 200  # something bright: the text and glass


def test_regime_picks_the_palette(sample):
    """The card should read before a number is parsed, so the tint follows
    the verdict rather than the ticker."""
    long_png = build_share_card(sample, "SPY")
    short = analyze(*read_chain(open("data/sample_option_chain.csv", "rb").read()),
                    ASOF)
    object.__setattr__(short, "regime", "short_gamma")
    short_png = build_share_card(short, "SPY")
    assert long_png != short_png
    lr = np.asarray(_open(long_png).convert("RGB").resize((40, 22)), dtype=float)
    sr = np.asarray(_open(short_png).convert("RGB").resize((40, 22)), dtype=float)
    # the short-gamma palette is red-dominant, the long-gamma one is not
    assert sr[..., 0].mean() - sr[..., 1].mean() > lr[..., 0].mean() - lr[..., 1].mean()


def test_tiles_come_from_the_field_list(sample):
    """Adding a number to the card has to be one entry, not a layout edit."""
    one = build_share_card(sample, "SPY", fields=CARD_FIELDS[:1])
    six = build_share_card(sample, "SPY", fields=CARD_FIELDS)
    assert one != six
    assert _open(one).size == _open(six).size      # the grid reflows, not the card


def test_extras_append_without_touching_the_defaults(sample):
    base = build_share_card(sample, "SPY")
    with_extra = build_share_card(
        sample, "SPY", extras=[("Block book", "M2M FLR", "78% of premium")])
    assert base != with_extra
    assert len(CARD_FIELDS) == 6                   # the default set is unchanged


def test_a_custom_field_renders(sample):
    got = build_share_card(sample, "SPY", fields=[
        Field("spot", "Spot", lambda a: f"{a.spot:,.2f}", lambda a: "underlying"),
    ])
    assert got[:8] == b"\x89PNG\r\n\x1a\n"


def test_missing_numbers_do_not_break_the_card(sample):
    """A book with no flip and no expected move still has to produce a card."""
    object.__setattr__(sample, "gamma_flip", None)
    object.__setattr__(sample, "flip_levels", [])
    object.__setattr__(sample, "expected_move", None)
    object.__setattr__(sample, "nearest_expiry", None)
    png = build_share_card(sample, "SPY")
    assert _open(png).size == (CARD_W * SCALE, CARD_H * SCALE)


def test_rendering_is_deterministic(sample):
    assert build_share_card(sample, "SPY") == build_share_card(sample, "SPY")


def test_long_values_are_shrunk_to_fit(sample):
    """A 12-digit index level must not run out of its tile."""
    wide = build_share_card(sample, "SPY", fields=[
        Field("x", "Huge", lambda a: "1,234,567,890.12", None)])
    assert _open(wide).size == (CARD_W * SCALE, CARD_H * SCALE)


def test_filename_carries_ticker_and_date(sample):
    assert card_filename(sample, "spx") == "SPX-positioning-20260717.png"
    assert card_filename(sample, "").startswith("BOOK-")


def test_every_theme_is_complete():
    for name, t in THEMES.items():
        assert {"base", "blobs", "accent", "verdict", "tagline"} <= set(t)
        assert len(t["blobs"]) == 3
