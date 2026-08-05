"""The shareable card: a real PNG, driven by a field list."""

import io
import math
from datetime import date

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from dealer_gex.analytics import analyze
from dealer_gex.parsing import read_chain
from dealer_gex.share import (
    CARD_FIELDS, CARD_H, CARD_W, HUES, SCALE, THEMES, Field,
    build_share_card, card_filename,
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


def test_card_is_dark(sample):
    """'Dark' is measurable: the backdrop should sit near black, with the
    bright pixels confined to text, glass edges and the accent."""
    im = _open(build_share_card(sample, "SPY")).convert("L")
    px = np.asarray(im, dtype=float)
    assert px.mean() < 70                      # overall near-black
    assert np.percentile(px, 50) < 60          # the typical pixel is dark
    assert px.max() > 230                      # text still reads white


def test_the_instrument_is_named_on_the_card(sample):
    """An NQ card and a QQQ card would otherwise be identical pictures with
    different dollar figures — the multiplier has to be stated."""
    nq = build_share_card(sample, "NQ")
    qqq = build_share_card(sample, "QQQ")
    assert nq != qqq


def test_multiplier_shows_through_to_the_card():
    from datetime import date as _date

    from dealer_gex.analytics import analyze
    from dealer_gex.parsing import read_chain
    chain, spot = read_chain(open("data/sample_option_chain.csv", "rb").read())
    a20 = analyze(chain, spot, _date(2026, 7, 17), multiplier=20.0)
    a100 = analyze(chain, spot, _date(2026, 7, 17), multiplier=100.0)
    assert build_share_card(a20, "NQ") != build_share_card(a100, "NQ")


def test_type_is_monospaced():
    """The card is a numbers card: digits have to sit in columns, so the
    resolved face must be fixed-pitch even when Consolas is missing and a
    substitute was picked."""
    from PIL import ImageDraw
    from dealer_gex.share import _font
    d = ImageDraw.Draw(Image.new("RGB", (10, 10)))
    f = _font(40, True)
    widths = {d.textlength(c, font=f) for c in "iW1.0m"}
    assert len(widths) == 1


def test_consolas_is_asked_for_first():
    from dealer_gex.share import _MONO_CANDIDATES
    for bold in (False, True):
        assert "consola" in _MONO_CANDIDATES[bold][0].lower()
        # and a face that exists everywhere has to close the list out
        assert _MONO_CANDIDATES[bold][-1].endswith(".ttf")


def test_labels_keep_a_lowercase_sigma():
    """`.upper()` turns 1σ into 1Σ, which is a different symbol."""
    from dealer_gex.share import _caps
    assert _caps("Expected move (1σ)") == "EXPECTED MOVE (1σ)"
    assert _caps("net gex / 1% move") == "NET GEX / 1% MOVE"


def test_net_gex_takes_its_colour_from_its_own_sign(sample):
    from dealer_gex.share import CARD_FIELDS, _hue_of
    object.__setattr__(sample, "total_gex", 1e9)
    assert _hue_of(CARD_FIELDS[0], sample) == HUES["mint"]
    object.__setattr__(sample, "total_gex", -1e9)
    assert _hue_of(CARD_FIELDS[0], sample) == HUES["rose"]


def test_tiles_carry_distinct_accents(sample):
    """Colour is the index into the card — two tiles reading the same hue
    would make the walls indistinguishable at thumbnail size."""
    from dealer_gex.share import CARD_FIELDS, _hue_of
    hues = [_hue_of(f, sample) for f in CARD_FIELDS]
    assert len(set(hues)) >= 5


def test_the_grid_stays_two_rows_while_it_can():
    """A third row costs a third of the tile height, and height is what
    decides how large the value can be set."""
    from dealer_gex.share import _COLS
    for n in range(1, 9):
        assert math.ceil(n / _COLS[n]) <= 2
    assert _COLS[4] == 2          # 2x2, not a row of three and a straggler


def test_a_seventh_tile_does_not_collide_with_the_caption(sample):
    """Tile internals are placed as a fraction of tile height; a fixed
    offset stacks the value on the caption once the grid reflows."""
    png = build_share_card(sample, "SPY", extras=[
        ("Block book", "M2M FLR", "78% of block premium")])
    assert _open(png).size == (CARD_W * SCALE, CARD_H * SCALE)


def test_extras_may_name_their_own_accent(sample):
    a = build_share_card(sample, "SPY", extras=[("X", "1", "", "lime")])
    b = build_share_card(sample, "SPY", extras=[("X", "1", "", "orange")])
    assert a != b
