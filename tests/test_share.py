"""The shareable card: a real PNG, driven by a field list."""

import copy
import io
import os
import math
from datetime import date

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from dealer_gex.analytics import analyze
from dealer_gex.parsing import read_chain
from dealer_gex.share import (
    CARD_FIELDS, DEFAULT_SIZE, HUES, SCALE, SIZES, THEMES, Field,
    build_share_card, card_catalog, card_filename, card_size,
)

CARD_W = SIZES[DEFAULT_SIZE]

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
    assert im.size == card_size(len(CARD_FIELDS))
    assert im.width == CARD_W * SCALE


def test_card_is_not_a_flat_rectangle(sample):
    """Glass means the backdrop shows through with structure — a card that
    rendered as one flat colour would still be a valid PNG."""
    im = _open(build_share_card(sample, "SPY")).convert("RGB")
    small = im.resize((60, 34))
    colours = {small.getpixel((x, y)) for x in range(60) for y in range(34)}
    assert len(colours) > 300          # a gradient plus blurred blobs
    assert im.getextrema()[0][1] > 200  # something bright: the text and glass


def test_the_default_keys_are_all_offered(sample):
    """DEFAULT_KEYS seeds the picker, so every one of them has to resolve
    against a catalogue built from a book that can answer them."""
    from dealer_gex.share import DEFAULT_KEYS
    cat = card_catalog(sample)
    assert [k for k in DEFAULT_KEYS if k in cat] == DEFAULT_KEYS


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
    one = _open(build_share_card(sample, "SPY", fields=CARD_FIELDS[:1]))
    many = _open(build_share_card(
        sample, "SPY", fields=CARD_FIELDS,
        extras=[("a", "1"), ("b", "2"), ("c", "3")]))
    assert one.width == many.width == CARD_W * SCALE  # the width is the frame
    assert one.height < many.height                   # the height follows the rows


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
    blank = copy.copy(sample)
    object.__setattr__(blank, "gamma_flip", None)
    object.__setattr__(blank, "flip_levels", [])
    object.__setattr__(blank, "expected_move", None)
    object.__setattr__(blank, "nearest_expiry", None)
    png = build_share_card(blank, "SPY")
    assert _open(png).size == card_size(len(CARD_FIELDS))


def test_rendering_is_deterministic(sample):
    assert build_share_card(sample, "SPY") == build_share_card(sample, "SPY")


def test_long_values_are_shrunk_to_fit(sample):
    """A 12-digit index level must not run out of its tile."""
    wide = build_share_card(sample, "SPY", fields=[
        Field("x", "Huge", lambda a: "1,234,567,890.12", None)])
    assert _open(wide).size == card_size(1)


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


def test_no_row_is_ever_left_short(sample):
    """First-fit leaves a tail whenever the next tile is wider than the gap,
    and an empty cell reads as a missing tile rather than as spare room."""
    from dealer_gex.share import GRID_COLS, _place, _Tile

    def mk(span=1, rowspan=1):
        return _Tile("x", "1", "", HUES["slate"], "dot", None, span, None, rowspan)

    for tiles in ([mk()], [mk()] * 2, [mk()] * 5, [mk()] * 7,
                  [mk(2, 2)] + [mk()] * 6,
                  [mk(2, 2), mk(3), mk(), mk(), mk(6), mk(3)],
                  [mk(6)], [mk(3), mk()], [mk(2, 2)] + [mk()] * 3 + [mk(3)]):
        cols, places, heights = _place(tiles)
        assert cols == GRID_COLS
        filled = [[False] * cols for _ in heights]
        for (r, c, sp, rs) in places:
            for rr in range(r, r + rs):
                for cc in range(c, c + sp):
                    assert not filled[rr][cc], "tiles overlap"
                    filled[rr][cc] = True
        for r, row in enumerate(filled):
            assert all(row), f"row {r} left short: {row}"


def test_a_seventh_tile_does_not_collide_with_the_caption(sample):
    """Tile internals are placed as a fraction of tile height; a fixed
    offset stacks the value on the caption once the grid reflows."""
    png = build_share_card(sample, "SPY", extras=[
        ("Extra", "M2M FLR", "78% of block premium")])
    assert _open(png).size == card_size(len(CARD_FIELDS) + 1)


def test_extras_may_name_their_own_accent(sample):
    a = build_share_card(sample, "SPY", extras=[("X", "1", "", "lime")])
    b = build_share_card(sample, "SPY", extras=[("X", "1", "", "orange")])
    assert a != b


def test_the_catalog_only_offers_numbers_the_file_can_answer(sample):
    """The picker is built from this, so an entry that would render as an em
    dash must not be in it."""
    from dealer_gex.analytics import magnet_levels, oi_walls
    cat = card_catalog(sample, oi=oi_walls(sample), magnets=magnet_levels(sample))
    assert set(CARD_FIELDS[i].key for i in range(len(CARD_FIELDS))) <= set(cat)
    assert len(cat) > len(CARD_FIELDS)          # more than the default six
    for key, f in cat.items():
        assert f.key == key                     # keyed by its own name
        assert str(f.value(sample)) != "—"


def test_the_catalog_drops_tiles_whose_inputs_are_missing(sample):
    blank = copy.copy(sample)
    object.__setattr__(blank, "expected_move", None)
    object.__setattr__(blank, "gamma_flip", None)
    cat = card_catalog(blank)
    assert "em_band" not in cat                 # needs an expected move
    assert "flip_distance" not in cat           # needs a flip
    assert "call_oi_wall" not in cat            # needs the OI walls passed in
    assert "flip" not in cat                    # and neither of the defaults
    assert "expected_move" not in cat
    assert "max_pain" in cat                    # the answerable ones stay


def test_every_style_renders_and_they_differ(sample):
    from dealer_gex.share import STYLES
    seen = {name: build_share_card(sample, "SPY", style=name) for name in STYLES}
    for png in seen.values():
        assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(set(seen.values())) == len(STYLES)


def test_the_card_stays_dark_in_every_style(sample):
    from dealer_gex.share import STYLES
    for name in STYLES:
        px = np.asarray(_open(build_share_card(sample, "SPY", style=name)
                              ).convert("L"), dtype=float)
        assert px.mean() < 70, name
        assert px.max() > 230, name


def test_title_subtitle_and_footnote_are_the_callers(sample):
    plain = build_share_card(sample, "SPY")
    assert build_share_card(sample, "SPY", subtitle="my own line") != plain
    assert build_share_card(sample, "SPY", footnote="desk note") != plain
    # a footnote adds a footer band, so the card grows
    assert (_open(build_share_card(sample, "SPY", footnote="x")).height
            > _open(plain).height)


def test_a_signed_caption_is_coloured_by_its_sign():
    """The delta is the only coloured text on a midnight tile — that is what
    makes it readable at a glance, so the sign has to drive it."""
    from dealer_gex.share import _DELTA
    assert _DELTA.match("+3.21% vs spot").group(1) == "+3.21%"
    assert _DELTA.match("-4.44% vs spot").group(1) == "-4.44%"
    assert _DELTA.match("positive = stabilizing") is None


def test_slack_is_shared_out_rather_than_dumped_on_one_tile():
    """Growing only the last tile in a row would leave one narrow tile
    beside a very wide one."""
    from dealer_gex.share import _place, _Tile
    mk = lambda: _Tile("x", "1", "", HUES["slate"], "dot", None, 1, None, 1)
    _, places, _ = _place([mk(), mk()])
    spans = sorted(p[2] for p in places)
    assert spans == [3, 3]                      # six columns, split evenly
    _, places, _ = _place([mk(), mk(), mk(), mk()])
    assert sorted(p[2] for p in places) == [1, 1, 2, 2], (n, cols, last)


def test_ranked_list_tiles_are_offered_and_render(sample):
    """Top-5 GEX strikes and the magnet map are lists, not single numbers."""
    from dealer_gex.analytics import magnet_levels
    cat = card_catalog(sample, magnets=magnet_levels(sample))
    assert "top_gex" in cat and "magnets" in cat
    for key in ("top_gex", "magnets"):
        f = cat[key]
        assert f.span == 3                       # a list needs half the row
        rows = f.rows(sample)
        assert 1 <= len(rows) <= 5
        assert all(len(r) == 3 for r in rows)    # left, right, hue
    png = build_share_card(sample, "SPY",
                           fields=[cat["top_gex"], cat["magnets"]])
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_top_gex_strikes_are_the_heaviest_ones(sample):
    """Ranked by absolute net GEX — a big negative strike matters as much as
    a big positive one, and sorting signed would bury it."""
    cat = card_catalog(sample)
    got = [float(r[0].replace(",", "")) for r in cat["top_gex"].rows(sample)]
    bs = sample.by_strike
    want = list(bs.reindex(bs["net_gex"].abs().sort_values(ascending=False).index)
                .head(5)["strike"].astype(float))
    assert got == want


def test_a_list_tile_gets_a_taller_row(sample):
    """Five rows squeezed into a one-number tile shrink the type until the
    list is unreadable, which defeats putting it on the card."""
    from dealer_gex.share import TILE_H, _place, _Tile
    plain = _Tile("x", "1", "", HUES["slate"], "dot", None, 1)
    listy = _Tile("y", "1", "", HUES["slate"], "dot", [("a", "b", "mint")] * 5, 2)
    _, _, heights = _place([plain, listy])
    assert heights[0] > TILE_H


def test_a_wide_tile_is_never_split_across_a_row(sample):
    from dealer_gex.share import _place, _Tile
    plain = _Tile("x", "1", "", HUES["slate"], "dot", None, 1)
    wide = _Tile("y", "1", "", HUES["slate"], "dot", [("a", "b")], 2)
    cols, places, _ = _place([plain, plain, plain, wide])
    for (r, c, span, rspan) in places:
        assert c + span <= cols                  # fits in the row it starts


def test_the_gex_bar_tile_is_offered_and_renders(sample):
    cat = card_catalog(sample)
    assert "gex_bars" in cat
    f = cat["gex_bars"]
    bars = f.bars(sample)
    assert 5 <= len(bars) <= 40             # capped so the bars stay legible
    assert all(len(b) == 2 for b in bars)
    assert bars == sorted(bars)             # strike-ordered, left to right
    png = build_share_card(sample, "SPY", fields=[f])
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_the_bar_tile_gets_a_chart_sized_row(sample):
    from dealer_gex.share import BARS_H, TILE_H, _place, _Tile
    bar = _Tile("g", "1", "", HUES["sky"], "bars", None, 4, [(1.0, 2.0)])
    _, _, heights = _place([bar])
    assert heights[0] == BARS_H > TILE_H


def test_every_size_renders_at_its_declared_width(sample):
    from dealer_gex.share import SIZES
    for name, px in SIZES.items():
        im = _open(build_share_card(sample, "SPY", size=name))
        assert im.width == px * SCALE, name
        assert im.size == card_size(len(CARD_FIELDS), size=name)


def test_the_zero_line_follows_the_data_not_the_middle():
    """A book that is long gamma nearly everywhere should show one shallow
    red stub under a wall of green — centring the axis would draw that as a
    balanced book."""
    from PIL import Image as _I, ImageDraw as _D
    from dealer_gex.share import STYLES, _draw_bars
    im = _I.new("RGB", (200, 100), (0, 0, 0))
    d = _D.Draw(im)
    _draw_bars(d, (0, 0, 200, 100), [(1.0, 10.0), (2.0, 9.0), (3.0, -1.0)],
               2.0, STYLES["midnight"])
    px = np.asarray(im.convert("L"), dtype=float)
    lit = np.flatnonzero(px.max(axis=1) > 20)
    # most of the ink sits above the zero line, which is low in the frame
    assert lit.size and np.median(lit) < 60


def test_the_gex_ladder_is_a_tall_left_hand_tile(sample):
    """Price runs up the vertical axis, so the tile is narrow and tall and
    sits against the left edge with the numbers flowing to its right."""
    from dealer_gex.share import _place, _Tile
    cat = card_catalog(sample)
    f = cat["gex_bars"]
    assert f.span == 2 and f.rowspan == 2
    tiles = [_Tile(x.label, "1", "", HUES["slate"], "dot", None, x.span, None,
                   x.rowspan) for x in CARD_FIELDS]
    ladder = _Tile(f.label, "1", "", HUES["sky"], "bars", None, f.span,
                   f.bars(sample), f.rowspan)
    # even placed last in the reading order, the tall tile takes the corner
    cols, places, heights = _place(tiles + [ladder])
    assert places[-1][:2] == (0, 0)
    assert places[-1][3] == 2
    for (r, c, sp, rs) in places[:-1]:
        assert not (r < 2 and c == 0)            # nothing else in that corner


def test_nothing_overlaps_in_the_packed_grid():
    """Row spans mean a naive flow can write two tiles into one cell."""
    from dealer_gex.share import _place, _Tile
    def mk(span, rowspan):
        return _Tile("x", "1", "", HUES["slate"], "dot", None, span, None, rowspan)
    tiles = [mk(1, 2), mk(1, 1), mk(2, 1), mk(1, 1), mk(2, 1), mk(1, 1), mk(1, 2)]
    cols, places, heights = _place(tiles)
    seen = set()
    for (r, c, sp, rs) in places:
        for rr in range(r, r + rs):
            for cc in range(c, c + sp):
                assert (rr, cc) not in seen, (rr, cc)
                seen.add((rr, cc))
                assert cc < cols
    assert len(heights) == max(r + rs for r, _, _, rs in places)


def test_the_ladder_puts_high_strikes_at_the_top():
    """A price ladder read upside down is worse than no ladder."""
    from PIL import Image as _I, ImageDraw as _D
    from dealer_gex.share import STYLES, _draw_bars
    im = _I.new("RGB", (200, 120), (0, 0, 0))
    _draw_bars(_D.Draw(im), (0, 0, 200, 120),
               [(100.0, 1.0), (101.0, 0.0), (102.0, 9.0)], 101.0,
               STYLES["midnight"])
    px = np.asarray(im.convert("L"), dtype=float)
    # the biggest bar belongs to the highest strike, so the ink is up top
    rows = px.sum(axis=1)
    assert rows[:40].sum() > rows[80:].sum()


def _avatar_bytes(w=600, h=260):
    """Deliberately not square: a profile picture is cover-cropped, not
    squashed, and the wrong one shows up as a stretched face."""
    from PIL import Image as _I, ImageDraw as _D
    im = _I.new("RGB", (w, h), (18, 90, 140))
    _D.Draw(im).ellipse([w // 2 - 60, 10, w // 2 + 60, 130], fill=(250, 200, 60))
    buf = io.BytesIO(); im.save(buf, "PNG")
    return buf.getvalue()


def test_a_profile_changes_the_card(sample):
    plain = build_share_card(sample, "SPX")
    named = build_share_card(sample, "SPX", username="jsb8200")
    withpic = build_share_card(sample, "SPX", username="jsb8200",
                               avatar=_avatar_bytes())
    assert len({plain, named, withpic}) == 3


def test_initials_come_from_the_handle():
    from dealer_gex.share import _initials
    assert _initials("jsb8200") == "JS"
    assert _initials("@jane.doe") == "JD"
    assert _initials("Jane Doe") == "JD"
    assert _initials("") == "?"
    assert _initials(None) == "?"


def test_a_broken_avatar_still_produces_a_card(sample):
    """A picture that Pillow cannot open is not a reason to lose the
    numbers — it falls back to initials."""
    broken = build_share_card(sample, "SPX", username="jsb", avatar=b"not an image")
    assert broken[:8] == b"\x89PNG\r\n\x1a\n"
    assert broken == build_share_card(sample, "SPX", username="jsb")


def test_the_profile_side_of_the_header_holds_only_the_profile(sample):
    """The upper-right corner belongs to whoever made the card: the verdict
    pill moves inline after the spot and the tagline goes, so nothing else
    is drawn on that side."""
    im = _open(build_share_card(sample, "SPX", username="jsb8200")).convert("L")
    px = np.asarray(im, dtype=float)
    # header spans y in [pad, pad+head_h]; the profile sits on its top line,
    # so the band under it on the right must be empty
    pad, head_h = 44 * SCALE, 112 * SCALE
    band = px[pad + head_h // 2 + 12 * SCALE: pad + head_h - 4 * SCALE,
              im.width // 2:]
    assert band.max() < 90, band.max()

    plain = _open(build_share_card(sample, "SPX")).convert("L")
    same = np.asarray(plain, dtype=float)[
        pad + head_h // 2 + 12 * SCALE: pad + head_h - 4 * SCALE, im.width // 2:]
    assert same.max() > 90            # without a profile that band is used


def test_a_long_handle_does_not_collide_with_the_pill(sample):
    """Narrowest card, longest verdict, longest handle — the one case where
    the inline pill and the profile could meet."""
    short = copy.copy(sample)
    object.__setattr__(short, "regime", "short_gamma")
    png = build_share_card(short, "SPX", size="share",
                           username="@a_very_long_handle_here",
                           avatar=_avatar_bytes())
    im = _open(png).convert("L")
    px = np.asarray(im, dtype=float)
    pad, head_h = 44 * SCALE, 112 * SCALE
    row = px[pad + 20 * SCALE: pad + head_h // 2, :]
    # a gap of dark pixels must survive between the pill and the handle
    dark_cols = (row.max(axis=0) < 60)
    assert dark_cols.any()


def test_a_pil_image_is_accepted_as_an_avatar(sample):
    from PIL import Image as _I
    im = _I.new("RGB", (120, 120), (200, 40, 90))
    got = build_share_card(sample, "SPX", username="x", avatar=im)
    assert got != build_share_card(sample, "SPX", username="x")


def test_the_assumptions_tile_reports_the_real_settings(sample):
    """A card travels without the methodology panel, so the caveats have to
    travel with it — and they have to be the settings actually used, not a
    fixed blurb."""
    cat = card_catalog(sample)
    assert cat["assumptions"].columns == ("Assumption", "Setting", "If it is wrong")
    assert cat["assumptions"].span == 3
    rows = dict((r[0], r[1]) for r in cat["assumptions"].rows(sample))
    assert all(len(r) == 3 for r in cat["assumptions"].rows(sample))
    assert f"×{sample.multiplier:g}" in rows["Multiplier"]
    assert f"{sample.rate:.2%}" in rows["Rate"]
    assert f"{sample.n_contracts:,}" in rows["Book"]
    assert "long calls / short puts" == rows["Dealer sign"]


def test_the_assumptions_follow_the_weighting_and_multiplier(sample):
    other = copy.copy(sample)
    object.__setattr__(other, "weight_mode", "volume")
    object.__setattr__(other, "multiplier", 50.0)
    a_rows = dict((r[0], r[1]) for r in card_catalog(sample)["assumptions"].rows(sample))
    b_rows = dict((r[0], r[1]) for r in card_catalog(other)["assumptions"].rows(other))
    assert "open interest" in a_rows["Weighting"]
    assert "volume" in b_rows["Weighting"]
    assert "×50" in b_rows["Multiplier"]
    assert build_share_card(sample, "SPX", fields=[card_catalog(sample)["assumptions"]]) \
        != build_share_card(other, "SPX", fields=[card_catalog(other)["assumptions"]])


def test_a_table_tile_draws_a_header_band_and_rules(sample):
    """The ranked-list layout puts one value hard right, which reads as a
    mess for prose — a table needs a header and columns."""
    from dealer_gex.share import ROW_H, STYLES, _draw_table
    from PIL import Image as _I, ImageDraw as _D
    im = _I.new("RGB", (700, 260), STYLES["midnight"]["panel"])
    _draw_table(im, _D.Draw(im), (0, 0, 700, 260),
                ("A", "B", "C"),
                [("one", "two", "three"), ("four", "five", "six")],
                STYLES["midnight"])
    px = np.asarray(im.convert("L"), dtype=float)
    rh = min(ROW_H, 260 / 3)
    header = px[2:int(rh) - 2].mean()
    body = px[int(rh) + 4:int(rh * 2) - 4].mean()
    assert header > body          # the band is lighter than the rows


def test_a_table_row_is_reserved_for_the_header(sample):
    """Otherwise the header eats a data row's space and the last row is cut."""
    from dealer_gex.share import _Tile, _tile_need
    rows = [("a", "b", "c")] * 4
    plain = _Tile("x", "1", "", HUES["slate"], "dot", rows, 2)
    table = _Tile("x", "1", "", HUES["slate"], "dot", rows, 2, None, 1,
                  ("A", "B", "C"))
    assert _tile_need(table) > _tile_need(plain)


def test_the_assumptions_table_renders_on_a_card(sample):
    cat = card_catalog(sample)
    png = build_share_card(sample, "SPX", fields=[cat["assumptions"]])
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert _open(png).width == SIZES[DEFAULT_SIZE] * SCALE


def test_label_spacing_keeps_order_and_bounds():
    """The heaviest strikes cluster, so three of the top five can land on
    neighbouring rows; pushing them apart must not reorder them."""
    from dealer_gex.share import _spread
    got = _spread([10.0, 11.0, 12.0, 60.0, 61.0], 15.0, 0.0, 100.0)
    assert got == sorted(got)
    assert all(b - a >= 15.0 - 1e-9 for a, b in zip(got, got[1:]))
    assert min(got) >= 0.0 and max(got) <= 100.0


def test_label_spacing_shifts_the_stack_rather_than_piling_it_up():
    """Clamping each label to the bottom would stack them all on one row."""
    from dealer_gex.share import _spread
    got = _spread([90.0, 91.0, 92.0], 15.0, 0.0, 100.0)
    assert len(set(got)) == 3
    assert all(b - a >= 15.0 - 1e-9 for a, b in zip(got, got[1:]))
    assert max(got) <= 100.0


def test_only_five_strikes_are_named_on_the_ladder(sample):
    """Labelling twenty-six turns the axis into a wall of digits."""
    from dealer_gex.share import STYLES, _draw_bars
    from PIL import Image as _I, ImageDraw as _D
    bars = card_catalog(sample)["gex_bars"].bars(sample)
    im = _I.new("RGB", (600, 400), STYLES["midnight"]["panel"])
    _draw_bars(_D.Draw(im), (0, 0, 600, 400), bars, sample.spot, STYLES["midnight"])
    # the gutter is the right-hand strip; count rows carrying bright ink
    px = np.asarray(im.convert("L"), dtype=float)[:, -70:]
    lit = px.max(axis=1) > 140
    runs = int(np.sum(lit[1:] & ~lit[:-1])) + int(lit[0])
    assert 1 <= runs <= 5, runs


def test_the_interpretation_tile_keeps_the_numbers(sample):
    """Taking the bold lead looks tidier and loses data: the confluence
    bullet nests bold inside bold, and "**Passive flows:**" puts both dollar
    figures outside the bold entirely."""
    from dealer_gex.analytics import confluence_levels, magnet_levels, oi_levels
    m, o = magnet_levels(sample), oi_levels(sample)
    master = confluence_levels(sample, m, o)
    cat = card_catalog(sample, magnets=m, master=master)
    lines = [r[0] for r in cat["interpretation"].rows(sample)]
    assert lines and cat["interpretation"].span == 6   # full width
    assert not any("**" in ln for ln in lines)          # markdown stripped
    joined = " ".join(lines)
    assert "Passive flows" in joined
    passive = next(ln for ln in lines if "Passive flows" in ln)
    assert "vanna" in passive and "charm" in passive    # the figures survived
    conf = next((ln for ln in lines if "confluence" in ln), None)
    if conf is not None:
        assert f"{master.iloc[0]['level']:,.2f}" in conf


def test_sentences_split_on_a_capital_not_on_any_period(sample):
    """"700.03" and "±9.40" are full of periods — splitting on all of them
    truncates the line at its first number."""
    from dealer_gex.share import _playbook_heads
    from dealer_gex.analytics import confluence_levels, magnet_levels, oi_levels
    m, o = magnet_levels(sample), oi_levels(sample)
    heads = _playbook_heads(sample, confluence_levels(sample, m, o))
    flip = next((h for h in heads if "gamma flip" in h), None)
    assert flip is not None and f"{sample.gamma_flip:,.2f}" in flip


def test_missing_glyphs_are_replaced_not_drawn_as_tofu():
    """PIL's built-in bitmap font — what you get when no mono face resolves
    at all — has no sigma, em dash, en dash or multiplication sign, and
    draws each as a .notdef box. A card full of tofu is worse than one that
    says "sd"."""
    from PIL import ImageFont
    from dealer_gex.share import _safe
    poor = ImageFont.load_default(28)
    rich = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 28)
    src = "Regime — long gamma · ±392.39 (1σ) × 100"
    got = _safe(src, poor)
    assert "σ" not in got and "—" not in got and "×" not in got
    assert "1sd" in got
    assert _safe(src, rich) == src          # a capable face is left alone
    assert _safe("plain ascii", poor) == "plain ascii"


def test_a_card_survives_having_no_mono_font_at_all(monkeypatch):
    """The whole card must still render — and render readably — on a box
    where none of the candidate faces exist."""
    import dealer_gex.share as S
    S._font.cache_clear()
    monkeypatch.setattr(S, "_MONO_CANDIDATES", {False: ["/nope.ttf"],
                                                True: ["/nope.ttf"]})
    try:
        chain, spot = read_chain(open("data/sample_option_chain.csv", "rb").read())
        a = analyze(chain, spot, ASOF)
        png = build_share_card(a, "SPX", fields=CARD_FIELDS)
        assert png[:8] == b"\x89PNG\r\n\x1a\n"
        px = np.asarray(_open(png).convert("L"), dtype=float)
        assert px.max() > 200               # text still drew
    finally:
        S._font.cache_clear()


def test_the_ladder_strike_labels_are_centred_in_their_gutter(sample):
    """Right-aligned at the panel edge they float away from the bars they
    belong to."""
    from dealer_gex.share import STYLES, _draw_bars
    from PIL import Image as _I, ImageDraw as _D
    bars = card_catalog(sample)["gex_bars"].bars(sample)
    im = _I.new("RGB", (600, 400), STYLES["midnight"]["panel"])
    _draw_bars(_D.Draw(im), (0, 0, 600, 400), bars, sample.spot, STYLES["midnight"])
    px = np.asarray(im.convert("L"), dtype=float)
    lit = np.flatnonzero(px[:, -70:].max(axis=0) > 140)
    assert lit.size
    # ink sits away from both edges of the gutter, i.e. it is centred
    assert lit.min() > 2 and lit.max() < 68


def test_interpretation_lines_can_be_chosen_individually(sample):
    """The dashboard shows all eight; a card is usually making one point."""
    from dealer_gex.analytics import confluence_levels, magnet_levels, oi_levels
    from dealer_gex.share import playbook_choices
    m, o = magnet_levels(sample), oi_levels(sample)
    master = confluence_levels(sample, m, o)
    choices = playbook_choices(sample, master)
    keys = [k for k, _ in choices]
    assert len(set(keys)) == len(keys)          # keys identify a line uniquely
    assert "other" not in keys                  # every line was classified

    few = card_catalog(sample, magnets=m, master=master, lines={"regime", "flip"})
    got = [r[0] for r in few["interpretation"].rows(sample)]
    assert len(got) == 2
    assert any("Regime" in g for g in got) and any("gamma flip" in g for g in got)

    # no lines selected drops the tile rather than drawing an empty panel
    assert "interpretation" not in card_catalog(sample, magnets=m, master=master,
                                                lines=set())


def test_the_level_ladder_tile_matches_the_dashboard_table(sample):
    from dealer_gex.report import key_ladder
    cat = card_catalog(sample)
    f = cat["level_ladder"]
    assert f.columns == ("Level", "Price", "Reading") and f.span == 6
    rows = f.rows(sample)
    want = key_ladder(sample)
    assert len(rows) == len(want)
    assert [r[0] for r in rows] == list(want["Level"])       # same order
    assert rows[0][1] == f"{float(want.iloc[0]['Price']):,.2f}"
    # price-sorted, highest first
    prices = [float(r[1].replace(",", "")) for r in rows]
    assert prices == sorted(prices, reverse=True)


@pytest.mark.parametrize("weight", ["open_interest", "volume"])
def test_every_playbook_line_classifies_under_every_weighting(weight):
    """`build_playbook` adds a line explaining the weighting when it is not
    open interest. Two unclassified lines would collide on the same key and
    become indistinguishable in the picker."""
    from dealer_gex.analytics import confluence_levels, magnet_levels, oi_levels
    from dealer_gex.share import playbook_choices
    chain, spot = read_chain(open("data/sample_option_chain.csv", "rb").read())
    a = analyze(chain, spot, ASOF, weight=weight)
    m, o = magnet_levels(a), oi_levels(a)
    keys = [k for k, _ in playbook_choices(a, confluence_levels(a, m, o))]
    assert "other" not in keys, weight
    assert len(set(keys)) == len(keys), (weight, keys)


def test_the_weighting_view_lines_classify():
    """`build_playbook` adds a line explaining the weighting when it is not
    open interest. Signed-flow needs side codes the sample chain lacks, so
    its line is checked against the classifier directly rather than through
    an analysis that cannot be built."""
    from dealer_gex.share import _playbook_key
    assert _playbook_key(
        "Volume-weighted (intraday) view — levels reflect today's traded flow"
    ) == "volume_view"
    assert _playbook_key(
        "Signed order-flow view — dealer positioning is inferred from actual"
    ) == "flow_view"
    # and every lead is distinct, or two lines would share a key
    from dealer_gex.share import PLAYBOOK_LINES
    leads = list(PLAYBOOK_LINES.values())
    assert len(set(leads)) == len(leads)
    for i, a in enumerate(leads):
        for b in leads[i + 1:]:
            assert not a.startswith(b) and not b.startswith(a), (a, b)


@pytest.fixture(scope="module")
def flow_prints():
    from dealer_gex.parsing import parse_file
    path = ("/root/.claude/uploads/c39ce9c9-b3cb-5bd5-9d5e-e27db2e98dad/"
            "f9f9e43a-QuantData_OptionsOrderFlow_2026_07_30_to_2026_07_31_2.csv")
    if not os.path.exists(path):
        pytest.skip("real flow export not present")
    pf = parse_file(open(path, "rb").read())
    return pf.prints[pf.prints["ticker"] == "SPX"], pf.spots.get("SPX"), pf.asof


def test_block_tiles_appear_only_when_prints_are_given(sample):
    """A chain-only upload has no prints, so there is no block book to show
    and the picker must not offer one."""
    assert not [k for k in card_catalog(sample) if k.startswith("block")]


def test_block_tiles_come_from_the_real_flow_export(flow_prints):
    from dealer_gex.analytics import block_type_breakdown
    from dealer_gex.parsing import aggregate_prints
    prints, spot, asof = flow_prints
    a = analyze(aggregate_prints(prints), spot, asof, multiplier=100.0)
    cat = card_catalog(a, prints=prints)
    assert {"block_book", "block_types", "block_tiers",
            "block_strikes"} <= set(cat)

    bt = block_type_breakdown(prints, a.spot)
    rows = cat["block_types"].rows(a)
    assert [r[0] for r in rows] == list(bt.head(5)["block_type"])
    # premium-ranked, biggest first
    assert cat["block_types"].value(a) == str(bt.iloc[0]["block_type"])

    # the dominance sub-line reports counts, and `lenses` is an int not a list
    sub = cat["block_book"].sub(a)
    assert "of 3 lenses" in sub

    strikes = [float(r[0].replace(",", "")) for r in cat["block_strikes"].rows(a)]
    assert len(strikes) <= 5
    png = build_share_card(a, "SPX", fields=[cat["block_types"],
                                             cat["block_strikes"]])
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
