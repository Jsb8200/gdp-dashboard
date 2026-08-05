"""Shareable card: the headline read as a downloadable picture.

The dashboard is twenty tables deep. This is the opposite — one image with
the numbers you would actually paste into a chat, rendered in the
liquid-glass idiom: translucent panels floating over a blurred, tinted
backdrop, with a specular edge where the light catches.

Real glass, not a flat grey box with rounded corners: each panel blurs the
backdrop *behind it* (``ImageFilter.GaussianBlur`` on the cropped region),
then layers a coloured glow, a translucent fill, a bright hairline border
and a top-edge highlight. That is what makes it read as depth rather than
decoration.

Type is monospaced throughout — Consolas where it exists, a close free
substitute where it does not. Numbers in a mono face line up column-wise
between tiles, which is the whole reason a trading card is legible at a
glance.

Colour carries meaning rather than mood: every tile owns an accent, and the
accent says what the number is. Walls are green above and red below, the
flip is violet, expected move amber, and net GEX takes its colour from its
own sign. Read the card by hue before reading a digit.

Pillow only — it already ships with Streamlit, so the picture costs no new
dependency.

The tile set is data-driven (``CARD_FIELDS``): adding a number to the card
is one entry, not a layout change.
"""

from __future__ import annotations

import io
import math
import os
from dataclasses import dataclass
from datetime import date
from functools import lru_cache

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

from dealer_gex.analytics import Analysis, fmt_dollars
from dealer_gex.instruments import detect_instrument

# Rendered at 2x and kept there: the card is meant to survive being
# screenshotted, cropped and re-posted.
SCALE = 2
CARD_W, CARD_H = 1200, 675

#: Consolas first — it is what the card was drawn for, and it is present on
#: Windows and on any box with the ms core fonts. Everything after it is a
#: humanist mono of the same flavour, so the layout holds even where the
#: licensed face is missing (Linux CI, Streamlit Cloud). Ordered by how
#: close the metrics sit to Consolas, not by preference.
_MONO_CANDIDATES = {
    False: [
        "C:/Windows/Fonts/consola.ttf",
        "/Library/Fonts/Consolas.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/Consolas.ttf",
        os.path.expanduser("~/.fonts/consola.ttf"),
        os.path.expanduser("~/.local/share/fonts/consola.ttf"),
        "C:/Windows/Fonts/CascadiaMono.ttf",
        "/usr/share/fonts/truetype/jetbrains-mono/JetBrainsMono-Regular.ttf",
        "/usr/share/fonts/truetype/cascadia-code/CascadiaMono.ttf",
        "/System/Library/Fonts/Menlo.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    ],
    True: [
        "C:/Windows/Fonts/consolab.ttf",
        "/Library/Fonts/Consolas Bold.ttf",
        "/usr/share/fonts/truetype/msttcorefonts/Consolas_Bold.ttf",
        os.path.expanduser("~/.fonts/consolab.ttf"),
        os.path.expanduser("~/.local/share/fonts/consolab.ttf"),
        "C:/Windows/Fonts/CascadiaMono-Bold.ttf",
        "/usr/share/fonts/truetype/jetbrains-mono/JetBrainsMono-Bold.ttf",
        "/System/Library/Fonts/Menlo.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf",
    ],
}

#: Palette per regime. Near-black base with the tint carried by low-alpha
#: blobs rather than the background itself — the card reads dark, and the
#: colour is a signal (which way dealers are forced to hedge) rather than a
#: wash. ``accent`` drives the spot price and the verdict pill.
THEMES = {
    "long_gamma": {
        "base": (8, 11, 15), "blobs": [(16, 84, 70), (14, 48, 92), (28, 24, 86)],
        "accent": (94, 240, 186), "verdict": "LONG GAMMA",
        "tagline": "dealers hedge against the move — dampening",
    },
    "short_gamma": {
        "base": (14, 8, 11), "blobs": [(104, 26, 40), (78, 18, 60), (56, 14, 32)],
        "accent": (255, 116, 128), "verdict": "SHORT GAMMA",
        "tagline": "dealers hedge with the move — amplifying",
    },
}

#: Named accents. Semantic, not decorative — see the module docstring.
HUES = {
    "mint": (74, 222, 145),
    "rose": (251, 113, 133),
    "violet": (167, 139, 250),
    "amber": (251, 191, 36),
    "sky": (56, 189, 248),
    "fuchsia": (232, 121, 249),
    "lime": (163, 230, 53),
    "orange": (251, 146, 60),
}

#: Accents handed to ``extras`` tiles in order, so a caller that appends a
#: number does not have to pick a colour.
_EXTRA_HUES = ["fuchsia", "sky", "lime", "orange", "violet", "amber"]

#: Tiles per row, by tile count. Anything past eight takes four columns.
_COLS = {1: 1, 2: 2, 3: 3, 4: 2, 5: 3, 6: 3, 7: 4, 8: 4}


@dataclass(frozen=True)
class Field:
    """One tile. ``value`` takes the Analysis and returns display text.

    ``hue`` is a key into ``HUES``, or a callable taking the Analysis and
    returning one — that is how net GEX colours itself by its own sign.
    """
    key: str
    label: str
    value: object
    sub: object = None
    hue: object = "sky"


def _pct_of_spot(level, a: Analysis) -> str:
    if level is None or not a.spot:
        return ""
    return f"{(level / a.spot - 1) * 100:+.2f}% vs spot"


#: The headline set. Extend this list to put another number on the card.
CARD_FIELDS = [
    Field("net_gex", "Net GEX / 1% move",
          lambda a: fmt_dollars(a.total_gex),
          lambda a: "positive = stabilizing" if a.total_gex >= 0 else "negative = amplifying",
          hue=lambda a: "mint" if a.total_gex >= 0 else "rose"),
    Field("flip", "Gamma flip",
          lambda a: f"{a.gamma_flip:,.2f}" if a.gamma_flip is not None else "—",
          lambda a: (_pct_of_spot(a.gamma_flip, a) +
                     (f" · {len(a.flip_levels)} crossings" if len(a.flip_levels or []) > 1 else ""))
          if a.gamma_flip is not None else "no crossing in ±15%",
          hue="violet"),
    Field("call_wall", "Call wall",
          lambda a: f"{a.call_wall:,.2f}",
          lambda a: _pct_of_spot(a.call_wall, a), hue="mint"),
    Field("put_wall", "Put wall",
          lambda a: f"{a.put_wall:,.2f}",
          lambda a: _pct_of_spot(a.put_wall, a), hue="rose"),
    Field("expected_move", "Expected move (1σ)",
          lambda a: f"±{a.expected_move:,.2f}" if a.expected_move is not None else "—",
          lambda a: f"into {a.nearest_expiry}" if a.expected_move is not None else "",
          hue="amber"),
    Field("max_pain", "Max pain",
          lambda a: f"{a.max_pain:,.2f}",
          lambda a: _pct_of_spot(a.max_pain, a), hue="sky"),
]


@lru_cache(maxsize=64)
def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    for path in _MONO_CANDIDATES[bold]:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def _caps(s: str) -> str:
    """Upper-case the Latin letters and leave everything else alone.

    ``"Expected move (1σ)".upper()`` yields ``1Σ`` — a capital sigma is a
    different symbol from the one-standard-deviation sigma, and on a card
    about expected move that is a wrong label, not a style choice.
    """
    return "".join(c.upper() if c.isascii() else c for c in s)


def _mix(a, b, t: float):
    """Blend two RGB triples — used to lighten an accent for body text."""
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def _text(d: ImageDraw.ImageDraw, xy, s, font, fill, anchor="la"):
    d.text(xy, s, font=font, fill=fill, anchor=anchor)


def _tracked(d: ImageDraw.ImageDraw, xy, s, font, fill, track: float) -> float:
    """Draw letter-spaced text and return its width.

    Mono caps read as a label rather than a value once they are tracked out;
    Pillow has no letter-spacing, so the glyphs are placed one at a time.
    """
    x, y = xy
    for ch in s:
        d.text((x, y), ch, font=font, fill=fill, anchor="la")
        x += d.textlength(ch, font=font) + track
    return x - xy[0]


def _fit(d: ImageDraw.ImageDraw, s: str, max_w: float, start: int,
         floor: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """Largest font from ``start`` down to ``floor`` that keeps ``s`` inside
    ``max_w``. Mono is wide, so long index levels need this more than the
    proportional face did."""
    size = start
    while size > floor and d.textlength(s, font=_font(size * SCALE, bold)) > max_w:
        size -= 1
    return _font(size * SCALE, bold)


def _grain(w: int, h: int, strength: float = 3.5) -> Image.Image:
    """Deterministic fine noise. Large flat gradients on a dark card band
    badly once the PNG is re-compressed by a chat app; a couple of levels of
    grain hides the steps and costs nothing visually."""
    rng = np.random.default_rng(7)
    n = rng.normal(0, strength, (h, w, 1)).repeat(3, axis=2)
    return Image.fromarray(np.clip(n + 128, 0, 255).astype(np.uint8), "RGB")


def _backdrop(w: int, h: int, theme: dict) -> Image.Image:
    """A tinted gradient with soft colour blobs — the thing the glass has
    to refract. Without something structured behind it, a blurred panel is
    just grey."""
    grad = Image.new("RGB", (1, h))
    top, bot = theme["base"], tuple(int(c * 0.4) for c in theme["base"])
    for y in range(h):
        f = y / max(h - 1, 1)
        grad.putpixel((0, y), tuple(int(top[i] + (bot[i] - top[i]) * f) for i in range(3)))
    base = grad.resize((w, h))

    blobs = Image.new("RGB", (w, h), (0, 0, 0))
    bd = ImageDraw.Draw(blobs)
    spots = [(0.14, 0.18, 0.44), (0.86, 0.26, 0.38), (0.52, 0.94, 0.48)]
    for (cx, cy, r), colour in zip(spots, theme["blobs"]):
        rr = int(min(w, h) * r)
        bd.ellipse([cx * w - rr, cy * h - rr, cx * w + rr, cy * h + rr], fill=colour)
    blobs = blobs.filter(ImageFilter.GaussianBlur(int(min(w, h) * 0.16)))
    # keep it dark: the blobs are a tint over near-black, not a background
    img = Image.blend(base, blobs, 0.44)

    # vignette pulls the eye to the middle and keeps the corners near black
    vig = Image.new("L", (w, h), 0)
    ImageDraw.Draw(vig).ellipse(
        [-w * 0.22, -h * 0.34, w * 1.22, h * 1.34], fill=255)
    vig = vig.filter(ImageFilter.GaussianBlur(int(min(w, h) * 0.10)))
    img = Image.composite(img, Image.new("RGB", (w, h), (0, 0, 0)), vig)

    return ImageChops.overlay(img, _grain(w, h))


def _rounded_mask(size, radius: int) -> Image.Image:
    m = Image.new("L", size, 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, size[0] - 1, size[1] - 1],
                                        radius=radius, fill=255)
    return m


def _glow(img: Image.Image, box, radius: int, colour, strength: float) -> None:
    """Bleed the tile's accent into the backdrop behind it, in place.

    This is the step that makes the glass read as coloured rather than
    merely tinted: the light appears to come through the panel and land on
    what is behind it. Added, not blended, so black stays black.
    """
    x0, y0, x1, y1 = (int(v) for v in box)
    pad = int(radius * 1.6)
    gx0, gy0 = max(x0 - pad, 0), max(y0 - pad, 0)
    gx1, gy1 = min(x1 + pad, img.width), min(y1 + pad, img.height)
    w, h = gx1 - gx0, gy1 - gy0
    if w <= 2 or h <= 2:
        return
    layer = Image.new("RGB", (w, h), (0, 0, 0))
    ImageDraw.Draw(layer).rounded_rectangle(
        [x0 - gx0, y0 - gy0, x1 - gx0 - 1, y1 - gy0 - 1], radius=radius,
        fill=tuple(int(c * strength) for c in colour))
    layer = layer.filter(ImageFilter.GaussianBlur(pad * 0.7))
    img.paste(ImageChops.add(img.crop((gx0, gy0, gx1, gy1)), layer), (gx0, gy0))


def _glass(img: Image.Image, box, radius: int, *, blur: int = 26,
           tint: int = 16, border: int = 62, highlight: int = 78,
           accent=None) -> None:
    """Frost the backdrop inside ``box`` and lay glass over it, in place.

    The order matters: blur what is behind, lift it slightly, add a
    translucent white fill, then the hairline border and the top specular
    edge. Skipping the blur gives a sticker; skipping the highlight gives a
    flat panel. ``accent`` tints the border and the bottom inner edge, so
    the panel picks up the colour of the number it holds.
    """
    x0, y0, x1, y1 = (int(v) for v in box)
    w, h = x1 - x0, y1 - y0
    if w <= 2 or h <= 2:
        return
    mask = _rounded_mask((w, h), radius)

    region = img.crop((x0, y0, x1, y1)).filter(ImageFilter.GaussianBlur(blur))
    region = Image.blend(region, Image.new("RGB", (w, h), (255, 255, 255)), 0.026)
    img.paste(region, (x0, y0), mask)

    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ld = ImageDraw.Draw(layer)
    ld.rounded_rectangle([0, 0, w - 1, h - 1], radius=radius,
                         fill=(255, 255, 255, tint))
    edge = _mix((255, 255, 255), accent, 0.55) if accent else (255, 255, 255)
    ld.rounded_rectangle([0, 0, w - 1, h - 1], radius=radius,
                         outline=edge + (border,), width=max(1, SCALE))
    if accent:
        # light that entered the top edge leaving at the bottom: an accent
        # bloom rising off the lower edge. Drawn as fading lines rather than
        # an arc — an arc inscribed in the panel is an ellipse the size of
        # the panel, which crosses the middle of the tile instead of hugging
        # its edge.
        wash = max(min(h // 4, 26 * SCALE), 1)
        for i in range(wash):
            alpha = int(border * 0.72 * (1 - i / wash) ** 2)
            ld.line([(0, h - 1 - i), (w - 1, h - 1 - i)], fill=accent + (alpha,))

    # specular edge: brightest at the top, fading over ~a third of the panel
    spec = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    sd = ImageDraw.Draw(spec)
    span = max(h // 3, 1)
    for i in range(span):
        alpha = int(highlight * (1 - i / span) ** 2)
        sd.line([(0, i), (w - 1, i)], fill=(255, 255, 255, alpha))
    layer.alpha_composite(Image.composite(
        spec, Image.new("RGBA", (w, h), (0, 0, 0, 0)), mask))
    img.paste(Image.alpha_composite(
        img.crop((x0, y0, x1, y1)).convert("RGBA"), layer).convert("RGB"),
        (x0, y0), mask)


def _paint_rgba(img: Image.Image, box, draw_fn) -> None:
    """Run ``draw_fn`` against a transparent overlay and composite it.

    An RGBA fill handed straight to a draw on an RGB canvas silently loses
    its alpha, which paints translucent shapes solid — that is what once hid
    the verdict label inside its own pill. ``box`` bounds the overlay:
    compositing the full 2400x1350 canvas once per translucent shape is the
    difference between a card that renders instantly and one that stalls a
    Streamlit rerun.
    """
    x0, y0, x1, y1 = (int(v) for v in box)
    x0, y0 = max(x0, 0), max(y0, 0)
    x1, y1 = min(x1, img.width), min(y1, img.height)
    if x1 - x0 <= 0 or y1 - y0 <= 0:
        return
    overlay = Image.new("RGBA", (x1 - x0, y1 - y0), (0, 0, 0, 0))
    draw_fn(ImageDraw.Draw(overlay), x0, y0)
    img.paste(Image.alpha_composite(
        img.crop((x0, y0, x1, y1)).convert("RGBA"), overlay).convert("RGB"),
        (x0, y0))


def _hue_of(field: Field, a: Analysis) -> tuple:
    h = field.hue(a) if callable(field.hue) else field.hue
    return HUES.get(h, HUES["sky"]) if isinstance(h, str) else tuple(h)


def build_share_card(a: Analysis, ticker: str = "", *,
                     extras: list | None = None,
                     fields: list | None = None,
                     footnote: str = "") -> bytes:
    """Render the headline read as a PNG and return its bytes.

    ``fields`` overrides the tile set (defaults to ``CARD_FIELDS``);
    ``extras`` appends ready-made ``(label, value, sub)`` tuples for
    numbers that do not live on the Analysis — a dominant block type, a
    model forecast, whatever gets added next. A fourth element may carry an
    accent name from ``HUES``; without one the tiles cycle through
    ``_EXTRA_HUES``.
    """
    theme = THEMES.get(a.regime, THEMES["long_gamma"])
    w, h = CARD_W * SCALE, CARD_H * SCALE
    pad = 44 * SCALE

    img = _backdrop(w, h, theme)
    ink = (255, 255, 255)

    # --- header ---------------------------------------------------------
    head_h = 112 * SCALE
    head_box = (pad, pad, w - pad, pad + head_h)
    _glow(img, head_box, 28 * SCALE, theme["accent"], 0.16)
    _glass(img, head_box, 28 * SCALE, blur=30, accent=theme["accent"])
    d = ImageDraw.Draw(img)
    hx, hy = pad + 32 * SCALE, pad + head_h // 2
    label = (ticker or "").upper().strip() or "OPTIONS BOOK"
    tf = _font(38 * SCALE, True)
    _text(d, (hx, hy - 20 * SCALE), label, tf, ink, "lm")
    lw = d.textlength(label, font=tf)
    _text(d, (hx + lw + 20 * SCALE, hy - 19 * SCALE), f"{a.spot:,.2f}",
          _font(29 * SCALE, True), theme["accent"], "lm")

    inst = detect_instrument(ticker)
    # the multiplier changes every dollar figure on the card, so it is stated
    # rather than assumed: an NQ card and a QQQ card are otherwise identical
    mult = a.multiplier
    inst_bit = f"{inst.name} · ×{mult:g}" if inst.root else f"×{mult:g} per contract"
    sub = f"{inst_bit} · {a.asof:%d %b %Y} · {a.n_contracts:,} contracts"
    _text(d, (hx, hy + 25 * SCALE), sub,
          _fit(d, sub, w - 2 * pad - 300 * SCALE, 16, 11),
          _mix((255, 255, 255), theme["accent"], 0.35), "lm")

    # verdict pill, right-aligned in the header
    vtxt = theme["verdict"]
    vf = _font(21 * SCALE, True)
    vw = d.textlength(vtxt, font=vf) + 3 * 1.6 * SCALE + 40 * SCALE
    vx1 = w - pad - 30 * SCALE
    vy0 = pad + head_h // 2 - 40 * SCALE
    pill = (vx1 - vw, vy0, vx1, vy0 + 44 * SCALE)
    _paint_rgba(img, pill, lambda od, ox, oy: od.rounded_rectangle(
        [pill[0] - ox, pill[1] - oy, pill[2] - ox - 1, pill[3] - oy - 1],
        radius=22 * SCALE, fill=theme["accent"] + (54,),
        outline=theme["accent"] + (215,), width=max(2, SCALE)))
    d = ImageDraw.Draw(img)
    pw = d.textlength(vtxt, font=vf) + 3 * 1.6 * SCALE
    _tracked(d, ((pill[0] + pill[2]) / 2 - pw / 2,
                 (pill[1] + pill[3]) / 2 - 13 * SCALE),
             vtxt, vf, theme["accent"], 1.6 * SCALE)
    # tagline sits inside the header panel, not below it
    tag = theme["tagline"]
    _text(d, (vx1, pad + head_h - 24 * SCALE), tag,
          _fit(d, tag, w / 2 - 40 * SCALE, 15, 10),
          _mix((235, 235, 245), theme["accent"], 0.28), "rs")

    # --- tiles ----------------------------------------------------------
    spec = fields if fields is not None else CARD_FIELDS
    tiles = [(f.label, f.value(a), (f.sub(a) if f.sub else ""), _hue_of(f, a))
             for f in spec]
    for i, extra in enumerate(extras or []):
        lab, val, sb = extra[0], extra[1], (extra[2] if len(extra) > 2 else "")
        hue = extra[3] if len(extra) > 3 else _EXTRA_HUES[i % len(_EXTRA_HUES)]
        tiles.append((lab, val, sb, HUES.get(hue, HUES["fuchsia"])
                      if isinstance(hue, str) else tuple(hue)))

    # Column count is chosen to avoid a lonely tile on the last row and to
    # keep the grid at two rows for as long as it can: a third row cuts tile
    # height by a third, and it is height that decides whether the value can
    # be set large enough to read at chat-thumbnail size.
    cols = _COLS.get(len(tiles), 4)
    rows = max(1, math.ceil(len(tiles) / cols))
    top = pad + head_h + 24 * SCALE
    foot_h = 46 * SCALE
    grid_h = h - top - pad - foot_h
    gap = 20 * SCALE
    tw = (w - 2 * pad - gap * (cols - 1)) / cols
    th = (grid_h - gap * (rows - 1)) / rows

    for i, (lab, val, sb, hue) in enumerate(tiles):
        r, c = divmod(i, cols)
        x0 = pad + c * (tw + gap)
        y0 = top + r * (th + gap)
        box = (x0, y0, x0 + tw, y0 + th)
        _glow(img, box, 24 * SCALE, hue, 0.13)
        _glass(img, box, 24 * SCALE, accent=hue)
        d = ImageDraw.Draw(img)
        cx = x0 + 26 * SCALE
        # everything inside the tile is placed as a fraction of its height:
        # a seventh tile pushes the grid to a third row, and fixed offsets
        # then stack the label, the value and the caption on top of one
        # another instead of shrinking with the box
        inset = min(22 * SCALE, th * 0.11)

        # accent rule above the label — the tile's colour stated as a mark of
        # its own, so the hue is not the only thing carrying it
        ry = y0 + inset
        rule = (cx, ry, cx + 26 * SCALE, ry + 3 * SCALE)
        _paint_rgba(img, rule, lambda od, ox, oy, r=rule, c=hue: od.rounded_rectangle(
            [r[0] - ox, r[1] - oy, r[2] - ox - 1, r[3] - oy - 1],
            radius=SCALE, fill=c + (235,)))
        d = ImageDraw.Draw(img)
        _tracked(d, (cx, ry + 12 * SCALE), _caps(str(lab)),
                 _font(13 * SCALE, True), hue + (215,), 1.1 * SCALE)

        # the value gets whatever vertical room is left between the label and
        # the caption, so it is capped by height as well as by width
        val = str(val)
        cap_h = (18 * SCALE if sb else 0)
        room = th - inset - 34 * SCALE - cap_h
        vfont = _fit(d, val, tw - 52 * SCALE,
                     max(int(min(36, room / SCALE * 0.62)), 16), 15, bold=True)
        _text(d, (cx, ry + 34 * SCALE + (room - cap_h * 0.2) / 2), val, vfont,
              _mix(ink, hue, 0.22), "lm")
        if sb:
            sb = str(sb)
            _text(d, (cx, y0 + th - inset), sb,
                  _fit(d, sb, tw - 44 * SCALE, 14, 9),
                  (255, 255, 255, 150), "ls")

    # --- footer ---------------------------------------------------------
    d = ImageDraw.Draw(img)
    note = footnote or "positioning analysis from the option book — not trading advice"
    dot = (pad + 4 * SCALE, h - pad + 1 * SCALE,
           pad + 11 * SCALE, h - pad + 8 * SCALE)
    _paint_rgba(img, dot, lambda od, ox, oy: od.ellipse(
        [dot[0] - ox, dot[1] - oy, dot[2] - ox - 1, dot[3] - oy - 1],
        fill=theme["accent"] + (230,)))
    d = ImageDraw.Draw(img)
    _text(d, (pad + 22 * SCALE, h - pad + 9 * SCALE), note, _font(14 * SCALE),
          (255, 255, 255, 140), "ls")

    buf = io.BytesIO()
    # compress_level=6 rather than optimize=True: the grain makes the card
    # nearly incompressible, so exhaustive optimisation spends three seconds
    # to save 4% — long enough to be felt on every Streamlit rerun.
    img.save(buf, format="PNG", compress_level=6)
    return buf.getvalue()


def card_filename(a: Analysis, ticker: str = "") -> str:
    stem = (ticker or "book").upper().replace(" ", "-")
    return f"{stem}-positioning-{a.asof:%Y%m%d}.png"
