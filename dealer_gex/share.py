"""Shareable card: the headline read as a downloadable picture.

The dashboard is twenty tables deep. This is the opposite — one image with
the numbers you would actually paste into a chat, rendered in the
liquid-glass idiom: translucent panels floating over a blurred, tinted
backdrop, with a specular edge where the light catches.

Real glass, not a flat grey box with rounded corners: each panel blurs the
backdrop *behind it* (``ImageFilter.GaussianBlur`` on the cropped region),
then layers a translucent fill, a bright hairline border and a top-edge
highlight. That is what makes it read as depth rather than decoration.

Pillow only — it already ships with Streamlit, so the picture costs no new
dependency.

The tile set is data-driven (``CARD_FIELDS``): adding a number to the card
is one entry, not a layout change.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass
from datetime import date

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from dealer_gex.analytics import Analysis, fmt_dollars

# Rendered at 2x and kept there: the card is meant to survive being
# screenshotted, cropped and re-posted.
SCALE = 2
CARD_W, CARD_H = 1200, 675

_FONT_CANDIDATES = {
    False: [
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "C:/Windows/Fonts/segoeui.ttf",
    ],
    True: [
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "C:/Windows/Fonts/segoeuib.ttf",
    ],
}

#: Palette per regime — the backdrop is tinted by what the book is doing,
#: so the card reads before a single number is parsed.
THEMES = {
    "long_gamma": {
        "base": (14, 24, 38), "blobs": [(34, 150, 120), (26, 92, 168), (18, 60, 120)],
        "accent": (74, 222, 168), "verdict": "LONG GAMMA",
        "tagline": "dealers hedge against the move — dampening",
    },
    "short_gamma": {
        "base": (30, 14, 22), "blobs": [(190, 60, 78), (150, 40, 110), (90, 24, 60)],
        "accent": (255, 122, 130), "verdict": "SHORT GAMMA",
        "tagline": "dealers hedge with the move — amplifying",
    },
}


@dataclass(frozen=True)
class Field:
    """One tile. ``value`` takes the Analysis and returns display text."""
    key: str
    label: str
    value: object
    sub: object = None


def _pct_of_spot(level, a: Analysis) -> str:
    if level is None or not a.spot:
        return ""
    return f"{(level / a.spot - 1) * 100:+.2f}% vs spot"


#: The headline set. Extend this list to put another number on the card.
CARD_FIELDS = [
    Field("net_gex", "Net GEX / 1% move",
          lambda a: fmt_dollars(a.total_gex),
          lambda a: "positive = stabilizing" if a.total_gex >= 0 else "negative = amplifying"),
    Field("flip", "Gamma flip",
          lambda a: f"{a.gamma_flip:,.2f}" if a.gamma_flip is not None else "—",
          lambda a: (_pct_of_spot(a.gamma_flip, a) +
                     (f" · {len(a.flip_levels)} crossings" if len(a.flip_levels or []) > 1 else ""))
          if a.gamma_flip is not None else "no crossing in ±15%"),
    Field("call_wall", "Call wall",
          lambda a: f"{a.call_wall:,.2f}",
          lambda a: _pct_of_spot(a.call_wall, a)),
    Field("put_wall", "Put wall",
          lambda a: f"{a.put_wall:,.2f}",
          lambda a: _pct_of_spot(a.put_wall, a)),
    Field("expected_move", "Expected move (1σ)",
          lambda a: f"±{a.expected_move:,.2f}" if a.expected_move is not None else "—",
          lambda a: f"into {a.nearest_expiry}" if a.expected_move is not None else ""),
    Field("max_pain", "Max pain",
          lambda a: f"{a.max_pain:,.2f}",
          lambda a: _pct_of_spot(a.max_pain, a)),
]


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    for path in _FONT_CANDIDATES[bold]:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default(size)


def _text(d: ImageDraw.ImageDraw, xy, s, font, fill, anchor="la"):
    d.text(xy, s, font=font, fill=fill, anchor=anchor)


def _backdrop(w: int, h: int, theme: dict) -> Image.Image:
    """A tinted gradient with soft colour blobs — the thing the glass has
    to refract. Without something structured behind it, a blurred panel is
    just grey."""
    base = Image.new("RGB", (w, h), theme["base"])
    grad = Image.new("RGB", (1, h))
    top, bot = theme["base"], tuple(int(c * 0.45) for c in theme["base"])
    for y in range(h):
        f = y / max(h - 1, 1)
        grad.putpixel((0, y), tuple(int(top[i] + (bot[i] - top[i]) * f) for i in range(3)))
    base = grad.resize((w, h))

    blobs = Image.new("RGB", (w, h), (0, 0, 0))
    bd = ImageDraw.Draw(blobs)
    spots = [(0.16, 0.20, 0.42), (0.84, 0.30, 0.36), (0.55, 0.92, 0.46)]
    for (cx, cy, r), colour in zip(spots, theme["blobs"]):
        rr = int(min(w, h) * r)
        bd.ellipse([cx * w - rr, cy * h - rr, cx * w + rr, cy * h + rr], fill=colour)
    blobs = blobs.filter(ImageFilter.GaussianBlur(int(min(w, h) * 0.16)))
    return Image.blend(base, Image.blend(base, blobs, 0.55), 0.9)


def _rounded_mask(size, radius: int) -> Image.Image:
    m = Image.new("L", size, 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, size[0] - 1, size[1] - 1],
                                        radius=radius, fill=255)
    return m


def _glass(img: Image.Image, box, radius: int, *, blur: int = 26,
           tint: int = 30, border: int = 74, highlight: int = 96) -> None:
    """Frost the backdrop inside ``box`` and lay glass over it, in place.

    The order matters: blur what is behind, lift it slightly, add a
    translucent white fill, then the hairline border and the top specular
    edge. Skipping the blur gives a sticker; skipping the highlight gives a
    flat panel.
    """
    x0, y0, x1, y1 = (int(v) for v in box)
    w, h = x1 - x0, y1 - y0
    if w <= 2 or h <= 2:
        return
    mask = _rounded_mask((w, h), radius)

    region = img.crop((x0, y0, x1, y1)).filter(ImageFilter.GaussianBlur(blur))
    region = Image.blend(region, Image.new("RGB", (w, h), (255, 255, 255)), 0.05)
    img.paste(region, (x0, y0), mask)

    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    ld = ImageDraw.Draw(layer)
    ld.rounded_rectangle([0, 0, w - 1, h - 1], radius=radius,
                         fill=(255, 255, 255, tint))
    ld.rounded_rectangle([0, 0, w - 1, h - 1], radius=radius,
                         outline=(255, 255, 255, border), width=max(1, SCALE))
    # specular edge: brightest at the top, fading over ~a third of the panel
    spec = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    sd = ImageDraw.Draw(spec)
    span = max(h // 3, 1)
    for i in range(span):
        a = int(highlight * (1 - i / span) ** 2)
        sd.line([(radius * 0.6, i), (w - radius * 0.6, i)], fill=(255, 255, 255, a))
    layer.alpha_composite(Image.composite(
        spec, Image.new("RGBA", (w, h), (0, 0, 0, 0)), mask))
    img.paste(Image.alpha_composite(
        img.crop((x0, y0, x1, y1)).convert("RGBA"), layer).convert("RGB"),
        (x0, y0), mask)


def build_share_card(a: Analysis, ticker: str = "", *,
                     extras: list | None = None,
                     fields: list | None = None,
                     footnote: str = "") -> bytes:
    """Render the headline read as a PNG and return its bytes.

    ``fields`` overrides the tile set (defaults to ``CARD_FIELDS``);
    ``extras`` appends ready-made ``(label, value, sub)`` tuples for
    numbers that do not live on the Analysis — a dominant block type, a
    model forecast, whatever gets added next.
    """
    theme = THEMES.get(a.regime, THEMES["long_gamma"])
    w, h = CARD_W * SCALE, CARD_H * SCALE
    pad = 44 * SCALE

    img = _backdrop(w, h, theme)
    d = ImageDraw.Draw(img)
    ink, dim = (255, 255, 255), (255, 255, 255, 170)

    # --- header ---------------------------------------------------------
    head_h = 112 * SCALE
    _glass(img, (pad, pad, w - pad, pad + head_h), 28 * SCALE, blur=30)
    d = ImageDraw.Draw(img)
    hx, hy = pad + 32 * SCALE, pad + head_h // 2
    label = (ticker or "").upper().strip() or "OPTIONS BOOK"
    _text(d, (hx, hy - 20 * SCALE), label, _font(40 * SCALE, True), ink, "lm")
    lw = d.textlength(label, font=_font(40 * SCALE, True))
    _text(d, (hx + lw + 18 * SCALE, hy - 18 * SCALE), f"{a.spot:,.2f}",
          _font(30 * SCALE, True), theme["accent"], "lm")
    _text(d, (hx, hy + 24 * SCALE),
          f"dealer positioning · {a.asof:%d %b %Y} · {a.n_contracts:,} contracts",
          _font(17 * SCALE), (255, 255, 255, 165), "lm")

    # verdict pill, right-aligned in the header. Drawn on an RGBA layer and
    # composited: an RGBA fill passed straight to a draw on an RGB canvas
    # silently loses its alpha, which paints the pill solid and hides the
    # label inside it.
    vtxt = theme["verdict"]
    vf = _font(24 * SCALE, True)
    vw = d.textlength(vtxt, font=vf) + 46 * SCALE
    vx1 = w - pad - 30 * SCALE
    vy0 = pad + head_h // 2 - 40 * SCALE
    pill = (vx1 - vw, vy0, vx1, vy0 + 46 * SCALE)
    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    od.rounded_rectangle(pill, radius=23 * SCALE,
                         fill=theme["accent"] + (58,),
                         outline=theme["accent"] + (215,), width=max(2, SCALE))
    img.paste(Image.alpha_composite(img.convert("RGBA"), overlay).convert("RGB"),
              (0, 0))
    d = ImageDraw.Draw(img)
    _text(d, ((pill[0] + pill[2]) / 2, (pill[1] + pill[3]) / 2 + SCALE), vtxt, vf,
          theme["accent"], "mm")
    # tagline sits inside the header panel, not below it
    _text(d, (vx1, pad + head_h - 26 * SCALE), theme["tagline"],
          _font(16 * SCALE), (235, 235, 245), "rs")

    # --- tiles ----------------------------------------------------------
    tiles = [(f.label, f.value(a), (f.sub(a) if f.sub else ""))
             for f in (fields if fields is not None else CARD_FIELDS)]
    tiles += list(extras or [])
    cols = 3
    rows = max(1, math.ceil(len(tiles) / cols))
    top = pad + head_h + 26 * SCALE
    foot_h = 46 * SCALE
    grid_h = h - top - pad - foot_h
    gap = 20 * SCALE
    tw = (w - 2 * pad - gap * (cols - 1)) / cols
    th = (grid_h - gap * (rows - 1)) / rows

    for i, (lab, val, sub) in enumerate(tiles):
        r, c = divmod(i, cols)
        x0 = pad + c * (tw + gap)
        y0 = top + r * (th + gap)
        _glass(img, (x0, y0, x0 + tw, y0 + th), 24 * SCALE)
        d = ImageDraw.Draw(img)
        cx = x0 + 26 * SCALE
        _text(d, (cx, y0 + 24 * SCALE), str(lab).upper(),
              _font(15 * SCALE, True), (255, 255, 255, 150))
        # shrink the value until it fits the tile
        size = 42
        while size > 20 and d.textlength(str(val), font=_font(size * SCALE, True)) > tw - 52 * SCALE:
            size -= 2
        _text(d, (cx, y0 + th / 2 + 4 * SCALE), str(val), _font(size * SCALE, True),
              ink, "lm")
        if sub:
            _text(d, (cx, y0 + th - 26 * SCALE), str(sub), _font(15 * SCALE),
                  (255, 255, 255, 145), "ls")

    # --- footer ---------------------------------------------------------
    d = ImageDraw.Draw(img)
    note = footnote or "positioning analysis from the option book — not trading advice"
    _text(d, (pad + 4 * SCALE, h - pad + 6 * SCALE), note, _font(16 * SCALE),
          (255, 255, 255, 135), "ls")

    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def card_filename(a: Analysis, ticker: str = "") -> str:
    stem = (ticker or "book").upper().replace(" ", "-")
    return f"{stem}-positioning-{a.asof:%Y%m%d}.png"
