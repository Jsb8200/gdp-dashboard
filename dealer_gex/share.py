"""Shareable card: the headline read as a downloadable picture.

The dashboard is twenty tables deep. This is the opposite — one image with
the numbers you would actually paste into a chat.

Two looks, both dark:

``midnight`` (default) is the flat fintech-dashboard idiom — near-black
page, opaque panels a shade lighter, a hairline border, an icon chip beside
a muted label, a large white value, and a small coloured delta underneath.
Restraint is the point: colour appears on the delta, the icon and the
verdict pill, and nowhere else, so the eye lands on the numbers.

``glass`` is the liquid-glass alternative: the panel is a real lens. Near
the rim it drags the blurred backdrop outward along the surface normal,
fringes the channels (chromatic aberration), and carries a specular that
follows the curvature of the corner rather than running straight across the
top. That is what makes something read as a solid transparent object
instead of a frosted sticker.

Type is monospaced throughout — Consolas where it exists, a close free
substitute where it does not. Numbers in a mono face line up column-wise
between tiles, which is the whole reason a card of figures is legible at a
glance.

Pillow only — it already ships with Streamlit, so the picture costs no new
dependency.

Which numbers appear is entirely the caller's choice: ``card_catalog``
returns every tile the current analysis can support, keyed by name, and
``build_share_card(fields=...)`` renders whichever subset is handed to it,
in that order.
"""

from __future__ import annotations

import io
import math
import os
import re
from dataclasses import dataclass, replace
from datetime import date
from functools import lru_cache

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter, ImageFont

from dealer_gex.analytics import Analysis, fmt_dollars
from dealer_gex.instruments import detect_instrument

# Rendered at 2x and kept there: the card is meant to survive being
# screenshotted, cropped and re-posted. Width is a preset; height is
# whatever the chosen tiles need — see ``card_size``.
SCALE = 2

#: Card widths. ``desktop`` is a 1080p-class picture — the card at the size
#: a desktop actually shows it, rather than a chat thumbnail blown up.
SIZES = {"desktop": 1920, "wide": 1600, "share": 1200}
DEFAULT_SIZE = "desktop"

#: Consolas first — it is what the card was drawn for, and it is present on
#: Windows and on any box with the ms core fonts. Everything after it is a
#: humanist mono of the same flavour, so the layout holds even where the
#: licensed face is missing (Linux CI, Streamlit Cloud).
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

#: Regime palette. The accent is the only saturated colour on a midnight
#: card: the spot price, the verdict pill and a whisper of tint on the page.
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

#: Named accents, used for a tile's icon and its delta.
HUES = {
    "mint": (74, 222, 145),
    "rose": (251, 113, 133),
    "violet": (167, 139, 250),
    "amber": (251, 191, 36),
    "sky": (56, 189, 248),
    "fuchsia": (232, 121, 249),
    "lime": (163, 230, 53),
    "orange": (251, 146, 60),
    "slate": (148, 163, 184),
}

#: Accents handed to ``extras`` tiles in order, so a caller that appends a
#: number does not have to pick a colour.
_EXTRA_HUES = ["fuchsia", "sky", "lime", "orange", "violet", "amber"]

#: How tall a tile wants to be, and how tall one row of a ranked-list tile
#: is. See ``_tile_need``.
TILE_H = 158 * SCALE
ROW_H = 26 * SCALE
BARS_H = 268 * SCALE

#: The grid is six columns wide and every row is filled: a tile's ``span``
#: is how many sixths of the card it asks for, and whatever slack is left on
#: a row is shared out among the tiles on it. See ``_place``.
GRID_COLS = 6

#: The two looks. ``glass`` panels are lenses over a tinted backdrop;
#: ``midnight`` panels are opaque and flat, which is what a dashboard card
#: actually looks like.
STYLES = {
    "midnight": {
        "glass": False,
        "page": (10, 10, 12),
        "panel": (23, 23, 26),
        "panel_head": (28, 28, 32),
        "border": (44, 44, 50),
        "label": (139, 139, 149),
        "value": (246, 246, 248),
        "sub": (112, 112, 122),
        "radius": 20,
        "wash": 0.10,          # how much regime tint reaches the page
    },
    "glass": {
        "glass": True,
        "page": None,          # comes from the regime blobs
        "panel": None,
        "panel_head": None,
        "border": (255, 255, 255),
        "label": (170, 170, 180),
        "value": (255, 255, 255),
        "sub": (150, 150, 160),
        "radius": 24,
        "wash": 1.0,
    },
}
DEFAULT_STYLE = "midnight"


@dataclass(frozen=True)
class Field:
    """One tile. ``value`` takes the Analysis and returns display text.

    ``hue`` is a key into ``HUES``, or a callable taking the Analysis and
    returning one — that is how net GEX colours itself by its own sign.
    ``icon`` names the mark drawn in the tile's chip.

    A ranked list — the top strikes, the magnet map — is not one number, so
    it gets ``rows`` instead: a callable returning ``(left, right, hue)``
    triples, drawn as a small table in place of the big value. Those tiles
    also set ``span`` to take more than one grid column, because five rows
    of "level … dollars … what it is" need the width.
    """
    key: str
    label: str
    value: object
    sub: object = None
    hue: object = "slate"
    icon: str = "dot"
    rows: object = None
    #: Column headers. Set these and ``rows`` renders as a real table —
    #: header band, hairline rules, left-aligned columns — instead of the
    #: two-column ranked list.
    columns: tuple = ()
    #: ``(strike, value)`` pairs drawn as a price ladder — see ``_draw_bars``.
    bars: object = None
    span: int = 1
    #: Grid rows the tile occupies. A ladder wants to be tall and narrow.
    rowspan: int = 1


def _pct_of_spot(level, a: Analysis) -> str:
    if level is None or not a.spot:
        return ""
    return f"{(level / a.spot - 1) * 100:+.2f}% vs spot"


#: The headline set — the default six, and the order they appear in.
CARD_FIELDS = [
    Field("net_gex", "Net GEX / 1% move",
          lambda a: fmt_dollars(a.total_gex),
          lambda a: "positive = stabilizing" if a.total_gex >= 0 else "negative = amplifying",
          hue=lambda a: "mint" if a.total_gex >= 0 else "rose", icon="coin"),
    Field("flip", "Gamma flip",
          lambda a: f"{a.gamma_flip:,.2f}" if a.gamma_flip is not None else "—",
          lambda a: (_pct_of_spot(a.gamma_flip, a) +
                     (f" · {len(a.flip_levels)} crossings" if len(a.flip_levels or []) > 1 else ""))
          if a.gamma_flip is not None else "no crossing in ±15%",
          hue="violet", icon="target"),
    Field("call_wall", "Call wall",
          lambda a: f"{a.call_wall:,.2f}",
          lambda a: _pct_of_spot(a.call_wall, a), hue="mint", icon="up"),
    Field("put_wall", "Put wall",
          lambda a: f"{a.put_wall:,.2f}",
          lambda a: _pct_of_spot(a.put_wall, a), hue="rose", icon="down"),
    Field("expected_move", "Expected move (1σ)",
          lambda a: f"±{a.expected_move:,.2f}" if a.expected_move is not None else "—",
          lambda a: f"into {a.nearest_expiry}" if a.expected_move is not None else "",
          hue="amber", icon="range"),
    Field("max_pain", "Max pain",
          lambda a: f"{a.max_pain:,.2f}",
          lambda a: _pct_of_spot(a.max_pain, a), hue="sky", icon="pin"),
]

DEFAULT_KEYS = [f.key for f in CARD_FIELDS]


def _em_band(a: Analysis) -> str:
    if a.expected_move is None:
        return "—"
    return f"{a.spot - a.expected_move:,.0f} – {a.spot + a.expected_move:,.0f}"


def _playbook_heads(a: Analysis, master) -> list[str]:
    """The bold lead of each trading-interpretation bullet.

    The dashboard's bullets run two or three sentences each — the reasoning
    belongs on a screen you can scroll. What survives on a card is the first
    sentence, which carries the numbers.

    Taking the **bold** lead instead looks tidier and loses data: the
    confluence bullet nests bold inside bold, and "**Passive flows:**" puts
    its two dollar figures outside the bold entirely, so both would arrive
    on the card as a heading with nothing under it.

    Sentences are split on a period followed by a capital, not on any
    period — "700.03" and "±9.40" are full of them.
    """
    from dealer_gex.report import build_playbook
    heads = []
    for line in build_playbook(a, master=master):
        plain = re.sub(r"\*\*", "", line)
        plain = re.sub(r"\s+", " ", plain).strip()
        first = re.split(r"\.\s+(?=[A-Z])", plain, maxsplit=1)[0]
        heads.append(first.strip().rstrip("."))
    return heads


def card_catalog(a: Analysis, *, oi=None, magnets=None,
                 forecast=None, master=None) -> dict[str, Field]:
    """Every tile this analysis can support, keyed by name.

    The dashboard renders the keys as checkboxes; anything the current file
    cannot answer is simply absent, so the picker never offers a number that
    would come out as an em dash. Pass the optional analytics (OI walls,
    magnet map, model forecast) to unlock the tiles that need them.
    """
    # The default six go in only where the book can answer them: the
    # catalogue is what the dashboard's picker is built from, so an entry
    # that would render as an em dash must not appear in it at all.
    dead = set()
    if a.gamma_flip is None:
        dead |= {"flip"}
    if a.expected_move is None:
        dead |= {"expected_move"}
    cat: dict[str, Field] = {f.key: f for f in CARD_FIELDS if f.key not in dead}

    cat["spot"] = Field("spot", "Spot", lambda a: f"{a.spot:,.2f}",
                        lambda a: f"{a.asof:%d %b %Y}", hue="slate", icon="dot")
    cat["regime"] = Field(
        "regime", "Regime",
        lambda a: "LONG GAMMA" if a.regime == "long_gamma" else "SHORT GAMMA",
        lambda a: THEMES[a.regime]["tagline"],
        hue=lambda a: "mint" if a.regime == "long_gamma" else "rose", icon="flag")
    cat["contracts"] = Field(
        "contracts", "Contracts", lambda a: f"{a.n_contracts:,}",
        lambda a: f"{len(a.expiries)} expiries · ×{a.multiplier:g}",
        hue="slate", icon="stack")
    cat["range_frame"] = Field(
        "range_frame", "Wall-to-wall range",
        lambda a: f"{a.put_wall:,.0f} – {a.call_wall:,.0f}",
        lambda a: f"{(a.call_wall - a.put_wall) / a.spot * 100:.2f}% wide"
        if a.spot else "", hue="sky", icon="range")
    cat["dex"] = Field(
        "dex", "Net dealer delta", lambda a: fmt_dollars(a.dex),
        lambda a: "dealers long stock-equiv." if a.dex >= 0 else "dealers short stock-equiv.",
        hue=lambda a: "mint" if a.dex >= 0 else "rose", icon="coin")
    cat["vanna"] = Field(
        "vanna", "Vanna flow / 1pt IV", lambda a: fmt_dollars(a.vanna_flow),
        lambda a: "buying if IV falls" if a.vanna_flow >= 0 else "selling if IV falls",
        hue=lambda a: "mint" if a.vanna_flow >= 0 else "rose", icon="bolt")
    cat["charm"] = Field(
        "charm", "Charm flow / day", lambda a: fmt_dollars(a.charm_flow),
        lambda a: "buying from decay" if a.charm_flow >= 0 else "selling from decay",
        hue=lambda a: "mint" if a.charm_flow >= 0 else "rose", icon="bolt")

    if a.expected_move is not None:
        cat["em_band"] = Field(
            "em_band", "1σ range", _em_band,
            lambda a: f"into {a.nearest_expiry}", hue="amber", icon="range")
    if a.gamma_flip is not None and a.spot:
        cat["flip_distance"] = Field(
            "flip_distance", "Distance to flip",
            lambda a: f"{(a.gamma_flip / a.spot - 1) * 100:+.2f}%",
            lambda a: f"flip at {a.gamma_flip:,.2f}", hue="violet", icon="target")
    if oi is not None and getattr(oi, "call", None) is not None:
        cat["call_oi_wall"] = Field(
            "call_oi_wall", "Call OI wall", lambda a, v=oi.call: f"{v:,.2f}",
            lambda a, v=oi.call: _pct_of_spot(v, a), hue="mint", icon="wall")
    if oi is not None and getattr(oi, "put", None) is not None:
        cat["put_oi_wall"] = Field(
            "put_oi_wall", "Put OI wall", lambda a, v=oi.put: f"{v:,.2f}",
            lambda a, v=oi.put: _pct_of_spot(v, a), hue="rose", icon="wall")

    if magnets is not None and len(magnets):
        above = magnets[magnets["level"] > a.spot]
        below = magnets[magnets["level"] < a.spot]
        if len(above):
            r = above.iloc[0]
            cat["magnet_above"] = Field(
                "magnet_above", "Magnet above",
                lambda a, v=float(r["level"]): f"{v:,.2f}",
                lambda a, v=float(r["level"]), k=str(r["kind"]),
                s=float(r["strength"]): f"{k} · pull {s:.0f}",
                hue="lime", icon="up")
        if len(below):
            r = below.iloc[-1]
            cat["magnet_below"] = Field(
                "magnet_below", "Magnet below",
                lambda a, v=float(r["level"]): f"{v:,.2f}",
                lambda a, v=float(r["level"]), k=str(r["kind"]),
                s=float(r["strength"]): f"{k} · pull {s:.0f}",
                hue="orange", icon="down")

    # --- ranked lists ---------------------------------------------------
    bs = a.by_strike
    if bs is not None and len(bs):
        top = bs.reindex(bs["net_gex"].abs().sort_values(ascending=False).index)
        top = top.head(5)
        if len(top):
            cat["top_gex"] = Field(
                "top_gex", "Top 5 GEX strikes",
                lambda a, t=top: f"{float(t.iloc[0]['strike']):,.2f}",
                lambda a, t=top: f"heaviest of {len(bs):,} strikes",
                hue="sky", icon="wall", span=3,
                rows=lambda a, t=top: [
                    (f"{float(r['strike']):,.2f}",
                     fmt_dollars(float(r["net_gex"])),
                     "mint" if r["net_gex"] >= 0 else "rose")
                    for _, r in t.iterrows()])

    if bs is not None and len(bs) > 2 and a.spot:
        # a window around spot, capped: forty bars is the most that stays
        # legible across the card, and the ones far from spot are the ones
        # that matter least
        near = bs[(bs["strike"] >= a.spot * 0.92) & (bs["strike"] <= a.spot * 1.08)]
        if len(near) < 5:
            near = bs
        if len(near) > 26:
            near = near.reindex(
                (near["strike"] - a.spot).abs().sort_values().index).head(26)
        near = near.sort_values("strike")
        cat["gex_bars"] = Field(
            "gex_bars", "Net GEX by strike",
            lambda a: fmt_dollars(a.total_gex),
            lambda a, n=len(near): f"{n} strikes around spot",
            hue="sky", icon="bars", span=2, rowspan=2,
            bars=lambda a, t=near: [(float(r.strike), float(r.net_gex))
                                    for r in t.itertuples()])

    if magnets is not None and len(magnets):
        top_m = magnets.head(5)
        cat["magnets"] = Field(
            "magnets", "Magnet map",
            lambda a, t=top_m: f"{float(t.iloc[0]['level']):,.2f}",
            lambda a, t=top_m: f"{len(t)} levels within ±10%",
            hue="lime", icon="target", span=3,
            rows=lambda a, t=top_m: [
                (f"{float(r['level']):,.2f}",
                 f"{str(r['kind'])[:11]} {float(r['strength']):.0f}",
                 "lime" if str(r["kind"]).startswith("magnet") else "orange")
                for _, r in t.iterrows()])

    # What the numbers rest on. A card travels without the dashboard's
    # methodology panel attached, so the caveats have to be able to travel
    # with it — every level here is model output, not an observed price.
    _weights = {"open_interest": "open interest (standing book)",
                "volume": "volume (today's tape)",
                "flow": "signed order flow"}
    cat["assumptions"] = Field(
        "assumptions", "Assumptions",
        lambda a: "estimate",
        lambda a: "levels are model output, not observed prices",
        hue="slate", icon="flag", span=3,
        columns=("Assumption", "Setting", "If it is wrong"),
        rows=lambda a: [
            ("Dealer sign", "long calls / short puts",
             "every GEX sign flips"),
            ("Weighting", _weights.get(a.weight_mode, a.weight_mode),
             "levels describe a different book"),
            ("Multiplier", f"×{a.multiplier:g}",
             "every dollar figure rescales"),
            ("Rate", f"{a.rate:.2%}",
             "only fills missing greeks"),
            ("Greeks", "Black-Scholes, IV fixed",
             "no vol response as spot moves"),
            ("Book", f"{a.n_contracts:,} contracts · {len(a.expiries)} expiries",
             "expired contracts excluded"),
        ])

    heads = _playbook_heads(a, master)
    if heads:
        cat["interpretation"] = Field(
            "interpretation", "Trading interpretation",
            lambda a: "read",
            lambda a: "",
            hue="amber", icon="flag", span=6,
            rows=lambda a, h=heads: [(f"· {t}", "", "slate") for t in h])

    fmove = getattr(forecast, "blended_pct", None) if forecast is not None else None
    if fmove is not None and getattr(forecast, "status", "") == "ok":
        cat["forecast"] = Field(
            "forecast", "Model next-session move",
            lambda a, p=fmove: f"±{a.spot * p:,.2f}",
            lambda a, w=getattr(forecast, "weight", 0.0): f"{w:.0%} model weight",
            hue="fuchsia", icon="bolt")
    return cat


@lru_cache(maxsize=96)
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
    """Blend two RGB triples."""
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))


def _text(d: ImageDraw.ImageDraw, xy, s, font, fill, anchor="la"):
    d.text(xy, s, font=font, fill=fill, anchor=anchor)


def _tracked(d: ImageDraw.ImageDraw, xy, s, font, fill, track: float) -> float:
    """Draw letter-spaced text and return its width. Mono caps read as a
    label rather than a value once tracked out; Pillow has no letter
    spacing, so the glyphs are placed one at a time."""
    x, y = xy
    for ch in s:
        d.text((x, y), ch, font=font, fill=fill, anchor="la")
        x += d.textlength(ch, font=font) + track
    return x - xy[0]


def _fit(d: ImageDraw.ImageDraw, s: str, max_w: float, start: int,
         floor: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """Largest font from ``start`` down to ``floor`` that keeps ``s`` inside
    ``max_w``. Mono is wide, so long index levels need this."""
    size = start
    while size > floor and d.textlength(s, font=_font(size * SCALE, bold)) > max_w:
        size -= 1
    return _font(size * SCALE, bold)


def _paint_rgba(img: Image.Image, box, draw_fn) -> None:
    """Run ``draw_fn`` against a transparent overlay and composite it.

    An RGBA fill handed straight to a draw on an RGB canvas silently loses
    its alpha, which paints translucent shapes solid. ``box`` bounds the
    overlay: compositing the full canvas once per translucent shape is the
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


# --------------------------------------------------------------------------
# backdrops
# --------------------------------------------------------------------------

def _grain(w: int, h: int, strength: float = 3.5) -> Image.Image:
    """Deterministic fine noise. Large flat gradients on a dark card band
    badly once a chat app re-compresses the PNG; a couple of levels of grain
    hides the steps and costs nothing visually."""
    rng = np.random.default_rng(7)
    n = rng.normal(0, strength, (h, w, 1)).repeat(3, axis=2)
    return Image.fromarray(np.clip(n + 128, 0, 255).astype(np.uint8), "RGB")


def _backdrop(w: int, h: int, theme: dict, style: dict) -> Image.Image:
    """The page the panels sit on."""
    if not style["glass"]:
        # flat near-black with a faint centre lift and a whisper of regime
        # tint, so the card still reads long/short before a digit is parsed
        page = np.zeros((h, w, 3), dtype=np.float32)
        page += np.asarray(style["page"], dtype=np.float32)
        ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
        r = np.hypot((xs - w / 2) / (w / 2), (ys - h / 2) / (h / 2))
        lift = np.clip(1.0 - r / 1.35, 0, 1) ** 2
        page += lift[..., None] * np.asarray(theme["accent"], dtype=np.float32) * style["wash"]
        img = Image.fromarray(np.clip(page, 0, 255).astype(np.uint8), "RGB")
        return ImageChops.overlay(img, _grain(w, h, 2.2))

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
    img = Image.blend(base, blobs, 0.44)

    vig = Image.new("L", (w, h), 0)
    ImageDraw.Draw(vig).ellipse([-w * 0.22, -h * 0.34, w * 1.22, h * 1.34], fill=255)
    vig = vig.filter(ImageFilter.GaussianBlur(int(min(w, h) * 0.10)))
    img = Image.composite(img, Image.new("RGB", (w, h), (0, 0, 0)), vig)
    return ImageChops.overlay(img, _grain(w, h))


# --------------------------------------------------------------------------
# panels
# --------------------------------------------------------------------------

def _sdf(w: int, h: int, radius: float, power: float = 4.0) -> np.ndarray:
    """Signed distance, in pixels, to a squircle-cornered rounded rectangle.

    Negative inside, zero on the edge, positive outside. Everything the
    glass does is derived from this one field: the antialiased mask, the
    depth-from-edge that drives refraction, and — through its gradient — the
    surface normal that makes the highlight bend around the corners.
    """
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    qx = np.abs(xs - (w - 1) / 2.0) - (w / 2.0 - radius)
    qy = np.abs(ys - (h - 1) / 2.0) - (h / 2.0 - radius)
    corner = (np.maximum(qx, 0.0) ** power
              + np.maximum(qy, 0.0) ** power) ** (1.0 / power)
    return np.minimum(np.maximum(qx, qy), 0.0) + corner - radius


def _sample(src: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    """Bilinear lookup into an HxWx3 array at fractional coordinates."""
    h, w = src.shape[:2]
    fx = np.clip(xs, 0, w - 1.001)
    fy = np.clip(ys, 0, h - 1.001)
    x0, y0 = fx.astype(np.int32), fy.astype(np.int32)
    x1, y1 = np.minimum(x0 + 1, w - 1), np.minimum(y0 + 1, h - 1)
    ax, ay = (fx - x0)[..., None], (fy - y0)[..., None]
    return (src[y0, x0] * (1 - ax) * (1 - ay) + src[y0, x1] * ax * (1 - ay)
            + src[y1, x0] * (1 - ax) * ay + src[y1, x1] * ax * ay)


def _glow(img: Image.Image, box, radius: int, colour, strength: float) -> None:
    """Bleed a panel's accent into the backdrop behind it, in place. Added,
    not blended, so black stays black."""
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


def _glass_panel(img: Image.Image, box, radius: int, *, blur: int = 26,
                 tint: int = 10, border: int = 78, highlight: int = 96,
                 accent=None, thickness: float = 0.32) -> None:
    """Lay a pane of glass over ``box``, in place.

    A blurred crop with a bright top edge is a frosted sticker, not glass.
    What makes something read as a solid transparent object is that it bends
    what is behind it: near the rim the pane acts as a lens, so the backdrop
    is dragged outward, colour-fringed by dispersion, and the highlight
    tracks the curvature of the edge instead of running across the top.
    """
    x0, y0, x1, y1 = (int(v) for v in box)
    w, h = x1 - x0, y1 - y0
    if w <= 2 or h <= 2:
        return

    # crop with margin: the rim bends in what sits *outside* the panel, so
    # the sampler needs real surroundings to reach for
    reach = int(max(radius * 0.9, 16))
    sx0, sy0 = max(x0 - reach, 0), max(y0 - reach, 0)
    sx1, sy1 = min(x1 + reach, img.width), min(y1 + reach, img.height)
    src = np.asarray(
        img.crop((sx0, sy0, sx1, sy1)).filter(ImageFilter.GaussianBlur(blur)),
        dtype=np.float32)

    d = _sdf(w, h, float(radius))
    depth = -d                                   # distance inward from the rim
    cover = np.clip(0.5 - d, 0.0, 1.0)           # antialiased coverage
    gy, gx = np.gradient(d)
    glen = np.hypot(gx, gy) + 1e-6
    nx, ny = gx / glen, gy / glen                # unit normal, pointing outward

    # Keep the lensing band narrow — spread it over most of the panel and
    # the tile stops looking like glass and starts looking embossed.
    edge = max(radius * 0.55, 10.0)
    lens = np.clip(1.0 - depth / edge, 0.0, 1.0) ** 2 * (radius * thickness)

    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    bx, by = xs + (x0 - sx0), ys + (y0 - sy0)
    out = np.empty((h, w, 3), dtype=np.float32)
    for ch, disp in ((0, 1.07), (1, 1.0), (2, 0.93)):   # chromatic aberration
        out[..., ch] = _sample(src, bx + nx * lens * disp,
                               by + ny * lens * disp)[..., ch]

    out = out * (1.0 - 0.018) + 255.0 * 0.018            # a hint of milkiness
    out = out * (1.0 - tint / 255.0) + 255.0 * (tint / 255.0)

    # speculars from the normal: bright where the surface faces the light, a
    # dimmer accent rim where it faces away. Because the normal rotates
    # around the corners, so does the highlight.
    lx, ly = -0.52, -0.85
    facing = nx * lx + ny * ly
    lit = np.clip(facing, 0.0, 1.0) ** 3.0 * np.exp(-depth / (edge * 0.32))
    out += (lit * highlight)[..., None]
    if accent:
        rim = np.clip(-facing, 0.0, 1.0) ** 3.0 * np.exp(-depth / (edge * 0.35))
        out += rim[..., None] * np.asarray(accent, dtype=np.float32) * 0.5

    line = np.exp(-((depth - 1.0 * SCALE) / (1.4 * SCALE)) ** 2)
    edge_rgb = np.asarray(_mix((255, 255, 255), accent, 0.5) if accent
                          else (255, 255, 255), dtype=np.float32)
    out += (line * (border / 255.0))[..., None] * edge_rgb

    img.paste(Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB"),
              (x0, y0), Image.fromarray((cover * 255).astype(np.uint8), "L"))


def _flat_panel(img: Image.Image, box, radius: int, style: dict, *,
                fill=None, accent=None, wash: float = 0.0) -> None:
    """An opaque dashboard card: solid fill, hairline border, and a single
    lighter pixel along the top edge for the suggestion of a raised surface.

    ``wash`` bleeds the accent across the panel from the left — used on the
    header so the regime is legible without colouring every tile.
    """
    x0, y0, x1, y1 = (int(v) for v in box)
    w, h = x1 - x0, y1 - y0
    if w <= 2 or h <= 2:
        return
    body = fill or style["panel"]
    panel = Image.new("RGB", (w, h), body)
    if wash and accent:
        grad = np.linspace(1.0, 0.0, w, dtype=np.float32)[None, :, None] ** 2
        arr = np.asarray(panel, dtype=np.float32)
        arr += grad * np.asarray(accent, dtype=np.float32) * wash
        panel = Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8), "RGB")

    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, w - 1, h - 1], radius=radius,
                                           fill=255)
    img.paste(panel, (x0, y0), mask)

    _paint_rgba(img, box, lambda od, ox, oy: (
        od.rounded_rectangle([0, 0, w - 1, h - 1], radius=radius,
                             outline=style["border"] + (255,), width=SCALE),
        od.line([(radius, SCALE // 2), (w - radius, SCALE // 2)],
                fill=(255, 255, 255, 26), width=SCALE),
    ))


# --------------------------------------------------------------------------
# icon marks
# --------------------------------------------------------------------------

def _icon(d: ImageDraw.ImageDraw, box, kind: str, colour) -> None:
    """A small geometric mark inside the tile's chip.

    Glyphs are drawn rather than typed: an emoji font is not guaranteed
    anywhere this runs, and a missing glyph renders as a tofu box.
    """
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    r = (x1 - x0) * 0.30
    lw = max(int(1.6 * SCALE), 2)
    if kind == "up":
        d.polygon([(cx, cy - r), (cx + r, cy + r * 0.7), (cx - r, cy + r * 0.7)],
                  fill=colour)
    elif kind == "down":
        d.polygon([(cx, cy + r), (cx + r, cy - r * 0.7), (cx - r, cy - r * 0.7)],
                  fill=colour)
    elif kind == "target":
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=colour, width=lw)
        d.ellipse([cx - r * 0.3, cy - r * 0.3, cx + r * 0.3, cy + r * 0.3], fill=colour)
    elif kind == "range":
        d.line([(cx - r, cy), (cx + r, cy)], fill=colour, width=lw)
        d.line([(cx - r, cy - r * 0.6), (cx - r, cy + r * 0.6)], fill=colour, width=lw)
        d.line([(cx + r, cy - r * 0.6), (cx + r, cy + r * 0.6)], fill=colour, width=lw)
    elif kind == "pin":
        d.ellipse([cx - r * 0.8, cy - r, cx + r * 0.8, cy + r * 0.6],
                  outline=colour, width=lw)
        d.line([(cx, cy + r * 0.4), (cx, cy + r)], fill=colour, width=lw)
    elif kind == "coin":
        d.ellipse([cx - r, cy - r, cx + r, cy + r], outline=colour, width=lw)
        d.line([(cx, cy - r * 0.55), (cx, cy + r * 0.55)], fill=colour, width=lw)
    elif kind == "bolt":
        d.polygon([(cx + r * 0.4, cy - r), (cx - r * 0.6, cy + r * 0.15),
                   (cx, cy + r * 0.15), (cx - r * 0.4, cy + r),
                   (cx + r * 0.6, cy - r * 0.15), (cx, cy - r * 0.15)], fill=colour)
    elif kind == "wall":
        for i, f in enumerate((-0.62, 0.0, 0.62)):
            d.line([(cx - r, cy + r * f), (cx + r, cy + r * f)],
                   fill=colour, width=lw if i == 1 else max(lw - 1, 1))
    elif kind == "stack":
        for f in (-0.6, 0.0, 0.6):
            d.rounded_rectangle([cx - r, cy + r * f - lw, cx + r, cy + r * f + lw],
                                radius=lw, fill=colour)
    elif kind == "flag":
        d.line([(cx - r * 0.6, cy - r), (cx - r * 0.6, cy + r)], fill=colour, width=lw)
        d.polygon([(cx - r * 0.6, cy - r), (cx + r, cy - r * 0.45),
                   (cx - r * 0.6, cy + r * 0.1)], fill=colour)
    elif kind == "bars":
        for i, f in enumerate((-0.66, 0.0, 0.66)):
            hh = r * (0.5 if i == 1 else 1.0)
            d.rectangle([cx + r * f - lw * 0.9, cy + r - hh * 2,
                         cx + r * f + lw * 0.9, cy + r], fill=colour)
    else:                                    # dot
        d.ellipse([cx - r * 0.55, cy - r * 0.55, cx + r * 0.55, cy + r * 0.55],
                  fill=colour)


def _spread(ys, sep: float, lo: float, hi: float) -> list[float]:
    """Push overlapping label positions apart to at least ``sep``, keeping
    their order and staying inside ``[lo, hi]``.

    ``ys`` must be sorted. Labels are placed greedily downward, then the
    whole stack is shifted back up if it overran the bottom — shifting is
    what keeps the order intact, where clamping each label individually
    would pile them all on the last row.
    """
    out = []
    for y in ys:
        out.append(max(y, out[-1] + sep) if out else y)
    if out:
        over = out[-1] - hi
        if over > 0:
            out = [max(y - over, lo) for y in out]
            # re-spread downward in case the clamp at ``lo`` recreated an overlap
            for i in range(1, len(out)):
                out[i] = max(out[i], out[i - 1] + sep)
    return out


def _draw_bars(d: ImageDraw.ImageDraw, box, data, spot: float, sty: dict) -> None:
    """Net GEX by strike as a price ladder: strike up the vertical axis,
    each strike's gamma as a horizontal bar off a zero line.

    Price runs vertically here because that is how a trader reads a level —
    high strikes at the top, low at the bottom, spot marked between them.
    A strike axis laid out horizontally forces you to re-map the picture
    onto the chart you already have in your head.

    The zero line is placed by the data, not down the middle: a book that
    is long gamma nearly everywhere should show one shallow red stub beside
    a wall of green, and centring the axis would draw that as balanced.

    Only the five heaviest strikes are named. Labelling twenty-six of them
    turns the axis into a wall of digits and hides the thing worth reading;
    the rest stay dimmed so the top five carry the eye, and the top and
    bottom of the window are marked so the range is still legible.
    """
    x0, y0, x1, y1 = box
    if not data:
        return
    vals = [v for _, v in data]
    hi, lo = max(max(vals), 0.0), min(min(vals), 0.0)
    span = (hi - lo) or 1.0

    f = _font(12 * SCALE)
    fb = _font(12 * SCALE, True)
    ks = [k for k, _ in data]
    # a gutter on the right for the strike labels, so the longest bar cannot
    # run into its own number
    gutter = max(d.textlength(f"{k:,.0f}", font=fb) for k in ks) + 14 * SCALE
    px1 = x1 - gutter
    zero_x = x0 + (-lo / span) * (px1 - x0)
    bh = (y1 - y0) / len(data)

    # rank by absolute gamma: a big negative strike matters as much as a big
    # positive one, and ranking signed would bury it
    top = set(sorted(range(len(vals)), key=lambda i: -abs(vals[i]))[:5])
    dim = sty["panel"] or (20, 20, 24)

    # highest strike at the top: the data arrives strike-ascending
    marks = []
    for i, (k, v) in enumerate(data):
        cy = y1 - (i + 0.5) * bh
        tip = zero_x + (v / span) * (px1 - x0)
        colour = HUES["mint"] if v >= 0 else HUES["rose"]
        if i not in top:
            colour = _mix(colour, dim, 0.55)
        d.rectangle([min(tip, zero_x), cy - bh * 0.33,
                     max(tip, zero_x), cy + bh * 0.33], fill=colour)
        if i in top:
            marks.append([cy, cy, f"{k:,.0f}",
                          HUES["mint"] if v >= 0 else HUES["rose"]])

    # The heaviest strikes cluster, so three of the five can land on
    # neighbouring rows and overprint each other. Push the labels apart to a
    # legible spacing, then run a leader back to the bar each one belongs to
    # — a moved label with nothing tying it to its row is worse than none.
    marks.sort(key=lambda m: m[1])
    for m, ly in zip(marks, _spread([m[1] for m in marks], 15 * SCALE, y0, y1)):
        m[0] = ly
    for ly, cy, txt, colour in marks:
        tw = d.textlength(txt, font=fb)
        if abs(ly - cy) > 1.5 * SCALE:
            d.line([(px1 + 4 * SCALE, cy), (x1 - tw - 7 * SCALE, ly)],
                   fill=_mix(colour, dim, 0.45), width=max(1, SCALE // 2))
        _text(d, (x1, ly), txt, fb, colour, "rm")

    d.line([(zero_x, y0), (zero_x, y1)], fill=sty["border"], width=SCALE)

    if spot and len(ks) > 1 and ks[0] <= spot <= ks[-1]:
        j = max(i for i, k in enumerate(ks) if k <= spot)
        frac = ((spot - ks[j]) / (ks[j + 1] - ks[j])) if j + 1 < len(ks) else 0.0
        sy = y1 - (j + 0.5 + frac) * bh
        for xx in range(int(x0), int(px1), 8 * SCALE):
            d.line([(xx, sy), (xx + 4 * SCALE, sy)], fill=sty["label"], width=SCALE)
        _text(d, (x0, sy - 4 * SCALE), f"{spot:,.0f}", f, sty["label"], "ld")

    _text(d, (x0, y0), f"{ks[-1]:,.0f}", f, sty["sub"], "la")
    _text(d, (x0, y1), f"{ks[0]:,.0f}", f, sty["sub"], "ld")


def _draw_table(img: Image.Image, d: ImageDraw.ImageDraw, box, columns, rows,
                sty: dict) -> None:
    """A real table: header band, hairline rules, left-aligned columns.

    The ranked-list layout puts one value hard right, which works for a
    price ladder and reads as a mess for prose. Columns are sized to their
    own widest cell, and the last one takes whatever is left, so a short
    key column does not steal room from a sentence.
    """
    x0, y0, x1, y1 = box
    n = len(rows) + 1
    rh = min(ROW_H, (y1 - y0) / max(n, 1))
    size = max(int(min(14, rh / SCALE * 0.60)), 9)
    f, fb = _font(size * SCALE), _font(size * SCALE, True)

    ncol = len(columns)
    pad_c = 14 * SCALE
    # every column but the last is as wide as its widest cell
    widths = []
    for i in range(ncol - 1):
        cells = [str(columns[i])] + [str(r[i]) for r in rows if len(r) > i]
        widths.append(max(d.textlength(c, font=fb) for c in cells) + pad_c * 2)
    widths.append(max((x1 - x0) - sum(widths), pad_c * 4))

    # header band, then a rule under it and one between each pair of rows
    _paint_rgba(img, (x0, y0, x1, y0 + rh),
                lambda od, ox, oy: od.rectangle(
                    [0, 0, int(x1 - x0), int(rh)], fill=(255, 255, 255, 12)))
    d = ImageDraw.Draw(img)
    for i in range(n + 1):
        yy = y0 + i * rh
        d.line([(x0, yy), (x1, yy)], fill=sty["border"], width=max(1, SCALE // 2))

    cx = x0
    for i, head in enumerate(columns):
        _text(d, (cx + pad_c, y0 + rh / 2), _caps(str(head)), fb,
              sty["label"], "lm")
        cx += widths[i]

    for j, row in enumerate(rows):
        ry = y0 + (j + 1) * rh + rh / 2
        cx = x0
        for i in range(ncol):
            cell = str(row[i]) if len(row) > i else ""
            # the first column names the thing; the rest report it
            colour = sty["value"] if i == 0 else sty["sub"]
            _text(d, (cx + pad_c, ry),
                  cell, fb if i == 0 else f, colour, "lm")
            cx += widths[i]


def _initials(name: str) -> str:
    """Up to two initials from a handle, for when there is no picture."""
    parts = [p for p in re.split(r"[\s._\-@]+", str(name or "")) if p]
    if not parts:
        return "?"
    if len(parts) == 1:
        return parts[0][:2].upper()
    return (parts[0][:1] + parts[1][:1]).upper()


def _draw_avatar(img: Image.Image, box, avatar, name: str, accent, sty: dict) -> None:
    """A round profile picture, or initials on a tinted disc.

    The image is cover-cropped to a square before the circular mask, so a
    portrait or a banner-shaped file is centred rather than squashed. A file
    that Pillow cannot open falls back to initials instead of failing the
    whole card — a broken avatar is not a reason to lose the numbers.
    """
    x0, y0, x1, y1 = (int(v) for v in box)
    size = min(x1 - x0, y1 - y0)
    if size <= 2:
        return
    mask = Image.new("L", (size * 4, size * 4), 0)
    ImageDraw.Draw(mask).ellipse([0, 0, size * 4 - 1, size * 4 - 1], fill=255)
    mask = mask.resize((size, size), Image.LANCZOS)   # smooth rim at 1x cost

    face = None
    if avatar is not None:
        try:
            src = (avatar if isinstance(avatar, Image.Image)
                   else Image.open(io.BytesIO(avatar)))
            src = src.convert("RGB")
            side = min(src.size)
            left, top = (src.width - side) // 2, (src.height - side) // 2
            face = src.crop((left, top, left + side, top + side)).resize(
                (size, size), Image.LANCZOS)
        except Exception:                              # noqa: BLE001 - see docstring
            face = None
    if face is None:
        face = Image.new("RGB", (size, size), _mix(sty["panel"], accent, 0.28))
        fd = ImageDraw.Draw(face)
        txt = _initials(name)
        fd.text((size / 2, size / 2 + SCALE), txt,
                font=_font(int(size * 0.42), True), fill=accent, anchor="mm")

    img.paste(face, (x0, y0), mask)
    _paint_rgba(img, (x0, y0, x0 + size, y0 + size),
                lambda od, ox, oy: od.ellipse(
                    [0, 0, size - 1, size - 1],
                    outline=tuple(accent) + (170,), width=max(1, SCALE)))


_DELTA = re.compile(r"^([+\-−][\d.,]+%?)(.*)$")


def _draw_sub(d: ImageDraw.ImageDraw, xy, text: str, font, style: dict,
              anchor: str = "ls") -> None:
    """Draw a caption, colouring a leading signed number green or red.

    The reference dashboards all do this: the delta is the only coloured
    thing on an otherwise monochrome card, which is what makes it readable
    at a glance.
    """
    m = _DELTA.match(text)
    if not m:
        _text(d, xy, text, font, style["sub"], anchor)
        return
    head, tail = m.group(1), m.group(2)
    colour = HUES["rose"] if head[0] in "-−" else HUES["mint"]
    x, y = xy
    _text(d, (x, y), head, font, colour, anchor)
    if tail:
        _text(d, (x + d.textlength(head, font=font), y), tail, font,
              style["sub"], anchor)


def _hue_of(field: Field, a: Analysis) -> tuple:
    h = field.hue(a) if callable(field.hue) else field.hue
    return HUES.get(h, HUES["slate"]) if isinstance(h, str) else tuple(h)


@dataclass(frozen=True)
class _Tile:
    """A resolved tile: the Field's callables already applied to the book."""
    label: str
    value: object
    sub: str
    hue: tuple
    icon: str
    rows: list | None
    span: int
    bars: list | None = None
    rowspan: int = 1
    columns: tuple = ()


def _tile_need(t: _Tile) -> float:
    """How tall this tile wants to be.

    A single number is happy at ``TILE_H``. A ranked list needs room for its
    rows — squeezing five of them into a one-number tile shrinks the type
    until the list is unreadable, which defeats the point of putting it on
    the card.
    """
    if t.bars:
        return BARS_H
    if not t.rows:
        return TILE_H
    # a table carries a header row of its own
    n = len(t.rows) + (1 if t.columns else 0)
    return 22 * SCALE + 30 * SCALE + 16 * SCALE + n * ROW_H + 22 * SCALE


def _place(tiles: list) -> tuple[int, list, list]:
    """Pack the tiles into a grid, honouring column *and* row spans.

    Returns ``(cols, [(row, col, span, rowspan), ...], [row_height, ...])``.

    Placement is first-fit over an occupancy grid, but tiles taller than one
    row are placed first. A full-height price ladder belongs against the
    left edge with the single-number tiles flowing to its right; threading
    it into the reading order instead would leave a hole wherever it landed.
    """
    cols = max(GRID_COLS, max(t.span for t in tiles))

    grid: list[list[bool]] = []

    def row(r: int) -> list[bool]:
        while len(grid) <= r:
            grid.append([False] * cols)
        return grid[r]

    def fits(r: int, c: int, sp: int, rs: int) -> bool:
        return all(not any(row(rr)[c:c + sp]) for rr in range(r, r + rs))

    places: list = [None] * len(tiles)
    for i in sorted(range(len(tiles)), key=lambda i: -tiles[i].rowspan):
        t = tiles[i]
        sp, rs = min(t.span, cols), t.rowspan
        r = 0
        while True:
            row(r + rs - 1)
            hit = next((c for c in range(cols - sp + 1) if fits(r, c, sp, rs)), None)
            if hit is not None:
                for rr in range(r, r + rs):
                    for cc in range(hit, hit + sp):
                        row(rr)[cc] = True
                places[i] = (r, hit, sp, rs)
                break
            r += 1

    while grid and not any(grid[-1]):
        grid.pop()

    # No row may end short. First-fit leaves a tail whenever the next tile
    # is wider than the gap, and an empty cell reads as a missing tile
    # rather than as spare room. The slack is shared out evenly across the
    # single-row tiles on that row and the row is re-flowed left to right;
    # tiles that span rows keep their column, since widening one of those
    # would punch a hole in the other row it occupies.
    for r in range(len(grid)):
        on_row = sorted((i for i, p in enumerate(places)
                         if p[0] <= r < p[0] + p[3]), key=lambda i: places[i][1])
        if not on_row:
            continue
        used = sum(places[i][2] for i in on_row)
        slack = cols - used
        growable = [i for i in on_row if places[i][3] == 1]
        if slack <= 0 or not growable:
            continue
        for n, i in enumerate(growable):
            r0, c0, sp0, rs0 = places[i]
            extra = slack // len(growable) + (1 if n < slack % len(growable) else 0)
            places[i] = (r0, c0, sp0 + extra, rs0)
        c = 0
        for i in on_row:
            r0, c0, sp0, rs0 = places[i]
            if rs0 > 1:                       # anchored: it owns another row too
                c = c0 + sp0
                continue
            places[i] = (r0, c, sp0, rs0)
            c += sp0

    heights = [TILE_H] * len(grid)
    # A tile spanning rows takes its height from the rows it covers, so only
    # single-row tiles get a say in how tall a row has to be.
    for t, (r, _, _, rs) in zip(tiles, places):
        if rs == 1:
            heights[r] = max(heights[r], _tile_need(t))
    return cols, places, heights


def build_share_card(a: Analysis, ticker: str = "", *,
                     extras: list | None = None,
                     fields: list | None = None,
                     footnote: str = "",
                     style: str = DEFAULT_STYLE,
                     size: str = DEFAULT_SIZE,
                     subtitle: str | None = None,
                     username: str = "",
                     avatar: bytes | None = None) -> bytes:
    """Render the headline read as a PNG and return its bytes.

    ``fields`` is the tile set and its order (defaults to ``CARD_FIELDS``);
    ``card_catalog`` builds the menu it is chosen from. ``extras`` appends
    ready-made ``(label, value, sub[, hue])`` tuples for numbers that do not
    live on the Analysis. ``style`` picks the look, ``subtitle`` overrides
    the header's second line, ``footnote`` adds a line under the grid — all
    of it driven from the dashboard's card controls.

    ``username`` and ``avatar`` put a profile in the upper-right corner so
    the card carries whose read it is once it leaves the app. ``avatar``
    takes image bytes or a PIL image; without one the initials of the handle
    are drawn instead.
    """
    sty = STYLES.get(style, STYLES[DEFAULT_STYLE])
    theme = THEMES.get(a.regime, THEMES["long_gamma"])
    pad = 44 * SCALE
    glass = sty["glass"]
    rad = sty["radius"] * SCALE

    spec = fields if fields is not None else CARD_FIELDS
    tiles = [_Tile(f.label, f.value(a), (f.sub(a) if f.sub else ""),
                   _hue_of(f, a), f.icon,
                   f.rows(a) if f.rows else None, max(1, f.span),
                   f.bars(a) if f.bars else None, max(1, f.rowspan),
                   tuple(f.columns))
             for f in spec]
    for i, extra in enumerate(extras or []):
        lab, val, sb = extra[0], extra[1], (extra[2] if len(extra) > 2 else "")
        hue = extra[3] if len(extra) > 3 else _EXTRA_HUES[i % len(_EXTRA_HUES)]
        tiles.append(_Tile(lab, val, sb,
                           HUES.get(hue, HUES["fuchsia"]) if isinstance(hue, str)
                           else tuple(hue), "dot", None, 1))
    if not tiles:
        tiles = [_Tile("", "—", "", HUES["slate"], "dot", None, 1)]

    # Column count avoids a lonely tile on the last row and keeps the grid
    # wide rather than tall; the card then grows to fit the rows that fall
    # out of it. A fixed canvas would either crop a nine-tile card or leave
    # a lake of empty page under a three-tile one.
    cols, places, row_h = _place(tiles)
    head_h = 112 * SCALE
    gap = 20 * SCALE
    head_gap = 24 * SCALE
    foot_h = 46 * SCALE if footnote else 0
    w = SIZES.get(size, SIZES[DEFAULT_SIZE]) * SCALE
    h = int(pad + head_h + head_gap + sum(row_h) + (len(row_h) - 1) * gap
            + foot_h + pad)
    tw = (w - 2 * pad - gap * (cols - 1)) / cols
    # y of each grid row, so a tall list row pushes what follows it down
    row_y = []
    _y = pad + head_h + head_gap
    for rh in row_h:
        row_y.append(_y)
        _y += rh + gap

    img = _backdrop(w, h, theme, sty)

    # --- header ---------------------------------------------------------
    head_box = (pad, pad, w - pad, pad + head_h)
    if glass:
        _glow(img, head_box, rad, theme["accent"], 0.16)
        _glass_panel(img, head_box, rad, blur=30, accent=theme["accent"])
    else:
        _flat_panel(img, head_box, rad, sty, fill=sty["panel_head"],
                    accent=theme["accent"], wash=0.11)
    d = ImageDraw.Draw(img)
    hx, hy = pad + 32 * SCALE, pad + head_h // 2
    label = (ticker or "").upper().strip() or "OPTIONS BOOK"
    tf = _font(38 * SCALE, True)
    _text(d, (hx, hy - 20 * SCALE), label, tf, sty["value"], "lm")
    lw = d.textlength(label, font=tf)
    sf = _font(29 * SCALE, True)
    spot_txt = f"{a.spot:,.2f}"
    spot_x = hx + lw + 20 * SCALE
    _text(d, (spot_x, hy - 19 * SCALE), spot_txt, sf, theme["accent"], "lm")
    left_end = spot_x + d.textlength(spot_txt, font=sf)

    inst = detect_instrument(ticker)
    # the multiplier changes every dollar figure on the card, so it is stated
    # rather than assumed: an NQ card and a QQQ card are otherwise identical
    if subtitle is None:
        inst_bit = (f"{inst.name} · ×{a.multiplier:g}" if inst.root
                    else f"×{a.multiplier:g} per contract")
        subtitle = f"{inst_bit} · {a.asof:%d %b %Y} · {a.n_contracts:,} contracts"
    if subtitle:
        _text(d, (hx, hy + 25 * SCALE), subtitle,
              _fit(d, subtitle, w - 2 * pad - 300 * SCALE, 16, 11),
              sty["label"], "lm")

    vx1 = w - pad - 30 * SCALE
    # Profile in the upper-right corner, and nothing else on that side: the
    # corner belongs to whoever made the card. The verdict pill moves inline
    # after the spot price, which reads as one phrase — "SPX 7,484.84, long
    # gamma" — and the tagline goes, since the pill already says it.
    has_profile = bool(username or avatar)
    if has_profile:
        av = 40 * SCALE
        acy = hy - 20 * SCALE
        abox = (vx1 - av, acy - av / 2, vx1, acy + av / 2)
        _draw_avatar(img, abox, avatar, username, theme["accent"], sty)
        d = ImageDraw.Draw(img)
        if username:
            handle = username if username.startswith("@") else f"@{username}"
            _text(d, (vx1 - av - 14 * SCALE, acy), handle,
                  _fit(d, handle, w / 3, 19, 11, bold=True), sty["value"], "rm")

    vtxt = theme["verdict"]
    vf = _font(21 * SCALE, True)
    track = 1.6 * SCALE
    vw = d.textlength(vtxt, font=vf) + 3 * track + 40 * SCALE
    if has_profile:
        vy0 = hy - 42 * SCALE
        pill = (left_end + 22 * SCALE, vy0,
                left_end + 22 * SCALE + vw, vy0 + 44 * SCALE)
    else:
        vy0 = pad + head_h // 2 - 40 * SCALE
        pill = (vx1 - vw, vy0, vx1, vy0 + 44 * SCALE)
    _paint_rgba(img, pill, lambda od, ox, oy: od.rounded_rectangle(
        [pill[0] - ox, pill[1] - oy, pill[2] - ox - 1, pill[3] - oy - 1],
        radius=22 * SCALE, fill=theme["accent"] + (54,),
        outline=theme["accent"] + (215,), width=max(2, SCALE)))
    d = ImageDraw.Draw(img)
    pw = d.textlength(vtxt, font=vf) + 3 * track
    _tracked(d, ((pill[0] + pill[2]) / 2 - pw / 2,
                 (pill[1] + pill[3]) / 2 - 13 * SCALE),
             vtxt, vf, theme["accent"], track)
    if not has_profile:
        tag = theme["tagline"]
        _text(d, (vx1, pad + head_h - 24 * SCALE), tag,
              _fit(d, tag, w / 2 - 40 * SCALE, 15, 10), sty["sub"], "rs")

    # --- tiles ----------------------------------------------------------
    for t, (r, c, span, rspan) in zip(tiles, places):
        lab, val, sb, hue, icon = t.label, t.value, t.sub, t.hue, t.icon
        x0 = pad + c * (tw + gap)
        y0 = row_y[r]
        th = sum(row_h[r:r + rspan]) + gap * (rspan - 1)
        tile_w = span * tw + (span - 1) * gap
        box = (x0, y0, x0 + tile_w, y0 + th)
        if glass:
            _glow(img, box, rad, hue, 0.13)
            _glass_panel(img, box, rad, accent=hue)
        else:
            _flat_panel(img, box, rad, sty)
        d = ImageDraw.Draw(img)
        cx = x0 + 24 * SCALE
        # everything inside the tile is placed as a fraction of its height:
        # a seventh tile pushes the grid to a third row, and fixed offsets
        # then stack the label, the value and the caption on each other
        inset = min(22 * SCALE, th * 0.11)

        # chip + label on one row, the way a dashboard card opens
        chip = 30 * SCALE
        cbox = (cx, y0 + inset, cx + chip, y0 + inset + chip)
        _paint_rgba(img, cbox, lambda od, ox, oy, b=cbox, c=hue: od.rounded_rectangle(
            [b[0] - ox, b[1] - oy, b[2] - ox - 1, b[3] - oy - 1],
            radius=9 * SCALE, fill=c + (34,), outline=c + (92,), width=max(1, SCALE)))
        _paint_rgba(img, cbox, lambda od, ox, oy, b=cbox, c=hue, k=icon: _icon(
            od, (b[0] - ox, b[1] - oy, b[2] - ox, b[3] - oy), k, c + (255,)))
        d = ImageDraw.Draw(img)
        _tracked(d, (cx + chip + 12 * SCALE, y0 + inset + chip / 2 - 7 * SCALE),
                 _caps(str(lab)), _font(13 * SCALE, True), sty["label"], 1.1 * SCALE)

        # Value and caption stack down from the label row rather than
        # floating in the middle of the panel — the reading order is
        # label, number, delta, and centring the number breaks it.
        vy = y0 + inset + chip + 16 * SCALE
        room = th - (vy - y0) - inset

        if t.columns and t.rows:
            _draw_table(img, d, (cx, vy, x0 + tile_w - 24 * SCALE,
                                 y0 + th - inset), t.columns, t.rows, sty)
            d = ImageDraw.Draw(img)
        elif t.bars:
            _draw_bars(d, (cx, vy, x0 + tile_w - 24 * SCALE,
                           y0 + th - inset - 20 * SCALE),
                       t.bars, a.spot, sty)
            if sb:
                _draw_sub(d, (x0 + tile_w - 24 * SCALE, y0 + inset + chip / 2 + 5 * SCALE),
                          str(sb), _font(13 * SCALE), sty, "rm")
        elif t.rows:
            # a ranked list: level on the left, the number that ranks it on
            # the right, its own colour on the number. Row height comes from
            # the room left over so the list never overruns the panel.
            # a fixed rhythm, not room/n: two list tiles side by side must
            # share a baseline grid, and a three-row list dividing the same
            # room as a five-row one drifts out of step with it
            n = len(t.rows)
            rh = min(ROW_H, room / max(n, 1))
            rf = _font(max(int(min(15, rh / SCALE * 0.62)), 9) * SCALE)
            rb = _font(rf.size, True)
            for j, row in enumerate(t.rows):
                left, right = str(row[0]), str(row[1])
                rhue = row[2] if len(row) > 2 else None
                colour = (HUES.get(rhue, hue) if isinstance(rhue, str)
                          else (tuple(rhue) if rhue else hue))
                ry = vy + j * rh + rh / 2
                _text(d, (cx, ry), left, rb, sty["value"], "lm")
                _text(d, (x0 + tile_w - 24 * SCALE, ry), right, rf, colour, "rm")
        else:
            val = str(val)
            cap_h = 22 * SCALE if sb else 0
            vsize = max(int(min(38, (room - cap_h) / SCALE * 0.86)), 15)
            vfont = _fit(d, val, tile_w - 48 * SCALE, vsize, 15, bold=True)
            _text(d, (cx, vy), val, vfont, sty["value"], "la")
            if sb:
                sb = str(sb)
                _draw_sub(d, (cx, vy + vfont.size * 1.16 + 10 * SCALE), sb,
                          _fit(d, sb, tile_w - 44 * SCALE, 14, 9), sty, "la")

    # --- footer ---------------------------------------------------------
    # Nothing by default: the card is the numbers. A caller that wants a
    # line under them passes ``footnote``.
    if footnote:
        dot = (pad + 4 * SCALE, h - pad + 1 * SCALE,
               pad + 11 * SCALE, h - pad + 8 * SCALE)
        _paint_rgba(img, dot, lambda od, ox, oy: od.ellipse(
            [dot[0] - ox, dot[1] - oy, dot[2] - ox - 1, dot[3] - oy - 1],
            fill=theme["accent"] + (230,)))
        d = ImageDraw.Draw(img)
        _text(d, (pad + 22 * SCALE, h - pad + 9 * SCALE), footnote,
              _font(14 * SCALE), sty["sub"], "ls")

    buf = io.BytesIO()
    # compress_level=6 rather than optimize=True: the grain makes the card
    # nearly incompressible, so exhaustive optimisation spends three seconds
    # to save 4% — long enough to be felt on every Streamlit rerun.
    img.save(buf, format="PNG", compress_level=6)
    return buf.getvalue()


def card_size(n_tiles: int, footnote: str = "",
              size: str = DEFAULT_SIZE) -> tuple[int, int]:
    """The pixel size a card with ``n_tiles`` tiles will come out at.

    Exposed so a caller can reserve layout for the picture — and so a test
    can state the height rule rather than hard-coding a number.
    """
    n = max(1, n_tiles)
    _, _, row_h = _place([_Tile("", "", "", HUES["slate"], "dot", None, 1)] * n)
    w = SIZES.get(size, SIZES[DEFAULT_SIZE]) * SCALE
    pad, head_h, gap, head_gap = 44 * SCALE, 112 * SCALE, 20 * SCALE, 24 * SCALE
    foot_h = 46 * SCALE if footnote else 0
    return (w, int(pad + head_h + head_gap + sum(row_h) + (len(row_h) - 1) * gap
                   + foot_h + pad))


def card_filename(a: Analysis, ticker: str = "") -> str:
    stem = (ticker or "book").upper().replace(" ", "-")
    return f"{stem}-positioning-{a.asof:%Y%m%d}.png"
