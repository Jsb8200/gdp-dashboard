"""Option-chain CSV parsing and normalization.

Accepts the two common export shapes and normalizes both into a canonical
long-format DataFrame with columns:

    expiry (datetime64), strike (float), type ('C'/'P'),
    open_interest (float), volume (float), iv (decimal, e.g. 0.22),
    gamma (float, NaN when the file has no greeks)

Supported shapes:

* **Long format** — one row per contract with a call/put column (typical
  broker export). Columns are matched case-insensitively against synonyms.
* **CBOE side-by-side** — calls and puts share a row per strike (CBOE
  ``quotedata`` downloads). Duplicated column names are read by pandas as
  ``Bid`` / ``Bid.1`` etc.; the unsuffixed set is calls, ``.1`` is puts.
* **Trade-level order flow** — one row per print (QuantData "Options Order
  Flow" exports and similar), detected by Trade ID/Time + Size columns.
  Prints are collapsed to one row per contract (OI = max, never summed),
  and ask/bid side codes are folded into a signed ``net_customer_size``.
"""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field
from datetime import date

import pandas as pd


class ChainParseError(Exception):
    """Raised when a file cannot be auto-detected as an option chain."""


CANONICAL_FIELDS = ["expiry", "strike", "type", "open_interest", "volume", "iv", "gamma"]
# carried through when the source provides them (trade-flow files)
OPTIONAL_FIELDS = ["ticker", "net_customer_size"]

SYNONYMS = {
    "expiry": [
        "expiry", "expiration", "expirationdate", "expdate", "exp",
        "expirydate", "maturity", "expiredate",
    ],
    "strike": ["strike", "strikeprice", "k", "exerciseprice"],
    "type": [
        "type", "optiontype", "cp", "callput", "putcall", "right",
        "optionright", "pc", "contracttype",
    ],
    "open_interest": ["openinterest", "openint", "oi", "opint"],
    "volume": ["volume", "vol", "totalvolume"],
    "iv": [
        "iv", "impliedvolatility", "implvol", "impliedvol", "midiv", "ivmid",
        "volatility", "impvol",
    ],
    "gamma": ["gamma"],
    "underlying_price": [
        "underlyingprice", "underlying", "spot", "underlyinglast",
        "stockprice", "last", "underlyingspot", "referenceprice",
    ],
    "ticker": ["ticker", "symbol", "underlyingsymbol"],
    "size": ["size", "quantity", "qty", "tradesize"],
    "side": ["sidecode", "side", "tradeside"],
    "trade_time": ["tradetime", "timestamp", "datetime", "time"],
    "trade_id": ["tradeid"],
    "premium": ["premiumprice", "premium", "totalpremium"],
    "consolidation": ["consolidationtype", "consolidation"],
    "trade_type": ["tradetype", "executiontype", "tradekind"],
    "is_golden": ["isgoldensweep", "goldensweep"],
    "is_unusual": ["isunusual", "unusual"],
    "is_opening": ["isopeningposition", "openingposition", "isopening"],
}


def _yes(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.upper().isin(["YES", "TRUE", "Y", "1"])


# A consolidated-flow type mixes two independent things: *where/how* the
# print executed (floor, electronic, cross) and *what shape* the order took
# (block, sweep, split, multi-leg). "FLR BLOCK" is both, "AUTO" is only a
# venue. Collapsing them into one flag loses the distinction that matters
# most — a floor-negotiated block is upstairs institutional size, an
# auto-executed print is just the electronic default.
FLOW_VENUES = {
    "floor": ("FLR", "FLOOR", "PIT", "OPENOUTCRY", "OPEN OUTCRY"),
    "auto":  ("AUTO", "ELECTRONIC", "ELEC", "AUTOEX"),
    "cross": ("CROSS", "XCRS", "QCC", "COMBOCROSS"),
}
FLOW_SHAPES = {
    "block": ("BLOCK", "BLK"),
    "sweep": ("SWEEP", "SWP"),
    "split": ("SPLIT",),
    "multi": ("MULTI", "MLEG", "MULTILEG", "COMBO", "SPREAD"),
}
#: Venues whose prints are negotiated size, not just an execution route.
INSTITUTIONAL_VENUES = ("floor", "cross")


def _has_any(s: pd.Series, needles: tuple[str, ...]) -> pd.Series:
    hit = pd.Series(False, index=s.index)
    for n in needles:
        hit |= s.str.contains(n, na=False, regex=False)
    return hit


# QuantData splits the two axes across two columns: *Consolidation Type* is
# the shape (SWEEP / BLOCK / SPLIT) and *Trade Type* is the mechanism —
# AUTO, FLR, CROSS, COB, AUCT, ISO — with two modifying prefixes:
#
#   SPRD_ / SPRD_LEG_  the print is a package, or one leg of one
#   TIED_              the print is stock-tied, i.e. delta-hedged on the trade
#
# A leg of a spread is not an outright bet, and a tied print is a volatility
# bet whose delta is neutralized by the accompanying stock — so both change
# what a "block" means, and both have to survive parsing.
TRADE_MECHANISMS = {   # first match wins, so compound codes come first
    "floor":       ("FLR", "FLOOR", "PIT", "OPENOUTCRY"),
    "cross":       ("CROSS", "XCRS", "QCC", "FACILITATION"),
    "cob auction": ("COBAUCT", "COMPLEXAUCT"),
    "cob":         ("COB", "COMPLEX"),
    "auction":     ("AUCT", "AUCTION", "PIM", "AIM", "SOLICIT"),
    "iso":         ("ISO", "INTERMARKET"),
    "auto":        ("AUTO", "ELECTRONIC", "ELEC"),
}
#: How much a block of each mechanism actually says, most meaningful first.
BLOCK_TIERS = {
    "negotiated": ("floor", "cross"),   # agreed upstairs, off the public book
    "facilitated": ("cob", "cob auction", "auction"),  # worked for price improvement
    "electronic": ("auto", "iso"),      # the default route
}
CANCEL_CODES = ("CANCEL", "CXL", "BUST")


def _clean_codes(values) -> pd.Series:
    s = pd.Series(values).fillna("").astype(str).str.upper()
    s = s.str.replace(r"[_\-/|]+", " ", regex=True).str.strip()
    return s.replace({"NAN": "", "NONE": "", "NAT": ""})


def classify_trade_type(trade_type) -> pd.DataFrame:
    """Split an execution-mechanism column into its independent parts.

    Returns ``trade_type`` (raw), ``mechanism`` (floor / cross / cob /
    auction / iso / auto / ""), ``is_spread``, ``is_spread_leg``,
    ``is_tied``, ``is_cancelled`` and ``block_tier`` (negotiated /
    facilitated / electronic / fragment / cancelled / "").

    A spread *leg* is its own tier: ``SPRD_LEG_AUTO`` prints are fragments
    of a package — in a real export they can be the majority of block
    prints while carrying ~1% of block premium, so counting them as
    outright blocks buries the trades that matter.
    """
    s = _clean_codes(trade_type)
    squeezed = s.str.replace(" ", "", regex=False)

    mechanism = pd.Series("", index=s.index, dtype=object)
    for name, needles in TRADE_MECHANISMS.items():
        mechanism = mechanism.mask((mechanism == "") & _has_any(squeezed, needles), name)

    is_spread_leg = squeezed.str.startswith("SPRDLEG", na=False)
    is_spread = squeezed.str.startswith("SPRD", na=False)
    is_tied = squeezed.str.contains("TIED", na=False, regex=False)
    is_cancelled = _has_any(squeezed, CANCEL_CODES)

    tier = pd.Series("", index=s.index, dtype=object)
    for name, mechs in BLOCK_TIERS.items():
        tier = tier.mask((tier == "") & mechanism.isin(mechs), name)
    tier = tier.mask(is_spread_leg, "fragment")
    tier = tier.mask(is_cancelled, "cancelled")
    return pd.DataFrame({
        "trade_type": s, "mechanism": mechanism, "is_spread": is_spread,
        "is_spread_leg": is_spread_leg, "is_tied": is_tied,
        "is_cancelled": is_cancelled, "block_tier": tier,
    })


def classify_flow(consolidation) -> pd.DataFrame:
    """Split a consolidated-flow type column into precise, orthogonal parts.

    Returns a frame indexed like the input with:

    * ``consolidation`` — the raw value, upper-cased and squeezed (never
      dropped, so an unmapped code stays visible downstream);
    * ``flow_venue`` — ``floor`` / ``auto`` / ``cross`` / ``""``;
    * ``flow_shape`` — ``block`` / ``sweep`` / ``split`` / ``multi`` /
      ``single``;
    * ``flow_type`` — the display label, e.g. ``floor block``, ``auto
      sweep``, ``block``, ``floor``. A value matching nothing keeps its raw
      text rather than being silently bucketed as ``single``.
    """
    s = _clean_codes(consolidation)
    squeezed = s.str.replace(" ", "", regex=False)

    venue = pd.Series("", index=s.index, dtype=object)
    for name, needles in FLOW_VENUES.items():
        venue = venue.mask((venue == "") & _has_any(squeezed, needles), name)

    shape = pd.Series("single", index=s.index, dtype=object)
    for name, needles in FLOW_SHAPES.items():   # block beats sweep beats split
        shape = shape.mask((shape == "single") & _has_any(squeezed, needles), name)

    label = (venue + " " + shape.where(shape != "single", "")).str.strip()
    # nothing recognized: show the raw code so unmapped vocabularies surface
    unknown = (venue == "") & (shape == "single")
    label = label.mask(unknown & (s != ""), s)
    label = label.mask(unknown & (s == ""), "single")
    return pd.DataFrame({"consolidation": s, "flow_venue": venue,
                         "flow_shape": shape, "flow_type": label})


_REQUIRED = ["expiry", "strike", "type", "open_interest", "iv"]


@dataclass
class ParsedFile:
    """A parsed option file plus everything inferable about its context."""
    chain: pd.DataFrame
    spots: dict[str, float] = field(default_factory=dict)  # ticker -> spot ("" = unknown ticker)
    asof: date | None = None          # last trade date for flow files
    tickers: list[str] = field(default_factory=list)  # by activity, descending
    prints: pd.DataFrame | None = None  # normalized per-print rows (flow files only)
    dark: pd.DataFrame | None = None    # normalized dark-pool prints (equity blocks)


# Dark-pool (equity block) exports: no strike/expiry/option-type, just
# price + size per print. Matched with a dedicated vocabulary.
DP_PRICE = {"price", "tradeprice", "fillprice", "executionprice", "lastprice", "fill"}
DP_SIZE = {"size", "quantity", "qty", "shares", "tradesize"}
DP_VALUE = {"premium", "value", "notional", "dollarvolume", "tradevalue",
            "premiumprice", "dollars"}
DP_TIME = {"time", "tradetime", "timestamp", "datetime"}
DP_TICKER = {"ticker", "symbol", "underlyingsymbol"}


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def _strip_dup_suffix(name: str) -> str:
    return re.sub(r"\.\d+$", "", str(name))


def _match_field(colname: str) -> str | None:
    n = _norm(_strip_dup_suffix(colname))
    for field, names in SYNONYMS.items():
        if n in names:
            return field
    return None


def _find_header_row(text: str) -> tuple[int, float | None]:
    """Locate the header line and pull a spot price from any preamble.

    CBOE downloads carry 2-3 metadata lines (ticker description, ``Last:
    601.23`` quote line) before the real header. Returns (header_row_index,
    inferred_spot_or_None).
    """
    lines = text.splitlines()
    spot = None
    for i, line in enumerate(lines[:10]):
        m = re.search(r"last(?:\s*sale)?[:\s,]+\$?([0-9]+(?:\.[0-9]+)?)", line, re.I)
        if m and "strike" not in line.lower():
            spot = float(m.group(1))
        cells = [_norm(c) for c in line.split(",")]
        if len(cells) >= 3 and any(c in SYNONYMS["strike"] for c in cells):
            return i, spot
    raise ChainParseError(
        "Could not find a header row containing a 'Strike' column in the first 10 lines."
    )


def _try_dark_pool(text: str) -> ParsedFile | None:
    """Detect and parse an equity dark-pool / block-print export: price +
    size per row, with no strike/expiry/option-type. Returns a ParsedFile
    whose ``dark`` frame is set, or None if the file isn't dark-pool."""
    try:
        raw = pd.read_csv(io.StringIO(text), nrows=200)
    except Exception:
        return None
    norm = {c: _norm(_strip_dup_suffix(c)) for c in raw.columns}
    vals = set(norm.values())

    def pick(vocab):
        for col, n in norm.items():
            if n in vocab:
                return col
        return None

    price_c, size_c = pick(DP_PRICE), pick(DP_SIZE)
    has_strike = any(v in SYNONYMS["strike"] for v in vals)
    has_type = any(v in SYNONYMS["type"] for v in vals)
    if price_c is None or size_c is None or has_strike or has_type:
        return None

    raw = pd.read_csv(io.StringIO(text))
    tkr_c, val_c, time_c = pick(DP_TICKER), pick(DP_VALUE), pick(DP_TIME)
    df = pd.DataFrame({
        "ticker": raw[tkr_c].astype(str) if tkr_c else "",
        "price": _to_num(raw[price_c]),
        "size": _to_num(raw[size_c]).fillna(0),
    })
    df["premium"] = _to_num(raw[val_c]) if val_c else df["price"] * df["size"]
    df["premium"] = df["premium"].fillna(df["price"] * df["size"]).fillna(0)
    df["time"] = (pd.to_datetime(raw[time_c], errors="coerce", utc=True, format="ISO8601")
                  if time_c else pd.NaT)
    df = df.dropna(subset=["price"])
    df = df[df["price"] > 0]
    if df.empty:
        return None

    spots, tickers = {}, []
    if tkr_c:
        ordered = df.sort_values("time", na_position="first") if time_c else df
        for tk, sub in ordered.groupby("ticker"):
            spots[str(tk)] = float(sub["price"].iloc[-1])
        tickers = list(df["ticker"].value_counts().index.astype(str))
    asof = df["time"].max().date() if time_c and df["time"].notna().any() else None
    return ParsedFile(chain=df.iloc[0:0], spots=spots, asof=asof,
                      tickers=tickers, dark=df)


def parse_file(data: bytes | str) -> ParsedFile:
    """Parse raw CSV bytes/text into a canonical chain plus file context.

    Raises ChainParseError when the shape cannot be auto-detected — the
    caller can then fall back to a manual column mapping with
    :func:`normalize_chain`.
    """
    text = data.decode("utf-8-sig", errors="replace") if isinstance(data, bytes) else data
    dp = _try_dark_pool(text)
    if dp is not None:
        return dp
    header_row, spot = _find_header_row(text)
    raw = pd.read_csv(io.StringIO(text), skiprows=header_row)

    if _is_trade_flow(raw):
        return _parse_trade_flow(raw)
    if _is_side_by_side(raw):
        chain = _parse_side_by_side(raw)
    else:
        chain = _parse_long(raw)

    if spot is None:
        for col in raw.columns:
            if _match_field(col) == "underlying_price" and _norm(col) != "last":
                vals = pd.to_numeric(_to_num(raw[col]), errors="coerce").dropna()
                if not vals.empty:
                    spot = float(vals.iloc[0])
                break
    spots = {"": spot} if spot is not None else {}
    return ParsedFile(chain=chain, spots=spots)


def read_chain(data: bytes | str) -> tuple[pd.DataFrame, float | None]:
    """Back-compat wrapper over :func:`parse_file`: (chain, inferred_spot)."""
    pf = parse_file(data)
    spot = None
    if pf.spots:
        key = pf.tickers[0] if pf.tickers else next(iter(pf.spots))
        spot = pf.spots.get(key)
    return pf.chain, spot


def _is_trade_flow(raw: pd.DataFrame) -> bool:
    fields = {_match_field(c) for c in raw.columns}
    return ("trade_id" in fields or "trade_time" in fields) and "size" in fields \
        and "strike" in fields and "type" in fields


def _parse_trade_flow(raw: pd.DataFrame) -> ParsedFile:
    """Collapse a trade-level (one row per print) file into a per-contract
    chain. OI is a property of the contract, not the print — take max, never
    sum. Side codes give each print a customer direction: at/above ask =
    customer buy (+), at/below bid = customer sell (−), mid = unknown (0)."""
    cols: dict[str, str] = {}
    for col in raw.columns:
        f = _match_field(col)
        if f and f not in cols:
            cols[f] = col
    missing = [f for f in ("expiry", "strike", "type", "size") if f not in cols]
    if missing:
        raise ChainParseError(f"Trade-flow file is missing columns for: {', '.join(missing)}")

    df = pd.DataFrame({
        "ticker": raw[cols["ticker"]].astype(str) if "ticker" in cols else "",
        "expiry": raw[cols["expiry"]],
        "strike": _to_num(raw[cols["strike"]]),
        "type": raw[cols["type"]],
        "size": _to_num(raw[cols["size"]]).fillna(0),
        "open_interest": _to_num(raw[cols["open_interest"]]) if "open_interest" in cols else 0.0,
        "volume": _to_num(raw[cols["volume"]]) if "volume" in cols else float("nan"),
        "iv": _to_num(raw[cols["iv"]]) if "iv" in cols else float("nan"),
        "gamma": _to_num(raw[cols["gamma"]]) if "gamma" in cols else float("nan"),
        "ref_price": _to_num(raw[cols["underlying_price"]]) if "underlying_price" in cols else float("nan"),
        "premium": _to_num(raw[cols["premium"]]).fillna(0) if "premium" in cols else 0.0,
    })
    if "side" in cols:
        side = raw[cols["side"]].astype(str).str.strip().str.upper()
        sign = side.map({"A": 1.0, "AA": 1.0, "B": -1.0, "BB": -1.0}).fillna(0.0)
        df["side"] = side
    else:
        sign = pd.Series(0.0, index=df.index)
        df["side"] = ""
    df["signed_size"] = sign * df["size"]

    # conviction flags: shape and venue come from the consolidation type,
    # the rest from QuantData's Yes/No columns
    if "consolidation" in cols:
        kinds = classify_flow(raw[cols["consolidation"]])
    else:
        kinds = classify_flow(pd.Series("", index=df.index))
    kinds.index = df.index
    df["consolidation"] = kinds["consolidation"]
    df["flow_shape"] = kinds["flow_shape"]
    df["is_sweep"] = kinds["flow_shape"] == "sweep"
    df["is_block"] = kinds["flow_shape"] == "block"
    # SPLIT: one order worked across executions — between a sweep and a
    # block; large patient flow, still conviction
    df["is_split"] = kinds["flow_shape"] == "split"

    # The mechanism lives in its own column when the export has one (this is
    # where FLR / AUTO / CROSS actually are); otherwise fall back to whatever
    # the consolidation column carried.
    if "trade_type" in cols:
        mech = classify_trade_type(raw[cols["trade_type"]])
    else:
        mech = classify_trade_type(pd.Series("", index=df.index))
        mech["mechanism"] = kinds["flow_venue"].to_numpy()
        mech["block_tier"] = [
            next((t for t, ms in BLOCK_TIERS.items() if m in ms), "")
            for m in kinds["flow_venue"]
        ]
    mech.index = df.index
    df["trade_type"] = mech["trade_type"]
    df["mechanism"] = mech["mechanism"]
    df["flow_venue"] = mech["mechanism"]
    df["is_spread"] = mech["is_spread"]
    df["is_spread_leg"] = mech["is_spread_leg"]
    df["is_tied"] = mech["is_tied"]
    df["block_tier"] = mech["block_tier"]
    df["is_floor"] = mech["mechanism"] == "floor"
    df["is_auto"] = mech["mechanism"] == "auto"
    df["is_cross"] = mech["mechanism"] == "cross"

    # display label: how it printed + its shape, e.g. "floor block", "auto sweep"
    parts = kinds["flow_shape"].where(kinds["flow_shape"] != "single", "")
    label = (mech["mechanism"] + " " + parts).str.strip()
    label = label.mask(mech["is_spread"] & ~mech["is_spread_leg"], label + " spread")
    label = label.mask(mech["is_spread_leg"], label + " leg")
    label = label.mask(mech["is_tied"], "tied " + label)
    df["flow_type"] = label.where(label != "", kinds["flow_type"])

    # Negotiated size, whatever it was tagged: a BLOCK-shaped print, or one
    # that printed on the floor / as a cross. AUTO is deliberately excluded —
    # electronic execution is the default route, not a size signal.
    df["is_institutional"] = (
        (df["is_block"] | mech["mechanism"].isin(INSTITUTIONAL_VENUES))
        & ~mech["is_cancelled"]
    )

    # Cancelled/busted prints never happened — drop them rather than let
    # them inflate block premium (they are large and rare, so they land
    # straight in the "biggest prints" table if kept).
    cancelled = mech["is_cancelled"]
    if cancelled.any():
        df = df[~cancelled]
        raw = raw.loc[df.index]
    for flag in ("is_golden", "is_unusual", "is_opening"):
        df[flag] = _yes(raw[cols[flag]]) if flag in cols else False

    # format="ISO8601" handles mixed with/without-milliseconds timestamps;
    # NaT rows sort first so the per-ticker "last" pick is a real trade.
    ttime = (pd.to_datetime(raw[cols["trade_time"]], errors="coerce",
                            utc=True, format="ISO8601")
             if "trade_time" in cols else pd.Series(pd.NaT, index=df.index))
    df["trade_time"] = ttime
    df = df.sort_values("trade_time", na_position="first").reset_index(drop=True)

    chain = aggregate_prints(df)

    # last observed reference price per ticker = spot; last trade date = as-of
    spots: dict[str, float] = {}
    with_ref = df[df["ref_price"].notna()]
    for tkr, sub in with_ref.groupby("ticker"):
        spots[str(tkr)] = float(sub["ref_price"].iloc[-1])
    asof = None
    if df["trade_time"].notna().any():
        asof = df["trade_time"].max().date()
    tickers = list(df["ticker"].value_counts().index.astype(str))
    return ParsedFile(chain=chain, spots=spots, asof=asof, tickers=tickers, prints=df)


def aggregate_prints(prints: pd.DataFrame) -> pd.DataFrame:
    """Collapse normalized per-print rows into a per-contract chain.

    Callers may pre-filter `prints` (e.g. sweeps/golden/opening only) to
    build a conviction-weighted book; OI and cumulative volume stay
    contract-level properties (max), while flow and signed flow sum over
    whatever prints remain."""
    grouped = (
        prints.groupby(["ticker", "expiry", "strike", "type"], dropna=False)
        .agg(
            open_interest=("open_interest", "max"),
            volume=("volume", "max"),          # cumulative day volume: max ≈ total
            flow=("size", "sum"),
            net_customer_size=("signed_size", "sum"),
            iv=("iv", "median"),
            gamma=("gamma", "median"),
        )
        .reset_index()
    )
    grouped["volume"] = grouped["volume"].fillna(grouped["flow"])
    return _finalize(grouped.drop(columns=["flow"]))


def _is_side_by_side(raw: pd.DataFrame) -> bool:
    fields = [_match_field(c) for c in raw.columns]
    # Duplicated greeks/OI columns around a single strike column ⇒ CBOE shape.
    dup_keys = [f for f in ("open_interest", "iv", "gamma", "volume") if fields.count(f) >= 2]
    return len(dup_keys) >= 2 and fields.count("strike") == 1


def _parse_side_by_side(raw: pd.DataFrame) -> pd.DataFrame:
    call_cols: dict[str, str] = {}
    put_cols: dict[str, str] = {}
    shared: dict[str, str] = {}
    for col in raw.columns:
        field = _match_field(col)
        if field is None or field == "underlying_price":
            continue
        if field in ("strike", "expiry"):
            shared.setdefault(field, col)
        elif re.search(r"\.\d+$", str(col)):
            put_cols.setdefault(field, col)
        else:
            call_cols.setdefault(field, col)

    if "strike" not in shared:
        raise ChainParseError("Side-by-side chain is missing a strike column.")

    sides = []
    for cp, cols in (("C", call_cols), ("P", put_cols)):
        side = pd.DataFrame({"strike": raw[shared["strike"]]})
        if "expiry" in shared:
            side["expiry"] = raw[shared["expiry"]]
        for field, col in cols.items():
            side[field] = raw[col]
        side["type"] = cp
        sides.append(side)
    return _finalize(pd.concat(sides, ignore_index=True))


def _parse_long(raw: pd.DataFrame) -> pd.DataFrame:
    mapping: dict[str, str] = {}
    for col in raw.columns:
        field = _match_field(col)
        if field and field != "underlying_price" and field not in mapping:
            mapping[field] = col
    missing = [f for f in _REQUIRED if f not in mapping]
    if missing:
        raise ChainParseError(
            f"Could not auto-detect columns for: {', '.join(missing)}. "
            f"Found columns: {list(raw.columns)}"
        )
    return normalize_chain(raw, {f: c for f, c in mapping.items()})


def normalize_chain(raw: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    """Build a canonical chain from an explicit field→column mapping.

    Used both by auto-detection and by the manual mapping UI fallback.
    """
    missing = [f for f in _REQUIRED if f not in mapping or not mapping[f]]
    if missing:
        raise ChainParseError(f"Missing required fields: {', '.join(missing)}")
    df = pd.DataFrame({field: raw[col] for field, col in mapping.items() if col})
    return _finalize(df)


def _to_num(s: pd.Series) -> pd.Series:
    if not pd.api.types.is_numeric_dtype(s):
        s = s.astype(str).str.replace(r"[,%$\s]", "", regex=True)
    return pd.to_numeric(s, errors="coerce")


def _finalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["strike"] = _to_num(df["strike"])
    for col in ("open_interest", "volume", "iv", "gamma"):
        if col in df.columns:
            df[col] = _to_num(df[col])
        else:
            df[col] = float("nan")

    if "type" in df.columns and df["type"].notna().any():
        t = df["type"].astype(str).str.strip().str.upper().str[0]
        df["type"] = t.where(t.isin(["C", "P"]))
    df = df.dropna(subset=["strike", "type"])

    if "expiry" in df.columns:
        df["expiry"] = pd.to_datetime(df["expiry"], errors="coerce", format="mixed")
    else:
        df["expiry"] = pd.NaT

    # Files quote IV either as a decimal (0.22) or a percent (22.0).
    iv_med = df["iv"].median()
    if pd.notna(iv_med) and iv_med > 3:
        df["iv"] = df["iv"] / 100.0

    df["open_interest"] = df["open_interest"].fillna(0)
    df["volume"] = df["volume"].fillna(0)
    # Keep zero-OI rows that traded today: they matter in volume-weighted
    # (intraday/0DTE) mode where fresh positioning has no OI yet.
    df = df[(df["open_interest"] > 0) | (df["volume"] > 0)]

    if df.empty:
        raise ChainParseError("No rows with positive open interest or volume after normalization.")
    keep = CANONICAL_FIELDS + [c for c in OPTIONAL_FIELDS if c in df.columns]
    return df[keep].reset_index(drop=True)
