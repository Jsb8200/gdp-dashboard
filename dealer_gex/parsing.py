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
"""

from __future__ import annotations

import io
import re

import pandas as pd


class ChainParseError(Exception):
    """Raised when a file cannot be auto-detected as an option chain."""


CANONICAL_FIELDS = ["expiry", "strike", "type", "open_interest", "volume", "iv", "gamma"]

SYNONYMS = {
    "expiry": [
        "expiry", "expiration", "expirationdate", "expdate", "exp",
        "expirydate", "maturity", "expiredate",
    ],
    "strike": ["strike", "strikeprice", "k", "exerciseprice"],
    "type": [
        "type", "optiontype", "cp", "callput", "putcall", "right", "side",
        "optionright", "pc",
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
        "stockprice", "last", "underlyingspot",
    ],
}

_REQUIRED = ["expiry", "strike", "type", "open_interest", "iv"]


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
        if "strike" in cells and len(cells) >= 3:
            return i, spot
    raise ChainParseError(
        "Could not find a header row containing a 'Strike' column in the first 10 lines."
    )


def read_chain(data: bytes | str) -> tuple[pd.DataFrame, float | None]:
    """Parse raw CSV bytes/text into a canonical chain.

    Returns (chain_df, inferred_spot). Raises ChainParseError when the
    shape cannot be auto-detected — the caller can then fall back to a
    manual column mapping with :func:`normalize_chain`.
    """
    text = data.decode("utf-8-sig", errors="replace") if isinstance(data, bytes) else data
    header_row, spot = _find_header_row(text)
    raw = pd.read_csv(io.StringIO(text), skiprows=header_row)

    if _is_side_by_side(raw):
        chain = _parse_side_by_side(raw)
    else:
        chain = _parse_long(raw)

    if spot is None:
        for col in raw.columns:
            if _match_field(col) == "underlying_price" and _norm(col) != "last":
                vals = pd.to_numeric(raw[col], errors="coerce").dropna()
                if not vals.empty:
                    spot = float(vals.iloc[0])
                break
    return chain, spot


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
    if s.dtype == object:
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
    return df[CANONICAL_FIELDS].reset_index(drop=True)
