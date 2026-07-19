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
}

_REQUIRED = ["expiry", "strike", "type", "open_interest", "iv"]


@dataclass
class ParsedFile:
    """A parsed option file plus everything inferable about its context."""
    chain: pd.DataFrame
    spots: dict[str, float] = field(default_factory=dict)  # ticker -> spot ("" = unknown ticker)
    asof: date | None = None          # last trade date for flow files
    tickers: list[str] = field(default_factory=list)  # by activity, descending


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


def parse_file(data: bytes | str) -> ParsedFile:
    """Parse raw CSV bytes/text into a canonical chain plus file context.

    Raises ChainParseError when the shape cannot be auto-detected — the
    caller can then fall back to a manual column mapping with
    :func:`normalize_chain`.
    """
    text = data.decode("utf-8-sig", errors="replace") if isinstance(data, bytes) else data
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
    })
    if "side" in cols:
        side = raw[cols["side"]].astype(str).str.strip().str.upper()
        sign = side.map({"A": 1.0, "AA": 1.0, "B": -1.0, "BB": -1.0}).fillna(0.0)
    else:
        sign = pd.Series(0.0, index=df.index)
    df["signed_size"] = sign * df["size"]

    # format="ISO8601" handles mixed with/without-milliseconds timestamps;
    # NaT rows sort first so the per-ticker "last" pick is a real trade.
    ttime = (pd.to_datetime(raw[cols["trade_time"]], errors="coerce",
                            utc=True, format="ISO8601")
             if "trade_time" in cols else pd.Series(pd.NaT, index=df.index))
    df["trade_time"] = ttime
    df = df.sort_values("trade_time", na_position="first")

    grouped = (
        df.groupby(["ticker", "expiry", "strike", "type"], dropna=False)
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
    chain = _finalize(grouped.drop(columns=["flow"]))

    # last observed reference price per ticker = spot; last trade date = as-of
    spots: dict[str, float] = {}
    with_ref = df[df["ref_price"].notna()]
    for tkr, sub in with_ref.groupby("ticker"):
        spots[str(tkr)] = float(sub["ref_price"].iloc[-1])
    asof = None
    if ttime.notna().any():
        asof = ttime.max().date()
    tickers = list(df["ticker"].value_counts().index.astype(str))
    return ParsedFile(chain=chain, spots=spots, asof=asof, tickers=tickers)


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
