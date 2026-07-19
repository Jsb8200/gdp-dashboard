"""Canonical schema and column mapping for uploaded options CSVs.

Pure pandas/numpy — no Streamlit imports, so everything here can be
exercised from a plain python shell.
"""

import re

import numpy as np
import pandas as pd

# Canonical column -> accepted aliases (compared after normalize_header()).
# The canonical name itself is always accepted.
ALIASES = {
    'symbol': ['ticker', 'underlyingsymbol', 'root', 'symb'],
    'type': ['cpflag', 'cp', 'callput', 'optiontype', 'putcall', 'right',
             'side', 'kind', 'contracttype'],
    'strike': ['strikeprice', 'k', 'exerciseprice'],
    'expiration': ['expirationdate', 'exp', 'expdate', 'expiry', 'expirydate',
                   'maturity', 'maturitydate'],
    'bid': ['bidprice'],
    'ask': ['askprice', 'offer'],
    'last': ['lastprice', 'lasttrade', 'close', 'price', 'mark', 'mid',
             'optionprice', 'tradeprice', 'fillprice'],
    'volume': ['vol', 'totalvolume', 'tradevolume'],
    'open_interest': ['openinterest', 'oi', 'openint', 'opint'],
    'implied_volatility': ['impliedvolatility', 'iv', 'impliedvol', 'impvol',
                           'sigma', 'midiv'],
    'underlying_price': ['underlyingprice', 'underlying', 'underlyinglast',
                         'spot', 'spotprice', 'stockprice', 'referenceprice'],
    'timestamp': ['date', 'quotedate', 'datadate', 'snapshottime', 'asofdate',
                  'tradedate', 'tradetime', 'time', 'datetime'],
}

REQUIRED = ['type', 'strike', 'expiration']
NUMERIC_COLS = ['strike', 'bid', 'ask', 'last', 'volume', 'open_interest',
                'implied_volatility', 'underlying_price']
DATE_COLS = ['expiration', 'timestamp']


def normalize_header(name):
    """Lowercase and strip everything that isn't a letter or digit."""
    return re.sub(r'[^a-z0-9]', '', str(name).lower())


def map_columns(raw_df, overrides=None):
    """Map raw CSV columns onto the canonical schema.

    overrides: optional {canonical: raw_column} dict that wins over the
    alias table (used by the manual-mapping UI).

    Returns (df, mapping, unmapped_canonical, ignored_raw) where df has
    canonical column names, mapping is {canonical: raw_column}.
    """
    overrides = overrides or {}
    lookup = {}
    for canonical, aliases in ALIASES.items():
        lookup[normalize_header(canonical)] = canonical
        for alias in aliases:
            lookup[alias] = canonical

    mapping = {}
    for raw in raw_df.columns:
        canonical = lookup.get(normalize_header(raw))
        if canonical and canonical not in mapping:
            mapping[canonical] = raw
    for canonical, raw in overrides.items():
        if raw in raw_df.columns:
            mapping[canonical] = raw

    df = pd.DataFrame({canonical: raw_df[raw] for canonical, raw in mapping.items()})
    unmapped = [c for c in ALIASES if c not in mapping]
    ignored = [c for c in raw_df.columns if c not in mapping.values()]
    return df, mapping, unmapped, ignored


def coerce_types(df):
    """Coerce mapped columns to canonical dtypes/values, in place-ish.

    Returns (df, notes) where notes is a list of human-readable messages
    about normalization decisions (e.g. IV rescaled from percent).
    """
    df = df.copy()
    notes = []

    if 'type' in df:
        first = df['type'].astype(str).str.strip().str.lower().str[:1]
        df['type'] = first.map({'c': 'call', 'p': 'put'})

    for col in DATE_COLS:
        if col in df:
            df[col] = pd.to_datetime(df[col], errors='coerce', utc=True)
            df[col] = df[col].dt.tz_localize(None)

    for col in NUMERIC_COLS:
        if col in df and not pd.api.types.is_numeric_dtype(df[col]):
            cleaned = df[col].astype(str).str.replace(r'[$,%\s]', '', regex=True)
            df[col] = pd.to_numeric(cleaned, errors='coerce')

    if 'implied_volatility' in df:
        median_iv = df['implied_volatility'].median()
        if pd.notna(median_iv) and median_iv > 3.0:
            df['implied_volatility'] = df['implied_volatility'] / 100.0
            notes.append('Implied volatility looked like percent values '
                         f'(median {median_iv:.1f}); rescaled to decimals.')

    return df, notes


def add_derived(df):
    """Add mid, as_of and dte columns. Requires coerce_types() first."""
    df = df.copy()

    bid = df['bid'] if 'bid' in df else pd.Series(np.nan, index=df.index)
    ask = df['ask'] if 'ask' in df else pd.Series(np.nan, index=df.index)
    last = df['last'] if 'last' in df else pd.Series(np.nan, index=df.index)
    both_quoted = (bid > 0) & (ask > 0)
    df['mid'] = np.where(both_quoted, (bid + ask) / 2, last)

    if 'timestamp' in df and df['timestamp'].notna().any():
        as_of = df['timestamp'].max()
    else:
        as_of = pd.Timestamp.today().normalize()
    # Store as str: non-JSON-serializable attrs make st.dataframe complain.
    df.attrs['as_of'] = as_of.isoformat()

    if 'expiration' in df:
        dte = (df['expiration'] - as_of).dt.days
        df['dte'] = dte.clip(lower=1)

    return df


def contract_keys(df):
    """Columns identifying one contract, for the columns present in df."""
    return [c for c in ('symbol', 'type', 'strike', 'expiration') if c in df]


def collapse_trades(df):
    """Aggregate per-trade rows (order-flow exports) to one row per contract.

    Takes the latest quote fields per contract and the max of the running
    totals. Returns (collapsed, n_trades); n_trades is 0 when the frame
    already has one row per contract (no collapsing happened).
    """
    keys = contract_keys(df)
    if not keys or not df.duplicated(keys).any():
        return df, 0

    out = df.copy()
    if 'timestamp' in out:
        out = out.sort_values('timestamp')
    agg = {}
    for col in ('bid', 'ask', 'last', 'implied_volatility', 'underlying_price'):
        if col in out:
            agg[col] = 'last'
    for col in ('volume', 'open_interest', 'timestamp'):
        if col in out:
            agg[col] = 'max'
    collapsed = out.groupby(keys, as_index=False, dropna=False).agg(agg)
    return collapsed, len(df)


def load_chain(raw_df, overrides=None):
    """Full pipeline: map, coerce, collapse trade rows, derive.

    Returns (df, info) where info carries the mapping details, notes for
    display in the UI, and `trades` — the pre-collapse frame when the CSV
    was per-trade order-flow data (None otherwise).
    """
    df, mapping, unmapped, ignored = map_columns(raw_df, overrides)
    coerced, notes = coerce_types(df)
    df, n_trades = collapse_trades(coerced)
    trades = None
    if n_trades:
        trades = coerced
        notes.append(f'Detected order-flow data: aggregated {n_trades:,} '
                     f'trades into {len(df):,} contracts.')
    df = add_derived(df)
    info = {
        'mapping': mapping,
        'unmapped': unmapped,
        'ignored': ignored,
        'notes': notes,
        'as_of': df.attrs.get('as_of'),
        'trades': trades,
    }
    return df, info


def missing_columns(df, required):
    """Canonical columns from `required` absent or entirely null in df."""
    return [c for c in required if c not in df or df[c].isna().all()]


def alias_help(canonical):
    """Human-readable list of accepted header names for a canonical column."""
    return ', '.join([canonical] + ALIASES.get(canonical, []))
