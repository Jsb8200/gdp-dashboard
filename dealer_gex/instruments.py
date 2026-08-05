"""Instrument presets: what one contract is actually worth.

Every dollar figure in the app — GEX, DEX, vanna, charm, block premium —
scales with the contract multiplier, and the multiplier is not 100 outside
equities. An NQ book priced at 100 overstates its hedging flow five-fold;
an ES book understates it by half. Levels are unaffected, dollars are not.

Futures symbols arrive with a month and year glued on (``ESU6``,
``NQZ25``, ``/MESH6``), so detection strips the contract code and matches
the root — longest root first, or ``MES`` would resolve as ``ES`` and
``MNQ`` as ``NQ``, each off by a factor of ten.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

#: CME month codes, stripped off a symbol before matching the root.
_MONTH_CODES = "FGHJKMNQUVXZ"


@dataclass(frozen=True)
class Instrument:
    root: str
    name: str
    multiplier: float
    kind: str          # 'future' | 'index' | 'etf' | 'equity'
    note: str = ""

    @property
    def label(self) -> str:
        return f"{self.root} · {self.name} (×{self.multiplier:g})"


#: Roots we know. Anything else falls back to the equity/ETF convention of
#: 100, which is right for stocks, ETFs and cash index options.
INSTRUMENTS: dict[str, Instrument] = {
    i.root: i for i in [
        # --- index futures -------------------------------------------------
        Instrument("ES", "S&P 500 E-mini", 50.0, "future", "$50 × index"),
        Instrument("MES", "S&P 500 Micro", 5.0, "future", "$5 × index"),
        Instrument("NQ", "Nasdaq-100 E-mini", 20.0, "future", "$20 × index"),
        Instrument("MNQ", "Nasdaq-100 Micro", 2.0, "future", "$2 × index"),
        Instrument("RTY", "Russell 2000 E-mini", 50.0, "future", "$50 × index"),
        Instrument("M2K", "Russell 2000 Micro", 5.0, "future", "$5 × index"),
        Instrument("YM", "Dow E-mini", 5.0, "future", "$5 × index"),
        Instrument("MYM", "Dow Micro", 2.0, "future", "$2 × index"),
        # --- metals & energy ----------------------------------------------
        Instrument("GC", "Gold", 100.0, "future", "100 troy oz"),
        Instrument("MGC", "Gold Micro", 10.0, "future", "10 troy oz"),
        Instrument("SI", "Silver", 5000.0, "future", "5,000 troy oz"),
        Instrument("CL", "Crude oil (WTI)", 1000.0, "future", "1,000 barrels"),
        Instrument("MCL", "Crude oil Micro", 100.0, "future", "100 barrels"),
        Instrument("NG", "Natural gas", 10000.0, "future", "10,000 MMBtu"),
        # --- cash indices and their ETFs ----------------------------------
        Instrument("SPX", "S&P 500 index", 100.0, "index"),
        Instrument("XSP", "S&P 500 Mini index", 100.0, "index", "1/10th of SPX"),
        Instrument("NDX", "Nasdaq-100 index", 100.0, "index"),
        Instrument("RUT", "Russell 2000 index", 100.0, "index"),
        Instrument("VIX", "Volatility index", 100.0, "index"),
        Instrument("SPY", "S&P 500 ETF", 100.0, "etf"),
        Instrument("QQQ", "Nasdaq-100 ETF", 100.0, "etf"),
        Instrument("IWM", "Russell 2000 ETF", 100.0, "etf"),
        Instrument("GLD", "Gold ETF", 100.0, "etf"),
        Instrument("SLV", "Silver ETF", 100.0, "etf"),
        Instrument("USO", "Crude oil ETF", 100.0, "etf"),
    ]
}

DEFAULT_INSTRUMENT = Instrument("", "Equity / ETF / index", 100.0, "equity",
                                "100 shares per contract")

#: Roots longest-first, so MES matches before ES and MNQ before NQ.
_ROOTS = sorted(INSTRUMENTS, key=len, reverse=True)


def normalize_symbol(ticker: str) -> str:
    """Strip the decoration a futures symbol arrives with.

    ``/ESU6`` → ``ES``, ``NQZ25`` → ``NQ``, ``.SPX`` → ``SPX``. Only a
    trailing month+year is removed, so ``SI`` (silver) does not lose its
    letters and an equity ticker passes through untouched.
    """
    s = re.sub(r"[^A-Za-z0-9]", "", str(ticker or "")).upper()
    if not s:
        return ""
    # trailing CME contract code: one month letter plus 1-2 year digits
    m = re.match(rf"^([A-Z]{{1,4}})([{_MONTH_CODES}])(\d{{1,2}})$", s)
    if m and m.group(1) in INSTRUMENTS:
        return m.group(1)
    return s


def detect_instrument(ticker: str) -> Instrument:
    """Best-known instrument for a ticker, or the equity default.

    Never guesses past an exact root match: a symbol that merely *starts*
    with a known root (``ESGV``, ``GLDM``) is a different product, and
    silently applying a futures multiplier to it would be worse than the
    100 default.
    """
    sym = normalize_symbol(ticker)
    if not sym:
        return DEFAULT_INSTRUMENT
    if sym in INSTRUMENTS:
        return INSTRUMENTS[sym]
    return DEFAULT_INSTRUMENT


def multiplier_for(ticker: str) -> float:
    return detect_instrument(ticker).multiplier


def instrument_choices() -> list[str]:
    """Labels for a picker, futures first — they are the ones whose
    multiplier a user cannot guess."""
    order = {"future": 0, "index": 1, "etf": 2, "equity": 3}
    items = sorted(INSTRUMENTS.values(),
                   key=lambda i: (order.get(i.kind, 9), i.root))
    return [DEFAULT_INSTRUMENT.label.replace(" · ", " ")] + [i.label for i in items]


def instrument_from_choice(choice: str) -> Instrument:
    for i in INSTRUMENTS.values():
        if i.label == choice:
            return i
    return DEFAULT_INSTRUMENT
