"""Dealer positioning analytics from option-chain data.

Computes net gamma exposure (GEX), the zero-gamma flip level, call/put
walls, and a dealer-hedging verdict from a normalized option chain.
"""

from dealer_gex.analytics import Analysis, analyze, bs_gamma
from dealer_gex.forecast import ExpectedMoveForecast, forecast_expected_move
from dealer_gex.parsing import (
    ChainParseError, ParsedFile, parse_file, read_chain, normalize_chain,
)

__all__ = [
    "Analysis",
    "analyze",
    "bs_gamma",
    "ExpectedMoveForecast",
    "forecast_expected_move",
    "ChainParseError",
    "ParsedFile",
    "parse_file",
    "read_chain",
    "normalize_chain",
]
