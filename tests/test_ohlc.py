"""Futures candles: IST in, real sessions out."""

from datetime import date

import pandas as pd
import pytest

from dealer_gex.analytics import range_from_ohlc, session_ohlc
from dealer_gex.parsing import ChainParseError, is_ohlc, parse_ohlc

# The common futures shape: a separate Date and Time, local wall clock, no
# zone anywhere in the file.
IST_CSV = b"""Symbol,Date,Time,Open,High,Low,Close,Volume
NQ,2026-07-31,19:15,23110.25,23140.00,23100.50,23132.75,4210
NQ,2026-07-31,19:30,23132.75,23180.25,23128.00,23175.50,3980
NQ,2026-07-31,23:45,23175.50,23190.00,23050.75,23064.25,5120
NQ,2026-08-01,02:30,23064.25,23099.00,23040.00,23088.50,3300
"""


def test_a_naive_stamp_is_read_as_ist_not_utc():
    """The whole point: 19:15 on the file's clock is the 09:45 New York
    open. Reading it as UTC would put it at 15:15 ET, after the close."""
    c = parse_ohlc(IST_CSV)
    assert str(c["time"].iloc[0]) == "2026-07-31 13:45:00+00:00"
    et = c["time"].dt.tz_convert("America/New_York")
    assert et.iloc[0].strftime("%H:%M") == "09:45"


def test_the_date_and_time_columns_are_both_used():
    """Regression: 'Date' claiming the timestamp slot dropped 'Time' and
    collapsed every bar in a session onto midnight."""
    c = parse_ohlc(IST_CSV)
    assert c["time"].nunique() == 4          # four distinct bars, not one
    assert (c["time"].diff().dropna() > pd.Timedelta(0)).all()


def test_an_after_midnight_ist_bar_belongs_to_the_previous_us_session():
    """01 Aug 02:30 IST is 31 Jul 17:00 in New York — the same session as
    the 19:15 bar. Grouping on the file's own date would split them."""
    sess = session_ohlc(parse_ohlc(IST_CSV))
    assert len(sess) == 1
    assert sess["date"].iloc[0] == date(2026, 7, 31)
    assert sess["bars"].iloc[0] == 4


def test_session_ohlc_aggregates_by_time_not_row_order():
    shuffled = parse_ohlc(IST_CSV).sample(frac=1, random_state=0)
    s = session_ohlc(shuffled).iloc[0]
    assert s["open"] == 23110.25            # first bar's open
    assert s["close"] == 23088.50           # last bar's close
    assert s["high"] == 23190.00 and s["low"] == 23040.00
    assert s["volume"] == 16610


def test_an_explicit_offset_is_respected_not_shifted_again():
    csv = b"""Timestamp,Open,High,Low,Close
2026-07-31T13:45:00+00:00,100,101,99,100.5
2026-07-31T14:45:00+00:00,100.5,102,100,101.5
"""
    c = parse_ohlc(csv, tz="Asia/Kolkata")
    assert str(c["time"].iloc[0]) == "2026-07-31 13:45:00+00:00"


def test_the_timezone_is_a_choice():
    utc = parse_ohlc(IST_CSV, tz="UTC")["time"].iloc[0]
    ist = parse_ohlc(IST_CSV, tz="Asia/Kolkata")["time"].iloc[0]
    assert (utc - ist) == pd.Timedelta(hours=5, minutes=30)


def test_range_matches_the_shape_session_range_reconstructs():
    """(low, high, close, open) — the same tuple, measured instead of
    inferred, so it drops straight into the hit-rate."""
    got = range_from_ohlc(parse_ohlc(IST_CSV), date(2026, 7, 31))
    assert got == (23040.0, 23190.0, 23088.5, 23110.25)
    assert range_from_ohlc(parse_ohlc(IST_CSV), date(2026, 8, 5)) is None


def test_a_title_row_above_the_header_is_skipped():
    csv = b"""NQ continuous, 15 minute
    
Date,Time,Open,High,Low,Close
2026-07-31,19:15,1,2,0.5,1.5
"""
    assert len(parse_ohlc(csv)) == 1


def test_a_chain_is_not_mistaken_for_candles():
    chain = pd.DataFrame({"Strike": [1], "Expiry": ["2026-08-21"],
                          "Open Interest": [1], "IV": [0.2]})
    assert not is_ohlc(chain)
    with pytest.raises(ChainParseError):
        parse_ohlc(b"Strike,Expiration Date,Open Interest,IV\n100,2026-08-21,5,0.2\n")


def test_candles_without_a_clock_are_rejected():
    with pytest.raises(ChainParseError):
        parse_ohlc(b"Open,High,Low,Close\n1,2,0.5,1.5\n")


def test_empty_and_missing_inputs():
    assert session_ohlc(None).empty
    assert session_ohlc(pd.DataFrame()).empty
    assert range_from_ohlc(pd.DataFrame(), date(2026, 7, 31)) is None
