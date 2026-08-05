"""Contract multipliers: futures are not 100."""

import pytest

from dealer_gex.instruments import (
    DEFAULT_INSTRUMENT, INSTRUMENTS, detect_instrument, instrument_choices,
    instrument_from_choice, multiplier_for, normalize_symbol,
)


def test_the_micros_do_not_resolve_to_their_parents():
    """MES matching as ES, or MNQ as NQ, is a factor-of-ten error in every
    dollar figure on the page. Longest root wins."""
    assert multiplier_for("MES") == 5.0 and multiplier_for("ES") == 50.0
    assert multiplier_for("MNQ") == 2.0 and multiplier_for("NQ") == 20.0
    assert multiplier_for("MGC") == 10.0 and multiplier_for("GC") == 100.0
    assert multiplier_for("MCL") == 100.0 and multiplier_for("CL") == 1000.0


def test_the_ones_asked_for():
    assert multiplier_for("NQ") == 20.0            # Nasdaq E-mini
    assert multiplier_for("ES") == 50.0            # S&P 500 E-mini
    assert multiplier_for("SPX") == 100.0          # S&P 500 cash index
    assert multiplier_for("GLD") == 100.0          # gold ETF
    assert multiplier_for("GC") == 100.0           # gold future, 100 oz


def test_contract_month_codes_are_stripped():
    for sym, root in (("ESU6", "ES"), ("NQZ25", "NQ"), ("/MESH6", "MES"),
                      ("GCM6", "GC"), ("/NQ", "NQ")):
        assert normalize_symbol(sym) == root, sym
        assert detect_instrument(sym).root == root


def test_an_unknown_ticker_falls_back_to_the_equity_convention():
    for sym in ("AAPL", "TSLA", "", "   ", "ZZZZ"):
        assert detect_instrument(sym) is DEFAULT_INSTRUMENT
        assert multiplier_for(sym) == 100.0


def test_a_lookalike_ticker_is_not_given_a_futures_multiplier():
    """ESGV is an ETF, not the S&P future; GLDM is not GLD. Guessing past an
    exact root match would silently misprice the whole book."""
    assert detect_instrument("ESGV") is DEFAULT_INSTRUMENT
    assert detect_instrument("GLDM") is DEFAULT_INSTRUMENT
    assert detect_instrument("NQXY") is DEFAULT_INSTRUMENT


def test_silver_keeps_its_letters():
    """SI is two letters and looks like a stripped code; it must survive."""
    assert normalize_symbol("SI") == "SI"
    assert multiplier_for("SI") == 5000.0


def test_choices_are_round_trippable_and_lead_with_futures():
    choices = instrument_choices()
    assert len(choices) == len(INSTRUMENTS) + 1
    futures = [c for c in choices[1:6]]
    assert all(instrument_from_choice(c).kind == "future" for c in futures)
    for c in choices[1:]:
        i = instrument_from_choice(c)
        assert i.label == c
        assert i.multiplier > 0


def test_every_instrument_is_well_formed():
    for root, i in INSTRUMENTS.items():
        assert i.root == root
        assert i.multiplier > 0
        assert i.kind in {"future", "index", "etf", "equity"}
        assert i.name and f"×{i.multiplier:g}" in i.label


def test_multiplier_scales_dollars_but_not_levels():
    """The reason the preset matters: an NQ book priced at 100 overstates
    its hedging flow five-fold, while every level stays put."""
    from datetime import date

    from dealer_gex.analytics import analyze
    from dealer_gex.parsing import read_chain

    chain, spot = read_chain(open("data/sample_option_chain.csv", "rb").read())
    at100 = analyze(chain, spot, date(2026, 7, 17), multiplier=100.0)
    at20 = analyze(chain, spot, date(2026, 7, 17), multiplier=20.0)
    assert at20.total_gex == pytest.approx(at100.total_gex * 0.2)
    assert at20.call_wall == pytest.approx(at100.call_wall)
    assert at20.put_wall == pytest.approx(at100.put_wall)
    assert at20.gamma_flip == pytest.approx(at100.gamma_flip)
