from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from dealer_gex.analytics import TRADING_DAYS, Analysis, analyze
from dealer_gex.forecast import (
    FEATURES, MEAN_ABS_OVER_SIGMA, ExpectedMoveForecast, build_dataset,
    forecast_expected_move, forecast_summary, implied_session_sigma,
    lightgbm_available, positioning_features, skill_score, walk_forward,
)
from dealer_gex.parsing import read_chain

REPO = Path(__file__).resolve().parent.parent
ASOF = date(2026, 7, 17)

ENGINES = ["ridge"] + (["lightgbm"] if lightgbm_available() else [])


def _mk_analysis(asof: date, spot: float, *, regime: str = "long_gamma",
                 implied_1d: float = 0.008, dte: int = 7,
                 call_oi: float = 1000.0, put_oi: float = 1000.0,
                 scale: float = 1.0) -> Analysis:
    """A minimal Analysis carrying exactly the fields the forecaster reads."""
    strikes = np.array([0.95, 1.0, 1.05]) * spot
    gex = np.array([1.0, 3.0, 1.0]) * 1e9 * scale
    if regime == "short_gamma":
        gex = -gex
    by_strike = pd.DataFrame({
        "strike": strikes,
        "call_gex": gex, "put_gex": -gex * 0.2, "net_gex": gex * 0.8,
        "call_oi": np.full(3, call_oi / 3), "put_oi": np.full(3, put_oi / 3),
    })
    t = dte / 365.0
    # invert implied_session_sigma: em = spot * sigma_1d * sqrt(252 * T)
    em = spot * implied_1d * np.sqrt(TRADING_DAYS * t)
    return Analysis(
        spot=spot, asof=asof, rate=0.045,
        total_gex=float(by_strike["net_gex"].sum()), regime=regime,
        gamma_flip=spot * 0.98, call_wall=spot * 1.05, put_wall=spot * 0.95,
        call_wall_strike=spot * 1.05, put_wall_strike=spot * 0.95,
        max_pain=spot, by_strike=by_strike, curve=pd.DataFrame(),
        by_expiry=pd.DataFrame(), n_contracts=6,
        dex=1e8 * scale, vanna_flow=1e7 * scale, charm_flow=-1e6 * scale,
        expected_move=em, nearest_expiry=asof + timedelta(days=dte),
    )


def _sessions(n: int, start: date = date(2026, 6, 1)) -> list[date]:
    """``n`` consecutive weekday dates."""
    out, d = [], start
    while len(out) < n:
        if d.weekday() < 5:
            out.append(d)
        d += timedelta(days=1)
    return out


def _days(closes, *, regimes=None, implied=0.008) -> list[dict]:
    dates = _sessions(len(closes))
    regimes = regimes or ["long_gamma"] * len(closes)
    imp = implied if isinstance(implied, (list, tuple)) else [implied] * len(closes)
    return [
        {"date": d, "analysis": _mk_analysis(d, c, regime=r, implied_1d=i),
         "close": c, "low": c * 0.99, "high": c * 1.01}
        for d, c, r, i in zip(dates, closes, regimes, imp)
    ]


# --- features ----------------------------------------------------------------

def test_implied_session_sigma_rescales_to_one_session():
    a = _mk_analysis(ASOF, 100.0, implied_1d=0.008, dte=7)
    assert implied_session_sigma(a) == pytest.approx(0.008, rel=1e-9)
    # same annualized vol, longer expiry -> same per-session number
    long_dated = _mk_analysis(ASOF, 100.0, implied_1d=0.008, dte=45)
    assert implied_session_sigma(long_dated) == pytest.approx(0.008, rel=1e-9)
    # ...but a bigger headline expected move, since it covers more days
    assert long_dated.expected_move > a.expected_move


def test_positioning_features_are_scale_free():
    """The same book quoted on a $100 stock and a $6,000 index must produce
    the same features — otherwise the model learns the ticker, not the setup."""
    small = positioning_features(_mk_analysis(ASOF, 100.0))
    big = positioning_features(_mk_analysis(ASOF, 6000.0, scale=60.0))
    for k in ("gex_ratio", "put_oi_share", "flip_dist", "call_wall_dist",
              "put_wall_dist", "wall_width", "max_pain_dist", "implied_1d",
              "iv_atm", "short_gamma"):
        assert small[k] == pytest.approx(big[k], rel=1e-9), k


def test_positioning_features_on_a_real_chain():
    chain, spot = read_chain((REPO / "data" / "sample_option_chain.csv").read_bytes())
    f = positioning_features(analyze(chain, spot, ASOF))
    assert set(FEATURES) >= set(f)
    assert 0.0 < f["implied_1d"] < 0.1          # a sane daily sigma
    assert 0.0 < f["iv_atm"] < 2.0
    assert -1.0 <= f["gex_ratio"] <= 1.0
    assert 0.0 <= f["put_oi_share"] <= 1.0
    assert f["dte"] == 7


def test_regime_flag_follows_net_gamma():
    assert positioning_features(_mk_analysis(ASOF, 100.0))["short_gamma"] == 0.0
    short = positioning_features(_mk_analysis(ASOF, 100.0, regime="short_gamma"))
    assert short["short_gamma"] == 1.0
    assert short["gex_ratio"] < 0


# --- dataset -----------------------------------------------------------------

def test_target_is_the_next_session_move_and_last_row_is_live():
    closes = [100.0, 101.0, 99.0, 99.5]
    f = build_dataset(_days(closes))
    assert len(f) == 4
    assert f["y"].iloc[0] == pytest.approx(1.0 / 100.0)
    assert f["y"].iloc[1] == pytest.approx(2.0 / 101.0)
    assert f["y"].iloc[2] == pytest.approx(0.5 / 99.0)
    assert pd.isna(f["y"].iloc[-1])          # nothing to score the live row on
    assert pd.isna(f["target_date"].iloc[-1])
    assert f["target_date"].iloc[0] == f["date"].iloc[1]


def test_features_never_see_the_future():
    """Row t's lags are the moves that already happened; the move it is
    asked to predict must not appear among them."""
    closes = [100.0, 104.0, 101.0, 103.0, 100.0]
    f = build_dataset(_days(closes))
    assert pd.isna(f["realized_lag1"].iloc[0])                 # nothing before day 0
    assert f["realized_lag1"].iloc[1] == pytest.approx(4.0 / 100.0)
    assert f["realized_lag1"].iloc[2] == pytest.approx(3.0 / 104.0)
    assert f["realized_lag2"].iloc[2] == pytest.approx(4.0 / 100.0)
    for i in range(len(f) - 1):
        assert f["realized_lag1"].iloc[i] != pytest.approx(f["y"].iloc[i])


def test_baseline_converts_sigma_to_an_absolute_move():
    f = build_dataset(_days([100.0, 101.0], implied=0.01))
    assert f["baseline"].iloc[0] == pytest.approx(0.01 * MEAN_ABS_OVER_SIGMA)


def test_multi_session_gaps_are_sqrt_scaled_and_stale_pairs_dropped():
    mon, wed, fri = date(2026, 6, 1), date(2026, 6, 3), date(2026, 6, 5)
    much_later = date(2026, 7, 20)
    closes = {mon: 100.0, wed: 102.0, fri: 102.0, much_later: 110.0}
    days = [{"date": d, "analysis": _mk_analysis(d, c), "close": c}
            for d, c in closes.items()]
    f = build_dataset(days)
    # Mon -> Wed spans two sessions: a 2% move is ~1.41% per session
    assert f["sessions"].iloc[0] == 2.0
    assert f["y"].iloc[0] == pytest.approx(0.02 / np.sqrt(2))
    # Fri -> a month later is too stale to be a one-session target
    assert pd.isna(f["y"].iloc[2])
    assert pd.isna(f["sessions"].iloc[2])


def test_weekend_is_one_session():
    fri, mon = date(2026, 6, 5), date(2026, 6, 8)
    days = [{"date": d, "analysis": _mk_analysis(d, c), "close": c}
            for d, c in [(fri, 100.0), (mon, 102.0)]]
    f = build_dataset(days)
    assert f["sessions"].iloc[0] == 1.0
    assert f["y"].iloc[0] == pytest.approx(0.02)


def test_duplicate_dates_collapse_and_order_is_normalized():
    d0, d1 = date(2026, 6, 1), date(2026, 6, 2)
    days = [
        {"date": d1, "analysis": _mk_analysis(d1, 101.0), "close": 101.0},
        {"date": d0, "analysis": _mk_analysis(d0, 100.0), "close": 100.0},
        {"date": d1, "analysis": _mk_analysis(d1, 102.0), "close": 102.0},
    ]
    f = build_dataset(days)
    assert list(f["date"]) == [d0, d1]
    assert f["spot"].iloc[1] == 102.0        # the later upload for that date wins
    assert f["y"].iloc[0] == pytest.approx(0.02)


def test_close_falls_back_to_spot_when_no_prints():
    days = [{"date": d, "analysis": _mk_analysis(d, c)}
            for d, c in zip(_sessions(3), [100.0, 103.0, 103.0])]
    f = build_dataset(days)
    assert f["y"].iloc[0] == pytest.approx(0.03)
    assert pd.isna(f["range_pct"]).all()      # no reconstructed range available


def test_every_declared_feature_is_actually_populated():
    """A misspelled key would leave an all-NaN column the model silently
    ignores — catch that here rather than in a quiet loss of skill."""
    f = build_dataset(_days([100.0, 101.0, 99.0, 100.5, 102.0]))
    assert list(f.columns[-len(FEATURES):]) == FEATURES
    assert not f[FEATURES].isna().all().any()


def test_empty_input_is_empty_frame():
    assert build_dataset([]).empty
    assert build_dataset([{"date": None, "analysis": None}]).empty


# --- walk-forward evaluation -------------------------------------------------

@pytest.mark.parametrize("engine", ENGINES)
def test_walk_forward_is_expanding_and_out_of_sample(engine):
    rng = np.random.default_rng(0)
    closes = 100 * np.cumprod(1 + rng.normal(0, 0.01, 15))
    f = build_dataset(_days(list(closes)))
    bt = walk_forward(f, engine=engine, min_train=6)
    labelled = int(f["y"].notna().sum())
    assert len(bt) == labelled - 6            # one prediction per unseen row
    assert (bt["predicted"] >= 0).all()
    # every scored row is later than the training window it was fit on
    assert list(bt["date"]) == list(f["date"].iloc[6:labelled])


def test_walk_forward_needs_a_minimum_history():
    f = build_dataset(_days([100.0, 101.0, 100.5, 101.5]))
    assert walk_forward(f, min_train=6).empty


def test_skill_score_arithmetic():
    bt = pd.DataFrame({"actual": [0.01, 0.02], "predicted": [0.011, 0.021],
                       "baseline": [0.012, 0.022]})
    s = skill_score(bt)
    assert s["n"] == 2
    assert s["mae_model"] == pytest.approx(0.001)
    assert s["mae_baseline"] == pytest.approx(0.002)
    assert s["skill"] == pytest.approx(0.5)
    assert skill_score(pd.DataFrame())["skill"] is None


# --- the pipeline ------------------------------------------------------------

def test_thin_history_falls_back_to_the_implied_move():
    f = forecast_expected_move(_days([100.0, 101.0, 100.0], implied=0.01))
    assert f.status == "insufficient_history"
    assert not f.used_model and f.weight == 0.0
    assert f.blended_pct == pytest.approx(f.baseline_pct)
    assert f.baseline_pct == pytest.approx(0.01 * MEAN_ABS_OVER_SIGMA)
    assert f.predicted_sigma == pytest.approx(100.0 * 0.01)   # back to 1 sigma
    assert "more daily export" in f.message


def test_no_implied_move_is_reported_not_guessed():
    d = date(2026, 6, 1)
    a = _mk_analysis(d, 100.0)
    a.expected_move, a.nearest_expiry = None, None
    f = forecast_expected_move([{"date": d, "analysis": a, "close": 100.0}])
    assert f.status == "no_data"
    assert f.blended_pct is None
    assert forecast_summary(f) == "Expected move: not available."


def test_no_days_at_all():
    f = forecast_expected_move([])
    assert f.status == "no_data" and f.blended_pct is None


@pytest.mark.parametrize("engine", ENGINES)
def test_model_earns_weight_when_positioning_actually_predicts(engine):
    """Short-gamma days really do move more here, and the implied move is
    flat and wrong. The model should find that and beat the baseline."""
    n = 26
    regimes = ["short_gamma" if i % 2 else "long_gamma" for i in range(n)]
    rng = np.random.default_rng(3)
    closes = [100.0]
    for i in range(n - 1):
        step = (0.020 if regimes[i] == "short_gamma" else 0.002)
        closes.append(closes[-1] * (1 + step * rng.choice([-1.0, 1.0])))
    days = _days(closes, regimes=regimes, implied=0.011)

    f = forecast_expected_move(days, engine=engine)
    assert f.status == "ok", f.message
    assert f.engine == engine
    assert f.skill > 0 and 0 < f.weight <= 1
    assert f.mae_model < f.mae_baseline
    assert f.n_oos == f.n_train - 6
    # the blend sits between the implied move and the model's own number
    assert min(f.baseline_pct, f.model_pct) <= f.blended_pct <= max(f.baseline_pct, f.model_pct)
    assert not f.importance.empty
    assert f.importance["share"].sum() == pytest.approx(1.0)


@pytest.mark.parametrize("engine", ENGINES)
def test_model_stands_down_when_implied_is_already_right(engine):
    """Moves are pure noise around a correctly-priced implied move: there is
    nothing in the features to learn, so the model must take zero weight."""
    rng = np.random.default_rng(11)
    n, sigma = 30, 0.01
    steps = rng.normal(0, sigma, n - 1)
    closes = [100.0]
    for s in steps:
        closes.append(closes[-1] * (1 + s))
    f = forecast_expected_move(_days(closes, implied=sigma), engine=engine)
    assert f.status in ("ok", "no_skill")
    if f.status == "no_skill":
        assert f.weight == 0.0
        assert f.blended_pct == pytest.approx(f.baseline_pct)
    else:
        # any edge on pure noise must be marginal, not a takeover
        assert f.weight < 0.35


def test_richness_and_summary_read_the_right_way():
    base = ExpectedMoveForecast(status="ok", message="", spot=100.0,
                                baseline_pct=0.008, model_pct=0.016,
                                blended_pct=0.012, weight=0.5, skill=0.5,
                                engine="lightgbm")
    assert base.used_model
    assert base.richness == pytest.approx(1.5)
    assert "wider than the market" in forecast_summary(base)

    tight = ExpectedMoveForecast(status="ok", message="", spot=100.0,
                                 baseline_pct=0.008, model_pct=0.004,
                                 blended_pct=0.006, weight=0.5, skill=0.5,
                                 engine="lightgbm")
    assert "tighter than the market" in forecast_summary(tight)

    off = ExpectedMoveForecast(status="no_skill", message="", spot=100.0,
                               baseline_pct=0.008, blended_pct=0.008)
    assert not off.used_model
    assert "implied" in forecast_summary(off)
    # a fraction-of-spot forecast reads back as a 1-sigma dollar move
    assert off.predicted_sigma == pytest.approx(0.008 / MEAN_ABS_OVER_SIGMA * 100)


def test_engine_selection_and_fallback():
    from dealer_gex.forecast import _resolve_engine

    assert _resolve_engine("ridge") == "ridge"
    assert _resolve_engine("auto") == ("lightgbm" if lightgbm_available() else "ridge")
    # ridge is always available, so the pipeline runs without lightgbm
    f = forecast_expected_move(_days([100.0, 101.0, 100.0]), engine="ridge")
    assert f.engine == "ridge"


@pytest.mark.skipif(not lightgbm_available(), reason="lightgbm not installed")
def test_lightgbm_is_deterministic():
    rng = np.random.default_rng(5)
    closes = list(100 * np.cumprod(1 + rng.normal(0, 0.012, 22)))
    a = forecast_expected_move(_days(closes), engine="lightgbm")
    b = forecast_expected_move(_days(closes), engine="lightgbm")
    assert a.engine == "lightgbm"
    assert a.blended_pct == pytest.approx(b.blended_pct)
    assert a.skill == pytest.approx(b.skill)


# --- report integration ------------------------------------------------------

def _skilled_forecast() -> ExpectedMoveForecast:
    return ExpectedMoveForecast(
        status="ok", message="Model beat the implied move by 30% (MAE).",
        spot=100.0, asof=ASOF, baseline_pct=0.008, model_pct=0.016,
        blended_pct=0.010, weight=0.3, skill=0.3, mae_model=0.004,
        mae_baseline=0.0057, n_oos=5, n_train=11, engine="lightgbm",
        importance=pd.DataFrame({"feature": ["short_gamma", "implied_1d"],
                                 "importance": [3.0, 1.0],
                                 "share": [0.75, 0.25]}),
    )


def test_report_shows_implied_and_model_side_by_side():
    from dealer_gex.report import build_markdown

    chain, spot = read_chain((REPO / "data" / "sample_option_chain.csv").read_bytes())
    a = analyze(chain, spot, ASOF)
    f = _skilled_forecast()
    f.spot = a.spot
    md = build_markdown(a, ticker="TEST", forecast=f)
    assert "## Expected move — implied vs model" in md
    assert "Implied (options, per session)" in md
    assert "Model (lightgbm, blended)" in md
    assert "walk-forward" in md
    assert "short_gamma (75%)" in md
    # no forecast -> no section at all
    assert "Expected move — implied vs model" not in build_markdown(a, ticker="TEST")


def test_report_omits_the_model_row_when_it_has_no_edge():
    from dealer_gex.report import build_markdown

    chain, spot = read_chain((REPO / "data" / "sample_option_chain.csv").read_bytes())
    a = analyze(chain, spot, ASOF)
    f = ExpectedMoveForecast(status="no_skill", message="No edge.", spot=a.spot,
                             baseline_pct=0.008, blended_pct=0.008, skill=-0.1,
                             engine="lightgbm")
    md = build_markdown(a, forecast=f)
    assert "Implied (options, per session)" in md
    assert "Model (lightgbm, blended)" not in md   # no row for a model at zero weight
    assert "No edge." in md


def test_executive_summary_quotes_the_model_only_when_it_is_used():
    from dealer_gex.report import executive_summary

    chain, spot = read_chain((REPO / "data" / "sample_option_chain.csv").read_bytes())
    a = analyze(chain, spot, ASOF)
    f = _skilled_forecast()
    f.spot = a.spot
    assert "the model expects" in executive_summary(a, forecast=f).lower()
    assert "under-pricing" in executive_summary(a, forecast=f)
    off = ExpectedMoveForecast(status="no_skill", message="", spot=a.spot,
                               baseline_pct=0.008, blended_pct=0.008)
    assert "model expects" not in executive_summary(a, forecast=off).lower()
    assert "model expects" not in executive_summary(a).lower()


def test_pipeline_on_analyses_from_the_real_sample_chain():
    """End to end on chains the parser actually produced, one per session."""
    chain, spot = read_chain((REPO / "data" / "sample_option_chain.csv").read_bytes())
    rng = np.random.default_rng(2)
    days, px = [], spot
    for d in _sessions(14, start=date(2026, 7, 6)):
        px *= 1 + rng.normal(0, 0.01)
        c = chain.copy()
        c = c[c["expiry"].dt.date > d]
        days.append({"date": d, "analysis": analyze(c, px, d), "close": px,
                     "low": px * 0.995, "high": px * 1.006})
    f = forecast_expected_move(days)
    assert f.status in ("ok", "no_skill")
    assert f.blended_pct > 0 and f.baseline_pct > 0
    assert f.n_days == 14 and f.n_oos >= 3
    assert not f.backtest.empty
    assert f.spot == pytest.approx(px)
    assert 0 < f.predicted_sigma < px          # a sane dollar move
