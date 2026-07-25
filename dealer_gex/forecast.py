"""Expected-move forecasting with gradient-boosted trees (LightGBM).

The options-implied expected move is a *price*, not a forecast: the 1-sigma
straddle approximation from at-the-money IV. It only predicts realized
movement under the assumption that implied equals realized — which it
usually does not (the variance risk premium). This module learns the map
from *dealer positioning state* to the **next session's realized move**,
using the day-over-day history the dashboard already reconstructs, and
reports a model number only when it beats the implied baseline out of
sample.

Everything here is gated. With a handful of days there is nothing to learn,
so the module says exactly that and falls back to the implied baseline
rather than dressing up noise as a model:

* features for day *t* use only information available at day *t*'s close;
  the target is the move from *t* to *t+1* (no lookahead, ever);
* the model is scored by **expanding-window walk-forward** — every
  evaluated prediction is out of sample;
* the blend weight is the measured skill against the implied baseline, so
  a model that cannot beat implied gets weight zero.

LightGBM is optional: without it the same pipeline runs on a ridge
regression fallback, clearly labelled in the output.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from dealer_gex.analytics import MIN_T, TRADING_DAYS, Analysis

# E|X| = sigma * sqrt(2/pi) for a zero-mean normal: the implied 1-sigma move
# has to be scaled by this before it can be compared with a realized
# *absolute* move.
MEAN_ABS_OVER_SIGMA = float(np.sqrt(2.0 / np.pi))

EPS = 1e-8            # keeps the log-target finite on a zero-move session
MIN_TRAIN_ROWS = 6    # rows needed before a fit is attempted at all
MIN_OOS_ROWS = 3      # out-of-sample predictions needed to measure skill
MAX_GAP_SESSIONS = 5  # drop day-pairs further apart than this (stale target)

#: Model inputs, in a fixed order (reproducible importances).
FEATURES = [
    "implied_1d",        # implied 1-session sigma, fraction of spot
    "iv_atm",            # annualized ATM IV backed out of the implied move
    "dte",               # calendar days to the nearest expiry
    "short_gamma",       # 1.0 when net dealer gamma is negative
    "gex_ratio",         # net / gross dealer gamma, in [-1, 1]
    "gex_per_contract",  # net gamma dollars per contract per spot dollar
    "flip_dist",         # (spot - gamma flip) / spot
    "has_flip",          # 1.0 when a flip level exists in the +/-15% grid
    "call_wall_dist",    # (call wall - spot) / spot
    "put_wall_dist",     # (spot - put wall) / spot
    "wall_width",        # (call wall - put wall) / spot
    "max_pain_dist",     # (max pain - spot) / spot
    "put_oi_share",      # put OI / total OI
    "dex_per_contract",  # dealer delta per contract per spot dollar
    "vanna_per_contract",
    "charm_per_contract",
    "realized_lag1",     # previous session's realized |move|, fraction
    "realized_lag2",
    "realized_mean3",    # 3-session mean realized |move|
    "range_pct",         # this session's reconstructed range / close
    "vrp_lag1",          # realized_lag1 / implied_1d of the day that priced it
]


# --- feature engineering -----------------------------------------------------

def _safe(x, default: float = np.nan) -> float:
    """Coerce to a finite float, or ``default``."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return v if np.isfinite(v) else default


def horizon_years(a: Analysis) -> float | None:
    """Years to the nearest expiry the implied move was measured against —
    the same convention ``analytics._years_to_expiry`` uses (calendar days
    over 365, floored at half a trading day)."""
    if a.nearest_expiry is None:
        return None
    days = (a.nearest_expiry - a.asof).days
    return max(days / 365.0, MIN_T)


def implied_session_sigma(a: Analysis) -> float | None:
    """The implied move rescaled to **one trading session**, as a fraction
    of spot.

    ``a.expected_move`` is a 1-sigma move over ``horizon_years`` years, so
    the implied annualized vol is ``em/spot / sqrt(T)`` and one session is
    that over ``sqrt(1/252)``. This is what the model competes against.
    """
    t = horizon_years(a)
    if a.expected_move is None or t is None or a.spot <= 0:
        return None
    iv = (a.expected_move / a.spot) / np.sqrt(t)
    return float(iv / np.sqrt(TRADING_DAYS))


def positioning_features(a: Analysis) -> dict:
    """Scale-free snapshot of dealer positioning as of ``a.asof``'s close.

    Dollar quantities are divided by ``spot x multiplier x total OI`` so a
    feature means the same thing on a $30 stock and a $6,000 index, and so
    a book that simply grew does not look like a regime change.
    """
    spot = float(a.spot)
    oi_total = 0.0
    if not a.by_strike.empty:
        oi_total = float(a.by_strike["call_oi"].sum() + a.by_strike["put_oi"].sum())
    notional = spot * a.multiplier * max(oi_total, 1.0)

    gross = 0.0
    if not a.by_strike.empty:
        gross = float(a.by_strike["net_gex"].abs().sum())

    sigma_1d = implied_session_sigma(a)
    t = horizon_years(a)
    iv_atm = np.nan
    if a.expected_move is not None and t:
        iv_atm = (a.expected_move / spot) / np.sqrt(t)

    call_oi = float(a.by_strike["call_oi"].sum()) if not a.by_strike.empty else 0.0
    put_oi = float(a.by_strike["put_oi"].sum()) if not a.by_strike.empty else 0.0

    return {
        "implied_1d": _safe(sigma_1d),
        "iv_atm": _safe(iv_atm),
        "dte": float((a.nearest_expiry - a.asof).days) if a.nearest_expiry else np.nan,
        "short_gamma": 1.0 if a.regime == "short_gamma" else 0.0,
        "gex_ratio": _safe(a.total_gex / gross, 0.0) if gross > 0 else 0.0,
        "gex_per_contract": _safe(a.total_gex / notional, 0.0),
        "flip_dist": _safe((spot - a.gamma_flip) / spot, 0.0)
                     if a.gamma_flip is not None else 0.0,
        "has_flip": 1.0 if a.gamma_flip is not None else 0.0,
        "call_wall_dist": _safe((a.call_wall - spot) / spot),
        "put_wall_dist": _safe((spot - a.put_wall) / spot),
        "wall_width": _safe((a.call_wall - a.put_wall) / spot),
        "max_pain_dist": _safe((a.max_pain - spot) / spot),
        "put_oi_share": _safe(put_oi / (call_oi + put_oi), 0.5)
                        if (call_oi + put_oi) > 0 else 0.5,
        "dex_per_contract": _safe(a.dex / notional, 0.0),
        "vanna_per_contract": _safe(a.vanna_flow / notional, 0.0),
        "charm_per_contract": _safe(a.charm_flow / notional, 0.0),
    }


def _sessions_between(d0: date, d1: date) -> int:
    """Trading sessions from ``d0`` to ``d1`` (weekends only; holidays are
    not modelled — they make a gap look one session longer, which the gap
    filter and the sqrt-time scaling both absorb)."""
    n = int(np.busday_count(np.datetime64(d0, "D"), np.datetime64(d1, "D")))
    return max(n, 1)


def build_dataset(days: list[dict], max_gap: int = MAX_GAP_SESSIONS) -> pd.DataFrame:
    """Assemble the supervised frame from a list of daily observations.

    ``days``: ``[{"date": date, "analysis": Analysis, "close": float,
    "low": float, "high": float}, ...]`` — ``close``/``low``/``high`` are
    the reconstructed session values (see ``analytics.session_range``);
    ``close`` falls back to the analysis spot and the range features are
    simply missing when a day carries no prints.

    One row per day, sorted by date. Row *t* holds features known at *t*'s
    close and ``y``, the realized absolute move from *t* to the next
    uploaded day, **per session** (divided by sqrt of the session gap so a
    Friday-to-Monday pair and a Wednesday-to-Friday pair are comparable).
    The final row has no target — that is the live row to predict. Pairs
    more than ``max_gap`` sessions apart get no target: too stale to learn
    a one-session move from.
    """
    cols = ["date", "target_date", "sessions", "y", "baseline", "spot"] + FEATURES
    clean = []
    for d in days:
        a = d.get("analysis")
        if a is None or d.get("date") is None:
            continue
        close = _safe(d.get("close"), np.nan)
        if not np.isfinite(close) or close <= 0:
            close = float(a.spot)
        clean.append({
            "date": d["date"], "analysis": a, "close": close,
            "low": _safe(d.get("low")), "high": _safe(d.get("high")),
        })
    clean.sort(key=lambda r: r["date"])
    # collapse duplicate dates, keeping the last upload for that day
    dedup: dict[date, dict] = {r["date"]: r for r in clean}
    clean = [dedup[k] for k in sorted(dedup)]
    if not clean:
        return pd.DataFrame(columns=cols)

    rows = []
    for i, cur in enumerate(clean):
        a = cur["analysis"]
        feat = positioning_features(a)

        # --- lagged realized behaviour (known at this close) ---
        realized_hist = []
        for j in range(max(i - 3, 0), i):
            prev, nxt = clean[j], clean[j + 1]
            gap = _sessions_between(prev["date"], nxt["date"])
            if gap > max_gap or prev["close"] <= 0:
                continue
            move = abs(nxt["close"] - prev["close"]) / prev["close"]
            realized_hist.append((j, move / np.sqrt(gap)))
        lags = [m for _, m in realized_hist]
        feat["realized_lag1"] = lags[-1] if len(lags) >= 1 else np.nan
        feat["realized_lag2"] = lags[-2] if len(lags) >= 2 else np.nan
        feat["realized_mean3"] = float(np.mean(lags[-3:])) if lags else np.nan

        # this session's own range is known at its close — not a lag
        rng = np.nan
        if np.isfinite(cur["low"]) and np.isfinite(cur["high"]) and cur["close"] > 0:
            rng = (cur["high"] - cur["low"]) / cur["close"]
        feat["range_pct"] = rng

        vrp = np.nan
        if realized_hist:
            j, move = realized_hist[-1]
            prior_implied = implied_session_sigma(clean[j]["analysis"])
            if prior_implied and prior_implied > 0:
                vrp = move / prior_implied
        feat["vrp_lag1"] = vrp

        # --- target: the move this day's state has to predict ---
        y = np.nan
        target_date = None
        sessions = np.nan
        if i + 1 < len(clean):
            nxt = clean[i + 1]
            gap = _sessions_between(cur["date"], nxt["date"])
            if gap <= max_gap and cur["close"] > 0:
                y = abs(nxt["close"] - cur["close"]) / cur["close"] / np.sqrt(gap)
                target_date, sessions = nxt["date"], float(gap)

        base = feat["implied_1d"]
        rows.append({
            "date": cur["date"], "target_date": target_date, "sessions": sessions,
            "y": y,
            # the implied baseline predicts a 1-sigma move; the target is an
            # absolute move, so scale by E|X|/sigma before comparing
            "baseline": base * MEAN_ABS_OVER_SIGMA if np.isfinite(base) else np.nan,
            "spot": float(a.spot),
            **feat,
        })
    return pd.DataFrame(rows, columns=cols)


# --- models ------------------------------------------------------------------

def lightgbm_available() -> bool:
    try:
        import lightgbm  # noqa: F401
    except Exception:
        return False
    return True


LGBM_ROUNDS = 300


def _lgbm_params(n_rows: int) -> dict:
    """Hyperparameters for a *tiny*, noisy financial sample: shallow trees,
    heavy shrinkage, and splits allowed on very few rows (with this much
    data the alternative is no splits at all). Fixed seed and single-thread
    deterministic mode so the same history always gives the same forecast."""
    return {
        "objective": "regression_l1",   # absolute moves are heavy-tailed
        "learning_rate": 0.03,
        "num_leaves": 4,
        "max_depth": 3,
        "min_data_in_leaf": max(2, n_rows // 10),
        "min_sum_hessian_in_leaf": 1e-3,
        "min_gain_to_split": 0.0,
        "min_data_in_bin": 1,
        "bagging_fraction": 0.9,
        "bagging_freq": 1,
        "feature_fraction": 0.8,
        "lambda_l2": 1.0,
        "seed": 7,
        "deterministic": True,
        "force_col_wise": True,
        "num_threads": 1,
        "verbosity": -1,
    }


class _Ridge:
    """Standardized ridge regression — the fallback learner when LightGBM
    is not installed. Closed form, no dependencies beyond numpy."""

    name = "ridge"

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_Ridge":
        X = np.nan_to_num(np.asarray(X, dtype=float), nan=0.0)
        self.mu_ = X.mean(axis=0)
        sd = X.std(axis=0)
        self.sd_ = np.where(sd > 1e-12, sd, 1.0)
        Z = (X - self.mu_) / self.sd_
        self.y0_ = float(np.mean(y))
        A = Z.T @ Z + self.alpha * np.eye(Z.shape[1])
        self.w_ = np.linalg.solve(A, Z.T @ (y - self.y0_))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.nan_to_num(np.asarray(X, dtype=float), nan=0.0)
        return ((X - self.mu_) / self.sd_) @ self.w_ + self.y0_

    @property
    def importance_(self) -> np.ndarray:
        return np.abs(self.w_)


class _LightGBM:
    """LightGBM regressor on the log-move target.

    Uses the native training API rather than the scikit-learn wrapper, so
    ``lightgbm`` is the only extra dependency this feature adds.
    """

    name = "lightgbm"

    def fit(self, X: np.ndarray, y: np.ndarray) -> "_LightGBM":
        import lightgbm as lgb

        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        ds = lgb.Dataset(X, label=y, feature_name=list(FEATURES), free_raw_data=False)
        self.booster_ = lgb.train(_lgbm_params(len(y)), ds,
                                  num_boost_round=LGBM_ROUNDS)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.atleast_1d(self.booster_.predict(np.asarray(X, dtype=float)))

    @property
    def importance_(self) -> np.ndarray:
        return np.asarray(self.booster_.feature_importance(importance_type="gain"),
                          dtype=float)


def _make_model(engine: str):
    if engine == "lightgbm":
        return _LightGBM()
    return _Ridge()


def _resolve_engine(engine: str) -> str:
    if engine == "ridge":
        return "ridge"
    return "lightgbm" if lightgbm_available() else "ridge"


def _fit(frame: pd.DataFrame, engine: str):
    """Fit on the labelled rows of ``frame``. The target is log-move: the
    distribution of absolute moves is right-skewed and strictly positive,
    and errors matter proportionally, not in dollars."""
    X = frame[FEATURES].to_numpy(dtype=float)
    y = np.log(frame["y"].to_numpy(dtype=float) + EPS)
    return _make_model(engine).fit(X, y)


def _predict(model, frame: pd.DataFrame) -> np.ndarray:
    pred = model.predict(frame[FEATURES].to_numpy(dtype=float))
    return np.clip(np.exp(pred) - EPS, 0.0, None)


# --- evaluation --------------------------------------------------------------

def walk_forward(frame: pd.DataFrame, engine: str = "auto",
                 min_train: int = MIN_TRAIN_ROWS) -> pd.DataFrame:
    """Expanding-window walk-forward over the labelled rows.

    For each row from ``min_train`` onward: fit on everything strictly
    before it, predict it. Every returned prediction is out of sample, and
    no fit ever sees a future row.

    Columns: date, target_date, actual, predicted, baseline.
    """
    cols = ["date", "target_date", "actual", "predicted", "baseline"]
    lab = frame[frame["y"].notna() & frame["baseline"].notna()].reset_index(drop=True)
    if len(lab) <= min_train:
        return pd.DataFrame(columns=cols)

    engine = _resolve_engine(engine)
    rows = []
    for i in range(min_train, len(lab)):
        train, test = lab.iloc[:i], lab.iloc[[i]]
        try:
            model = _fit(train, engine)
            pred = float(_predict(model, test)[0])
        except Exception:
            continue
        rows.append({
            "date": test["date"].iloc[0], "target_date": test["target_date"].iloc[0],
            "actual": float(test["y"].iloc[0]), "predicted": pred,
            "baseline": float(test["baseline"].iloc[0]),
        })
    return pd.DataFrame(rows, columns=cols)


def skill_score(bt: pd.DataFrame) -> dict:
    """Out-of-sample error of the model against the implied baseline.

    ``skill = 1 - MAE_model / MAE_baseline``: positive means the model
    beat the price of the move, 0 means it matched it, negative means the
    implied move was better and the model should be ignored.
    """
    if bt is None or bt.empty:
        return {"n": 0, "mae_model": None, "mae_baseline": None, "skill": None}
    err_m = float(np.mean(np.abs(bt["predicted"] - bt["actual"])))
    err_b = float(np.mean(np.abs(bt["baseline"] - bt["actual"])))
    skill = None if err_b <= 0 else float(1.0 - err_m / err_b)
    return {"n": int(len(bt)), "mae_model": err_m, "mae_baseline": err_b,
            "skill": skill}


# --- the public result -------------------------------------------------------

@dataclass
class ExpectedMoveForecast:
    """What to actually show. ``blended_pct`` is the number to use; every
    other field exists so a reader can see how much of it is model."""

    status: str                       # ok | no_skill | insufficient_history | no_data
    message: str
    spot: float = float("nan")
    asof: date | None = None
    baseline_pct: float | None = None  # implied E|move|, fraction of spot
    model_pct: float | None = None     # raw model prediction
    blended_pct: float | None = None   # what to use
    weight: float = 0.0                # model's share of the blend
    skill: float | None = None
    mae_model: float | None = None
    mae_baseline: float | None = None
    n_train: int = 0
    n_oos: int = 0
    n_days: int = 0
    engine: str = "none"
    horizon_days: float = 1.0
    backtest: pd.DataFrame = field(default_factory=pd.DataFrame)
    importance: pd.DataFrame = field(default_factory=pd.DataFrame)

    @property
    def used_model(self) -> bool:
        return self.status == "ok" and self.weight > 0

    def _dollars(self, pct: float | None) -> float | None:
        """Absolute-move fraction -> a 1-sigma dollar move, so it lines up
        with the implied +/-1 sigma band drawn everywhere else."""
        if pct is None or not np.isfinite(self.spot):
            return None
        return float(pct / MEAN_ABS_OVER_SIGMA * self.spot)

    @property
    def implied_sigma(self) -> float | None:
        return self._dollars(self.baseline_pct)

    @property
    def predicted_sigma(self) -> float | None:
        return self._dollars(self.blended_pct)

    @property
    def richness(self) -> float | None:
        """Predicted / implied. Above 1 the market is under-pricing the
        next session relative to the model; below 1, over-pricing."""
        if not self.baseline_pct or self.blended_pct is None:
            return None
        return float(self.blended_pct / self.baseline_pct)


def _importance_frame(model, engine: str) -> pd.DataFrame:
    try:
        imp = np.asarray(model.importance_, dtype=float)
    except Exception:
        return pd.DataFrame(columns=["feature", "importance", "share"])
    total = float(np.sum(np.abs(imp)))
    share = np.abs(imp) / total if total > 0 else np.zeros_like(imp)
    df = pd.DataFrame({"feature": FEATURES, "importance": imp, "share": share})
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


def forecast_expected_move(days: list[dict], engine: str = "auto",
                           min_train: int = MIN_TRAIN_ROWS,
                           min_oos: int = MIN_OOS_ROWS) -> ExpectedMoveForecast:
    """Full pipeline: build the dataset, walk it forward, and blend.

    The returned ``blended_pct`` is ``baseline^(1-w) * model^w`` with
    ``w = clip(skill, 0, 1)`` — a geometric blend, because both numbers are
    positive scales. A model that cannot beat implied out of sample gets
    ``w = 0`` and the caller sees the implied move, unchanged.
    """
    frame = build_dataset(days)
    if frame.empty:
        return ExpectedMoveForecast(
            status="no_data",
            message="No usable day: each day needs an analysis with a dated "
                    "nearest expiry.",
        )

    live = frame.iloc[-1]
    spot = float(live["spot"])
    base = float(live["baseline"]) if np.isfinite(live["baseline"]) else None
    labelled = int(frame["y"].notna().sum())
    common = dict(spot=spot, asof=live["date"], baseline_pct=base,
                  blended_pct=base, n_days=int(len(frame)), n_train=labelled,
                  horizon_days=1.0)

    if base is None:
        return ExpectedMoveForecast(
            status="no_data",
            message="The latest day has no implied expected move (no dated "
                    "expiry or no near-the-money IV) — nothing to forecast "
                    "against.",
            **common,
        )

    need = min_train + min_oos + 1 - labelled
    if need > 0:
        return ExpectedMoveForecast(
            status="insufficient_history",
            message=(
                f"{labelled} labelled day-pair(s) — the model needs "
                f"{min_train + min_oos + 1} ({min_train} to fit, {min_oos} to "
                f"score out of sample). Upload ~{need} more daily export(s); "
                "until then this is the implied move only."
            ),
            engine=_resolve_engine(engine), **common,
        )

    resolved = _resolve_engine(engine)
    bt = walk_forward(frame, engine=resolved, min_train=min_train)
    sc = skill_score(bt)
    if sc["n"] < min_oos:
        return ExpectedMoveForecast(
            status="insufficient_history",
            message=f"Only {sc['n']} out-of-sample prediction(s) could be "
                    f"scored — need {min_oos}. Showing the implied move.",
            engine=resolved, backtest=bt, n_oos=sc["n"], **common,
        )

    lab = frame[frame["y"].notna() & frame["baseline"].notna()]
    try:
        model = _fit(lab, resolved)
        model_pct = float(_predict(model, frame.iloc[[-1]])[0])
    except Exception as exc:  # a fit failure is a reason to stand down
        return ExpectedMoveForecast(
            status="no_skill", engine=resolved, backtest=bt, n_oos=sc["n"],
            skill=sc["skill"], mae_model=sc["mae_model"],
            mae_baseline=sc["mae_baseline"],
            message=f"Model fit failed ({type(exc).__name__}) — showing the "
                    "implied move.",
            **common,
        )

    skill = sc["skill"]
    scored = dict(common, model_pct=model_pct, engine=resolved, backtest=bt,
                  importance=_importance_frame(model, resolved),
                  n_oos=sc["n"], skill=skill, mae_model=sc["mae_model"],
                  mae_baseline=sc["mae_baseline"])

    if skill is None or skill <= 0 or not np.isfinite(model_pct) or model_pct <= 0:
        pct = f"{-skill:.0%} worse" if skill is not None else "unmeasurable"
        return ExpectedMoveForecast(
            status="no_skill", weight=0.0,
            message=(
                f"The model did not beat the implied move out of sample "
                f"({pct} on {sc['n']} prediction(s)) — it gets zero weight and "
                "the number above is the implied move. On this ticker, "
                "recently, the option market priced the move better than the "
                "positioning features did."
            ),
            **scored,
        )

    w = float(np.clip(skill, 0.0, 1.0))
    blended = float(np.exp((1 - w) * np.log(base) + w * np.log(model_pct)))
    scored["blended_pct"] = blended
    return ExpectedMoveForecast(
        status="ok", weight=w,
        message=(
            f"Model beat the implied move by {skill:.0%} (MAE) across "
            f"{sc['n']} out-of-sample session(s); blended at {w:.0%} model / "
            f"{1 - w:.0%} implied."
        ),
        **scored,
    )


def forecast_summary(f: ExpectedMoveForecast) -> str:
    """One-line read for the TL;DR and the report."""
    if f.blended_pct is None or not np.isfinite(f.spot):
        return "Expected move: not available."
    lo = f.spot - (f.predicted_sigma or 0.0)
    hi = f.spot + (f.predicted_sigma or 0.0)
    if not f.used_model:
        return (f"Expected move (implied) ±{f.predicted_sigma:,.2f} "
                f"({lo:,.2f} – {hi:,.2f}); no model edge over the option "
                "market yet.")
    rich = f.richness or 1.0
    lean = ("wider than the market is pricing" if rich > 1.05 else
            "tighter than the market is pricing" if rich < 0.95 else
            "in line with the market")
    return (f"Model expects ±{f.predicted_sigma:,.2f} next session "
            f"({lo:,.2f} – {hi:,.2f}) — {lean} "
            f"({rich:.0%} of implied, {f.engine}, skill {f.skill:+.0%}).")
