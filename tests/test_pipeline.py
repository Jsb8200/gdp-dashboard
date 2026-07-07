import numpy as np
import pandas as pd
import pytest

from magnitude_detector import (
    Config,
    MagnitudeDetector,
    build_features,
    build_labels,
    make_synthetic_ohlcv,
)
from magnitude_detector.model import walk_forward_splits
from magnitude_detector.train import prepare_dataset


@pytest.fixture(scope="module")
def df():
    return make_synthetic_ohlcv(n_bars=6000, seed=3)


@pytest.fixture(scope="module")
def cfg():
    return Config(horizon=12, magnitude_threshold=3.0)


def test_labels_use_only_future_bars(cfg):
    """Changing past bars must not change the label at t."""
    df = make_synthetic_ohlcv(n_bars=800, seed=1)
    y1 = build_labels(df, cfg)

    df2 = df.copy()
    # ATR at t depends on the recent past, so perturb well before t.
    df2.iloc[:200, df2.columns.get_loc("high")] *= 1.5
    y2 = build_labels(df2, cfg)

    t = 700  # far from the perturbed region
    assert y1["big_move"].iloc[t] == y2["big_move"].iloc[t]


def test_labels_tail_is_nan(df, cfg):
    y = build_labels(df, cfg)
    assert y["fwd_mag"].iloc[-cfg.horizon:].isna().all()
    assert y["fwd_mag"].iloc[: -cfg.horizon].notna().sum() > 0


def test_label_magnitude_manual_check(cfg):
    df = make_synthetic_ohlcv(n_bars=500, seed=5)
    y = build_labels(df, cfg)
    t = 300
    from magnitude_detector.features import atr

    fwd = df.iloc[t + 1 : t + 1 + cfg.horizon]
    expected = max(
        fwd["high"].max() - df["close"].iloc[t],
        df["close"].iloc[t] - fwd["low"].min(),
    ) / atr(df, cfg.atr_period).iloc[t]
    assert np.isclose(y["fwd_mag"].iloc[t], expected)


def test_features_are_backward_looking(cfg):
    """Changing future bars must not change features at t."""
    df = make_synthetic_ohlcv(n_bars=1000, seed=2)
    f1 = build_features(df, cfg)

    df2 = df.copy()
    df2.iloc[801:] *= 1.3  # perturb strictly after t=800
    f2 = build_features(df2, cfg)

    pd.testing.assert_series_equal(f1.iloc[800], f2.iloc[800])


def test_walk_forward_splits_have_embargo():
    splits = list(walk_forward_splits(1000, n_splits=4, embargo=12))
    assert len(splits) == 4
    for tr, te in splits:
        assert tr.max() < te.min()
        assert te.min() - tr.max() > 12  # embargo gap
    # Test windows are consecutive and expanding-train.
    assert splits[-1][1][-1] == 999


def test_end_to_end_train_predict(tmp_path, df, cfg):
    X, y = prepare_dataset(df, cfg)
    assert not X.isna().any().any()
    assert 0 < y["big_move"].mean() < 1

    n = len(X)
    det = MagnitudeDetector(cfg).fit(
        X.iloc[: int(n * 0.7)],
        y["big_move"].iloc[: int(n * 0.7)],
        y["fwd_mag"].iloc[: int(n * 0.7)],
        X.iloc[int(n * 0.7) : int(n * 0.85)],
        y["big_move"].iloc[int(n * 0.7) : int(n * 0.85)],
        y["fwd_mag"].iloc[int(n * 0.7) : int(n * 0.85)],
    )
    pred = det.predict(X.iloc[int(n * 0.85) :])
    assert pred["p_big_move"].between(0, 1).all()
    assert pred["expected_mag_atr"].notna().all()

    # Round-trip persistence.
    det.save(str(tmp_path / "m"))
    det2 = MagnitudeDetector.load(str(tmp_path / "m"))
    pred2 = det2.predict(X.iloc[int(n * 0.85) :])
    np.testing.assert_allclose(pred["p_big_move"], pred2["p_big_move"])
