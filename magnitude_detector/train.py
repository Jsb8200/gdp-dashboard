"""Train the magnitude detector with walk-forward evaluation.

Usage:
    python -m magnitude_detector.train --synthetic
    python -m magnitude_detector.train --csv data/eurusd_1h.csv \
        --horizon 12 --threshold 3.0 --out models/eurusd
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .config import Config
from .data import load_ohlcv, make_synthetic_ohlcv
from .features import build_features
from .labels import build_labels
from .model import MagnitudeDetector, walk_forward_splits


def prepare_dataset(df: pd.DataFrame, cfg: Config):
    X = build_features(df, cfg)
    y = build_labels(df, cfg)
    mask = X.notna().all(axis=1) & y["big_move"].notna()
    return X[mask], y[mask]


def precision_at_top(y_true: np.ndarray, scores: np.ndarray, frac: float = 0.1):
    """Precision among the top `frac` highest-scored bars."""
    k = max(int(len(scores) * frac), 1)
    top = np.argsort(scores)[-k:]
    return float(np.mean(y_true[top]))


def evaluate_walk_forward(X, y, cfg: Config) -> pd.DataFrame:
    rows = []
    for i, (tr, te) in enumerate(
        walk_forward_splits(len(X), cfg.n_splits, cfg.effective_embargo), start=1
    ):
        # Tail of the (embargoed) train window is the early-stopping set.
        n_val = max(int(len(tr) * cfg.val_fraction), 1)
        tr_fit, tr_val = tr[:-n_val], tr[-n_val:]

        det = MagnitudeDetector(cfg).fit(
            X.iloc[tr_fit], y["big_move"].iloc[tr_fit], y["fwd_mag"].iloc[tr_fit],
            X.iloc[tr_val], y["big_move"].iloc[tr_val], y["fwd_mag"].iloc[tr_val],
        )
        pred = det.predict(X.iloc[te])
        y_te = y["big_move"].iloc[te].to_numpy()
        p = pred["p_big_move"].to_numpy()

        row = {
            "fold": i,
            "test_bars": len(te),
            "base_rate": float(y_te.mean()),
            "mag_mae": float(
                (pred["expected_mag_atr"] - y["fwd_mag"].iloc[te]).abs().mean()
            ),
        }
        if 0 < y_te.mean() < 1:
            row["auc"] = roc_auc_score(y_te, p)
            row["avg_precision"] = average_precision_score(y_te, p)
            row["precision_top10pct"] = precision_at_top(y_te, p, 0.10)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="Train the LightGBM magnitude detector")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", help="OHLCV csv path")
    src.add_argument("--synthetic", action="store_true",
                     help="use synthetic demo data")
    ap.add_argument("--horizon", type=int, default=12,
                    help="look-ahead bars for the move to develop")
    ap.add_argument("--threshold", type=float, default=3.0,
                    help="big-move threshold in ATRs")
    ap.add_argument("--out", default="models/latest", help="model output dir")
    ap.add_argument("--n-bars", type=int, default=20000,
                    help="bars of synthetic data")
    args = ap.parse_args()

    cfg = Config(horizon=args.horizon, magnitude_threshold=args.threshold)
    df = make_synthetic_ohlcv(args.n_bars) if args.synthetic else load_ohlcv(args.csv)
    print(f"Loaded {len(df)} bars ({df.index[0]} .. {df.index[-1]})")

    X, y = prepare_dataset(df, cfg)
    print(f"Usable rows: {len(X)}  |  positive rate: {y['big_move'].mean():.3f}")

    print("\nWalk-forward evaluation")
    report = evaluate_walk_forward(X, y, cfg)
    print(report.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    metric_cols = [c for c in ("auc", "avg_precision", "precision_top10pct",
                               "base_rate", "mag_mae") if c in report]
    print("\nMeans:")
    print(report[metric_cols].mean().to_string(float_format=lambda v: f"{v:.4f}"))

    # Final model on all data (tail as early-stopping validation).
    n_val = max(int(len(X) * cfg.val_fraction), 1)
    fit_end = len(X) - n_val - cfg.effective_embargo
    det = MagnitudeDetector(cfg).fit(
        X.iloc[:fit_end], y["big_move"].iloc[:fit_end], y["fwd_mag"].iloc[:fit_end],
        X.iloc[-n_val:], y["big_move"].iloc[-n_val:], y["fwd_mag"].iloc[-n_val:],
    )
    det.save(args.out)
    print(f"\nSaved model to {args.out}/")

    print("\nTop features (classifier gain):")
    print(det.feature_importance().head(12).to_string(index=False))


if __name__ == "__main__":
    main()
