"""Score bars with a trained magnitude detector.

Usage:
    # score the most recent closed bar
    python -m magnitude_detector.predict --model models/latest --csv data/eurusd_1h.csv

    # score every bar and write a csv
    python -m magnitude_detector.predict --model models/latest --csv data/eurusd_1h.csv \
        --all --out signals.csv
"""

import argparse

from .data import load_ohlcv, make_synthetic_ohlcv
from .features import build_features
from .model import MagnitudeDetector


def main():
    ap = argparse.ArgumentParser(description="Score bars with the magnitude detector")
    ap.add_argument("--model", required=True, help="model directory from train.py")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--csv", help="OHLCV csv path")
    src.add_argument("--synthetic", action="store_true")
    ap.add_argument("--all", action="store_true", help="score every bar, not just the last")
    ap.add_argument("--out", help="write scored rows to this csv")
    ap.add_argument("--alert-threshold", type=float, default=0.6,
                    help="p_big_move level that flags an alert")
    args = ap.parse_args()

    det = MagnitudeDetector.load(args.model)
    df = make_synthetic_ohlcv() if args.synthetic else load_ohlcv(args.csv)

    X = build_features(df, det.cfg)
    X = X[X.notna().all(axis=1)]
    pred = det.predict(X)
    pred["alert"] = pred["p_big_move"] >= args.alert_threshold

    if args.all:
        if args.out:
            pred.to_csv(args.out)
            print(f"Wrote {len(pred)} scored bars to {args.out}")
        else:
            print(pred.tail(20).to_string())
        print(f"\nAlerts: {int(pred['alert'].sum())} / {len(pred)} bars")
    else:
        last = pred.iloc[-1]
        print(f"Bar:                {pred.index[-1]}")
        print(f"P(big move):        {last['p_big_move']:.3f}")
        print(f"Expected magnitude: {last['expected_mag_atr']:.2f} ATR "
              f"(threshold {det.cfg.magnitude_threshold} ATR "
              f"over next {det.cfg.horizon} bars)")
        print(f"Alert:              {'YES' if last['alert'] else 'no'}")


if __name__ == "__main__":
    main()
