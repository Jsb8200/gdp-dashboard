"""LightGBM model wrapper: classifier (move imminent) + regressor (size)."""

import json
import os

import lightgbm as lgb
import numpy as np
import pandas as pd

from .config import Config


class MagnitudeDetector:
    """Two-head detector.

    * classifier: P(max excursion over next H bars >= k * ATR)
    * regressor:  expected forward magnitude in ATR units
    """

    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or Config()
        self.clf: lgb.Booster | None = None
        self.reg: lgb.Booster | None = None
        self.feature_names: list[str] | None = None

    # ------------------------------------------------------------------ train
    def fit(
        self,
        X: pd.DataFrame,
        y_class: pd.Series,
        y_mag: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_class_val: pd.Series | None = None,
        y_mag_val: pd.Series | None = None,
    ) -> "MagnitudeDetector":
        self.feature_names = list(X.columns)
        cfg = self.cfg

        clf_params = dict(cfg.lgbm_params)
        clf_params["objective"] = "binary"
        clf_params["metric"] = "auc"
        pos_rate = float(y_class.mean())
        if 0 < pos_rate < 1:
            clf_params.setdefault("scale_pos_weight", (1 - pos_rate) / pos_rate)

        reg_params = dict(cfg.lgbm_params)
        reg_params["objective"] = "regression_l1"  # robust to fat tails
        reg_params["metric"] = "l1"
        reg_params.pop("scale_pos_weight", None)

        def _train(params, y_train, y_val):
            dtrain = lgb.Dataset(X, label=y_train)
            valid_sets, callbacks = [], []
            if X_val is not None and len(X_val) > 0:
                valid_sets = [lgb.Dataset(X_val, label=y_val, reference=dtrain)]
                callbacks = [lgb.early_stopping(cfg.early_stopping_rounds,
                                                verbose=False)]
            return lgb.train(
                params,
                dtrain,
                num_boost_round=cfg.num_boost_round,
                valid_sets=valid_sets,
                callbacks=callbacks,
            )

        self.clf = _train(clf_params, y_class, y_class_val)
        self.reg = _train(reg_params, y_mag, y_mag_val)
        return self

    # ---------------------------------------------------------------- predict
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.clf is None or self.reg is None:
            raise RuntimeError("Model is not trained/loaded")
        X = X[self.feature_names]
        out = pd.DataFrame(index=X.index)
        out["p_big_move"] = self.clf.predict(
            X, num_iteration=self.clf.best_iteration or None
        )
        out["expected_mag_atr"] = self.reg.predict(
            X, num_iteration=self.reg.best_iteration or None
        )
        return out

    def feature_importance(self) -> pd.DataFrame:
        imp = pd.DataFrame(
            {
                "feature": self.feature_names,
                "gain_clf": self.clf.feature_importance("gain"),
                "gain_reg": self.reg.feature_importance("gain"),
            }
        )
        return imp.sort_values("gain_clf", ascending=False).reset_index(drop=True)

    # ----------------------------------------------------------------- persist
    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        self.clf.save_model(os.path.join(directory, "classifier.txt"))
        self.reg.save_model(os.path.join(directory, "regressor.txt"))
        meta = {
            "feature_names": self.feature_names,
            "config": {
                "horizon": self.cfg.horizon,
                "atr_period": self.cfg.atr_period,
                "magnitude_threshold": self.cfg.magnitude_threshold,
            },
        }
        with open(os.path.join(directory, "meta.json"), "w") as fh:
            json.dump(meta, fh, indent=2)

    @classmethod
    def load(cls, directory: str) -> "MagnitudeDetector":
        with open(os.path.join(directory, "meta.json")) as fh:
            meta = json.load(fh)
        cfg = Config(**meta["config"])
        obj = cls(cfg)
        obj.clf = lgb.Booster(model_file=os.path.join(directory, "classifier.txt"))
        obj.reg = lgb.Booster(model_file=os.path.join(directory, "regressor.txt"))
        obj.feature_names = meta["feature_names"]
        return obj


# --------------------------------------------------------------------- splits
def walk_forward_splits(n: int, n_splits: int, embargo: int):
    """Expanding-window walk-forward splits with an embargo gap.

    Yields (train_idx, test_idx) index arrays. The embargo removes the last
    `embargo` bars of each training window so training labels never overlap
    the test window's future horizon.
    """
    fold = n // (n_splits + 1)
    for k in range(1, n_splits + 1):
        train_end = fold * k
        test_end = min(fold * (k + 1), n)
        train_idx = np.arange(0, max(train_end - embargo, 0))
        test_idx = np.arange(train_end, test_end)
        if len(train_idx) and len(test_idx):
            yield train_idx, test_idx
