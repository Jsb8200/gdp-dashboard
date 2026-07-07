"""Configuration for the magnitude detector."""

from dataclasses import dataclass, field


@dataclass
class Config:
    # --- Labeling -----------------------------------------------------------
    # Look-ahead horizon: how many bars after t the move must develop in.
    horizon: int = 12
    # ATR period used to normalize magnitudes.
    atr_period: int = 14
    # A bar is a positive ("big move imminent") if the max excursion over the
    # next `horizon` bars is >= `magnitude_threshold` ATRs.
    magnitude_threshold: float = 3.0

    # --- Features -----------------------------------------------------------
    ema_periods: tuple = (8, 21, 50, 200)
    bb_period: int = 20
    donchian_period: int = 20
    vol_lookback: int = 20
    squeeze_lookback: int = 120  # window for bandwidth percentile

    # --- Training -----------------------------------------------------------
    n_splits: int = 4          # walk-forward folds
    val_fraction: float = 0.15  # tail of each train fold used for early stop
    # Bars dropped between train and test to avoid label overlap leakage.
    # Defaults to `horizon` when 0.
    embargo: int = 0

    lgbm_params: dict = field(default_factory=lambda: {
        "objective": "binary",
        "learning_rate": 0.03,
        "num_leaves": 31,
        "min_child_samples": 40,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "verbosity": -1,
    })
    num_boost_round: int = 2000
    early_stopping_rounds: int = 100

    @property
    def effective_embargo(self) -> int:
        return self.embargo if self.embargo > 0 else self.horizon
