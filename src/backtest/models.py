"""ML forecasters that plug into the existing backtest runner.

Each implements the Forecaster protocol: .fit(train_df) then .predict(index).
The runner doesn't know or care whether it's a naive baseline or a model.

LEAKAGE NOTE: scalers/models are fit on TRAIN data only. Weather features are
actual observed values (Option A — "perfect weather information" study).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.backtest.features import build_features, split_X_y, FEATURE_COLS, TARGET_COL

try:
    from lightgbm import LGBMRegressor
    _HAS_LGBM = True
except (ImportError, OSError):
    _HAS_LGBM = False

class LGBMForecaster:
    def fit(self, train_df):
        if not _HAS_LGBM:
            raise RuntimeError("LGBMForecaster requires lightgbm to be installed")


class RidgeForecaster:
    """StandardScaler → Ridge regression on calendar + weather features.

    Implements the Forecaster protocol. Re-fit fresh on each backtest split.
    """

    def __init__(self, alpha: float = 1.0, name: str | None = None, horizon_hours: int = 24):
        self.alpha = alpha
        self.name = name or f"ridge_a{alpha}"
        self.horizon_hours = horizon_hours
        self._pipeline: Pipeline | None = None
        self._feature_lookup: pd.DataFrame | None = None

    def fit(self, train_df: pd.DataFrame) -> None:
        feats = build_features(train_df, horizon_hours=self.horizon_hours)
        X, y = split_X_y(feats)
        if len(X) == 0:
            raise RuntimeError(f"{self.name}: no training rows after dropping NaN")

        self._pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("ridge", Ridge(alpha=self.alpha)),
        ])
        self._pipeline.fit(X, y)

    def predict(self, index: pd.DatetimeIndex) -> pd.Series:
        if self._pipeline is None or self._feature_lookup is None:
            raise RuntimeError(f"{self.name}: must call set_context() + fit() before predict()")

        # We need the feature row for each timestamp in `index`. The runner gives
        # us only timestamps, so we look features up from the context frame.
        wanted = self._feature_lookup.reindex(index)
        # Rows with missing features → NaN prediction (metrics layer drops them)
        valid_mask = wanted[FEATURE_COLS].notna().all(axis=1)

        preds = pd.Series(np.nan, index=index, name=self.name)
        if valid_mask.any():
            X_valid = wanted.loc[valid_mask, FEATURE_COLS]
            preds.loc[valid_mask] = self._pipeline.predict(X_valid)
        return preds

    def set_context(self, region_df: pd.DataFrame) -> None:
        """Provide the full region dataframe so predict() can look up features
        for test timestamps. Called by the runner before predict().

        This is NOT leakage: we only use the FEATURE columns (calendar + actual
        weather under Option A) for test timestamps, never the target.
        """
        feats = build_features(region_df, horizon_hours=self.horizon_hours)
        self._feature_lookup = feats.set_index("period_utc")