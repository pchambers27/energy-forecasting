"""Time-series cross-validation splitter.

Expanding-window walk-forward: each fold uses all available history up to a cutoff for training, then tests on a fixed-length window after a gap.

The gap prevents leakage: if forecasting H hours ahead, the test set's first hour must be H hours after the last train hour, because in production you wouldn't have access to the hours in between wehn making the forecast."""


from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterator
import pandas as pd

@dataclass(frozen=True)
class Split:
  """One train/test split for a single region."""
  fold: int
  region: str
  train_start: datetime
  train_end: datetime
  test_start: datetime
  test_end: datetime

  def __repr__(self) -> str:
    return(
      f"Split(fold={self.fold} region={self.region}"
      f"train=[{self.train_start.date()} -> {self.train_end.date()}]"
      f"test=[{self.test_start.date()} -> {self.test_end.date()}])"
    )

def make_splits(
  df: pd.DataFrame,
  *,
  region_col: str = "region",
  time_col: str = "period_utc",
  horizon_hours: int = 24,
  test_window_hours: int = 24 * 7,
  n_folds: int = 12,
  min_train_hours: int = 24 * 365,
) -> Iterator[Split]:
   """Yield expanding-window splits, one per fold per region. 
   Args:
     df: long-format dataframe with at least region_col and time_col
     horizon_hours: forecast horizon (gap between train and test start)
     test_window_hours: length of test window
     n_folds: number of folds to generate
     min_train_hours: minimum length of training window
    The most recent test window is fold 0; older folds have higher fold numbers.
    This makes it easy to "use fold 0 for final eval, 1..N for tuning."
    """
  for region, group in df.groupby(region_col):
    times = group[time_col].sort_values().reset_index(drop=True)
    if len(times) < min_train_hours + horizon_hours + test_window_hours:
      continue
    last_ts = times.iloc[-1]
    for fold in range(n_folds):
      test_end = last_ts - timedelta(hours=fold * test_window_hours)
      test_start = test_end - timedelta(hours=test_window_hours -1)
      train_end = test_start - timedelta(hours=horizon_hours)
      train_start = times.iloc[0]
      train_hours = (train_end - train_start).total_seconds() / 3600
      if train_hours < min_train_hours:
        break
      yield Split(
        fold=fold,
        region=region,
        train_start=train_start,
        train_end=train_end,
        test_start=test_start,
        test_end=test_end,
          )