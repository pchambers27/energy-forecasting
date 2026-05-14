"""Seasonal naive baseline for hourly demand. 

Both implement the Forecaster protocol: .fit(train_df) then .predict(index)."""

from __future__ import annotations

from typing import Protocol
import pandas as pd


class Forecaster(Protocol):
  """Minimal interface every forecaster must implement."""

  def fit(self, train_df: pd.DataFrame) -> None: ...
  def predict(self, index: pd.DatetimeIndex) -> pd.Series: ...


class SeasonalNaive:
  """Predicts y(t) = y(t - period_hours).
  
  For hourly data: 
    - period_hours=168 -> same hour, one week ago
    - period_hours=8760 -> same hour, one year ago
    
  Assumes train_df has 'period_utc' (datetime, sorted) and 'demand_mwh' columns.
  """

  def __init__(self, period_hours: int, name: str | None = None):
    self.period_hours = period_hours
    self.name = name or f"seasonal_naive_{period_hours}h"
    self._train: pd.Series | None = None

  def fit(self, train_df: pd.DataFrame) -> None:
    s = train_df.set_index("period_utc")["demand_mwh"].sort_index()
    s = s[~s.index.duplicated(keep="last")]
    self._train = s

  def predict(self, index: pd.DatetimeIndex) -> pd.Series:
    if self._train is None:
      raise RuntimeError(f"{self.name}: must call fit() before predict()")
    period = pd.Timedelta(hours=self.period_hours)

    lookup_times = index - period
    preds = self._train.reindex(lookup_times).values
    return pd.Series(preds, index=index, name=self.name)