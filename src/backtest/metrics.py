"""Regression metrics for backtst evaluation.

All metrics handle NaN predictions and NaN actuals by dropping those rows *pairwise* before computing. Predicitions or actuals with mismatched lengths will raise."""

from __future__ import annotations


from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass(frozen=True)
class Metrics:
  """Container for evaluation metrics on a single forecast/actual pair."""
  n: int
  n_dropped: int
  mae: float
  rmse: float
  mape: float
  bias: float

  def as_dict(self) -> dict:
    return {
      "n": self.n,
      "n_dropped": self.n_dropped,
      "mae": self.mae,
      "rmse": self.rmse,
      "mape": self.mape,
      "bias": self.bias
    }

def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> Metrics:
  """Compute regression metrics, dropping pairs with NaN in either input.
  Args:
    y_true: actual values, indexed by timestamp
    y_pred: predicted values, same index"""
  if len(y_true) != len(y_pred):
    raise ValueError(f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}")

  df  = pd.DataFrame({"y_true": y_true.values, "y_pred": y_pred.values})
  n_total = len(df)
  df = df.dropna()
  n_valid = len(df)
  n_dropped = n_total - n_valid

  if n_valid == 0:
    return Metrics(n=0, n_dropped=n_dropped, mae=np.nan, rmse=np.nan, mape=np.nan, bias=np.nan)
  err = df["y_pred"] - df["y_true"]
  abs_err = err.abs()
  mae = abs_err.mean()
  rmse = np.sqrt((err ** 2).mean())
  bias = err.mean()
  nonzero = df[df["y_true"] != 0]
  if len(nonzero) > 0:
    mape = ((nonzero["y_pred"] - nonzero["y_true"]).abs() / nonzero["y_true"].abs()).mean() * 100
  else:
    mape = np.nan
  return Metrics(
    n=n_valid,
    n_dropped=n_dropped,
    mae=float(mae),
    rmse=float(rmse),
    mape=float(mape),
    bias=float(bias),
  )


def skill_score(model_mae: float, baseline_mae: float) -> float:
  """1 - (model_MAE / baseline_MAE).
  Returns:
    > 0: model beats baseline (1.0 = perfect, 0.2 = 20% improvement)
    = 0: tied with baseline
    < 0: model worse than baseline
  """

  if baseline_mae == 0 or np.isnan(baseline_mae):
    return np.nan
  return 1.0 - (model_mae / baseline_mae)