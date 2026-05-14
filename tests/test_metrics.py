"""Sanity tests for metrics computation."""

import numpy as np
import pandas as pd

from src.backtest.metrics import compute_metrics, skill_score

def test_perfect_prediction():
  idx = pd.date_range("2024-01-01", periods=24, freq="h")
  y_true = pd.Series([100.0] * 24, index=idx)
  y_pred = pd.Series([100.0] * 24, index=idx)

  m = compute_metrics(y_true, y_pred)
  assert m.mae == 0
  assert m.rmse == 0
  assert m.mape == 0
  assert m.bias == 0
  assert m.n == 24
  assert m.n_dropped == 0


def test_constant_offset():
  """Predicting 10 too high on every hour: MAE=10, bias=+10, MAPE=10%."""
  idx = pd.date_range("2024-01-01", periods=24, freq="h")
  y_true = pd.Series([100.0] * 24, index=idx)
  y_pred = pd.Series([110.0] * 24, index=idx)

  m = compute_metrics(y_true, y_pred)
  assert m.mae == 10.0
  assert m.rmse == 10.0
  assert m.bias == 10.0
  assert abs(m.mape - 10.0) < 1e-9


def test_nan_predictions_dropped():
  idx = pd.date_range("2024-01-01", periods=10, freq="h")
  y_true = pd.Series([100.0] * 10, index=idx)
  y_pred = pd.Series([100.0, np.nan, 100.0, np.nan, 100.0, 100.0, 100.0, 100.0, 100.0, 100.0], index=idx)
  m = compute_metrics(y_true, y_pred)
  assert m.n == 8
  assert m.n_dropped == 2
  assert m.mae == 0


def test_skill_score():
  assert abs(skill_score(80, 100) -0.2) < 1e-9
  assert abs(skill_score(120, 100) - (-0.2)) < 1e-9
  assert skill_score(100, 100) == 0.0


def test_rmse_penalizes_large_errors():
  """One big miss should make RMSE > MAE."""
  idx = pd.date_range("2024-01-01", periods=4, freq="h")
  y_true = pd.Series([100.0, 100.0, 100.0, 100.0], index=idx)
  y_pred = pd.Series([100.0, 100.0, 100.0, 200.0], index=idx)
  m = compute_metrics(y_true, y_pred)
  assert m.mae == 25.0
  assert m.rmse == 50.0
  assert m.rmse > m.mae

if __name__ == "__main__":
  test_perfect_prediction()
  test_constant_offset()
  test_nan_predictions_dropped()
  test_skill_score()
  test_rmse_penalizes_large_errors()
  print("✓ All metrics tests passed")