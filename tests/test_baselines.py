"""Sanity tests for seasonal naive baselines."""

import duckdb
import pandas as pd
from datetime import timedelta

from src.backtest.baselines import SeasonalNaive

def load_region(region: str) -> pd.DataFrame:
  con = duckdb.connect("/home/runner/workspace/data/warehouse_dev.duckdb")
  df = con.execute(f"""
  SELECT period_utc, demand_mwh
  FROM main_marts.fct_hourly_demand
  WHERE region = '{region}'
  ORDER BY period_utc""").df()
  con.close()
  return df

def test_weekly_naive_reproduces_lagged_value():
  """Prediciting hour T should return demand at T - 168h."""
  df = load_region("ERCO")
  forecaster = SeasonalNaive(period_hours=168)
  cutoff = int(len(df) * 0.8)
  train = df.iloc[:cutoff]
  lookup_ts = train["period_utc"].iloc[-200]
  test_hour = lookup_ts + pd.Timedelta(hours=168)

  forecaster.fit(train)
  pred = forecaster.predict(pd.DatetimeIndex([test_hour])).iloc[0]

  expected = train.set_index("period_utc").loc[lookup_ts, "demand_mwh"]
  assert pred == expected, f"Expected {expected}, got {pred}"


def test_annual_naive_works():
  """Predicting hour T should return demand at T - 8760h."""
  df = load_region("ERCO")
  forecaster = SeasonalNaive(period_hours=8760)
  cutoff = int(len(df) * 0.8)
  train = df.iloc[:cutoff]
  lookup_ts = train["period_utc"].iloc[-10000]
  test_hour = lookup_ts + pd.Timedelta(hours=8760)
  forecaster.fit(train)
  pred = forecaster.predict(pd.DatetimeIndex([test_hour])).iloc[0]
  expected = train.set_index("period_utc").loc[lookup_ts, "demand_mwh"]
  assert pred == expected


def test_predict_returns_correct_length():
  df = load_region("SWPP")
  forecaster = SeasonalNaive(period_hours=168)
  cutoff = int(len(df) * 0.8)
  train = df.iloc[:cutoff]
  forecaster.fit(train)
  start = train["period_utc"].iloc[-1] + pd.Timedelta(hours=24)
  index = pd.date_range(start=start, periods=168, freq="h")
  preds = forecaster.predict(index)
  assert len(preds) == 168
  nan_count = preds.isna().sum()
  assert nan_count < 30, f"Too many NaN predictions: {nan_count}/168 - check lookup logic"

if __name__ == "__main__":
  test_weekly_naive_reproduces_lagged_value()
  test_annual_naive_works()
  test_predict_returns_correct_length()
  print("✓ All baseline tests passed")