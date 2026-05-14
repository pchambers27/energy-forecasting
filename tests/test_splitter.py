"""Sanity tests for the time-series splitter."""

import duckdb
import pandas as pd
from datetime import timedelta

from src.backtest.splitter import make_splits

def load_marts() -> pd.DataFrame:
  con = duckdb.connect("/home/runner/workspace/data/warehouse_dev.duckdb")
  df = con.execute("""
  SELECT period_utc, region, demand_mwh
  FROM main_marts.fct_hourly_demand
  ORDER BY region, preiod_utc""").df()
  con.close()
  return df

def test_splits_have_correct_structure():
  df = load_marts()
  splits = list(make_splits(df, horizon_hours=24, test_window_hours=168, n_folds=12))


  assert len(splits) > 0, "No splits produced"
  assert len(splits) <= 24

  regions_seen = {s.region for s in splits}
  assert regions_seen == {"ERCO", "SWPP"}, f"Unexpected regions: {regions_seen}"


def test_no_leakage_gap():
  """The gap between train_end and test_start must equal horizon."""
  df = load_marts()
  horizon = 24
  for split in make_splits(df, horizon_hours=horizon, test_window_hours=168, n_folds=12):
    gap = split.test_start - split.train_end
    assert gap == timedelta(hours=horizon), (
      f"Leakage risk: gap is {gap}, expected {timedelta(hours=horizon)} in {split}"
    )


def test_train_before_test():
  """Train must end before test starts. Always."""
  df = load_marts()
  for split in make_splits(df):
    assert split.train_end < split.test_start, f"Bad ordering in {split}"
    assert split.train_start < split.train_end, f"Bad train range in {split}"
    assert split.test_start < split.test_end, f"Bad test range in {split}"


def test_test_windows_dont_overlap_within_region():
  """Within a region, test windows must not overlap."""
  df = load_marts()
  by_region = dict[str, list] = {}
  for split in make_splits(df):
     by_region.setdefault(split.region, []).append(split)
  for region, splits in by_region.items():
     splits_sorted = sorted(splits, key=lambda s: s.test_start)
     for a, b in zip(splits_sorted, splits_sorted[1:]):
       assert a.test_end < b.test_start, f"Overlapping test windows in {region}: {a}, {b}"


if __name__ == "__main__":

  test_splits_have_correct_structure()
  test_no_leakage_gap()
  test_train_before_test()
  test_test_windows_dont_overlap_within_region()
  print("✓ All splitter tests passed")