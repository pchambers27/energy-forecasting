"""Backtest runner: executes forecasters across splits, returns predictions + metrics."""

from __future__ import annotations

import logging
from typing import Sequence

import pandas as pd

from src.backtest.splitter import Split, make_splits
from src.backtest.baselines import Forecaster
from src.backtest.metrics import compute_metrics, skill_score

logger = logging.getLogger(__name__)

def run_backtest(
  df: pd.DataFrame,
  forecasters: dict[str, Forecaster],
  *,
  splitter_kwargs: dict | None = None,) -> tuple[pd.DataFrame, pd.DataFrame]:
  """Run all forecasters across all splits.
  Args:
    df: long-format dataframe with region, period_utc, and demand_mwh
    forecasters: name -> Forecaster
    splitter_kwargs: passed to make_splits
  Returns:
    predictions_df: columns [forecaster, region, fold, period_utc, y_true, y_pred]
    metrics_df: columns [forecaster, region, fold, n, n_dropped, mae, rmse, mape, bias]
  """
  splitter_kwargs = splitter_kwargs or {}
  splits = list(make_splits(df, **splitter_kwargs))
  logger.info(f"Running backtest with {len(forecasters)} forecaster(s) across {len(splits)} split(s)")
  pred_rows = []
  metric_rows = []
  for split in splits:
    region_df = df[df["region"] == split.region].copy()
    train_mask = (region_df["period_utc"] >= split.train_start) & (region_df["period_utc"] <= split.train_end)
    test_mask = (region_df["period_utc"] >= split.test_start) & (region_df["period_utc"] <= split.test_end)
    train_df = region_df[train_mask]
    test_df = region_df[test_mask].sort_values("period_utc").reset_index(drop=True)
    if len(test_df) == 0:
      logger.warning(f"Empty test set for {split} - skipping")
      continue
    y_true = test_df.set_index("period_utc")["demand_mwh"]
    for name, forecaster in forecasters.items():
      forecaster.fit(train_df)
      y_pred = forecaster.predict(y_true.index)
      for ts, actual, pred in zip(y_true.index, y_true.values, y_pred.values):
        pred_rows.append({
          "forecaster": name,
          "region": split.region,
          "fold": split.fold,
          "period_utc": ts,
          "y_true": actual,
          "y_pred": pred
        })
      m = compute_metrics(y_true, y_pred)
      metric_rows.append({
        "forecaster": name,
        "region": split.region,
        "fold": split.fold,
        **m.as_dict()
      })
    logger.info(f" ✓ {split}")
  predictions_df = pd.DataFrame(pred_rows)
  metrics_df = pd.DataFrame(metric_rows)
  return predictions_df, metrics_df


def summarize(
  metrics_df: pd.DataFrame,
  *,
  baseline: str | None = None,) -> pd.DataFrame:
  """Aggregate fold-level metrics into per-(forecaster, region) summarize.
  If a baseline forecaster name is provided, also computes skill socre against it.
  """


  def weighted_mean(group, col):
    weights = group["n"]
    if weights.sum() == 0:
      return float("nan")
    return (group[col] * weights).sum() / weights.sum()

  rows = []
  for (forecaster, region), group in metrics_df.groupby(["forecaster", "region"]):
    rows.append({
      "forecaster": forecaster,
      "region": region,
      "n_folds": len(group),
      "mae": weighted_mean(group, "mae"),
      "rmse": weighted_mean(group, "rmse"),
      "mape": weighted_mean(group, "mape"),
      "bias": weighted_mean(group, "bias"),
    })
  summary = pd.DataFrame(rows)
  if baseline is not None: 
    baseline_maes = (
      summary[summary["forecaster"] == baseline].set_index("region")["mae"].to_dict()
  )
    summary["skill_vs_" + baseline] = summary.apply(
      lambda r:
      skill_score(r["mae"], baseline_maes.get(r["region"], float("nan"))),
      axis=1
    )

  return summary.sort_values(["region", "forecaster"]).reset_index(drop=True)