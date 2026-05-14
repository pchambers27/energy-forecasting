"""Run the seasonal naive baselines and print results.

Output: a printed summary table per region.
"""

from __future__ import annotations

import logging
import duckdb
import pandas as pd

from src.backtest.baselines import SeasonalNaive
from src.backtest.runner import run_backtest, summarize

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

def load_data() -> pd.DataFrame:
  con = duckdb.connect("/home/runner/workspace/data/warehouse_dev.duckdb")
  df = con.execute("""
    SELECT period_utc, region, demand_mwh
    FROM main_marts.fct_hourly_demand
    ORDER BY region, period_utc""").df()
  con.close()
  return df

def main():
  df = load_data()
  logger.info(f"Loaded {len(df)} rows across {df['region'].nunique()} region(s)")
  forecasters = {
    "weekly_naive": SeasonalNaive(period_hours=168, name="weekly_naive"),
    "seasonal_naive": SeasonalNaive(period_hours=8760, name="annual_naive"),
  }
  predictions_df, metrics_df = run_backtest(
    df,
    forecasters,
    splitter_kwargs=dict(
      horizon_hours=24,
      test_window_hours=168,
      n_folds=12,
    ),
  )
  logger.info(f"Generated {len(predictions_df)} predictions, {len(metrics_df)} fold metrics")
  summary = summarize(metrics_df, baseline="weekly_naive")
  print("\n=== Backtest summary ===")

  print(summary.to_string(index=False))

  predictions_df.to_parquet("/home/runner/workspace/data/baseline_predictions.parquet", index=False)
  metrics_df.to_parquet("/home/runner/workspace/data/baseline_fold_metrics.parquet", index=False)
  summary.to_parquet("/home/runner/workspace/data/baseline_summary.parquet", index=False)
  logger.info("Wrote results to data/baseline_*parquet")


if __name__ == "__main__":
   main()
