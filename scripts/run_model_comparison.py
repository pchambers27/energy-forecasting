"""Three-way backtest comparison (weekly_naive vs Ridge vs GBM) with MLflow tracking.
MLflow logs one run per (forecaster, region) with params + metrics. Tracking store is local (./mlruns), no external service -- works on Replit."""

from __future__ import annotations

import logging
from pathlib import Path

import duckdb
import mlflow
import pandas as pd

from src.backtest.baselines import SeasonalNaive
from src.backtest.models import RidgeForecaster, GBMForecaster
from src.backtest.runner import run_backtest, summarize

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MLRUNS_DIR = Path("/home/runner/workspace/mlruns")
EXPERIMENT = "energy-demand-phase3"

def load_data() -> pd.DataFrame:
    con = duckdb.connect("/home/runner/workspace/data/warehouse_dev.duckdb")
    df = con.execute("""
        SELECT period_utc, region, demand_mwh, temp_c, humidity_pct, wind_ms, cloud_pct, solar_wm2
        FROM main_marts.fct_hourly_demand
        ORDER BY period_utc""").df()
    con.close()
    return df

def main():
    mlflow.set_tracking_uri(f"file://{MLRUNS_DIR}")
    mlflow.set_experiment(EXPERIMENT)
    df = load_data()
    logger.info(f"Loaded {len(df):,} rows from DuckDB")
    forecasters = {
        "weekly_naive": SeasonalNaive(period_hours=168, name="weekly_naive"),
        "ridge": RidgeForecaster(alpha=1.0, name="ridge", horizon_hours=24),
        "gbm": GBMForecaster(name="gbm", horizon_hours=24),
    }
    predictions_df, metrics_df = run_backtest(df, forecasters, splitter_kwargs=dict(horizon_hours=24, test_window_hours=168, n_folds=12),)
    summary = summarize(metrics_df, baseline="weekly_naive")
    print("\n=== Three-way comparison ===")
    print(summary.to_string(index=False))
    for _, row in summary.iterrows():
        run_name = f"{row['forecaster']}__{row['region']}"
        with mlflow.start_run(run_name=run_name):
            mlflow.log_param("forecaster", row["forecaster"])
            mlflow.log_param("region", row["region"])
            mlflow.log_param("n_folds", int(row["n_folds"]))
            mlflow.log_param("feature_set", "calendar+weather+lags(24/168/336/roll)")
            mlflow.log_param("weather_assumption", "Option A: actual observed weather")
            mlflow.log_metric("mae", float(row["mae"]))
            mlflow.log_metric("rmse", float(row["rmse"]))
            mlflow.log_metric("mape", float(row["mape"]))
            mlflow.log_metric("bias", float(row["bias"]))
            skill_col = "skill_vs_weekly_naive"
            if skill_col in row and pd.notna(row[skill_col]):
                mlflow.log_metric("skill_vs_weekly_naive", float(row[skill_col]))
    predictions_df.to_parquet("/home/runner/workspace/data/phase3_predictions.parquet", index=False)
    metrics_df.to_parquet("/home/runner/workspace/data/phase3_fold_metrics.parquet", index=False)
    summary.to_parquet("/home/runner/workspace/data/phase3_summary.parquet", index=False)
    logger.info(f"Logged {len(summary)} runs to MLflow experiment '{EXPERIMENT}'")


if __name__ == "__main__":
     main()
            