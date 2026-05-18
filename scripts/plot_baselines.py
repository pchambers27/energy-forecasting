"""Generate diagnostic plots for the baseline backtest.

Outputs PNGs to data/plots/ for viewing the Replit's file tree.
"""


from __future__ import annotations
import logging
from pathlib import Path

import duckdb
import matplotlib.pyplot as plt
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PLOTS_DIR = Path("/home/runner/workspace/data/plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

def load_predictions() -> pd.DataFrame:
  return pd.read_parquet("/home/runner/workspace/data/baseline_predictions.parquet")


def plot_fold(predictions: pd.DataFrame, region: str, fold: int) -> Path:
  """Plot actuals vs. weekly_naive for one fold."""
  sub = predictions[(predictions["region"] == region) & (predictions["fold"] == fold)].copy()
  sub = sub.sort_values("period_utc")
  fig, ax = plt.subplots(figsize=(14, 5))

  actual = sub[sub["forecaster"] == "weekly_naive"][["period_utc", "y_true"]]
  ax.plot(actual["period_utc"], actual["y_true"], label="Actual", color="black", linewidth=2.5, alpha=0.4, zorder=1)
  ann = sub[sub["forecaster"] == "annual_naive"]
  ax.plot(ann["period_utc"], ann["y_pred"], label="Annual naive", color="tab:orange", linewidth=1.5, alpha=0.9, zorder=2)
  wk = sub[sub["forecaster"] == "weekly_naive"]
  ax.plot(wk["period_utc"], wk["y_pred"], label="Weekly naive", color="tab:blue", linewidth=1.5, linestyle="--", zorder=3)
  ax.set_title(f"{region} -- Fold {fold} (1 week of hourly predictions)")
  ax.set_xlabel("Time (UTC)")
  ax.set_ylabel("Demand (MWh)")
  ax.legend()
  ax.grid(True, alpha=0.3)
  fig.autofmt_xdate()
  out = PLOTS_DIR / f"fold_{region}_{fold}.png"
  fig.tight_layout()
  fig.savefig(out, dpi=120)
  plt.close(fig)
  logger.info(f"Saved {out}")
  return out

def plot_weekly_trend(region: str) -> Path:
  """Weekly mean demand over the last ~6 months -- shows growth/trend."""
  con = duckdb.connect("/home/runner/workspace/data/warehouse_dev.duckdb")
  df = con.execute(f"""
    SELECT period_utc, demand_mwh
    FROM main_marts.fct_hourly_demand
    WHERE region = '{region}'
      AND period_utc > NOW() - INTERVAL 200 DAY
    ORDER BY period_utc""").df()
  con.close()
  df["week"] = df["period_utc"].dt.to_period("W").dt.start_time
  weekly = df.groupby("week")["demand_mwh"].mean().reset_index()
  fig, ax = plt.subplots(figsize=(12, 4))
  ax.plot(weekly["week"], weekly["demand_mwh"], marker="o")
  ax.set_title(f"{region} -- Weekly mean demand (last ~200 days)")
  ax.set_xlabel("Week")
  ax.set_ylabel("Mean MWh / hour")
  ax.grid(True, alpha=0.3)
  fig.autofmt_xdate()
  out = PLOTS_DIR / f"weekly_trend_{region}.png"
  fig.tight_layout()
  fig.savefig(out, dpi=120)
  plt.close(fig)
  logger.info(f"Saved {out}")
  return out

def plot_error_distribution(predictions: pd.DataFrame, region: str) -> Path:
  """Histogram of weekly_naive errors for a region across all folds"""
  sub = predictions[(predictions["region"] == region) & (predictions["forecaster"] == "weekly_naive")].copy()
  sub["error"] = sub["y_pred"] - sub["y_true"]
  sub = sub.dropna(subset=["error"])
  fig, ax = plt.subplots(figsize=(10, 4))
  ax.hist(sub["error"], bins=60, edgecolor="black", alpha=0.7)
  ax.axvline(0, color="red", linestyle="--", linewidth=1, label="Zero error")
  ax.axvline(sub["error"].mean(), color="green", linestyle="--", linewidth=1, label=f"mean = {sub['error'].mean():.0f}")
  ax.set_title(f"{region} -- Weekly naive prediction erros (all folds)")
  ax.set_xlabel("Error (predicted - actual, MWh)")
  ax.set_ylabel("Count")
  ax.legend()
  ax.grid(True, alpha=0.3)
  out = PLOTS_DIR / f"error_dist_{region}.png"
  fig.tight_layout()
  fig.savefig(out, dpi=120)
  plt.close(fig)
  logger.info(f"Saved {out}")
  return out


def main():
  predictions = load_predictions()
  for region in ["ERCO", "SWPP"]:
    plot_fold(predictions, region=region, fold=0)
    plot_weekly_trend(region)
    plot_error_distribution(predictions, region)

if __name__ == "__main__":
  main()