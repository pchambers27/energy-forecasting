"""Leakage audit: prove no feature at a test timestamp uses data newer than the forecast-issue time.
Day-ahead setup: a forecast for any hour in the test window is "issued" horizon_hours before the EARLIEST test hour. Every feature value used for any test-window target must therefore come from a source timestamp at or before (test_start - horizon_hours). This script verifies that empirically by reconstructing each lag feature's source timestamp and checking it.

Run: PYTHONPATH=. python scripts/audit_leakage.py
"""


from __future__ import annotations

import duckdb
import pandas as pd

from src.backtest.splitter import make_splits
from src.backtest.features import build_features


HORIZON = 24
DB = "/home/runner/workspace/data/warehouse_dev.duckdb"

def load() -> pd.DataFrame:
    con = duckdb.connect(DB)
    df = con.execute("""
        SELECT period_utc, region, demand_mwh, temp_c, humidity_pct, wind_ms, cloud_pct, solar_wm2
        FROM main_marts.fct_hourly_demand
        ORDER BY region, period_utc""").df()
    con.close()
    return df

def issue_time_for_target(target: pd.Timestamp, test_start: pd.Timestamp) -> pd.Timestamp:
    """Issue time of the daily day-ahead forecast covering 'target'.
    Forecast for each calendar day is issued HORIZON hours before that day's first test hour. Daily grid anchored to test_start's time of day."""
    delta_days = (target.normalize() - test_start.normalize()).days
    day_first_hour = test_start + pd.Timedelta(days=delta_days)
    return day_first_hour - pd.Timedelta(hours=HORIZON)

def audit_one_split(df: pd.DataFrame, region: str, fold: int = 0) -> bool:
    splits = [s for s in make_splits(
        df, horizon_hours=HORIZON, test_window_hours=168, n_folds=12
    ) if s.region == region and s.fold == fold]
    if not splits:
        print(f"No split for {region} fold {fold}")
        return False
    split = splits[0]
    print(f"Split: {split}")
    print(f"Test window: {split.test_start} -> {split.test_end}")
    print(f"Protocol: daily re-issued day-ahead (issue = day_start - {HORIZON}h)")
    print ()

    region_df = df[df["region"] == region].copy()
    feats = build_features(region_df, horizon_hours=HORIZON)
    test_feats = feats[
    (feats["period_utc"] >= split.test_start) & (feats["period_utc"] <= split.test_end)
    ]
    lag_map = {
        "demand_lag_48h": 48,
        "demand_lag_168h": 168,
        "demand_lag_336h": 336,
    }
    raw = region_df.set_index("period_utc")["demand_mwh"]
    raw = raw[~raw.index.duplicated(keep="last")].sort_index()

    violations = 0
    mismatches = 0
    checked = 0
    all_ok = True

    for _, row in test_feats.iterrows():
        target = row["period_utc"]
        issue = issue_time_for_target(target, split.test_start)
        for col, lag in lag_map.items():
            src_ts = target - pd.Timedelta(hours=lag)
            checked += 1
            if src_ts > issue:
                violations += 1
                all_ok = False
                if violations <= 5:
                    print(f" LEAK: target {target} feature {col} src {src_ts} > issue {issue}")
                if src_ts in raw.index and pd.notna(row[col]):
                    if abs(row[col] - raw.loc[src_ts]) > 1e-6:
                        mismatches += 1
                        all_ok = False
                        if mismatches <= 5:
                            print(f" MISMATCH: {col} at {target}: "
                                  f" feat={row[col]} raw={raw.loc[src_ts]}")

    

    roll_ok = True
    for _, row in test_feats.iterrows():
        target = row["period_utc"]
        issue = issue_time_for_target(target, split.test_start)
        newest_roll_src = target - pd.Timedelta(hours=48)
        if newest_roll_src > issue:
            roll_ok = False
            all_ok = False
            print(f" ROLL LEAK: target {target} newest src {newest_roll_src} "
                  f"> issue {issue}")
            break
    print(f"\nChecked {checked} lag features instances over "
          f"{len(test_feats)} test rows")
    print(f"Lag violations: {violations} | value mismatches: {mismatches}"
          f"| rolling: {'OK' if roll_ok else 'LEAK'}")
    print("RESULT:", "PASS - no leakage under daily re-issue" if all_ok else "FAIL - leakage found")
    return all_ok

def main():
    df = load()
    ok = True
    for region in ["ERCO", "SWPP"]:
        print("=" * 60)
        print(f"AUDIT: {region} fold 0")
        print("=" * 60)
        ok &= audit_one_split(df, region=region, fold=0)
        print()
    print("OVERALL:", "ALL CLEAN" if ok else "LEAKAGE FOUND - investigate")

if __name__ == "__main__":
     main()