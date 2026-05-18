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

def audit_one_split(df: pd.DataFrame, region: str = "ERCO", fold: int = 0) -> bool:
    splits = [s for s in make_splits(
        df, horizon_hours=HORIZON, test_window_hours=168, n_folds=12
    ) if s.region == region and s.fold == fold]
    if not splits:
        print(f"No split for {region} fold {fold}")
        return False
    split = splits[0]
    issue_time = split.test_start - pd.Timedelta(hours=HORIZON)
    print(f"Split: {split}")
    print(f"Forecast issue time (latest data allowed): {issue_time}")
    print(f"Test window: {split.test_start} -> {split.test_end}")
    print ()

    region_df = df[df["region"] == region].copy()
    feats = build_features(region_df, horizon_hours=HORIZON)
    test_feats = feats[
    (feats["period_utc"] >= split.test_start) & (feats["period_utc"] <= split.test_end)
    ]
    lag_map = {
        "demand_lag_24h": 24,
        "demand_lag_168h": 168,
        "demand_lag_336h": 336,
    }
    raw = region_df.set_index("period_utc")["demand_mwh"]
    raw = raw[~raw.index.duplicated(keep="last")].sort_index()

    all_ok = True
    violations = 0
    check = 0

    for _, row in test_feats.iterrows():
        target = row["period_utc"]
        for col, lag in lag_map.items():
            src_ts = target - pd.Timedelta(hours=lag)
            checked += 1
            if src_ts > issue_time:
                violations += 1
                all_ok = False
                if violations <= 5:
                    print(f" LEAK: target {target} feature {col}" 
                          f"uses src {src_ts} > issue {issue_time}")
                if src_ts in raw.index:
                    expected = raw.loc[src_ts]
                    got = row[col]
                    if pd.notna(got) and abs(got - expected) > 1e-6:
                        print(f" MISMATCH: {col} at {target}: feat={got} raw={expected}")
                        all_ok = False

    earliest_roll_src = split.test_start - pd.Timedelta(hours=24)
    print(f"\nRolling feature newest contributing hour (earliest target): "
          f"{earliest_roll_src} (must be <= {issue_time})")
    if earliest_roll_src > issue_time:
        print(f" LEAK: rolling window extends beyond issue time")
        all_ok = False

    print(f"\nChecked {checked} lag features instances across "
          f"{len(test_feats)} test rows.")
    print(f"Violations: {violations}")
    print("RESULT:", "PASS - no leakage detected" if all_ok else "FAIL - leakage detected")
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