"""Sanity tests for the feature builder."""
import numpy as np
import pandas as pd

from src.backtest.features import build_features, split_X_y, FEATURE_COLS


def _toy_df():
    idx = pd.date_range("2024-06-01", periods=400, freq="h", tz="UTC")
    n = len(idx)
    return pd.DataFrame({
        "period_utc": idx,
        "region": ["ERCO"] * n,
        "demand_mwh": np.linspace(40000, 55000, n),
        "temp_c": np.linspace(10, 35, n),
        "humidity_pct": np.linspace(40, 90, n),
        "wind_ms": np.linspace(1, 8, n),
        "cloud_pct": np.linspace(0, 100, n),
        "solar_wm2": np.linspace(0, 800, n),
    })

def test_unsafe_lag_raises():
    """With the corrected rule, a horizon large enough that 48h < horizon + block must raise."""
    df = _toy_df()
    try:
        build_features(df, horizon_hours=48)
        assert False, "Expected ValueError for unsafe lag"
    except ValueError as e:
        assert "Unsafe lag" in str(e)


def test_lag_values_are_time_correct():
    """demand_lag_48h at time t must equal demand at t-48h."""
    df = _toy_df()
    f = build_features(df, horizon_hours=24)
    s = df.set_index("period_utc")["demand_mwh"]
    row = f.iloc[380]
    expected = s.loc[row["period_utc"] - pd.Timedelta(hours=48)]
    assert np.isclose(row["demand_lag_48h"], expected)


def test_all_feature_cols_present():
    f = build_features(_toy_df(), horizon_hours=24)
    for col in FEATURE_COLS:
        assert col in f.columns, f"Missing feature column: {col}"


def test_cyclical_bounds():
    f = build_features(_toy_df(), horizon_hours=24)
    for col in ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos"]:
        assert f[col].between(-1.0, 1.0).all(), f"{col} out of [-1, 1]"


def test_degree_days_nonnegative():
    f = build_features(_toy_df(), horizon_hours=24)
    assert (f["cooling_degrees"] >= 0).all()
    assert (f["heating_degrees"] >= 0).all()
    # At exactly 18°C both should be zero
    assert np.isclose(f.loc[f["temp_c"].sub(18).abs().idxmin(), "cooling_degrees"], 0, atol=2)


def test_split_xy_drops_nan():
    df = _toy_df()
    df.loc[5, "temp_c"] = np.nan
    f = build_features(df, horizon_hours=24)
    X, y = split_X_y(f)
    assert len(X) == len(y)
    assert not X.isna().any().any()
    assert 0 < len(X) < 70, f"Unexpected valid row count after lag warmup: {len(X)}"


if __name__ == "__main__":
    test_all_feature_cols_present()
    test_cyclical_bounds()
    test_degree_days_nonnegative()
    test_split_xy_drops_nan()
    print("All feature tests passed ✓")