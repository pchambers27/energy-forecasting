"""Feature engineering for demand forecasting models.

LEAKAGE NOTE: we use ACTUAL observed weather as features, not forecasted weather. This measures model skill GIVEN perfect weather information. It is an upper bound: real day-ahead forecasts have weather forecast error. This assumption is intentional and documented. Phase 5 revisits with forecasted weather.

No demand lags are used in this feature set, so there is no demand-leakage risk: every feature here is known at forecast time under assumption."""

from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_COLS=[
  "period_utc",
  "region",
  "demand_mwh",
  "temp_c",
  "humidity_pct",
  "wind_ms",
  "cloud_pct",
  "solar_wm2",
]

FEATURE_COLS = [
  # Calendar - cyclical encodings
  "hour_sin",
  "hour_cos",
  "dow_sin",
  "dow_cos",
  "month_sin",
  "month_cos",
  "is_weekend",
  # Weather
  "temp_c",
  "humidity_pct",
  "wind_ms",
  "cloud_pct",
  "solar_wm2",
  #Weather interactions / derived
  "temp_sq",
  "cooling_degrees",
  "heating_degrees",
  # Lag features (horizon-safe -- see build_features docstring)
  "demand_lag_24h",
  "demand_lag_168h",
  "demand_lag_336h",
  "demand_roll_mean_168h_lag_24h",
]

_LAG_SPECS = [
  ("demand_lag_24h", 24),
  ("demand_lag_168h", 168),
  ("demand_lag_336h", 336),
]

TARGET_COL = "demand_mwh"

_ROLL_SPEC = ("demand_roll_mean_168h_lag_24h", 168, 24)

def _cyclical(series: pd.Series, period: int) -> tuple[pd.Series, pd.Series]:
  """Encode a cyclic integer feature as sin/cos so 23:00 is 'close to' 00:00."""
  radians = 2 * np.pi * series / period
  return np.sin(radians), np.cos(radians)


def build_features(df: pd.DataFrame, horizon_hours: int = 24) -> pd.DataFrame:
  """Transform raw long-format rows into a model-ready feature frame.

  Args:
    df: raw rows for ONE region, containing REQUIRED_COLS.
    horizon_hours: forecast horizon. Any lag shorter than this is unsafe at forecast time and will raise a ValueError. This makes the leakage rule a hard constraint, not a convention.

  Lag featurs are computed by looking demand up in TIME space (not row space), so data gaps cannot silently corrupt a "24h" lag into a "27h". Because every leag is >= horizon_hours, every feature value is knowable at forecast issue time. Rows whose lags fall before the start of available history (the first ~226h) will have NaN features and are dropped by split_X_y.
  
  The dataframe MUST be a single region. We assert this rather than assume it. 
  """
  missing = set(REQUIRED_COLS) - set(df.columns)
  if missing:
    raise ValueError(f"Input missing required columns: {missing}")

  if df["region"].nunique() > 1: 
    raise ValueError(
      "build_features expects a single region. Group by region before calling."
    )

  for col, lag in _LAG_SPECS:
    if lag < horizon_hours:
      raise ValueError(
        f"Unsafe lag '{col}' ({lag}h) < horizon ({horizon_hours}h)."
        "This would leak future information at forecast time."
      )
  _, _roll_window, _roll_lag = _ROLL_SPEC
  if _roll_lag < horizon_hours:
    raise ValueError(
      f"Unsafe rolling lag {_roll_lag}h < horizon ({horizon_hours}h)."
    )

  d = df.sort_values("period_utc").reset_index(drop=True)

  s = d.set_index("period_utc")["demand_mwh"]
  s = s[~s.index.duplicated(keep="last")].sort_index()

  out = d[["period_utc", "region", "demand_mwh"]].copy()

  ts = d["period_utc"]
  hour = ts.dt.hour
  dow = ts.dt.dayofweek
  month = ts.dt.month

  out["hour_sin"], out["hour_cos"] = _cyclical(hour, 24)
  out["dow_sin"], out["dow_cos"] = _cyclical(dow, 7)
  out["month_sin"], out["month_cos"] = _cyclical(month, 12)
  out["is_weekend"] = (dow >= 5).astype(int)
  out["temp_c"] = d["temp_c"]
  out["humidity_pct"] = d["humidity_pct"]
  out["wind_ms"] = d["wind_ms"]
  out["cloud_pct"] = d["cloud_pct"]
  out["solar_wm2"] = d["solar_wm2"]
  out["temp_sq"] = d["temp_c"] ** 2
  out["cooling_degrees"] = (d["temp_c"] - 18.0).clip(lower=0)
  out["heating_degrees"] = (18.0 - d["temp_c"]).clip(lower=0)

  target_ts = out["period_utc"]
  for col, lag in _LAG_SPECS:
    lookup = target_ts - pd.Timedelta(hours=lag)
    out[col] = s.reindex(lookup).values


  roll_col, roll_window, roll_lag = _ROLL_SPEC
  full_idx = pd.date_range(s.index.min(), s.index.max(), freq="h")
  s_full = s.reindex(full_idx)
  rolled = s_full.rolling(window=roll_window, min_periods=roll_window // 2).mean()
  rolled_lagged = rolled.copy()
  rolled_lagged.index = rolled_lagged.index + pd.Timedelta(hours=roll_lag)
  out[roll_col] = rolled_lagged.reindex(target_ts).values

  return out


def split_X_y(feature_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
  """Seperate the feature matrix and target, dropping rows with any NaN."""
  cols = FEATURE_COLS + [TARGET_COL]
  clean = feature_df.dropna(subset=cols)
  X = clean[FEATURE_COLS]
  y = clean[TARGET_COL]
  return X, y