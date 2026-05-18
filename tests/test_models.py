"""Sanity tests for the Ridge forecaster."""
import duckdb
import pandas as pd

from src.backtest.models import RidgeForecaster


def load_region(region: str) -> pd.DataFrame:
    con = duckdb.connect("/home/runner/workspace/data/warehouse_dev.duckdb")
    df = con.execute(f"""
        SELECT period_utc, region, demand_mwh,
               temp_c, humidity_pct, wind_ms, cloud_pct, solar_wm2
        FROM main_marts.fct_hourly_demand
        WHERE region = '{region}'
        ORDER BY period_utc
    """).df()
    con.close()
    return df


def test_ridge_fits_and_predicts():
    df = load_region("ERCO")
    cutoff = int(len(df) * 0.8)
    train = df.iloc[:cutoff]
    test = df.iloc[cutoff:cutoff + 168]

    model = RidgeForecaster(alpha=1.0)
    model.set_context(df)            # full region frame for feature lookup
    model.fit(train)
    preds = model.predict(pd.DatetimeIndex(test["period_utc"]))

    assert len(preds) == len(test)
    # Most predictions should be valid (some NaN possible at data gaps)
    assert preds.notna().sum() > 150
    # Predictions should be in a physically sane range for ERCO (tens of thousands MWh)
    valid = preds.dropna()
    assert valid.min() > 5000, f"Implausibly low prediction: {valid.min()}"
    assert valid.max() < 120000, f"Implausibly high prediction: {valid.max()}"


def test_ridge_beats_naive_in_sample_sanity():
    """Not a rigorous test — just confirms the model learns *something*.
    On training data, Ridge should have lower MAE than predicting the mean."""
    df = load_region("SWPP")
    cutoff = int(len(df) * 0.8)
    train = df.iloc[:cutoff]

    model = RidgeForecaster(alpha=1.0)
    model.set_context(df)
    model.fit(train)
    preds = model.predict(pd.DatetimeIndex(train["period_utc"]))

    merged = pd.DataFrame({
        "y": train.set_index("period_utc")["demand_mwh"],
        "p": preds,
    }).dropna()

    model_mae = (merged["p"] - merged["y"]).abs().mean()
    mean_mae = (merged["y"].mean() - merged["y"]).abs().mean()
    assert model_mae < mean_mae, (
        f"Ridge ({model_mae:.0f}) not better than predicting the mean ({mean_mae:.0f}) "
        "— features may be broken"
    )


def test_gbm_fits_and_predicts():
    from src.backtest.models import GBMForecaster
    df = load_region("ERCO")
    cutoff = int(len(df) * 0.8)
    train = df.iloc[:cutoff]
    test = df.iloc[cutoff:cutoff + 168]
    model = GBMForecaster()
    model.set_context(df)
    model.fit(train)
    preds = model.predict(pd.DatetimeIndex(test["period_utc"]))
    assert len(preds) == len(test)
    assert preds.notna().sum() > 100
    valid = preds.dropna()
    assert valid.min() > 5000, f"Implausibly low prediction: {valid.min()}"
    assert valid.max() < 120000, f"Implausibly high prediction: {valid.max()}"
    print(f" GBM backend: {model.backend}")


if __name__ == "__main__":
    test_ridge_fits_and_predicts()
    test_ridge_beats_naive_in_sample_sanity()
    test_gbm_fits_and_predicts()
    print("All model tests passed ✓")