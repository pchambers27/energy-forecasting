# Energy Demand Forecasting

End-to-end hourly electricity demand forecasting for SPP (Southwest Power Pool)
and ERCOT, with weather features. Built as a learning project for MS-level
data analytics, with emphasis on rigorous backtesting and beating seasonal
naive baselines.

## Stack
- Python 3.11+
- DuckDB (embedded analytical DB)
- dbt-core + dbt-duckdb (transformations & testing)
- Future: scikit-learn / XGBoost / LightGBM / statsmodels, MLflow, FastAPI, Streamlit

## Project structure

    workspace/
    ├── data/                          # DuckDB files (gitignored)
    │   └── warehouse_dev.duckdb
    ├── src/
    │   └── ingestion/
    │       ├── eia_client.py          # EIA v2 API → raw_eia.raw_demand
    │       └── weather_client.py      # Open-Meteo archive → raw_weather.raw_hourly
    ├── scripts/
    │   └── run_pipeline.py            # Ingest + dbt run + dbt test
    └── dbt/energy_forecasting/        # dbt project
        ├── models/staging/            # Cleaned, typed views
        ├── models/marts/              # Analytics-ready table
        └── tests/                     # Custom gap-detection test

## Setup
1. Replit Secrets: `EIA_API_KEY` (get one free at api.eia.gov/register)
2. `pip install duckdb dbt-core dbt-duckdb requests pandas pytest`
3. `cd dbt/energy_forecasting && dbt deps`
4. `python scripts/run_pipeline.py`

## Data sources
- **EIA v2 API**: hourly demand by balancing authority (`SWPP`, `ERCO`)
- **Open-Meteo archive**: hourly weather at one representative city per region

## Scope (Phase 1)
- Regions: SWPP, ERCO
- History: 5 years
- Grain: hourly, UTC
- Weather points: Houston (ERCO), Oklahoma City (SWPP)

## Data dictionary

### `main_marts.fct_hourly_demand`
One row per (region, hour). Inner join of demand + weather; weather lags ~3 days.

| Column         | Type        | Units / Notes                                        |
|----------------|-------------|------------------------------------------------------|
| period_utc     | TIMESTAMPTZ | Hour start, UTC                                      |
| region         | VARCHAR     | `ERCO` or `SWPP`                                     |
| demand_mwh     | DOUBLE      | Electricity demand, megawatt-hours                   |
| temp_c         | DOUBLE      | Temperature, °C                                      |
| humidity_pct   | DOUBLE      | Relative humidity, %                                 |
| wind_ms        | DOUBLE      | Wind speed at 10m, m/s                               |
| cloud_pct      | DOUBLE      | Cloud cover, %                                       |
| solar_wm2      | DOUBLE      | Shortwave solar radiation, W/m²                      |
| hour_of_day    | BIGINT      | 0–23                                                 |
| day_of_week    | BIGINT      | 0=Sunday … 6=Saturday (DuckDB convention)            |
| month_of_year  | BIGINT      | 1–12                                                 |
| year           | BIGINT      |                                                      |
| is_weekend     | BOOLEAN     | true if Saturday or Sunday                           |

## Data quality notes
- EIA hourly demand has ~0.06% missing hours over the 5-year window (~1 hour
  for ERCO, ~26 hours for SWPP). Left as-is; the gap-detection dbt test
  tolerates up to 50 missing hours per region before failing.
- Weather (Open-Meteo archive) lags real-time by ~3–5 days. The marts table
  uses an inner join, so demand hours without weather are dropped from
  analytics. Live serving (Phase 4) will switch to weather forecasts.
- All timestamps stored as UTC. Local-time features (e.g., business hours
  by region) will be added in feature engineering, not in marts.

## Phases
1. ✅ Data foundation
2. ⏳ Backtesting framework + seasonal naive baselines
3. ⏳ Real models (linear, gradient boosted)
4. ⏳ Serving (FastAPI) + dashboard (Streamlit)
5. ⏳ Extensions (probabilistic forecasts, conformal prediction)