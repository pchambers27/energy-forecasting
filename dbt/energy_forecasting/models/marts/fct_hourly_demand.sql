{{ config(materialized='table') }}

with demand as (
  select
    period_utc,
    region,
    demand_mwh
  from {{ ref('stg_eia_demand') }}
),

weather as (
  select
    period_utc,
    region,
    temp_c,
    humidity_pct,
    wind_ms,
    cloud_pct,
    solar_wm2
  from {{ ref('stg_weather_hourly') }}
),

joined as (
  select
    d.period_utc,
    d.region,
    d.demand_mwh,
    w.temp_c,
    w.humidity_pct,
    w.wind_ms,
    w.cloud_pct,
    w.solar_wm2,

    extract(hour from d.period_utc) as hour_of_day,
    extract(dow from d.period_utc) as day_of_week,
    extract(month from d.period_utc) as month_of_year,
    extract(year from d.period_utc) as year,
    case when extract(dow from d.period_utc) in (0,6) then 1 else false end as is_weekend
  from demand d
  inner join weather w
    on d.period_utc = w.period_utc and d.region = w.region)

select * from joined