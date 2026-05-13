{{ config(materialized='view') }}

with source as (
  select * from {{ source('raw_weather', 'raw_hourly') }}
),

renamed as (
  select
    -- Cast naive timestamp to TZ-aware UTC
    cast (period_utc as timestamptz) as time zone 'UTC' as period_utc,
    region,
    city,
    latitude,
    longitude,
    temperature_2m as temp_c,
    relative_humidity_2m as humidity_pct,
    wind_speed_10m as wind_ms,
    cloud_cover as cloud_pct,
    shortwave_radiation as solar_wm2,
    ingested_at
  from source
  where period_utc is not null
)
select * from renamed