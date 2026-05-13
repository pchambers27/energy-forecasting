{{ config(materialized='view') }}

with source as (
  select * from {{ source('raw_eia', 'raw_demand') }}
),

renamed as (
  select
    -- Cast naive timestamp to TZ-aware UTC
    cast (period_utc as timestamptz) as time zone 'UTC' as period_utc,
    respondent as region,
    respondent_name as region_name,
    type_code,
    value as demand_mwh,
    value_units,
    ingested_at
  from source
  where type_code = 'D' -- defensive: only demand records
   and value is not null
   and value > 0 -- demand can't be 0
)