-- Fails if any region is missing hours between its min and max timestamps.
-- Returns one row per region with gaps; zero rows = test passes.

with bounds as (
  select
    region,
    min(period_utc) as first_hour,
    max(period_utc) as last_hour,
    count(*) as actual_hours
  from {{ ref('fct_hourly_demand') }}
  group by region
),

expected as (
  select
    region,
    first_hour,
    last_hour,
    actual_hours,
    -- Hours in range, inclusive. epoch is seconds; /3600 = hours; +1 for inclusive.
    cast(
      (extract(epoch from last_hour) - extract(epoch from first_hour)) / 3600 + 1 as bigint
    ) as expected_hours
  from bounds
)

select
  region,
  first_hour,
  last_hour,
  expected_hours,
  actual_hours,
  expected_hours - actual_hours as missing_hours
from expected
where actual_hours <> expected_hours and (expected_hours - actual_hours) > 50