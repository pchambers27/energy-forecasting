"""Open-Meteo historical weather archive client.

Fetches hourly weather for representative cities per refion. Uses the archive-api endpoint (free, no key required)."""

from __future__ import annotations

import os
import time
import logging
from datetime import datetime, timedelta, timezone
from typing import Iterator

import requests
import duckdb

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
MAX_RETRIES = 5
BACKOFF_BASE = 2.0

LOCATIONS = {
  "ERCO": {"city": "Houston", "lat": 29.7604, "lon": -95.3698},
  "SWPP": {"city": "Oklahoma City", "lat": 35.4676, "lon": -97.5164},
}

HOURLY_VARS = [
  "temperature_2m",
  "relative_humidity_2m",
  "wind_speed_10m",
  "cloud_cover",
  "shortwave_radiation",
]

def _request_with_retry(params: dict) -> dict:
  for attempt in range(MAX_RETRIES):
    try:
      r = requests.get(ARCHIVE_URL, params=params, timeout=60)
      r.raise_for_status()
      return r.json()
    except (requests.RequestException, ValueError) as e:
      wait = BACKOFF_BASE ** attempt
      logger.warning(f"Request failed (attempt {attempt + 1}/{MAX_RETRIES}) {e}. Retrying in {wait}s...")
      time.sleep(wait)
  raise RuntimeError(f"Open-Meteo request failed after {MAX_RETRIES} attempts")


def fetch_weather(lat: float, lon: float, start: datetime, end: datetime) -> Iterator[dict]:
  """Yield hourly weather records between start and end (UTC).
  Open-Meteo archive returns the full range in one response (no pagination needed for reasonable windows), but we chunk by year to keep requests bounded."""
  cursor = start
  while cursor < end:
    chunk_end = min(cursor + timedelta(days=365), end)
    params = {
      "latitude": lat,
      "longitude": lon,
      "start_date": cursor.strftime("%Y-%m-%d"),
      "end_date": chunk_end.strftime("%Y-%m-%d"),
      "hourly": ",".join(HOURLY_VARS),
      "timezone": "UTC",
    }
    payload = _request_with_retry(params)
    hourly = payload.get("hourly", {})
    times = hourly.get("time", [])
    for i, ts in enumerate(times):
      yield {
        "period_utc": ts,
        "temperature_2m": hourly["temperature_2m"][i],
        "relative_humidity_2m": hourly["relative_humidity_2m"][i],
        "wind_speed_10m": hourly["wind_speed_10m"][i],
        "cloud_cover": hourly["cloud_cover"][i],
        "shortwave_radiation": hourly["shortwave_radiation"][i]
      }
    logger.info(f"({lat}, {lon}) fetched {cursor.date()} -> {chunk_end.date()} ({len(times)} hours)")
    cursor = chunk_end + timedelta(days=1)


def ensure_raw_table(con: duckdb.DuckDBPyConnection) -> None:
  con.execute("CREATE SCHEMA IF NOT EXISTS raw_weather")
  con.execute("""
  CREATE TABLE IF NOT EXISTS raw_weather.raw_hourly (
  period_utc TIMESTAMP,
  region VARCHAR,
  city VARCHAR,
  latitude DOUBLE,
  longitude DOUBLE,
  temperature_2m DOUBLE,
  relative_humidity_2m DOUBLE,
  wind_speed_10m DOUBLE,
  cloud_cover DOUBLE,
  shortwave_radiation DOUBLE,
  ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
  PRIMARY KEY (period_utc, region))""")


def get_last_timestamp(con: duckdb.DuckDBPyConnection, region: str) -> datetime | None:
  row = con.execute("SELECT MAX(period_utc) FROM raw_weather.raw_hourly WHERE region = ?", [region],).fetchone()
  return row[0].replace(tzinfo=timezone.utc) if row and row[0] else None


def ingest_region(
  con: duckdb.DuckDBPyConnection, region: str, start: datetime, end: datetime,) -> int:
  ensure_raw_table(con)
  loc = LOCATIONS[region]
  last_ts = get_last_timestamp(con, region)
  effective_start = (last_ts + timedelta(hours=1)) if last_ts else start
  if effective_start >= end:
    logger.info(f"[{region}] already up to date through {last_ts}")
    return 0
  logger.info(f"[{region}] ingesting {effective_start} -> {end}")
  rows = []
  for rec in fetch_weather(loc["lat"], loc["lon"], effective_start, end):
    rows.append((
      rec["period_utc"],
      region,
      loc["city"],
      loc["lat"],
      loc["lon"],
      rec["temperature_2m"],
      rec["relative_humidity_2m"],
      rec["wind_speed_10m"],
      rec["cloud_cover"],
      rec["shortwave_radiation"],
      ))
  if not rows: 
    logger.warning(f"[{region}] API returned 0 rows - check params")
    return 0
  con.executemany("""
    INSERT OR IGNORE INTO raw_weather.raw_hourly (period_utc, region, city, latitude, longitude, temperature_2m, relative_humidity_2m, wind_speed_10m, cloud_cover, shortwave_radiation) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", rows)
  logger.info(f"[{region}] inserted {len(rows)} rows")
  return len(rows)


def main():
  end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
  start = end - timedelta(days=365 * 5)
  db_path = "/home/runner/workspace/data/warehouse_dev.duckdb"
  con = duckdb.connect(db_path)
  try:
    for region in LOCATIONS:
      ingest_region(con, region, start, end)
  finally:
    con.close()


if __name__ == "__main__":
  main()