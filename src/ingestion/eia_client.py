"""EIA v2 API client for hourly electricity demand data.

Fetches hourly demand for specified balancing authorities (regions), handles pagination + retries, lands raw records into DuckDB."""

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

EIA_BASE_URL = "https://api.eia.gov/v2/electricity/rto/region-data/data/"
PAGE_SIZE = 5000 # EIA max per request
MAX_RETRIES = 5
BACKOFF_BASE = 2.0 # seconds

def _get_api_key() -> str:
    key = os.environ.get("EIA_API_KEY")
    if not key:
        raise RuntimeError("EIA_API_KEY not set in environment / Replit Secrets")
    return key




def _request_with_retry(params: dict) -> dict: """GET with exponential backoff on transient errors"""
    for attempt in range(MAX_RETRIES):
        try: 
          r = requests.get(EIA_BASE_URL, params=params, timeout=30) 
          r.raise_for_status() 
            return r.json()
        except (requests.RequestException, ValueError) 
            as e: wait = BACKOFF_BASE ** attempt
        
logger.warning(f"Request failed (attempt {attempt + 1}/{MAX_RETRIES}) {e}. Retrying in {wait}s...")
time.sleep(wait)
raise RuntimeError(f"EIA request failed after {MAX_RETRIES} attempts")




def fetch_demand(region: str, start: datetime, end: datetime) -> Iterator[dict]: 
  """Yield hourly demand records for a region between start and end (UTC).EIA returns records with: period (timestamp), respondent (region code), type (D= demand), value(MWh), value-units."""
  api_key = _get_api_key()
  offset = 0
  total_fetched =0
  while True:
    params = {
      "api_key": api_key,
      "frequency": "hourly",
      "data[0]": "value",
      "facets[respondent][]": region,
      "facets[type][]": "D",
      "start": start.strftime("%Y-%m-%dT%H"),
      "end": end.strftime("%Y-%m-%dT%H"),
      "sort[0][column]": "period",
      "sort[0][direction]": "asc",
      "length": PAGE_SIZE,
      "offset": offset,
    }
    payload = _request_with_retry(params)
    response = payload.get("response", {})
    data = response.get("data", [])
    total = int(response.get("total", 0))

    if not data: 
      break
    for record in data:
      yield record

    total_fetched += len(data)
    logger.info(f"[{region}] fetched {total_fetched}/{total}")
    if total_fetched >= total:
      break
    offset += PAGE_SIZE


def ensure_raw_table(con: duckdb.DuckDBPyConnection) -> None:
   """Create raw_eia schema + raw_demand table if missing."""
   con.execute("CREATE SCHEMA IF NOT EXISTS raw_eia")
   con.execute("""
      CREATE TABLE IF NOT EXISTS raw_eia.raw_demand (
        period_utc TIMESTAMP,
        respondent VARCHAR,
        respondent_name VARCHAR,
        type_code VARCHAR,
        type_name VARCHAR,
        value DOUBLE,
        value_units VARCHAR,
        ingested_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        PRIMARY KEY (period_utc, respondent, type_code))
        """
      )


def get_last_timestamp(con: duckdb.DuckDBPyConnection, region: str) -> datetime | None:
   """Return the most recent period_utc for a region, or None if empty."""
   row = con.execute("SELECT MAX(period_utc) FROM raw_eia.raw_demand WHERE respondent = ?", [region],).fetchone()
   return row[0] if row and row[0] else None


def ingest_region(con: duckdb.DuckDBPyConnection, region: str, start: datetime, end: datetime) -> int: 
    """Ingest hourly demand for a region. Resumes from last ingested timestamp."""
    ensure_raw_table(con)
    last_ts = get_last_timestamp(con, region)
    effective_start = (last_ts + timedelta(hours=1)) if last_ts else start
        if effective_start >= end: logger.info(f"[{region}] already up to date through {last_ts}") return 0
        logger.info(f"[{region}] ingesting {effective_start} -> {end}")
        rows = []
        for rec in fetch_demand(region, effective_start, end): rows.append((
          rec["period"],
          rec.get("respondent"),
          rec.get("respondent-name"),
          rec.get("type"),
          rec.get("type-name"),
          float(rec["value"]) if rec.get("value") is not None else None,
          rec.get("value-units"),
        ))
        if not rows: logger.info(f"[{region}] no new records found") return 0
        con.executemany("""INSERT OR IGNORE INTO raw_eia.raw_demand
          (period_utc, respondent, respondent_name, type_code, type_name, value, value_units)
          VALUES (?, ?, ?, ?, ?, ?, ?)""", rows)
        logger.info(f"[{region}] inserted {len(rows)} rows")
        return len(rows)


def main():
  regions = ["SPP", "ERCO"]
  end = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
  start = end - timedelta(days=365 * 5)

  db_path = "/home/runner/workspace/data/warehouse_dev.duckdb"

  os.makedirs(os.path.dirname(db_path), exist_ok=True)
  con = duckdb.connect(db_path)

  try: for region in regions:
    ingest_region(con, region, start, end)
  finally:
    con.close()


if __name__ == "__main__": main()
