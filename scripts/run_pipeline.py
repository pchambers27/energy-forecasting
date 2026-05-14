"""End-to-end pipeline runner for energy forecasting project."""

from __future__ import annotations

import os
import sys
import subprocess
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path("/home/runner/workspace")
DBT_PROJECT = PROJECT_ROOT / "dbt" / "energy_forecasting"

def run_step(name: str, cmd: list[str], cwd: Path | None = None) -> None:
  """Run a shell command and log output."""
  logger.info(f"Running {name}...")
  result = subprocess.run(cmd, cwd=cwd)
  if result.returncode !=0:
    logger.error(f"{name} failed with exit code {result.returncode}")
    sys.exit(result.returncode)
  logger.info(f"{name} completed successfully")


def main():
  # Make sure dbt finds the project-local profiles.yml
      os.environ["DBT_PROFILES_DIR"] = str(DBT_PROJECT)

      run_step(
          "Ingest EIA demand",
          [sys.executable, "-m", "src.ingestion.eia_client"],
          cwd=PROJECT_ROOT,
      )
      run_step(
          "Ingest weather",
          [sys.executable, "-m", "src.ingestion.weather_client"],
          cwd=PROJECT_ROOT,
      )
      run_step(
          "dbt run",
          ["dbt", "run"],
          cwd=DBT_PROJECT,
      )
      run_step(
          "dbt test",
          ["dbt", "test"],
          cwd=DBT_PROJECT,
      )
      logger.info("✓ Pipeline completed successfully")


if __name__ == "__main__":
    main()
