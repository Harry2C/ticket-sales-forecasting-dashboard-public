"""Data loading helpers with CSV and parquet support."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEMO_DATA_DIR = PROJECT_ROOT / "data" / "demo"
SAMPLE_DATA_DIR = PROJECT_ROOT / "data" / "sample"
DEFAULT_DATA_DIR = DEMO_DATA_DIR if (DEMO_DATA_DIR / "matches.csv").exists() else SAMPLE_DATA_DIR

MATCH_DATE_COLUMNS = [
    "event_date",
    "on_sale_date",
    "early_bird_start",
    "early_bird_end",
    "member_presale_start",
    "member_presale_end",
    "general_public_start",
    "marketing_start",
    "marketing_end",
    "campaign_burst_date",
]
DAILY_DATE_COLUMNS = ["date"]


def _read_table(path_without_suffix: Path) -> pd.DataFrame:
    parquet_path = path_without_suffix.with_suffix(".parquet")
    csv_path = path_without_suffix.with_suffix(".csv")

    prefer_parquet = os.getenv("TICKET_DASHBOARD_PREFER_PARQUET", "").lower() in {"1", "true", "yes"}
    if csv_path.exists() and not prefer_parquet:
        return pd.read_csv(csv_path)
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)

    raise FileNotFoundError(
        f"Expected {parquet_path.name} or {csv_path.name} in {path_without_suffix.parent}"
    )


def _parse_dates(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    parsed = df.copy()
    for column in columns:
        if column in parsed.columns:
            parsed[column] = pd.to_datetime(parsed[column])
    return parsed


def load_matches(data_dir: str | Path | None = None) -> pd.DataFrame:
    """Load match metadata from CSV or parquet."""

    root = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    matches = _read_table(root / "matches")
    matches = _parse_dates(matches, MATCH_DATE_COLUMNS)
    matches["match_id"] = matches["match_id"].astype(str)
    return matches.sort_values(["season", "event_date"]).reset_index(drop=True)


def load_daily_sales(data_dir: str | Path | None = None) -> pd.DataFrame:
    """Load daily ticket sales from CSV or parquet."""

    root = Path(data_dir) if data_dir else DEFAULT_DATA_DIR
    daily = _read_table(root / "daily_sales")
    daily = _parse_dates(daily, DAILY_DATE_COLUMNS)
    daily["match_id"] = daily["match_id"].astype(str)
    return daily.sort_values(["match_id", "date"]).reset_index(drop=True)


def load_dataset(data_dir: str | Path | None = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return match metadata and daily sales frames."""

    return load_matches(data_dir), load_daily_sales(data_dir)


def latest_data_date(daily_sales: pd.DataFrame) -> pd.Timestamp:
    """Return the latest date represented in the loaded sales data."""

    if daily_sales.empty:
        return pd.Timestamp.today().normalize()
    return pd.to_datetime(daily_sales["date"]).max().normalize()
