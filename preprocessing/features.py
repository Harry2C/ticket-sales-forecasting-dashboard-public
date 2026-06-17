"""Feature engineering for sales pace and machine learning models."""

from __future__ import annotations

import numpy as np
import pandas as pd


CATEGORICAL_FEATURES = [
    "opponent",
    "venue",
    "competition",
    "day_of_week",
    "event_category",
    "campaign_theme",
    "sales_window",
    "marketing_period",
]

NUMERIC_FEATURES = [
    "season",
    "round",
    "days_to_event",
    "days_on_sale",
    "sales_progress",
    "cumulative_sales",
    "daily_sales",
    "trailing_3_day_sales",
    "trailing_7_day_sales",
    "sales_acceleration",
    "opponent_strength",
    "historical_attendance",
    "ticket_price",
    "membership_base",
    "prior_season_performance",
    "weather_temp_c",
    "is_public_holiday",
    "is_campaign_burst",
    "membership_on_sale",
    "early_bird_active",
    "finals_contention",
    "school_holiday",
    "double_header",
    "event_month",
    "is_weekend_event",
    "days_since_early_bird_start",
]


def enrich_sales_frame(matches: pd.DataFrame, daily_sales: pd.DataFrame) -> pd.DataFrame:
    """Join match attributes to daily sales and create reusable modelling features."""

    match_cols = [
        "match_id",
        "season",
        "round",
        "opponent",
        "venue",
        "competition",
        "event_date",
        "on_sale_date",
        "early_bird_start",
        "early_bird_end",
        "member_presale_start",
        "member_presale_end",
        "general_public_start",
        "day_of_week",
        "event_category",
        "campaign_theme",
        "opponent_strength",
        "historical_attendance",
        "ticket_price",
        "membership_base",
        "prior_season_performance",
        "weather_temp_c",
        "finals_contention",
        "school_holiday",
        "double_header",
    ]
    frame = daily_sales.merge(matches[match_cols], on="match_id", how="left")

    frame["date"] = pd.to_datetime(frame["date"])
    frame["event_date"] = pd.to_datetime(frame["event_date"])
    frame["on_sale_date"] = pd.to_datetime(frame["on_sale_date"])
    frame["early_bird_start"] = pd.to_datetime(frame["early_bird_start"])
    frame = frame.sort_values(["match_id", "date"]).reset_index(drop=True)

    frame["days_on_sale"] = (frame["date"] - frame["on_sale_date"]).dt.days.clip(lower=0)
    frame["days_since_early_bird_start"] = (frame["date"] - frame["early_bird_start"]).dt.days
    sale_length = (frame["event_date"] - frame["on_sale_date"]).dt.days.replace(0, 1)
    frame["sales_progress"] = (frame["days_on_sale"] / sale_length).clip(0, 1)
    frame["event_month"] = frame["event_date"].dt.month
    frame["is_weekend_event"] = frame["day_of_week"].isin(["Saturday", "Sunday"]).astype(int)

    grouped = frame.groupby("match_id", group_keys=False)
    frame["trailing_3_day_sales"] = grouped["daily_sales"].rolling(3, min_periods=1).mean().reset_index(level=0, drop=True)
    frame["trailing_7_day_sales"] = grouped["daily_sales"].rolling(7, min_periods=1).mean().reset_index(level=0, drop=True)
    frame["sales_acceleration"] = grouped["trailing_3_day_sales"].diff().fillna(0)
    frame["final_sales"] = grouped["cumulative_sales"].transform("max")
    frame["cumulative_share_of_final"] = np.where(
        frame["final_sales"] > 0,
        frame["cumulative_sales"] / frame["final_sales"],
        0,
    )

    for column in CATEGORICAL_FEATURES:
        if column in frame.columns:
            frame[column] = frame[column].fillna("Unknown").astype(str)

    for column in NUMERIC_FEATURES:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0)

    return frame


def completed_match_ids(
    matches: pd.DataFrame,
    daily_sales: pd.DataFrame,
    reference_date: pd.Timestamp | None = None,
) -> set[str]:
    """Identify matches with complete sales histories available for training."""

    if reference_date is None:
        reference_date = pd.to_datetime(daily_sales["date"]).max()

    event_completed = set(
        matches.loc[pd.to_datetime(matches["event_date"]) <= reference_date, "match_id"].astype(str)
    )
    event_day_seen = set(
        daily_sales.loc[daily_sales["days_to_event"].eq(0), "match_id"].astype(str)
    )
    return event_completed | event_day_seen


def build_training_snapshots(
    matches: pd.DataFrame,
    daily_sales: pd.DataFrame,
    reference_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Create daily match snapshots that predict eventual final sales."""

    enriched = enrich_sales_frame(matches, daily_sales)
    complete_ids = completed_match_ids(matches, daily_sales, reference_date)
    training = enriched[enriched["match_id"].isin(complete_ids)].copy()
    training = training[training["final_sales"] > 0].reset_index(drop=True)
    return training
