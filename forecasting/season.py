"""Season-level aggregation helpers."""

from __future__ import annotations

import pandas as pd


SEASON_VALUE_COLUMNS = [
    "actual_daily",
    "target_daily",
    "required_daily",
    "forecast_expected_daily",
    "forecast_lower_daily",
    "forecast_upper_daily",
    "uplift_10_daily",
    "manual_target_daily",
]


def aggregate_match_outputs(match_outputs: list[dict]) -> pd.DataFrame:
    """Aggregate match-level daily curves into a calendar season curve."""

    rows = []
    for output in match_outputs:
        match_id = output["match_id"]
        target = output["target_curve"][["date", "target_daily", "required_daily"]].copy()
        forecast = output["forecast_curve"][
            [
                "date",
                "forecast_expected_cumulative",
                "forecast_lower_cumulative",
                "forecast_upper_cumulative",
                "uplift_10_cumulative",
            ]
        ].copy()
        manual = output["manual_curve"][["date", "target_daily"]].rename(columns={"target_daily": "manual_target_daily"})
        actual = output["actual_daily"][["date", "daily_sales"]].rename(columns={"daily_sales": "actual_daily"})

        frame = target.merge(forecast, on="date", how="outer").merge(manual, on="date", how="outer").merge(actual, on="date", how="outer")
        frame["match_id"] = match_id
        frame = frame.sort_values("date")
        for cumulative_col, daily_col in [
            ("forecast_expected_cumulative", "forecast_expected_daily"),
            ("forecast_lower_cumulative", "forecast_lower_daily"),
            ("forecast_upper_cumulative", "forecast_upper_daily"),
            ("uplift_10_cumulative", "uplift_10_daily"),
        ]:
            frame[daily_col] = frame[cumulative_col].diff().fillna(frame[cumulative_col]).clip(lower=0)
        rows.append(frame)

    if not rows:
        return pd.DataFrame(columns=["date"])

    combined = pd.concat(rows, ignore_index=True).fillna(0)
    season_daily = combined.groupby("date", as_index=False)[SEASON_VALUE_COLUMNS].sum()
    season_daily = season_daily.sort_values("date").reset_index(drop=True)

    for daily_col, cumulative_col in [
        ("actual_daily", "actual_cumulative"),
        ("target_daily", "target_cumulative"),
        ("required_daily", "required_cumulative"),
        ("forecast_expected_daily", "forecast_expected_cumulative"),
        ("forecast_lower_daily", "forecast_lower_cumulative"),
        ("forecast_upper_daily", "forecast_upper_cumulative"),
        ("uplift_10_daily", "uplift_10_cumulative"),
        ("manual_target_daily", "manual_target_cumulative"),
    ]:
        season_daily[cumulative_col] = season_daily[daily_col].cumsum()

    return season_daily

