"""Target curve generation from historical pace and planning assumptions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TargetAssumptions:
    early_bird_lift_pct: float = 0.16
    member_presale_lift_pct: float = 0.12
    marketing_lift_pct: float = 0.18
    campaign_burst_lift_pct: float = 0.28
    final_week_lift_pct: float = 0.20


def generate_target_curve(
    match: pd.Series | dict,
    pace_profile: pd.DataFrame,
    total_sales_target: float,
    uplift_pct: float = 0.0,
    manual_target: float | None = None,
    assumptions: TargetAssumptions | None = None,
) -> pd.DataFrame:
    """Create a daily and cumulative target curve for a match."""

    assumptions = assumptions or TargetAssumptions()
    match_dict = dict(match)
    on_sale_date = pd.to_datetime(match_dict["on_sale_date"]).normalize()
    event_date = pd.to_datetime(match_dict["event_date"]).normalize()
    if pd.isna(on_sale_date) or pd.isna(event_date):
        return _flat_target_curve(pd.Timestamp.today().normalize(), float(total_sales_target), uplift_pct, manual_target)
    if on_sale_date > event_date:
        on_sale_date = event_date
    dates = pd.date_range(on_sale_date, event_date, freq="D")

    target_total = float(manual_target) if manual_target else float(total_sales_target) * (1 + uplift_pct)
    target_total = max(target_total, 0)

    curve = pd.DataFrame({"date": dates})
    curve["days_to_event"] = (event_date - curve["date"]).dt.days
    curve = curve.merge(
        pace_profile[["days_to_event", "historical_daily_share"]],
        on="days_to_event",
        how="left",
    )
    curve["historical_daily_share"] = curve["historical_daily_share"].fillna(0)

    curve["weight"] = curve["historical_daily_share"].rolling(window=5, center=True, min_periods=1).mean().clip(lower=0.000001)
    if "early_bird_start" in match_dict and "early_bird_end" in match_dict:
        early_mask = curve["date"].between(match_dict["early_bird_start"], match_dict["early_bird_end"])
        if early_mask.any():
            progress = np.linspace(0.35, 1.0, int(early_mask.sum()))
            curve.loc[early_mask, "weight"] *= 1 + assumptions.early_bird_lift_pct * progress
    if "member_presale_start" in match_dict and "member_presale_end" in match_dict:
        presale_mask = curve["date"].between(match_dict["member_presale_start"], match_dict["member_presale_end"])
        if presale_mask.any():
            progress = np.linspace(0.4, 1.0, int(presale_mask.sum()))
            curve.loc[presale_mask, "weight"] *= 1 + assumptions.member_presale_lift_pct * progress
    marketing_mask = curve["date"].between(match_dict["marketing_start"], match_dict["marketing_end"])
    if marketing_mask.any():
        progress = np.linspace(0.45, 1.0, int(marketing_mask.sum()))
        curve.loc[marketing_mask, "weight"] *= 1 + assumptions.marketing_lift_pct * progress
    burst_date = pd.to_datetime(match_dict["campaign_burst_date"]).normalize()
    curve.loc[(curve["date"] - burst_date).abs().dt.days <= 2, "weight"] *= (
        1 + assumptions.campaign_burst_lift_pct
    )
    final_week_mask = curve["days_to_event"].between(0, 7)
    if final_week_mask.any():
        ramp = (8 - curve.loc[final_week_mask, "days_to_event"]) / 8
        curve.loc[final_week_mask, "weight"] *= 1 + assumptions.final_week_lift_pct * ramp
    curve.loc[curve["days_to_event"].eq(1), "weight"] *= 1.12
    curve.loc[curve["days_to_event"].eq(0), "weight"] *= 1.24

    curve["adjusted_daily_share"] = curve["weight"] / curve["weight"].sum()
    curve["target_daily"] = curve["adjusted_daily_share"] * target_total
    curve["target_cumulative"] = curve["target_daily"].cumsum()
    curve["target_total"] = target_total
    return curve.drop(columns=["weight"])


def _flat_target_curve(
    date: pd.Timestamp,
    total_sales_target: float,
    uplift_pct: float,
    manual_target: float | None,
) -> pd.DataFrame:
    target_total = float(manual_target) if manual_target else float(total_sales_target) * (1 + uplift_pct)
    target_total = max(target_total, 0)
    return pd.DataFrame(
        {
            "date": [date],
            "days_to_event": [0],
            "historical_daily_share": [1.0],
            "adjusted_daily_share": [1.0],
            "target_daily": [target_total],
            "target_cumulative": [target_total],
            "target_total": [target_total],
        }
    )


def add_required_daily_sales(target_curve: pd.DataFrame, actual_daily: pd.DataFrame) -> pd.DataFrame:
    """Calculate required future sales to finish on target from the latest actual value."""

    curve = target_curve.copy()
    curve["required_daily"] = curve["target_daily"]

    if actual_daily.empty:
        return curve

    actual = actual_daily.sort_values("date")
    latest_date = pd.to_datetime(actual["date"]).max()
    latest_cumulative = float(actual.loc[actual["date"].eq(latest_date), "cumulative_sales"].max())
    target_total = float(curve["target_total"].iloc[-1])
    remaining_target = max(target_total - latest_cumulative, 0)
    future_mask = curve["date"] > latest_date

    if future_mask.any():
        future_weights = curve.loc[future_mask, "adjusted_daily_share"]
        if future_weights.sum() == 0:
            normalized = np.ones(future_weights.shape[0]) / future_weights.shape[0]
        else:
            normalized = future_weights / future_weights.sum()
        curve.loc[future_mask, "required_daily"] = normalized * remaining_target
        curve.loc[~future_mask, "required_daily"] = 0

    return curve
