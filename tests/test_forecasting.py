from __future__ import annotations

import pandas as pd

from dashboard.config import PLANNING_SEASON_LABEL
from dashboard.services import build_season_comparison_frame, history_filters, season_matches
from forecasting.historical_pace import HistoricalPaceEngine
from forecasting.ml_model import TicketSalesForecaster, build_forecast_curve
from forecasting.season import aggregate_match_outputs
from forecasting.targets import TargetAssumptions, add_required_daily_sales, generate_target_curve
from utils.data_loader import load_dataset


def _sample_context():
    if hasattr(_sample_context, "_cached"):
        return _sample_context._cached

    matches, daily = load_dataset()
    reference_date = pd.Timestamp("2026-05-29")
    engine = HistoricalPaceEngine(max_window_days=120).fit(matches, daily)
    model = TicketSalesForecaster().fit(matches, daily, reference_date)
    match = matches[matches["season"].eq(2026)].sort_values("event_date").iloc[4]
    match_daily = daily[daily["match_id"].eq(match["match_id"])]
    pace = engine.get_profile(match).frame
    _sample_context._cached = matches, daily, engine, model, match, match_daily, pace
    return _sample_context._cached


def test_historical_pace_profile_is_normalized():
    _, _, _, _, _, _, pace = _sample_context()

    assert pace["historical_daily_share"].sum() == pytest_approx(1.0)
    assert pace["historical_cumulative_share"].iloc[-1] == pytest_approx(1.0)
    assert pace["historical_cumulative_share"].is_monotonic_increasing


def test_target_curve_reaches_requested_total():
    _, _, _, _, match, match_daily, pace = _sample_context()

    curve = generate_target_curve(
        match,
        pace,
        total_sales_target=30_000,
        uplift_pct=0.10,
        assumptions=TargetAssumptions(),
    )
    curve = add_required_daily_sales(curve, match_daily)

    assert round(curve["target_cumulative"].iloc[-1]) == 33_000
    assert "required_daily" in curve.columns


def test_forecaster_returns_non_decreasing_prediction_curve():
    matches, daily, _, model, match, match_daily, pace = _sample_context()

    snapshot = model.latest_snapshot(matches, daily, match["match_id"])
    forecast = model.predict_snapshot(snapshot)
    curve = build_forecast_curve(match, match_daily, pace, forecast)

    assert forecast.upper_final_sales >= forecast.expected_final_sales >= forecast.lower_final_sales
    assert curve["forecast_expected_cumulative"].is_monotonic_increasing
    assert curve["uplift_10_cumulative"].iloc[-1] > curve["forecast_expected_cumulative"].iloc[-1]


def test_season_aggregation_changes_with_home_match_count():
    matches, daily, engine, model, _, _, _ = _sample_context()
    assumptions = TargetAssumptions()

    outputs = []
    for _, match in matches[matches["season"].eq(2026)].sort_values("event_date").head(3).iterrows():
        match_daily = daily[daily["match_id"].eq(match["match_id"])]
        pace = engine.get_profile(match).frame
        forecast = model.predict_snapshot(model.latest_snapshot(matches, daily, match["match_id"]))
        forecast_curve = build_forecast_curve(match, match_daily, pace, forecast)
        target_curve = add_required_daily_sales(
            generate_target_curve(match, pace, match["baseline_target"], 0.05, assumptions=assumptions),
            match_daily,
        )
        manual_curve = generate_target_curve(match, pace, match["baseline_target"], manual_target=match["manual_target"])
        outputs.append(
            {
                "match_id": match["match_id"],
                "actual_daily": match_daily,
                "forecast_curve": forecast_curve,
                "target_curve": target_curve,
                "manual_curve": manual_curve,
            }
        )

    first_two = aggregate_match_outputs(outputs[:2])
    first_three = aggregate_match_outputs(outputs)

    assert first_three["target_cumulative"].iloc[-1] > first_two["target_cumulative"].iloc[-1]


def test_sample_data_is_strikers_specific_and_competition_scoped():
    matches, daily, *_ = _sample_context()

    assert set(matches["competition"].unique()) == {"BBL", "WBBL"}
    assert PLANNING_SEASON_LABEL in set(matches["season_label"].unique())
    assert matches["planning_season"].sum() == 10
    assert set(daily["sales_window"].unique()) >= {
        "early-bird",
        "member-pre-sale",
        "general-public",
        "campaign-live",
        "final-week",
    }


def test_historical_filters_and_season_comparison_use_selected_competition():
    matches, daily, engine, _, _, _, _ = _sample_context()
    planning_bbl = season_matches(matches, PLANNING_SEASON_LABEL, "BBL")
    selected_match = planning_bbl.iloc[0]
    filters = history_filters("BBL", ["2024/25", "2025/26"])

    pace = engine.get_profile(selected_match, filters=filters)
    comparables = engine.comparable_matches(selected_match, filters=filters)
    comparison_frame = build_season_comparison_frame(
        matches,
        daily,
        "BBL",
        [PLANNING_SEASON_LABEL, "2024/25", "2025/26"],
    )

    assert pace.sample_size >= 2
    assert comparables["competition"].eq("BBL").all()
    assert set(comparison_frame["season_label"].unique()) == {PLANNING_SEASON_LABEL, "2024/25", "2025/26"}


def test_target_curve_smooths_towards_campaign_end_and_matchday():
    match = {
        "on_sale_date": pd.Timestamp("2026-08-01"),
        "event_date": pd.Timestamp("2026-08-20"),
        "early_bird_start": pd.Timestamp("2026-08-01"),
        "early_bird_end": pd.Timestamp("2026-08-05"),
        "member_presale_start": pd.Timestamp("2026-08-06"),
        "member_presale_end": pd.Timestamp("2026-08-08"),
        "marketing_start": pd.Timestamp("2026-08-10"),
        "marketing_end": pd.Timestamp("2026-08-17"),
        "campaign_burst_date": pd.Timestamp("2026-08-15"),
    }
    pace = pd.DataFrame(
        {
            "days_to_event": list(range(0, 20)),
            "historical_daily_share": [1 / 20] * 20,
        }
    )

    curve = generate_target_curve(match, pace, total_sales_target=1_000)
    campaign_start = float(curve.loc[curve["date"].eq(pd.Timestamp("2026-08-10")), "target_daily"].iloc[0])
    campaign_end = float(curve.loc[curve["date"].eq(pd.Timestamp("2026-08-17")), "target_daily"].iloc[0])
    match_day = float(curve.loc[curve["days_to_event"].eq(0), "target_daily"].iloc[0])
    early_sales = float(curve.loc[curve["date"].eq(pd.Timestamp("2026-08-02")), "target_daily"].iloc[0])

    assert campaign_end > campaign_start
    assert match_day > early_sales
    assert round(curve["target_daily"].sum()) == 1_000


def pytest_approx(value: float):
    import pytest

    return pytest.approx(value, abs=1e-6)
