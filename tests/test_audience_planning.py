from __future__ import annotations

import pandas as pd

from dashboard.app import (
    NAV_SECTIONS,
    PAGES,
    _apply_demographic_filters,
    _aligned_historical_curve,
    _audience_actual_curve,
    _audience_tracking_chart_data,
    _default_page_for_section,
    _early_onsale_target_segments,
    _enrich_transactions_with_demographics,
    _generate_dummy_2627_sales,
    _load_fixture_targets,
    _page_section,
    _save_fixture_targets,
    _segment_level_projection,
    _should_add_dummy_2627_sales,
    _target_breakdown_last_season_curve,
    _targets_from_editor_frame,
    _transaction_sales_curve,
    _visible_projection_columns,
    csv_bytes,
    validate_fixture_target_frame,
    validate_fixture_targets,
)
from dashboard.services import (
    expected_purchase_index,
    page_state_key,
    report_html,
    segment_status,
    stale_snapshot_status,
)
from preprocessing.strikers_ingestion import StrikersDataBundle, load_demo_strikers_data


def _ticket_rows() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "season_label": "2025/26",
                "competition": "BBL",
                "transaction_date": "2025-10-01",
                "match_date": "2025-12-31",
                "customer_id": "hist-1",
                "order_id": "h1",
                "paid_tickets_sold": 100,
                "comp_tickets_sold": 0,
                "tickets_sold": 100,
                "gross_revenue": 5000,
                "price_type": "Adult",
                "ticket_type": "Public",
                "ticket_class": "Gold",
                "sales_window": "general-sale",
                "customer_purchase_status": "Returning",
                "purchaser_family_flag": False,
                "marketing_opt_in": True,
                "is_comp": False,
                "is_refund": False,
            },
            {
                "season_label": "2026/27",
                "competition": "BBL",
                "transaction_date": "2026-10-01",
                "match_date": "2026-12-31",
                "customer_id": "cur-1",
                "order_id": "c1",
                "paid_tickets_sold": 80,
                "comp_tickets_sold": 0,
                "tickets_sold": 80,
                "gross_revenue": 4200,
                "price_type": "Adult",
                "ticket_type": "Public",
                "ticket_class": "Gold",
                "sales_window": "general-sale",
                "customer_purchase_status": "Returning",
                "purchaser_family_flag": False,
                "marketing_opt_in": True,
                "is_comp": False,
                "is_refund": False,
            },
        ]
    )


def _customers() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "customer_id": ["hist-1", "cur-1"],
            "seasons_purchased_count": [2, 2],
            "lifetime_tickets": [12, 10],
            "lifetime_revenue": [6000, 5200],
            "preferred_ticket_class": ["Gold", "Gold"],
            "preferred_ticket_type": ["Public", "Public"],
            "usual_purchase_window": ["general-sale", "general-sale"],
            "marketing_opt_in": [True, True],
            "email_opt_in": [False, False],
            "sms_opt_in": [False, False],
            "family_flag": [False, False],
        }
    )


def _matches() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "season_label": ["2026/27"],
            "competition": ["BBL"],
            "event_date": pd.to_datetime(["2026-12-31"]),
        }
    )


def test_page_state_keys_are_page_scoped():
    assert page_state_key("historical_overview", "season") == "historical_overview_season"
    assert page_state_key("target_breakdown", "competition") == "target_breakdown_competition"
    assert PAGES[0] == "Fixture Forecasting"
    assert "Target Breakdown" in PAGES
    assert NAV_SECTIONS["Pre On Sale"] == ["Fixture Forecasting", "Target Breakdown", "Historic Sales"]
    assert _page_section("Audience Insights") == "In Season Sales"
    assert _default_page_for_section("Data & Admin") == "Data Admin"
    assert "Target Builder" not in PAGES
    assert "Historical Overview / Data QA" not in PAGES
    assert page_state_key("audience_planner", "season") != page_state_key("target_breakdown", "competition")


def test_expected_purchase_index_uses_same_point_in_cycle_for_ticket_class():
    frame = expected_purchase_index(
        _ticket_rows(),
        _customers(),
        _matches(),
        "2026/27",
        competition="BBL",
        as_at_date=pd.Timestamp("2026-10-01"),
        group_kind="ticket_class",
        baseline_seasons=["2025/26"],
        min_size=0,
    )
    row = frame[frame["analysis_group"].eq("Gold")].iloc[0]

    assert row["expected_paid_tickets_by_now"] == 100
    assert row["current_paid_tickets"] == 80
    assert round(row["ticket_index"], 2) == 0.80
    assert row["status"] == "Behind"


def test_expected_purchase_index_works_for_ticket_type():
    frame = expected_purchase_index(
        _ticket_rows(),
        _customers(),
        _matches(),
        "2026/27",
        competition="BBL",
        as_at_date=pd.Timestamp("2026-10-01"),
        group_kind="ticket_type",
        baseline_seasons=["2025/26"],
        min_size=0,
    )

    assert "Adult" in set(frame["analysis_group"])
    assert frame.loc[frame["analysis_group"].eq("Adult"), "paid_ticket_gap"].iloc[0] == -20


def test_not_due_yet_logic_protects_late_buyers():
    status = segment_status(
        index=0,
        actual_paid=0,
        expected_paid=0,
        expected_final_paid=1000,
        current_window="early-sales",
        usual_purchase_window="match-week",
    )

    assert status == "Not due yet"


def test_stale_snapshot_warning_and_report_index_metrics():
    freshness = stale_snapshot_status(pd.Timestamp("2026-01-01"), today=pd.Timestamp("2026-02-01"))
    assert freshness["is_stale"]

    html = report_html(
        "Weekly current-season update",
        {"paid_tickets_sold": 80, "comps_issued": 0, "gross_revenue": 4200, "forecast_total": 100, "gap_to_target": -20},
        pd.DataFrame(),
        ["Gold is behind expected paid-ticket volume."],
        "Assumed fixtures included.",
        ticket_class_index=pd.DataFrame(
            {
                "analysis_group": ["Gold"],
                "status": ["Behind"],
                "current_paid_tickets": [80],
                "expected_paid_tickets_by_now": [100],
                "ticket_index": [0.8],
                "paid_ticket_gap": [-20],
                "recommended_action": ["Prioritise this group."],
            }
        ),
        snapshot_note="Current uploaded snapshot latest transaction date: 01 Jan 2026.",
    )

    assert "Ticket class performance" in html
    assert "Current uploaded snapshot" in html


def test_historic_sales_curve_can_switch_between_paid_and_comps():
    rows = _ticket_rows()
    rows.loc[0, "comp_tickets_sold"] = 12
    rows.loc[0, "tickets_sold"] = 112

    paid_curve = _transaction_sales_curve(rows, "Paid tickets only")
    comp_curve = _transaction_sales_curve(rows, "Comps only")

    assert paid_curve["actual_cumulative"].iloc[-1] == 180
    assert comp_curve["actual_cumulative"].iloc[-1] == 12


def test_early_onsale_target_segments_use_august_first_month():
    rows = _ticket_rows()
    rows.loc[0, "transaction_date"] = "2025-08-10"
    rows.loc[0, "ticket_class"] = "Boundary Zone"

    segments = _early_onsale_target_segments(
        rows,
        _customers(),
        "2025/26",
        "BBL",
        "ticket_class",
        min_size=10,
    )

    assert segments.iloc[0]["segment"] == "Boundary Zone"
    assert segments.iloc[0]["early_paid_tickets"] == 100
    assert "family-value" in segments.iloc[0]["recommendation"]


def test_csv_export_returns_underlying_data_bytes():
    frame = pd.DataFrame({"date": ["2026-08-01"], "tickets": [123]})
    output = csv_bytes(frame)

    assert output.decode("utf-8").startswith("date,tickets")
    assert "2026-08-01,123" in output.decode("utf-8")


def test_gigyauid_demographic_enrichment_retains_unmatched_rows_and_filters():
    transactions = pd.DataFrame(
        {
            "customer_id": [" GIGYA-001 ", "unknown-customer", None],
            "purchaser_age_band": [None, None, None],
            "purchaser_gender": [None, None, None],
            "purchaser_postcode": [None, None, None],
            "paid_tickets_sold": [2, 3, 4],
        }
    )
    customers = pd.DataFrame(
        {
            "customer_id": ["gigya-001"],
            "age_band": ["18-30"],
            "gender": ["Female"],
            "postcode": ["5000"],
            "marketing_opt_in": [True],
        }
    )

    enriched, metrics = _enrich_transactions_with_demographics(transactions, customers)
    filtered = _apply_demographic_filters(enriched, ["18-30"], ["Female"], ["5000"])

    assert len(enriched) == 3
    assert metrics["ticketing_rows_with_gigyauid"] == 2
    assert metrics["demographic_matched_ticket_rows"] == 1
    assert round(metrics["demographic_match_rate"], 2) == 0.50
    assert set(enriched["age_band_filter"]) == {"18-30", "Unknown / unmatched"}
    assert filtered["paid_tickets_sold"].sum() == 2


def test_fixture_target_frame_validates_and_saves_all_targets():
    draft = pd.DataFrame(
        {
            "fixture_id": ["BBL_2026_ASSUMED_01", "BBL_2026_NYE_ASSUMED"],
            "fixture": ["Opening fixture", "NYE"],
            "base_target": ["10000", "12,500"],
            "stretch_target": ["11000", "14000"],
        }
    )

    validation = validate_fixture_target_frame(draft)
    targets = _targets_from_editor_frame(draft)

    assert validation.empty
    assert targets["BBL_2026_ASSUMED_01"]["base_target"] == 10000
    assert targets["BBL_2026_NYE_ASSUMED"]["stretch_target"] == 14000

    invalid = draft.copy()
    invalid.loc[0, "stretch_target"] = "9000"

    assert "Stretch target" in validate_fixture_target_frame(invalid).iloc[0]["issue"]


def test_target_breakdown_projection_columns_are_planning_only():
    chart = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-08-01"]),
            "forecast_expected_cumulative": [100],
            "forecast_plus_10_cumulative": [110],
            "base_cumulative": [105],
            "stretch_cumulative": [125],
            "actual_cumulative": [50],
        }
    )

    selected = _visible_projection_columns(chart, ["Forecast", "Forecast +10%", "Base", "Stretch"], suffix="cumulative")

    assert selected == ["forecast_expected_cumulative", "forecast_plus_10_cumulative", "stretch_cumulative", "base_cumulative"]
    assert "actual_cumulative" not in selected


def test_audience_actual_curve_stops_at_last_observed_sales_day():
    transactions = pd.DataFrame(
        {
            "transaction_date": pd.to_datetime(["2026-08-03", "2026-08-05"]),
            "paid_tickets_sold": [10, 15],
            "comp_tickets_sold": [0, 0],
            "tickets_sold": [10, 15],
        }
    )
    controls = {"ticket_status": "Paid tickets only"}

    curve = _audience_actual_curve(transactions, controls, max_offset=30)

    assert curve["date"].min() == pd.Timestamp("2026-08-03")
    assert curve["date"].max() == pd.Timestamp("2026-08-05")
    assert len(curve) == 3
    assert curve["actual_daily"].tolist() == [10, 0, 15]


def test_audience_tracking_chart_data_keeps_three_comparison_lines_and_uses_actual_window():
    actual = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-08-03", "2026-08-04"]),
            "actual_daily": [10, 0],
            "actual_cumulative": [10, 10],
        }
    )
    historical = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-08-01", "2026-08-03", "2026-08-04"]),
            "historical_daily": [4, 8, 6],
            "historical_cumulative": [4, 12, 18],
        }
    )
    target = pd.DataFrame(
        {
            "date": pd.to_datetime(["2026-08-01", "2026-08-03", "2026-08-04"]),
            "target_daily": [5, 12, 14],
            "target_cumulative": [5, 17, 31],
        }
    )

    chart = _audience_tracking_chart_data(actual, historical, target, max_offset=2)

    assert {"actual_cumulative", "historical_cumulative", "target_cumulative"}.issubset(chart.columns)
    assert chart["actual_cumulative"].iloc[0] == 10
    assert chart["date"].tolist() == [pd.Timestamp("2026-08-03"), pd.Timestamp("2026-08-04")]
    assert len(chart) == 2


def test_aligned_historical_curve_respects_comp_filter():
    history = pd.DataFrame(
        {
            "season_label": ["2025/26"],
            "transaction_date": ["2025-08-01"],
            "paid_tickets_sold": [0],
            "comp_tickets_sold": [42],
            "tickets_sold": [42],
        }
    )

    curve = _aligned_historical_curve(history, "Comps only", max_offset=2, blend=True)

    assert curve["historical_cumulative"].iloc[-1] == 42

def test_fixture_targets_are_validated_and_persisted(tmp_path):
    path = tmp_path / "fixture_targets.csv"
    targets = {"BBL_2026_ASSUMED_01": {"base_target": 10000, "stretch_target": 11200}}

    _save_fixture_targets(targets, path)
    loaded = _load_fixture_targets(path)

    assert loaded == targets
    assert validate_fixture_targets(100, 120) == []
    assert "Stretch target should be greater" in " ".join(validate_fixture_targets(120, 100))
    assert "Base target must be numeric" in " ".join(validate_fixture_targets(None, 100))


def test_dummy_2627_sales_cover_august_and_include_demo_demographics():
    bundle = load_demo_strikers_data()
    dummy_transactions, dummy_customers = _generate_dummy_2627_sales(bundle)

    dates = pd.to_datetime(dummy_transactions["transaction_date"])
    assert dates.min() == pd.Timestamp("2026-08-01")
    assert dates.max() == pd.Timestamp("2026-08-31")
    assert dummy_transactions["season_label"].eq("2026/27").all()
    assert dummy_transactions["paid_tickets_sold"].sum() > 0
    assert dummy_transactions["comp_tickets_sold"].sum() > 0
    assert {"Adult", "Junior", "Family", "Concession"}.issubset(set(dummy_transactions["ticket_type"]))
    assert dummy_transactions["customer_id"].notna().any()
    assert dummy_customers["customer_id"].isin(dummy_transactions["customer_id"].dropna()).any()
    assert _should_add_dummy_2627_sales(pd.DataFrame({"season_label": ["2025/26"]}))


def test_target_breakdown_segment_projection_changes_with_filters():
    transactions = pd.DataFrame(
        [
            {
                "season_label": "2025/26",
                "competition": "BBL",
                "fixture_id": "HIST",
                "transaction_date": "2025-08-01",
                "match_date": "2025-12-31",
                "ticket_type": "Family",
                "ticket_class": "Boundary Zone",
                "paid_tickets_sold": 1000,
                "comp_tickets_sold": 0,
                "tickets_sold": 1000,
                "gross_revenue": 30000,
                "customer_id": "hist-family",
                "order_id": "hf",
                "sales_window": "early-sales",
                "is_comp": False,
                "is_refund": False,
                "purchaser_family_flag": True,
                "marketing_opt_in": True,
                "customer_purchase_status": "Returning",
                "age_band_filter": "18-30",
                "gender_filter": "Female",
                "postcode_filter": "5000",
            },
            {
                "season_label": "2025/26",
                "competition": "BBL",
                "fixture_id": "HIST",
                "transaction_date": "2025-08-02",
                "match_date": "2025-12-31",
                "ticket_type": "Adult",
                "ticket_class": "Gold",
                "paid_tickets_sold": 250,
                "comp_tickets_sold": 0,
                "tickets_sold": 250,
                "gross_revenue": 15000,
                "customer_id": "hist-adult",
                "order_id": "ha",
                "sales_window": "early-sales",
                "is_comp": False,
                "is_refund": False,
                "purchaser_family_flag": False,
                "marketing_opt_in": True,
                "customer_purchase_status": "Returning",
                "age_band_filter": "31-45",
                "gender_filter": "Male",
                "postcode_filter": "5006",
            },
            {
                "season_label": "2026/27",
                "competition": "BBL",
                "fixture_id": "M1",
                "transaction_date": "2026-08-01",
                "match_date": "2026-12-31",
                "ticket_type": "Family",
                "ticket_class": "Boundary Zone",
                "paid_tickets_sold": 80,
                "comp_tickets_sold": 0,
                "tickets_sold": 80,
                "gross_revenue": 2400,
                "customer_id": "cur-family",
                "order_id": "cf",
                "sales_window": "early-sales",
                "is_comp": False,
                "is_refund": False,
                "purchaser_family_flag": True,
                "marketing_opt_in": True,
                "customer_purchase_status": "Returning",
                "age_band_filter": "18-30",
                "gender_filter": "Female",
                "postcode_filter": "5000",
            },
        ]
    )
    bundle = StrikersDataBundle(
        data_mode="test",
        matches=pd.DataFrame(),
        daily_sales=pd.DataFrame(),
        transactions=transactions,
        fixtures=pd.DataFrame(),
        customers=pd.DataFrame(),
        future_fixtures=pd.DataFrame(),
        file_status=pd.DataFrame(),
        column_mappings=pd.DataFrame(),
        validation_warnings=pd.DataFrame(),
        metrics={},
    )
    selected_output = {
        "match_id": "M1",
        "match": pd.Series({"event_date": pd.Timestamp("2026-12-31"), "competition": "BBL"}),
        "match_chart": pd.DataFrame(
            {
                "date": pd.date_range("2026-08-01", periods=3),
                "forecast_expected_daily": [100, 90, 80],
                "forecast_expected_cumulative": [100, 190, 270],
            }
        ),
    }
    base_controls = {
        "season_label": "2026/27",
        "competition": "BBL",
        "as_at_date": pd.Timestamp("2026-08-31"),
        "fixture_ids": [],
        "audience_segment": "All",
        "ticket_class": "All",
        "sales_window": "All",
        "ticket_status": "Paid + comps",
        "age_band": [],
        "gender": [],
        "postcode": [],
        "history_filters": {"season_label": ["2025/26"]},
    }

    family_projection = _segment_level_projection(bundle, {**base_controls, "ticket_type": "Family"}, selected_output, 2000)
    adult_projection = _segment_level_projection(bundle, {**base_controls, "ticket_type": "Adult"}, selected_output, 2000)

    assert family_projection["forecast_final_tickets"] != adult_projection["forecast_final_tickets"]
    assert family_projection["curve"]["forecast_expected_daily"].sum() > adult_projection["curve"]["forecast_expected_daily"].sum()


def test_target_breakdown_can_build_last_season_comparison_for_season_total():
    transactions = pd.DataFrame(
        [
            {
                "season_label": "2025/26",
                "competition": "BBL",
                "fixture_id": "HIST-1",
                "transaction_date": "2025-08-01",
                "match_date": "2025-12-31",
                "ticket_type": "Family",
                "ticket_class": "Boundary Zone",
                "paid_tickets_sold": 30,
                "comp_tickets_sold": 0,
                "tickets_sold": 30,
                "gross_revenue": 900,
                "customer_id": "hist-1",
                "order_id": "hist-1",
                "sales_window": "early-sales",
                "is_comp": False,
                "is_refund": False,
                "purchaser_family_flag": True,
                "marketing_opt_in": True,
                "customer_purchase_status": "Returning",
                "age_band_filter": "18-30",
                "gender_filter": "Female",
                "postcode_filter": "5000",
            },
            {
                "season_label": "2025/26",
                "competition": "BBL",
                "fixture_id": "HIST-1",
                "transaction_date": "2025-08-02",
                "match_date": "2025-12-31",
                "ticket_type": "Family",
                "ticket_class": "Boundary Zone",
                "paid_tickets_sold": 45,
                "comp_tickets_sold": 0,
                "tickets_sold": 45,
                "gross_revenue": 1350,
                "customer_id": "hist-2",
                "order_id": "hist-2",
                "sales_window": "early-sales",
                "is_comp": False,
                "is_refund": False,
                "purchaser_family_flag": True,
                "marketing_opt_in": True,
                "customer_purchase_status": "Returning",
                "age_band_filter": "18-30",
                "gender_filter": "Female",
                "postcode_filter": "5000",
            },
        ]
    )
    fixtures = pd.DataFrame(
        {
            "fixture_id": ["HIST-1"],
            "season_label": ["2025/26"],
            "competition": ["BBL"],
            "opponent": ["Sydney Sixers"],
            "match_date": pd.to_datetime(["2025-12-31"]),
        }
    )
    bundle = StrikersDataBundle(
        data_mode="test",
        matches=pd.DataFrame(),
        daily_sales=pd.DataFrame(),
        transactions=transactions,
        fixtures=fixtures,
        customers=pd.DataFrame(),
        future_fixtures=pd.DataFrame(),
        file_status=pd.DataFrame(),
        column_mappings=pd.DataFrame(),
        validation_warnings=pd.DataFrame(),
        metrics={},
    )
    controls = {
        "competition": "BBL",
        "ticket_type": "Family",
        "ticket_class": "Boundary Zone",
        "sales_window": "All",
        "ticket_status": "Paid + comps",
        "audience_segment": "All",
        "age_band": [],
        "gender": [],
        "postcode": [],
    }
    selected_output = {
        "match_id": "season_total",
        "match": pd.Series({"event_date": pd.Timestamp("2026-12-31"), "competition": "BBL"}),
    }

    curve = _target_breakdown_last_season_curve(bundle, controls, selected_output, max_offset=3)

    assert {"last_season_daily", "last_season_cumulative"}.issubset(curve.columns)
    assert curve["last_season_daily"].iloc[0] == 30
    assert curve["last_season_cumulative"].iloc[1] == 75
