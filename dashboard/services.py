"""View-model helpers for the Streamlit dashboard."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from forecasting.historical_pace import HistoricalPaceEngine
from forecasting.ml_model import TicketSalesForecaster, build_forecast_curve
from forecasting.season import aggregate_match_outputs
from forecasting.targets import TargetAssumptions, add_required_daily_sales, generate_target_curve


GLOBAL_STATE_KEYS = {
    "page",
    "fixture_seed",
    "mapping_overrides",
    "reprocess_token",
}

DEFAULT_INDEX_THRESHOLDS = {
    "ahead": 1.10,
    "on_track": 0.95,
    "watch": 0.85,
    "behind": 0.70,
}

SALES_WINDOW_ORDER = {
    "on-sale": 0,
    "member-pre-sale": 0,
    "pre-sale": 0,
    "early-sales": 1,
    "early-bird": 1,
    "general-sale": 2,
    "general-public": 2,
    "campaign-window": 3,
    "black-friday": 3,
    "christmas-nye-push": 3,
    "match-week": 4,
}

INDEX_COLUMNS = [
    "analysis_group",
    "definition",
    "current_paid_tickets",
    "current_comps",
    "current_total_tickets",
    "current_revenue",
    "current_unique_purchasers",
    "current_average_basket_size",
    "current_average_ticket_price",
    "expected_paid_tickets_by_now",
    "expected_total_tickets_by_now",
    "expected_revenue_by_now",
    "expected_purchasers_by_now",
    "expected_final_paid_tickets",
    "eligible_audience_size",
    "marketable_audience_size",
    "actual_purchase_rate",
    "expected_purchase_rate",
    "ticket_index",
    "revenue_index",
    "purchaser_index",
    "purchase_rate_index",
    "paid_ticket_gap",
    "revenue_gap",
    "usual_purchase_window",
    "current_sales_window",
    "status",
    "confidence",
    "recommended_action",
    "suggested_message_angle",
    "suggested_timing",
    "suggested_ticket_product",
    "rationale",
]


def page_state_key(page_key: str, control_name: str) -> str:
    """Return a Streamlit session-state key that cannot collide across pages."""

    cleaned_page = str(page_key).strip().lower().replace(" ", "_").replace("/", "_").replace("&", "and")
    cleaned_control = str(control_name).strip().lower().replace(" ", "_")
    return f"{cleaned_page}_{cleaned_control}"


def match_label(match: pd.Series) -> str:
    return (
        f"{match['competition']} | M{int(match['round'])} | v {match['opponent']} | "
        f"{pd.to_datetime(match['event_date']).strftime('%d %b %Y')}"
    )


def season_matches(
    matches: pd.DataFrame,
    planning_season_label: str,
    competition: str,
) -> pd.DataFrame:
    filtered = matches[
        matches["season_label"].eq(planning_season_label) & matches["competition"].eq(competition)
    ].copy()
    return filtered.sort_values("event_date").reset_index(drop=True)


def historical_season_labels(matches: pd.DataFrame, planning_season_label: str) -> list[str]:
    values = (
        matches.loc[matches["season_label"].ne(planning_season_label), "season_label"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    return sorted(values)


def history_filters(competition: str, selected_history: Iterable[str] | None) -> dict[str, object]:
    seasons = [value for value in (selected_history or []) if value]
    filters: dict[str, object] = {"competition": competition}
    if seasons:
        filters["season_label"] = seasons
    return filters


def fixed_window_summary(planning_matches: pd.DataFrame) -> dict[str, pd.Timestamp]:
    row = planning_matches.sort_values("event_date").iloc[0]
    return {
        "early_bird_start": pd.to_datetime(row["early_bird_start"]),
        "early_bird_end": pd.to_datetime(row["early_bird_end"]),
        "member_presale_start": pd.to_datetime(row["member_presale_start"]),
        "member_presale_end": pd.to_datetime(row["member_presale_end"]),
        "general_public_start": pd.to_datetime(row["general_public_start"]),
    }


def build_match_output(
    match: pd.Series,
    matches: pd.DataFrame,
    daily_sales: pd.DataFrame,
    pace_engine: HistoricalPaceEngine,
    forecaster: TicketSalesForecaster,
    assumptions: TargetAssumptions,
    target_uplift_pct: float,
    history_filter_values: dict[str, object] | None = None,
    planning_target_override: float | None = None,
    manual_target_override: float | None = None,
) -> dict:
    match_daily = daily_sales[daily_sales["match_id"].astype(str).eq(str(match["match_id"]))].copy()
    pace = pace_engine.get_profile(match, filters=history_filter_values).frame
    snapshot = forecaster.latest_snapshot(matches, daily_sales, str(match["match_id"]))
    forecast = forecaster.predict_snapshot(snapshot)
    forecast_curve = build_forecast_curve(match, match_daily, pace, forecast)

    planning_target = float(planning_target_override if planning_target_override is not None else match["baseline_target"])
    manual_target = float(manual_target_override if manual_target_override is not None else match["manual_target"])

    target_curve = generate_target_curve(
        match,
        pace,
        total_sales_target=planning_target,
        uplift_pct=target_uplift_pct,
        assumptions=assumptions,
    )
    target_curve = add_required_daily_sales(target_curve, match_daily)
    manual_curve = generate_target_curve(
        match,
        pace,
        total_sales_target=planning_target,
        uplift_pct=0,
        manual_target=manual_target,
        assumptions=assumptions,
    )

    match_chart = (
        forecast_curve.merge(
            target_curve[["date", "target_cumulative", "target_daily", "required_daily"]],
            on="date",
            how="left",
        )
        .merge(
            manual_curve[["date", "target_cumulative"]].rename(columns={"target_cumulative": "manual_target_cumulative"}),
            on="date",
            how="left",
        )
        .merge(
            match_daily[["date", "daily_sales"]].rename(columns={"daily_sales": "actual_daily"}),
            on="date",
            how="left",
        )
        .fillna({"actual_daily": 0})
    )

    return {
        "match_id": str(match["match_id"]),
        "match": match,
        "actual_daily": match_daily,
        "pace_profile": pace,
        "forecast": forecast,
        "forecast_curve": forecast_curve,
        "target_curve": target_curve,
        "manual_curve": manual_curve,
        "match_chart": match_chart,
        "planning_target": planning_target,
        "manual_target": manual_target,
    }


def build_season_comparison_frame(
    matches: pd.DataFrame,
    daily_sales: pd.DataFrame,
    competition: str,
    season_labels: list[str],
) -> pd.DataFrame:
    comparison_rows: list[pd.DataFrame] = []
    for season_label in season_labels:
        season_match_ids = matches.loc[
            matches["season_label"].eq(season_label) & matches["competition"].eq(competition),
            "match_id",
        ].astype(str)
        if season_match_ids.empty:
            continue

        season_daily = (
            daily_sales[daily_sales["match_id"].astype(str).isin(season_match_ids)]
            .groupby("date", as_index=False)["daily_sales"]
            .sum()
            .sort_values("date")
        )
        if season_daily.empty:
            continue

        season_daily["actual_cumulative"] = season_daily["daily_sales"].cumsum()
        season_daily["season_label"] = season_label
        season_daily["season_day"] = (
            pd.to_datetime(season_daily["date"]) - pd.to_datetime(season_daily["date"]).min()
        ).dt.days
        comparison_rows.append(season_daily)

    if not comparison_rows:
        return pd.DataFrame(columns=["date", "actual_cumulative", "season_label", "season_day"])

    return pd.concat(comparison_rows, ignore_index=True)


def scenario_summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["Scenario", "Projected finish", "Gap to plan"])

    target_total = float(frame["target_cumulative"].iloc[-1]) if "target_cumulative" in frame else 0.0
    rows = [
        ("ML expected forecast", float(frame["forecast_expected_cumulative"].iloc[-1])),
        ("10% uplift target", float(frame["uplift_10_cumulative"].iloc[-1])),
        ("Manual target", float(frame["manual_target_cumulative"].iloc[-1])),
    ]
    summary = pd.DataFrame(rows, columns=["Scenario", "Projected finish"])
    summary["Gap to plan"] = summary["Projected finish"] - target_total
    return summary


def aggregate_selected_outputs(outputs: list[dict], home_match_count: int) -> pd.DataFrame:
    selected_outputs = sorted(outputs, key=lambda row: row["match"]["event_date"])[:home_match_count]
    return aggregate_match_outputs(selected_outputs)


def apply_transaction_filters(
    transactions: pd.DataFrame,
    season_label: str | None = None,
    competition: str = "All",
    fixture_ids: Iterable[str] | None = None,
    as_at_date: pd.Timestamp | None = None,
    audience_segment: str = "All",
    ticket_type: str = "All",
    ticket_class: str = "All",
    section: str = "All",
    sales_channel: str = "All",
    sales_window: str = "All",
    ticket_status: str = "Paid + comps",
) -> pd.DataFrame:
    """Apply reusable dashboard filters to the normalized transaction model."""

    if transactions.empty:
        return transactions.copy()

    frame = transactions.copy()
    if season_label and season_label != "All" and "season_label" in frame:
        frame = frame[frame["season_label"].astype(str).eq(str(season_label))]
    if competition and competition != "All" and "competition" in frame:
        frame = frame[frame["competition"].astype(str).eq(str(competition))]
    if fixture_ids:
        ids = [str(value) for value in fixture_ids if value and value != "Tournament total"]
        if ids:
            frame = frame[frame["fixture_id"].astype(str).isin(ids)]
    if as_at_date is not None and "transaction_date" in frame:
        frame = frame[pd.to_datetime(frame["transaction_date"], errors="coerce") <= pd.to_datetime(as_at_date)]

    for column, value in {
        "ticket_type": ticket_type,
        "ticket_class": ticket_class,
        "section": section,
        "sales_channel": sales_channel,
        "sales_window": sales_window,
    }.items():
        if value and value != "All" and column in frame:
            frame = frame[frame[column].fillna("Unknown").astype(str).eq(str(value))]

    if audience_segment != "All":
        if audience_segment == "Families" and "purchaser_family_flag" in frame:
            frame = frame[frame["purchaser_family_flag"].fillna(False).astype(bool)]
        elif audience_segment == "Marketable" and "marketing_opt_in" in frame:
            frame = frame[frame["marketing_opt_in"].fillna(False).astype(bool)]
        elif audience_segment == "18-30s" and "purchaser_age_band" in frame:
            frame = frame[frame["purchaser_age_band"].fillna("").astype(str).str.contains("18|19|20|21|22|23|24|25|26|27|28|29|30", regex=True)]
        elif audience_segment == "Returning purchasers" and "customer_purchase_status" in frame:
            frame = frame[frame["customer_purchase_status"].eq("Returning")]

    if ticket_status == "Paid tickets only":
        frame = frame[~frame["is_comp"].fillna(False) & ~frame["is_refund"].fillna(False)]
    elif ticket_status == "Comps only":
        frame = frame[frame["is_comp"].fillna(False)]
    elif ticket_status == "Refunds / voids":
        frame = frame[frame["is_refund"].fillna(False)]

    return frame.reset_index(drop=True)


def filter_options(frame: pd.DataFrame, column: str, include_all: bool = True) -> list[str]:
    if frame.empty or column not in frame:
        return ["All"] if include_all else []
    values = sorted(value for value in frame[column].dropna().astype(str).unique().tolist() if value)
    return (["All"] if include_all else []) + values


def kpi_summary(
    transactions: pd.DataFrame,
    all_transactions: pd.DataFrame,
    fixtures: pd.DataFrame,
    forecast_total: float = 0.0,
    target_total: float = 0.0,
    as_at_date: pd.Timestamp | None = None,
) -> dict[str, float]:
    """Calculate executive KPI values from normalized transactions."""

    if transactions.empty:
        return {
            "tickets_sold": 0.0,
            "paid_tickets_sold": 0.0,
            "comps_issued": 0.0,
            "gross_revenue": 0.0,
            "net_revenue": 0.0,
            "average_ticket_price": 0.0,
            "average_basket_size": 0.0,
            "unique_purchasers": 0.0,
            "retention_rate": 0.0,
            "new_purchasers": 0.0,
            "forecast_total": float(forecast_total),
            "target_total": float(target_total),
            "gap_to_target": float(forecast_total - target_total),
            "required_daily_run_rate": 0.0,
            "days_to_next_fixture": 0.0,
            "capacity_sold_pct": np.nan,
        }

    paid = float(transactions["paid_tickets_sold"].sum())
    tickets = float(transactions["tickets_sold"].sum())
    comps = float(transactions["comp_tickets_sold"].sum())
    gross = float(transactions["gross_revenue"].sum())
    net = float(transactions["net_revenue"].sum())
    orders = max(transactions["order_id"].dropna().nunique(), 1)
    customers = transactions["customer_id"].dropna().astype(str)
    unique_customers = float(customers.nunique())
    first_season = _first_purchase_season(all_transactions)
    current_seasons = transactions[["customer_id", "season_label"]].dropna().drop_duplicates()
    current_season = current_seasons["season_label"].mode().iloc[0] if not current_seasons.empty else None
    current_customers = set(current_seasons["customer_id"].astype(str))
    new_customers = {
        customer
        for customer in current_customers
        if current_season is not None and first_season.get(customer) == current_season
    }
    returning = len(current_customers - new_customers)
    denominator = len(current_customers) if current_customers else 0
    retention_rate = returning / denominator if denominator else 0.0

    next_fixture_days = _days_to_next_fixture(fixtures, as_at_date)
    capacity = float(fixtures["capacity_total"].dropna().sum()) if "capacity_total" in fixtures else np.nan
    capacity_pct = (tickets / capacity * 100) if capacity and capacity > 0 else np.nan
    remaining_days = max(next_fixture_days, 1)
    required = max(float(target_total) - paid, 0) / remaining_days if target_total else 0.0

    return {
        "tickets_sold": tickets,
        "paid_tickets_sold": paid,
        "comps_issued": comps,
        "gross_revenue": gross,
        "net_revenue": net,
        "average_ticket_price": gross / paid if paid else 0.0,
        "average_basket_size": paid / orders if orders else 0.0,
        "unique_purchasers": unique_customers,
        "retention_rate": retention_rate,
        "new_purchasers": float(len(new_customers)),
        "forecast_total": float(forecast_total),
        "target_total": float(target_total),
        "gap_to_target": float(forecast_total - target_total),
        "required_daily_run_rate": float(required),
        "days_to_next_fixture": float(next_fixture_days),
        "capacity_sold_pct": float(capacity_pct) if not pd.isna(capacity_pct) else np.nan,
    }


def fixture_sales_summary(
    transactions: pd.DataFrame,
    fixtures: pd.DataFrame,
    matches: pd.DataFrame,
    forecast_outputs: list[dict] | None = None,
) -> pd.DataFrame:
    """Fixture-level summary with paid/comp/revenue, target, forecast, and risk."""

    if fixtures.empty:
        return pd.DataFrame()
    sales = (
        transactions.groupby("fixture_id", dropna=False).agg(
            sold=("tickets_sold", "sum"),
            paid_sold=("paid_tickets_sold", "sum"),
            comps=("comp_tickets_sold", "sum"),
            gross_revenue=("gross_revenue", "sum"),
            unique_purchasers=("customer_id", "nunique"),
        )
        if not transactions.empty
        else pd.DataFrame(
            columns=["fixture_id", "sold", "paid_sold", "comps", "gross_revenue", "unique_purchasers"]
        )
    )
    frame = fixtures.merge(sales, on="fixture_id", how="left")
    for column in ["sold", "paid_sold", "comps", "gross_revenue", "unique_purchasers"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").fillna(0)

    target_lookup = matches.set_index("match_id")["manual_target"].to_dict() if not matches.empty else {}
    base_lookup = matches.set_index("match_id")["baseline_target"].to_dict() if not matches.empty else {}
    frame["target"] = frame["fixture_id"].map(target_lookup).fillna(frame["fixture_id"].map(base_lookup)).fillna(0)

    forecast_lookup: dict[str, float] = {}
    if forecast_outputs:
        for output in forecast_outputs:
            chart = output.get("match_chart")
            if chart is not None and not chart.empty and "forecast_expected_cumulative" in chart:
                forecast_lookup[str(output["match_id"])] = float(chart["forecast_expected_cumulative"].iloc[-1])
    frame["forecast"] = frame["fixture_id"].astype(str).map(forecast_lookup).fillna(0)
    frame["gap"] = frame["forecast"] - frame["target"]
    frame["capacity_sold_pct"] = np.where(frame["capacity_total"].fillna(0).gt(0), frame["sold"] / frame["capacity_total"] * 100, np.nan)
    frame["risk_status"] = frame.apply(_risk_status, axis=1)
    frame["match_date"] = pd.to_datetime(frame["match_date"], errors="coerce")
    return frame.sort_values(["competition", "match_date"]).reset_index(drop=True)


def audience_summary(
    transactions: pd.DataFrame,
    group_by: str = "ticket_class",
    min_size: int = 0,
) -> pd.DataFrame:
    if transactions.empty or group_by not in transactions:
        return pd.DataFrame()
    frame = transactions.copy()
    frame[group_by] = frame[group_by].fillna("Unknown").astype(str)
    grouped = frame.groupby(group_by).agg(
        paid_tickets=("paid_tickets_sold", "sum"),
        comps=("comp_tickets_sold", "sum"),
        gross_revenue=("gross_revenue", "sum"),
        customers=("customer_id", "nunique"),
        orders=("order_id", "nunique"),
    ).reset_index()
    grouped = grouped[grouped["customers"].ge(min_size) | grouped["paid_tickets"].ge(min_size)]
    grouped["average_ticket_price"] = np.where(grouped["paid_tickets"].gt(0), grouped["gross_revenue"] / grouped["paid_tickets"], 0)
    grouped["average_basket_size"] = np.where(grouped["orders"].gt(0), grouped["paid_tickets"] / grouped["orders"], 0)
    total_paid = max(float(grouped["paid_tickets"].sum()), 1.0)
    grouped["contribution_pct"] = grouped["paid_tickets"] / total_paid * 100
    grouped["status"] = np.select(
        [grouped["contribution_pct"].ge(20), grouped["comps"].gt(grouped["paid_tickets"] * 0.25)],
        ["growth engine", "comp masking risk"],
        default="watch",
    )
    return grouped.sort_values("paid_tickets", ascending=False).reset_index(drop=True)


def sales_window_summary(transactions: pd.DataFrame) -> pd.DataFrame:
    if transactions.empty:
        return pd.DataFrame()
    frame = transactions.copy()
    frame["sales_window"] = frame["sales_window"].fillna("unknown")
    grouped = frame.groupby("sales_window").agg(
        paid_tickets=("paid_tickets_sold", "sum"),
        comps=("comp_tickets_sold", "sum"),
        gross_revenue=("gross_revenue", "sum"),
        customers=("customer_id", "nunique"),
    ).reset_index()
    total = max(float(grouped["paid_tickets"].sum()), 1.0)
    grouped["window_share_pct"] = grouped["paid_tickets"] / total * 100
    grouped["recommended_weight"] = np.where(grouped["sales_window"].isin(["early-sales", "match-week"]), 1.15, 1.0)
    return grouped.sort_values("paid_tickets", ascending=False).reset_index(drop=True)


def current_sales_window(as_at_date: pd.Timestamp | None, fixtures: pd.DataFrame) -> str:
    """Infer the active sales window from the nearest future fixture."""

    if fixtures.empty or "match_date" not in fixtures:
        return "snapshot-review"
    date = pd.to_datetime(as_at_date if as_at_date is not None else pd.Timestamp.today()).normalize()
    fixture_dates = pd.to_datetime(fixtures["match_date"], errors="coerce").dropna()
    future_dates = fixture_dates[fixture_dates >= date]
    if future_dates.empty:
        return "match-week"
    days_to_next = int((future_dates.min().normalize() - date).days)
    if days_to_next <= 7:
        return "match-week"
    if days_to_next <= 35:
        return "campaign-window"
    if days_to_next <= 80:
        return "general-sale"
    return "early-sales"


def next_sales_window_label(window: str) -> str:
    order_to_label = {
        0: "early-sales",
        1: "general-sale",
        2: "campaign-window",
        3: "match-week",
        4: "match-week",
    }
    return order_to_label.get(_window_order(window) + 1, "match-week")


def stale_snapshot_status(
    latest_transaction_date: pd.Timestamp | None,
    today: pd.Timestamp | None = None,
    stale_after_days: int = 14,
) -> dict[str, object]:
    """Summarise whether the current extract looks stale."""

    if latest_transaction_date is None or pd.isna(latest_transaction_date):
        return {"is_stale": True, "age_days": None, "message": "No transaction date is available in the uploaded snapshot."}
    reference = pd.to_datetime(today if today is not None else pd.Timestamp.today()).normalize()
    latest = pd.to_datetime(latest_transaction_date).normalize()
    age_days = int((reference - latest).days)
    if age_days > stale_after_days:
        message = f"Latest ticketing extract is {age_days} days old. Upload a fresh snapshot before using this for current decisions."
    elif age_days < 0:
        message = "Latest transaction date is in the future relative to today. Check date parsing and extract timestamps."
    else:
        message = f"Latest ticketing extract is {age_days} days old."
    return {"is_stale": age_days > stale_after_days or age_days < 0, "age_days": age_days, "message": message}


def demographic_coverage(customers: pd.DataFrame) -> pd.DataFrame:
    """Return privacy-safe coverage rates for optional demographic fields."""

    rows = []
    if customers.empty:
        return pd.DataFrame(columns=["field", "known_records", "coverage_pct", "note"])
    for field in ["age_band", "gender", "postcode", "state", "family_flag", "marketing_opt_in", "email_opt_in", "sms_opt_in"]:
        if field not in customers:
            continue
        values = customers[field]
        known = values.notna()
        if values.dtype == bool:
            known = values.notna()
        known_count = int(known.sum())
        coverage = known_count / len(customers) * 100 if len(customers) else 0.0
        rows.append(
            {
                "field": field,
                "known_records": known_count,
                "coverage_pct": coverage,
                "note": "Low coverage - treat as directional" if coverage < 40 else "Usable for aggregated planning",
            }
        )
    return pd.DataFrame(rows)


def expected_purchase_index(
    transactions: pd.DataFrame,
    customers: pd.DataFrame,
    matches: pd.DataFrame,
    current_season_label: str,
    competition: str = "All",
    as_at_date: pd.Timestamp | None = None,
    group_kind: str = "segment",
    baseline_seasons: Iterable[str] | None = None,
    min_size: int = 100,
    thresholds: dict[str, float] | None = None,
) -> pd.DataFrame:
    """Compare actual sales with expected same-point-in-cycle sales by group."""

    if transactions.empty:
        return pd.DataFrame(columns=INDEX_COLUMNS)

    date = pd.to_datetime(as_at_date if as_at_date is not None else pd.Timestamp.today()).normalize()
    tx = transactions.copy()
    tx["transaction_date"] = pd.to_datetime(tx.get("transaction_date"), errors="coerce")
    tx["match_date"] = pd.to_datetime(tx.get("match_date"), errors="coerce")
    tx = tx[tx["transaction_date"].notna()].copy()
    if competition != "All" and "competition" in tx:
        tx = tx[tx["competition"].astype(str).eq(str(competition))]
    if tx.empty:
        return pd.DataFrame(columns=INDEX_COLUMNS)

    current_matches = matches[matches["season_label"].astype(str).eq(str(current_season_label))].copy() if not matches.empty else pd.DataFrame()
    if competition != "All" and not current_matches.empty and "competition" in current_matches:
        current_matches = current_matches[current_matches["competition"].astype(str).eq(str(competition))]
    current_window = current_sales_window(date, current_matches.rename(columns={"event_date": "match_date"}))
    days_threshold = _same_point_days_to_fixture(current_matches, date)

    available_history = sorted(
        season
        for season in tx["season_label"].dropna().astype(str).unique().tolist()
        if season != str(current_season_label)
    )
    history_seasons = [str(value) for value in (baseline_seasons or []) if str(value) in available_history]
    if not history_seasons:
        history_seasons = available_history[-3:]

    current_tx = tx[
        tx["season_label"].astype(str).eq(str(current_season_label))
        & tx["transaction_date"].le(date)
    ].copy()
    historical_tx = tx[tx["season_label"].astype(str).isin(history_seasons)].copy()

    current_groups = _group_transactions_for_index(current_tx, customers, group_kind)
    historical_groups = _group_transactions_for_index(historical_tx, customers, group_kind)
    if historical_groups.empty and current_groups.empty:
        return pd.DataFrame(columns=INDEX_COLUMNS)

    current_summary = _current_index_summary(current_groups)
    expected_summary = _historical_expected_summary(historical_groups, history_seasons, days_threshold)
    all_groups = sorted(set(current_summary.index.astype(str)) | set(expected_summary.index.astype(str)))
    if not all_groups:
        return pd.DataFrame(columns=INDEX_COLUMNS)

    audience_sizes = _segment_audience_sizes(customers)
    definitions = _group_definitions(group_kind)
    rows = []
    for group in all_groups:
        current = current_summary.loc[group] if group in current_summary.index else pd.Series(dtype=float)
        expected = expected_summary.loc[group] if group in expected_summary.index else pd.Series(dtype=float)
        eligible_size = float(audience_sizes.get(group, {}).get("eligible_audience_size", np.nan))
        marketable_size = float(audience_sizes.get(group, {}).get("marketable_audience_size", np.nan))
        expected_final_paid = float(expected.get("expected_final_paid_tickets", 0) or 0)
        if pd.isna(eligible_size):
            eligible_size = float(max(expected.get("expected_final_purchasers", 0) or 0, current.get("current_unique_purchasers", 0) or 0))
        actual_paid = float(current.get("current_paid_tickets", 0) or 0)
        expected_paid = float(expected.get("expected_paid_tickets_by_now", 0) or 0)
        actual_revenue = float(current.get("current_revenue", 0) or 0)
        expected_revenue = float(expected.get("expected_revenue_by_now", 0) or 0)
        actual_purchasers = float(current.get("current_unique_purchasers", 0) or 0)
        expected_purchasers = float(expected.get("expected_purchasers_by_now", 0) or 0)
        actual_rate = actual_purchasers / eligible_size if eligible_size else 0.0
        expected_rate = expected_purchasers / eligible_size if eligible_size else 0.0
        usual_window = str(expected.get("usual_purchase_window", "Unknown") or "Unknown")
        ticket_index = _safe_index(actual_paid, expected_paid)
        revenue_index = _safe_index(actual_revenue, expected_revenue)
        purchaser_index = _safe_index(actual_purchasers, expected_purchasers)
        purchase_rate_index = _safe_index(actual_rate, expected_rate)
        status = segment_status(
            ticket_index,
            actual_paid,
            expected_paid,
            expected_final_paid,
            current_window,
            usual_window,
            thresholds=thresholds,
        )
        confidence = _index_confidence(expected.get("history_seasons", 0), expected_final_paid, eligible_size)
        row = {
            "analysis_group": group,
            "definition": definitions.get(group, _definition_for_group(group_kind, group)),
            "current_paid_tickets": actual_paid,
            "current_comps": float(current.get("current_comps", 0) or 0),
            "current_total_tickets": float(current.get("current_total_tickets", 0) or 0),
            "current_revenue": actual_revenue,
            "current_unique_purchasers": actual_purchasers,
            "current_average_basket_size": float(current.get("current_average_basket_size", 0) or 0),
            "current_average_ticket_price": float(current.get("current_average_ticket_price", 0) or 0),
            "expected_paid_tickets_by_now": expected_paid,
            "expected_total_tickets_by_now": float(expected.get("expected_total_tickets_by_now", 0) or 0),
            "expected_revenue_by_now": expected_revenue,
            "expected_purchasers_by_now": expected_purchasers,
            "expected_final_paid_tickets": expected_final_paid,
            "eligible_audience_size": eligible_size,
            "marketable_audience_size": 0.0 if pd.isna(marketable_size) else marketable_size,
            "actual_purchase_rate": actual_rate,
            "expected_purchase_rate": expected_rate,
            "ticket_index": ticket_index,
            "revenue_index": revenue_index,
            "purchaser_index": purchaser_index,
            "purchase_rate_index": purchase_rate_index,
            "paid_ticket_gap": actual_paid - expected_paid,
            "revenue_gap": actual_revenue - expected_revenue,
            "usual_purchase_window": usual_window,
            "current_sales_window": current_window,
            "status": status,
            "confidence": confidence,
        }
        row.update(_recommendation_text(row, group_kind))
        rows.append(row)

    output = pd.DataFrame(rows)
    size_basis = output[["current_paid_tickets", "expected_paid_tickets_by_now", "expected_final_paid_tickets", "eligible_audience_size"]].max(axis=1)
    output = output[size_basis.ge(min_size)].copy()
    if output.empty:
        output = pd.DataFrame(rows).sort_values("expected_final_paid_tickets", ascending=False).head(12)
    status_order = {"At risk": 0, "Behind": 1, "Watch": 2, "Not due yet": 3, "On track": 4, "Ahead": 5}
    output["_status_order"] = output["status"].map(status_order).fillna(6)
    return output.sort_values(["_status_order", "paid_ticket_gap", "expected_final_paid_tickets"], ascending=[True, True, False]).drop(columns="_status_order").reset_index(drop=True)


def segment_status(
    index: float,
    actual_paid: float,
    expected_paid: float,
    expected_final_paid: float,
    current_window: str,
    usual_purchase_window: str,
    thresholds: dict[str, float] | None = None,
) -> str:
    """Classify index values while protecting late-buying groups from false churn flags."""

    threshold_values = {**DEFAULT_INDEX_THRESHOLDS, **(thresholds or {})}
    expected_share_due = expected_paid / expected_final_paid if expected_final_paid else 0.0
    later_window = _window_order(usual_purchase_window) > _window_order(current_window)
    if actual_paid <= 0 and (expected_paid <= 0 or (expected_share_due < 0.08 and later_window)):
        return "Not due yet"
    if later_window and expected_share_due < 0.12 and index < threshold_values["watch"]:
        return "Not due yet"
    if expected_paid <= 0:
        return "Ahead" if actual_paid > 0 else "Not due yet"
    if index >= threshold_values["ahead"]:
        return "Ahead"
    if index >= threshold_values["on_track"]:
        return "On track"
    if index >= threshold_values["watch"]:
        return "Watch"
    if index >= threshold_values["behind"]:
        return "Behind"
    return "At risk"


def marketing_recommendations(index_frame: pd.DataFrame, min_marketable_size: int = 5000) -> pd.DataFrame:
    """Return campaign-ready recommendations, falling back to diagnostic rows where opt-in data is sparse."""

    if index_frame.empty:
        return pd.DataFrame()
    frame = index_frame.copy()
    market_ready = frame[frame["marketable_audience_size"].fillna(0).ge(min_marketable_size)].copy()
    if market_ready.empty:
        market_ready = frame[frame["eligible_audience_size"].fillna(0).ge(min(250, min_marketable_size))].copy()
        market_ready["marketable_note"] = "Opt-in or marketable size is unavailable/low; use as diagnostic planning evidence."
    else:
        market_ready["marketable_note"] = "Campaign-ready marketable audience size meets the selected threshold."
    return market_ready.sort_values(["status", "paid_ticket_gap"], ascending=[True, True]).head(12).reset_index(drop=True)


def recommended_audiences(
    customers: pd.DataFrame,
    transactions: pd.DataFrame,
    planning_season_label: str,
    min_size: int = 5000,
    as_at_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Rank anonymised/aggregated marketable audiences not yet purchased."""

    if customers.empty:
        return pd.DataFrame()
    customer_frame = customers.copy()
    bought_current = set(
        transactions.loc[transactions["season_label"].eq(planning_season_label), "customer_id"].dropna().astype(str)
    ) if not transactions.empty and "season_label" in transactions else set()
    customer_frame["has_purchased_planning_season"] = customer_frame["customer_id"].astype(str).isin(bought_current)
    marketable_cols = [column for column in ["marketing_opt_in", "email_opt_in", "sms_opt_in"] if column in customer_frame]
    if marketable_cols:
        customer_frame["marketable"] = customer_frame[marketable_cols].fillna(False).astype(bool).any(axis=1)
    else:
        customer_frame["marketable"] = False

    candidates = customer_frame[~customer_frame["has_purchased_planning_season"] & customer_frame["marketable"]].copy()
    if candidates.empty:
        return pd.DataFrame()

    candidates["segment_family"] = np.where(_column_or_default(candidates, "family_flag", False).fillna(False).astype(bool), "family", "general")
    candidates["segment_age"] = _column_or_default(candidates, "age_band", "Unknown").fillna("Unknown").astype(str)
    candidates["segment_class"] = _column_or_default(candidates, "preferred_ticket_class", "Unknown").fillna("Unknown").astype(str)
    candidates["usual_purchase_window"] = _column_or_default(candidates, "usual_purchase_window", "Unknown").fillna("Unknown").astype(str)
    candidates["segment"] = (
        candidates["segment_family"].str.title()
        + " | "
        + candidates["segment_class"].replace("", "Unknown")
        + " | "
        + candidates["usual_purchase_window"].replace("", "Unknown")
    )
    grouped = candidates.groupby("segment").agg(
        audience_size=("customer_id", "nunique"),
        avg_lifetime_tickets=("lifetime_tickets", "mean"),
        avg_lifetime_revenue=("lifetime_revenue", "mean"),
        seasons_purchased=("seasons_purchased_count", "mean"),
        usual_purchase_window=("usual_purchase_window", _mode_or_unknown),
    ).reset_index()
    grouped = grouped[grouped["audience_size"].ge(min_size)]
    if grouped.empty:
        grouped = candidates.groupby("segment").agg(
            audience_size=("customer_id", "nunique"),
            avg_lifetime_tickets=("lifetime_tickets", "mean"),
            avg_lifetime_revenue=("lifetime_revenue", "mean"),
            seasons_purchased=("seasons_purchased_count", "mean"),
            usual_purchase_window=("usual_purchase_window", _mode_or_unknown),
        ).reset_index().sort_values("audience_size", ascending=False).head(10)
    grouped["likelihood_score"] = (
        grouped["seasons_purchased"].fillna(0).clip(0, 5) * 12
        + grouped["avg_lifetime_tickets"].fillna(0).clip(0, 20) * 1.7
        + np.log1p(grouped["avg_lifetime_revenue"].fillna(0).clip(lower=0)) * 4
    ).clip(0, 100)
    grouped["estimated_ticket_upside"] = (grouped["audience_size"] * (grouped["likelihood_score"] / 100) * 1.8).round()
    grouped["confidence"] = np.select(
        [grouped["audience_size"].ge(5000), grouped["audience_size"].ge(1000)],
        ["high", "medium"],
        default="directional",
    )
    grouped["suggested_timing"] = grouped["usual_purchase_window"].replace({"Unknown": "next available sales window"})
    grouped["suggested_message"] = np.where(
        grouped["segment"].str.contains("Family", case=False),
        "Family value, school-holiday timing, and urgency before window close.",
        "Availability, group purchase ease, and fixture-specific urgency.",
    )
    grouped["rationale"] = grouped.apply(
        lambda row: (
            f"{int(row['audience_size']):,} marketable customers have not purchased {planning_season_label}; "
            f"historic average is {row['avg_lifetime_tickets']:.1f} tickets across {row['seasons_purchased']:.1f} seasons."
        ),
        axis=1,
    )
    return grouped.sort_values(["estimated_ticket_upside", "likelihood_score"], ascending=False).reset_index(drop=True)


def insight_narrative(kpis: dict[str, float], fixture_summary_frame: pd.DataFrame, audience_frame: pd.DataFrame) -> list[str]:
    insights: list[str] = []
    gap = kpis.get("gap_to_target", 0.0)
    if gap < 0:
        insights.append(
            f"Forecast is {abs(gap):,.0f} paid tickets behind the selected target. Required run-rate is {kpis.get('required_daily_run_rate', 0):,.0f} paid tickets per day."
        )
    else:
        insights.append(f"Forecast is {gap:,.0f} paid tickets ahead of the selected target. Protect yield before using discount-led demand.")

    if not fixture_summary_frame.empty:
        at_risk = fixture_summary_frame.sort_values("gap").head(1).iloc[0]
        insights.append(
            f"Largest fixture watch item: {at_risk.get('fixture_label', 'selected fixture')} is {abs(float(at_risk.get('gap', 0))):,.0f} tickets "
            f"{'behind' if float(at_risk.get('gap', 0)) < 0 else 'ahead of'} target."
        )
    if not audience_frame.empty:
        top = audience_frame.iloc[0]
        insights.append(
            f"Biggest visible audience contributor is {top.iloc[0]} with {float(top['paid_tickets']):,.0f} paid tickets and {float(top['contribution_pct']):.1f}% share."
        )
    return insights


def report_html(
    title: str,
    kpis: dict[str, float],
    fixture_frame: pd.DataFrame,
    insights: list[str],
    assumed_note: str,
    audience_index: pd.DataFrame | None = None,
    ticket_type_index: pd.DataFrame | None = None,
    ticket_class_index: pd.DataFrame | None = None,
    snapshot_note: str = "",
) -> str:
    fixture_rows = fixture_frame.head(8).to_html(index=False, classes="fixture-table") if not fixture_frame.empty else "<p>No fixture rows available.</p>"
    insight_items = "".join(f"<li>{item}</li>" for item in insights)
    audience_rows = _report_index_rows(audience_index, "Audience opportunity")
    type_rows = _report_index_rows(ticket_type_index, "Ticket type performance")
    class_rows = _report_index_rows(ticket_class_index, "Ticket class performance")
    return f"""
    <html>
      <head>
        <meta charset="utf-8" />
        <style>
          body {{ font-family: Arial, sans-serif; color: #0f1728; margin: 28px; }}
          h1 {{ margin-bottom: 4px; }}
          .kpis {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin: 18px 0; }}
          .kpi {{ border: 1px solid #d7dbe3; padding: 12px; border-radius: 6px; }}
          .label {{ color: #67728a; font-size: 11px; text-transform: uppercase; }}
          .value {{ font-size: 22px; font-weight: 700; }}
          table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
          th, td {{ border-bottom: 1px solid #e5e7eb; padding: 6px; text-align: left; }}
        </style>
      </head>
      <body>
        <h1>{title}</h1>
        <p>{snapshot_note}</p>
        <p>{assumed_note}</p>
        <div class="kpis">
          <div class="kpi"><div class="label">Paid tickets</div><div class="value">{kpis.get('paid_tickets_sold', 0):,.0f}</div></div>
          <div class="kpi"><div class="label">Comps</div><div class="value">{kpis.get('comps_issued', 0):,.0f}</div></div>
          <div class="kpi"><div class="label">Gross revenue</div><div class="value">${kpis.get('gross_revenue', 0):,.0f}</div></div>
          <div class="kpi"><div class="label">Forecast</div><div class="value">{kpis.get('forecast_total', 0):,.0f}</div></div>
          <div class="kpi"><div class="label">Gap to target</div><div class="value">{kpis.get('gap_to_target', 0):,.0f}</div></div>
        </div>
        <h2>Recommended Actions</h2>
        <ul>{insight_items}</ul>
        {audience_rows}
        {type_rows}
        {class_rows}
        <h2>Fixture Summary</h2>
        {fixture_rows}
      </body>
    </html>
    """


def _same_point_days_to_fixture(matches: pd.DataFrame, as_at_date: pd.Timestamp) -> int:
    if matches.empty or "event_date" not in matches:
        return 0
    event_dates = pd.to_datetime(matches["event_date"], errors="coerce").dropna()
    if event_dates.empty:
        return 0
    future = event_dates[event_dates >= as_at_date]
    reference_event = future.min() if not future.empty else event_dates.max()
    return max(int((reference_event.normalize() - as_at_date).days), 0)


def _group_transactions_for_index(transactions: pd.DataFrame, customers: pd.DataFrame, group_kind: str) -> pd.DataFrame:
    if transactions.empty:
        return pd.DataFrame()
    frame = transactions.copy()
    if not customers.empty and "customer_id" in frame and "customer_id" in customers:
        customer_cols = [
            column
            for column in [
                "customer_id",
                "seasons_purchased_count",
                "lifetime_tickets",
                "lifetime_revenue",
                "preferred_ticket_class",
                "preferred_ticket_type",
                "usual_purchase_window",
                "marketing_opt_in",
                "email_opt_in",
                "sms_opt_in",
                "family_flag",
                "age_band",
            ]
            if column in customers
        ]
        frame = frame.merge(customers[customer_cols].drop_duplicates("customer_id"), on="customer_id", how="left", suffixes=("", "_customer"))

    if group_kind == "ticket_type":
        frame["_analysis_group"] = frame.get("price_type", frame.get("ticket_type", "Unknown")).fillna("Unknown").astype(str)
        return frame
    if group_kind == "ticket_class":
        frame["_analysis_group"] = frame.get("ticket_class", "Unknown").fillna("Unknown").astype(str)
        return frame
    if group_kind == "segment_product":
        segment_frame = _group_transactions_for_index(frame, customers=pd.DataFrame(), group_kind="segment")
        if segment_frame.empty:
            return segment_frame
        product = segment_frame.get("ticket_class", "Unknown").fillna("Unknown").astype(str)
        segment_frame["_analysis_group"] = segment_frame["_analysis_group"].astype(str) + " x " + product
        return segment_frame

    segment_specs = _segment_specs(frame)
    segment_frames = []
    for label, mask, _, _ in segment_specs:
        if mask.any():
            subset = frame[mask].copy()
            subset["_analysis_group"] = label
            segment_frames.append(subset)
    if not segment_frames:
        return pd.DataFrame()
    return pd.concat(segment_frames, ignore_index=True)


def _segment_specs(frame: pd.DataFrame) -> list[tuple[str, pd.Series, str, str]]:
    index = frame.index
    ticket_class = _string_col(frame, "ticket_class")
    price_type = _string_col(frame, "price_type")
    ticket_type = _string_col(frame, "ticket_type")
    purchase_status = _string_col(frame, "customer_purchase_status")
    competition = _string_col(frame, "competition")
    usual_window = _string_col(frame, "usual_purchase_window")
    seasons = pd.to_numeric(_column_or_default(frame, "seasons_purchased_count", 0), errors="coerce").fillna(0)
    lifetime_tickets = pd.to_numeric(_column_or_default(frame, "lifetime_tickets", 0), errors="coerce").fillna(0)
    lifetime_revenue = pd.to_numeric(_column_or_default(frame, "lifetime_revenue", 0), errors="coerce").fillna(0)
    match_dates = pd.to_datetime(_column_or_default(frame, "match_date", pd.NaT), errors="coerce")
    family_flag = _bool_col(frame, "purchaser_family_flag") | _bool_col(frame, "family_flag")
    marketable = _bool_col(frame, "marketing_opt_in") | _bool_col(frame, "email_opt_in") | _bool_col(frame, "sms_opt_in")
    paid_only = ~_bool_col(frame, "is_comp") & ~_bool_col(frame, "is_refund")
    comp = _bool_col(frame, "is_comp")
    age = _string_col(frame, "purchaser_age_band").where(_string_col(frame, "purchaser_age_band").ne(""), _string_col(frame, "age_band"))
    family_text = price_type.str.contains("family|fam4|fam5", case=False, regex=True) | ticket_class.str.contains("family", case=False, regex=True)
    ga = ticket_class.str.contains("general admission|\\bga\\b|hill", case=False, regex=True)
    gold = ticket_class.str.contains("gold", case=False, regex=True)
    boundary = ticket_class.str.contains("boundary", case=False, regex=True)
    premium = ticket_class.str.contains("platinum|premium", case=False, regex=True)
    early = usual_window.str.contains("early|pre-sale|on-sale", case=False, regex=True)
    late = usual_window.str.contains("campaign|match-week|late", case=False, regex=True)
    nye = match_dates.dt.month.eq(12) & match_dates.dt.day.eq(31)
    high_basket = lifetime_tickets.ge(12)
    low_basket = lifetime_tickets.gt(0) & lifetime_tickets.le(3)
    high_value = lifetime_revenue.ge(lifetime_revenue.quantile(0.75) if len(lifetime_revenue) else 0)
    age_18_30 = age.str.contains("18|19|20|21|22|23|24|25|26|27|28|29|30", regex=True)
    specs = [
        ("Returning purchasers", purchase_status.eq("Returning"), "Customers whose first purchase season was before the selected season.", "customer_purchase_status"),
        ("New purchasers", purchase_status.eq("New"), "Customers whose first purchase season is the selected season.", "customer_purchase_status"),
        ("Families", family_flag | family_text, "Family flag or family ticket price/product naming.", "purchaser_family_flag, price_type, ticket_class"),
        ("18-30s", age_18_30, "Customer age band indicates 18 to 30 where available.", "purchaser_age_band, age_band"),
        ("Premium buyers", premium, "Ticket class contains Platinum or Premium.", "ticket_class"),
        ("Gold ticket buyers", gold, "Ticket class contains Gold.", "ticket_class"),
        ("Boundary Zone buyers", boundary, "Ticket class contains Boundary Zone.", "ticket_class"),
        ("General Admission buyers", ga, "Ticket class contains General Admission, GA, or Hill.", "ticket_class"),
        ("High basket-size purchasers", high_basket, "Customer lifetime ticket volume indicates larger baskets.", "lifetime_tickets"),
        ("Low basket-size purchasers", low_basket, "Customer lifetime ticket volume indicates smaller baskets.", "lifetime_tickets"),
        ("High-value purchasers", high_value, "Customer lifetime revenue is in the upper quartile.", "lifetime_revenue"),
        ("Marketable customers", marketable, "Marketing, email, or SMS opt-in is true.", "marketing/email/sms opt-in"),
        ("Non-marketable customers", ~marketable, "No marketing, email, or SMS opt-in flag is available.", "marketing/email/sms opt-in"),
        ("Usual early-window buyers", early, "Customer usually buys in early-sales, pre-sale, or on-sale windows.", "usual_purchase_window"),
        ("Usual late-window buyers", late, "Customer usually buys in campaign or match-week windows.", "usual_purchase_window"),
        ("NYE purchasers", nye, "Historical purchase is attached to a 31 December fixture.", "match_date"),
        ("WBBL comp recipients", competition.eq("WBBL") & comp, "WBBL rows issued as complimentary tickets.", "competition, is_comp"),
        ("Paid-ticket-only groups", paid_only, "Rows representing paid demand, excluding comps and refunds.", "is_comp, is_refund"),
        ("Comp-heavy groups", comp, "Rows issued as complimentary tickets.", "is_comp"),
    ]
    return [(label, mask.reindex(index, fill_value=False).fillna(False), definition, fields) for label, mask, definition, fields in specs]


def _current_index_summary(grouped: pd.DataFrame) -> pd.DataFrame:
    if grouped.empty:
        return pd.DataFrame()
    summary = grouped.groupby("_analysis_group").agg(
        current_paid_tickets=("paid_tickets_sold", "sum"),
        current_comps=("comp_tickets_sold", "sum"),
        current_total_tickets=("tickets_sold", "sum"),
        current_revenue=("gross_revenue", "sum"),
        current_unique_purchasers=("customer_id", "nunique"),
        current_orders=("order_id", "nunique"),
    )
    summary["current_average_ticket_price"] = np.where(summary["current_paid_tickets"].gt(0), summary["current_revenue"] / summary["current_paid_tickets"], 0)
    summary["current_average_basket_size"] = np.where(summary["current_orders"].gt(0), summary["current_paid_tickets"] / summary["current_orders"], 0)
    return summary


def _historical_expected_summary(grouped: pd.DataFrame, history_seasons: list[str], days_threshold: int) -> pd.DataFrame:
    if grouped.empty or not history_seasons:
        return pd.DataFrame()
    hist = grouped.copy()
    hist["transaction_date"] = pd.to_datetime(hist["transaction_date"], errors="coerce")
    hist["match_date"] = pd.to_datetime(hist["match_date"], errors="coerce")
    hist["days_to_match"] = (hist["match_date"] - hist["transaction_date"]).dt.days
    if days_threshold > 0:
        by_now = hist[hist["days_to_match"].fillna(-9999).ge(days_threshold)].copy()
    else:
        by_now = hist.copy()
    final = _season_group_summary(hist, "final")
    current = _season_group_summary(by_now, "by_now")
    groups = sorted(set(final["_analysis_group"].astype(str)) | set(current["_analysis_group"].astype(str)))
    rows = []
    for group in groups:
        row = {"_analysis_group": group}
        final_group = final[final["_analysis_group"].astype(str).eq(group)]
        current_group = current[current["_analysis_group"].astype(str).eq(group)]
        row["expected_final_paid_tickets"] = _mean_by_season(final_group, history_seasons, "paid_tickets")
        row["expected_final_purchasers"] = _mean_by_season(final_group, history_seasons, "purchasers")
        row["expected_paid_tickets_by_now"] = _mean_by_season(current_group, history_seasons, "paid_tickets")
        row["expected_total_tickets_by_now"] = _mean_by_season(current_group, history_seasons, "total_tickets")
        row["expected_revenue_by_now"] = _mean_by_season(current_group, history_seasons, "revenue")
        row["expected_purchasers_by_now"] = _mean_by_season(current_group, history_seasons, "purchasers")
        row["history_seasons"] = len(history_seasons)
        usual = hist.loc[hist["_analysis_group"].astype(str).eq(group), "usual_purchase_window"] if "usual_purchase_window" in hist else pd.Series(dtype=object)
        if usual.empty or usual.dropna().empty:
            usual = hist.loc[hist["_analysis_group"].astype(str).eq(group), "sales_window"] if "sales_window" in hist else pd.Series(dtype=object)
        row["usual_purchase_window"] = _mode_or_unknown(usual)
        rows.append(row)
    return pd.DataFrame(rows).set_index("_analysis_group")


def _season_group_summary(frame: pd.DataFrame, _: str) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["season_label", "_analysis_group", "paid_tickets", "total_tickets", "revenue", "purchasers"])
    return frame.groupby(["season_label", "_analysis_group"]).agg(
        paid_tickets=("paid_tickets_sold", "sum"),
        total_tickets=("tickets_sold", "sum"),
        revenue=("gross_revenue", "sum"),
        purchasers=("customer_id", "nunique"),
    ).reset_index()


def _mean_by_season(frame: pd.DataFrame, seasons: list[str], column: str) -> float:
    if not seasons:
        return 0.0
    if frame.empty:
        return 0.0
    values = frame.set_index("season_label")[column].reindex(seasons).fillna(0)
    return float(values.mean())


def _segment_audience_sizes(customers: pd.DataFrame) -> dict[str, dict[str, float]]:
    if customers.empty or "customer_id" not in customers:
        return {}
    frame = customers.copy()
    marketable = _bool_col(frame, "marketing_opt_in") | _bool_col(frame, "email_opt_in") | _bool_col(frame, "sms_opt_in")
    ticket_class = _string_col(frame, "preferred_ticket_class")
    usual_window = _string_col(frame, "usual_purchase_window")
    seasons = pd.to_numeric(_column_or_default(frame, "seasons_purchased_count", 0), errors="coerce").fillna(0)
    lifetime_tickets = pd.to_numeric(_column_or_default(frame, "lifetime_tickets", 0), errors="coerce").fillna(0)
    lifetime_revenue = pd.to_numeric(_column_or_default(frame, "lifetime_revenue", 0), errors="coerce").fillna(0)
    family = _bool_col(frame, "family_flag")
    age = _string_col(frame, "age_band")
    masks = {
        "Returning purchasers": seasons.ge(1),
        "Families": family,
        "18-30s": age.str.contains("18|19|20|21|22|23|24|25|26|27|28|29|30", regex=True),
        "Premium buyers": ticket_class.str.contains("platinum|premium", case=False, regex=True),
        "Gold ticket buyers": ticket_class.str.contains("gold", case=False, regex=True),
        "Boundary Zone buyers": ticket_class.str.contains("boundary", case=False, regex=True),
        "General Admission buyers": ticket_class.str.contains("general admission|\\bga\\b|hill", case=False, regex=True),
        "High basket-size purchasers": lifetime_tickets.ge(12),
        "Low basket-size purchasers": lifetime_tickets.gt(0) & lifetime_tickets.le(3),
        "High-value purchasers": lifetime_revenue.ge(lifetime_revenue.quantile(0.75) if len(lifetime_revenue) else 0),
        "Marketable customers": marketable,
        "Non-marketable customers": ~marketable,
        "Usual early-window buyers": usual_window.str.contains("early|pre-sale|on-sale", case=False, regex=True),
        "Usual late-window buyers": usual_window.str.contains("campaign|match-week|late", case=False, regex=True),
    }
    sizes = {}
    for label, mask in masks.items():
        eligible = frame[mask.fillna(False)]
        sizes[label] = {
            "eligible_audience_size": float(eligible["customer_id"].nunique()),
            "marketable_audience_size": float(eligible.loc[marketable.reindex(eligible.index).fillna(False), "customer_id"].nunique()),
        }
    return sizes


def _group_definitions(group_kind: str) -> dict[str, str]:
    if group_kind == "ticket_type":
        return {}
    if group_kind == "ticket_class":
        return {}
    return {label: definition for label, _, definition, _ in _segment_specs(pd.DataFrame(index=[0]))}


def _definition_for_group(group_kind: str, group: str) -> str:
    if group_kind == "ticket_type":
        return f"Ticket type / price type equals {group}."
    if group_kind == "ticket_class":
        return f"Ticket class equals {group}."
    if group_kind == "segment_product":
        return "Combined audience segment and ticket product group."
    return "Aggregated audience segment generated from available ticket and customer fields."


def _safe_index(actual: float, expected: float) -> float:
    if expected and expected > 0:
        return float(actual) / float(expected)
    return 1.25 if actual and actual > 0 else 0.0


def _index_confidence(history_seasons: object, expected_final_paid: float, eligible_size: float) -> str:
    seasons = int(history_seasons or 0)
    size = max(float(expected_final_paid or 0), float(eligible_size or 0))
    if seasons >= 3 and size >= 5000:
        return "High"
    if seasons >= 2 and size >= 1000:
        return "Medium"
    return "Directional"


def _recommendation_text(row: dict[str, object], group_kind: str) -> dict[str, str]:
    group = str(row.get("analysis_group", "Selected group"))
    status = str(row.get("status", "Watch"))
    ticket_gap = float(row.get("paid_ticket_gap", 0) or 0)
    ticket_index = float(row.get("ticket_index", 0) or 0)
    current_window = str(row.get("current_sales_window", "current window"))
    usual_window = str(row.get("usual_purchase_window", "Unknown"))
    product = _suggested_product(group_kind, group)
    if status == "Not due yet":
        action = "Monitor now and prepare a later-window reminder rather than treating this group as churned."
        angle = "Timing-led reminder when the group reaches its usual purchase window."
        timing = f"Prepare for {usual_window if usual_window != 'Unknown' else next_sales_window_label(current_window)}."
        rationale = (
            f"{group} is below full-season conversion, but historical behaviour points to later-window purchasing "
            f"({usual_window}). Current window is {current_window}."
        )
    elif status in {"At risk", "Behind", "Watch"}:
        action = "Prioritise this group in the current marketing plan."
        angle = _message_angle(group, ticket_gap)
        timing = f"Act in the {current_window} window."
        rationale = (
            f"{group} is {abs(ticket_gap):,.0f} paid tickets {'behind' if ticket_gap < 0 else 'ahead of'} expected by now "
            f"with a ticket index of {ticket_index:.2f}."
        )
    elif status == "Ahead":
        action = "Protect yield and avoid discount-led messaging."
        angle = "Availability, scarcity, and premium positioning."
        timing = "Keep active while demand is ahead of expectation."
        rationale = f"{group} is ahead of expected paid-ticket volume with a ticket index of {ticket_index:.2f}."
    else:
        action = "Keep planned activity running and watch for mix shifts."
        angle = "Availability and fixture-specific value."
        timing = f"Continue through the {current_window} window."
        rationale = f"{group} is broadly on track with a ticket index of {ticket_index:.2f}."
    return {
        "recommended_action": action,
        "suggested_message_angle": angle,
        "suggested_timing": timing,
        "suggested_ticket_product": product,
        "rationale": rationale,
    }


def _message_angle(group: str, ticket_gap: float) -> str:
    lower = group.lower()
    if "family" in lower or "boundary" in lower:
        return "Family value, ease of attendance, and urgency before the window closes."
    if "gold" in lower or "premium" in lower or "platinum" in lower:
        return "Premium availability and scarcity, not discounting."
    if "general admission" in lower or "ga" in lower:
        return "Simple group-ticket value and fixture urgency."
    if ticket_gap < -1000:
        return "Direct response messaging with a clear fixture and product recommendation."
    return "Fixture-specific reminder with availability and timing cues."


def _suggested_product(group_kind: str, group: str) -> str:
    if group_kind == "ticket_type":
        return group
    lower = group.lower()
    if "boundary" in lower:
        return "Boundary Zone"
    if "gold" in lower:
        return "Gold"
    if "premium" in lower or "platinum" in lower:
        return "Premium / Platinum"
    if "general admission" in lower or "ga" in lower:
        return "General Admission"
    if "family" in lower:
        return "Family ticket products"
    return "Best available fixture inventory"


def _report_index_rows(frame: pd.DataFrame | None, title: str) -> str:
    if frame is None or frame.empty:
        return ""
    cols = [
        "analysis_group",
        "status",
        "current_paid_tickets",
        "expected_paid_tickets_by_now",
        "ticket_index",
        "paid_ticket_gap",
        "recommended_action",
    ]
    available = [column for column in cols if column in frame]
    table = frame[available].head(6).to_html(index=False)
    return f"<h2>{title}</h2>{table}"


def _window_order(window: object) -> int:
    value = str(window or "").strip().lower().replace("_", "-").replace(" ", "-")
    return SALES_WINDOW_ORDER.get(value, 2)


def _string_col(frame: pd.DataFrame, column: str) -> pd.Series:
    return _column_or_default(frame, column, "").fillna("").astype(str)


def _bool_col(frame: pd.DataFrame, column: str) -> pd.Series:
    values = _column_or_default(frame, column, False)
    if values.dtype == bool:
        return values.fillna(False).astype(bool)
    cleaned = values.astype(object).where(values.notna(), False)
    return cleaned.astype(str).str.lower().isin({"true", "1", "yes", "y"})


def _first_purchase_season(transactions: pd.DataFrame) -> dict[str, str]:
    if transactions.empty or "customer_id" not in transactions:
        return {}
    first = transactions.dropna(subset=["customer_id", "transaction_date"]).sort_values("transaction_date")
    return first.groupby("customer_id")["season_label"].first().astype(str).to_dict()


def _days_to_next_fixture(fixtures: pd.DataFrame, as_at_date: pd.Timestamp | None) -> int:
    if fixtures.empty or "match_date" not in fixtures:
        return 0
    date = pd.to_datetime(as_at_date if as_at_date is not None else pd.Timestamp.today()).normalize()
    future = pd.to_datetime(fixtures["match_date"], errors="coerce")
    future = future[future >= date]
    if future.empty:
        return 0
    return int((future.min().normalize() - date).days)


def _risk_status(row: pd.Series) -> str:
    target = float(row.get("target", 0) or 0)
    forecast = float(row.get("forecast", 0) or 0)
    if target <= 0 or forecast <= 0:
        return "watch"
    ratio = forecast / target
    if ratio >= 1.02:
        return "on track"
    if ratio >= 0.92:
        return "watch"
    return "at risk"


def _mode_or_unknown(series: pd.Series) -> object:
    values = series.dropna()
    if values.empty:
        return "Unknown"
    mode = values.mode()
    return mode.iloc[0] if not mode.empty else values.iloc[0]


def _column_or_default(frame: pd.DataFrame, column: str, default: object) -> pd.Series:
    if column in frame:
        return frame[column]
    return pd.Series([default] * len(frame), index=frame.index)
