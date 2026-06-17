"""Generate SACA / Adelaide Strikers sample data for ticket planning demos."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "demo"
PLANNING_SEASON = 2026
PLANNING_SEASON_LABEL = "2026/27"
REFERENCE_DATE = pd.Timestamp("2026-12-29")


@dataclass(frozen=True)
class CompetitionConfig:
    competition: str
    venue_sequence: tuple[str, ...]
    opponents: tuple[tuple[str, int], ...]
    start_month: int
    day_offsets: tuple[int, ...]
    base_ticket_price: float
    baseline_demand: float
    membership_window_length: int
    early_bird_length: int
    on_sale_month: int
    on_sale_day: int
    major_rounds: tuple[int, ...]


COMPETITIONS = {
    "BBL": CompetitionConfig(
        competition="BBL",
        venue_sequence=(
            "Adelaide Oval",
            "Adelaide Oval",
            "Adelaide Oval",
            "Adelaide Oval",
            "Adelaide Oval",
        ),
        opponents=(
            ("Perth Scorchers", 93),
            ("Sydney Sixers", 95),
            ("Brisbane Heat", 91),
            ("Melbourne Stars", 82),
            ("Hobart Hurricanes", 79),
        ),
        start_month=12,
        day_offsets=(18, 26, 30, 41, 52),
        base_ticket_price=49.0,
        baseline_demand=18_200,
        membership_window_length=7,
        early_bird_length=18,
        on_sale_month=8,
        on_sale_day=18,
        major_rounds=(2, 3, 5),
    ),
    "WBBL": CompetitionConfig(
        competition="WBBL",
        venue_sequence=(
            "Karen Rolton Oval",
            "Adelaide Oval",
            "Karen Rolton Oval",
            "Karen Rolton Oval",
            "Adelaide Oval",
        ),
        opponents=(
            ("Sydney Thunder", 88),
            ("Brisbane Heat", 92),
            ("Perth Scorchers", 90),
            ("Melbourne Renegades", 76),
            ("Sydney Sixers", 94),
        ),
        start_month=10,
        day_offsets=(12, 19, 28, 35, 44),
        base_ticket_price=29.0,
        baseline_demand=7_100,
        membership_window_length=6,
        early_bird_length=16,
        on_sale_month=7,
        on_sale_day=28,
        major_rounds=(2, 5),
    ),
}


def _season_rows() -> list[dict[str, float]]:
    return [
        {"season": 2023, "season_label": "2023/24", "membership_base": 21_800, "form_index": 0.58},
        {"season": 2024, "season_label": "2024/25", "membership_base": 22_950, "form_index": 0.62},
        {"season": 2025, "season_label": "2025/26", "membership_base": 24_300, "form_index": 0.67},
        {"season": 2026, "season_label": "2026/27", "membership_base": 25_900, "form_index": 0.71},
    ]


def _campaign_theme(competition: str, round_number: int) -> str:
    if competition == "BBL" and round_number == 3:
        return "New Year's Eve Bash"
    if round_number in {1, 2}:
        return "Member Renewal"
    if round_number == 5:
        return "Finals Push"
    return "School Holiday Push"


def _event_category(competition: str, opponent: str, round_number: int) -> str:
    if opponent in {"Sydney Sixers", "Brisbane Heat", "Perth Scorchers"}:
        return "marquee"
    if competition == "BBL" and round_number == 3:
        return "new-year"
    if competition == "WBBL" and round_number in {1, 4}:
        return "family"
    return "standard"


def _build_event_date(season: int, competition: str, offset_days: int) -> pd.Timestamp:
    if competition == "WBBL":
        start = pd.Timestamp(year=season, month=10, day=1)
    else:
        start = pd.Timestamp(year=season, month=12, day=1)
    return start + pd.Timedelta(days=offset_days)


def build_matches(rng: np.random.Generator) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for season_row in _season_rows():
        season = int(season_row["season"])
        season_label = str(season_row["season_label"])
        membership_base = int(season_row["membership_base"])
        form_index = float(season_row["form_index"])

        for competition, config in COMPETITIONS.items():
            early_bird_start = pd.Timestamp(year=season, month=config.on_sale_month, day=config.on_sale_day)
            early_bird_end = early_bird_start + pd.Timedelta(days=config.early_bird_length - 1)
            member_presale_start = early_bird_end + pd.Timedelta(days=7)
            member_presale_end = member_presale_start + pd.Timedelta(days=config.membership_window_length - 1)
            general_public_start = member_presale_end + pd.Timedelta(days=1)

            for round_number, ((opponent, strength), venue, day_offset) in enumerate(
                zip(config.opponents, config.venue_sequence, config.day_offsets),
                start=1,
            ):
                event_date = _build_event_date(season, competition, day_offset)
                campaign_theme = _campaign_theme(competition, round_number)
                category = _event_category(competition, opponent, round_number)
                school_holiday = int(event_date.month in {12, 1})
                double_header = int(competition == "WBBL" and round_number in {2, 4})
                finals_contention = int(season >= 2025 and round_number >= 4)
                marketing_start = event_date - pd.Timedelta(days=int(rng.integers(24, 33)))
                marketing_end = event_date - pd.Timedelta(days=int(rng.integers(6, 12)))
                burst_date = event_date - pd.Timedelta(days=int(rng.integers(4, 10)))
                public_holiday = int(event_date.strftime("%m-%d") in {"12-31", "01-26"})
                day_of_week = event_date.day_name()

                demand_index = (
                    config.baseline_demand
                    + membership_base * (0.37 if competition == "BBL" else 0.16)
                    + strength * (118 if competition == "BBL" else 52)
                    + form_index * (6200 if competition == "BBL" else 2600)
                    + (2300 if category == "marquee" else 0)
                    + (1600 if category == "new-year" else 0)
                    + (900 if category == "family" else 0)
                    + (700 if school_holiday else 0)
                    + (650 if double_header else 0)
                    + (780 if finals_contention else 0)
                    + rng.normal(0, 420 if competition == "BBL" else 220)
                )

                baseline_target = int(round(max(demand_index, 2500) * 1.05 / 100) * 100)
                manual_target = int(round(baseline_target * 1.09 / 100) * 100)

                rows.append(
                    {
                        "match_id": f"{season}-{competition}-H{round_number:02d}",
                        "season": season,
                        "season_label": season_label,
                        "competition": competition,
                        "team": "Adelaide Strikers",
                        "organisation": "SACA",
                        "round": round_number,
                        "home_match_number": round_number,
                        "opponent": opponent,
                        "venue": venue,
                        "event_date": event_date,
                        "on_sale_date": early_bird_start,
                        "early_bird_start": early_bird_start,
                        "early_bird_end": early_bird_end,
                        "member_presale_start": member_presale_start,
                        "member_presale_end": member_presale_end,
                        "general_public_start": general_public_start,
                        "day_of_week": day_of_week,
                        "event_category": category,
                        "campaign_theme": campaign_theme,
                        "opponent_strength": strength,
                        "historical_attendance": int(demand_index * rng.uniform(0.94, 1.05)),
                        "ticket_price": round(float(rng.normal(config.base_ticket_price, 3.2)), 2),
                        "membership_base": membership_base,
                        "prior_season_performance": form_index,
                        "weather_temp_c": round(float(rng.normal(28 if competition == "BBL" else 23, 4)), 1),
                        "marketing_start": marketing_start,
                        "marketing_end": marketing_end,
                        "campaign_burst_date": burst_date,
                        "public_holiday": public_holiday,
                        "school_holiday": school_holiday,
                        "double_header": double_header,
                        "finals_contention": finals_contention,
                        "baseline_target": baseline_target,
                        "manual_target": manual_target,
                        "planning_season": int(season_label == PLANNING_SEASON_LABEL),
                    }
                )

    return pd.DataFrame(rows)


def _sales_weight(row: pd.Series, date: pd.Timestamp) -> float:
    days_to_event = (row["event_date"] - date).days
    days_on_sale = (date - row["on_sale_date"]).days

    early_bird = 2.8 if row["early_bird_start"] <= date <= row["early_bird_end"] else 0.0
    member_presale = 2.1 if row["member_presale_start"] <= date <= row["member_presale_end"] else 0.0
    launch = 2.0 * np.exp(-days_on_sale / 8.5)
    campaign_live = 1.2 if row["marketing_start"] <= date <= row["marketing_end"] else 0.0
    burst = 2.4 if abs((date - row["campaign_burst_date"]).days) <= 1 else 0.0
    final_week = 4.6 * np.exp(-days_to_event / 4.8)
    marquee = 0.9 if row["event_category"] in {"marquee", "new-year"} else 0.0
    school_holiday = 0.6 if row["school_holiday"] else 0.0
    weekend = 1.1 if date.day_name() in {"Friday", "Saturday", "Sunday"} else 0.92
    base = 0.55 + (0.016 * row["opponent_strength"])
    return max((base + early_bird + member_presale + launch + campaign_live + burst + final_week + marquee + school_holiday) * weekend, 0.02)


def _sales_window(row: pd.Series, date: pd.Timestamp, days_to_event: int) -> str:
    if row["early_bird_start"] <= date <= row["early_bird_end"]:
        return "early-bird"
    if row["member_presale_start"] <= date <= row["member_presale_end"]:
        return "member-pre-sale"
    if days_to_event <= 7:
        return "final-week"
    if row["marketing_start"] <= date <= row["marketing_end"]:
        return "campaign-live"
    return "general-public"


def build_daily_sales(matches: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    visible_date_lookup = {}

    for _, match in matches.iterrows():
        all_dates = pd.date_range(match["on_sale_date"], match["event_date"], freq="D")
        reference_date = REFERENCE_DATE if match["season_label"] == PLANNING_SEASON_LABEL else match["event_date"]
        visible_dates = all_dates[all_dates <= reference_date]
        visible_date_lookup[match["match_id"]] = set(visible_dates)

        final_multiplier = rng.normal(1.0, 0.05)
        if match["competition"] == "WBBL" and match["event_category"] == "marquee":
            final_multiplier += 0.06
        if match["competition"] == "BBL" and match["event_category"] == "new-year":
            final_multiplier += 0.08
        if match["finals_contention"]:
            final_multiplier += 0.03

        final_sales = int(max(match["baseline_target"] * final_multiplier, 1800))
        weights = np.array([_sales_weight(match, date) for date in all_dates], dtype=float)
        weights = weights / weights.sum()
        full_daily = rng.multinomial(final_sales, weights)

        cumulative = 0
        for date, daily_sales in zip(all_dates, full_daily):
            if date not in visible_date_lookup[match["match_id"]]:
                continue

            cumulative += int(daily_sales)
            days_to_event = int((match["event_date"] - date).days)
            sales_window = _sales_window(match, date, days_to_event)

            rows.append(
                {
                    "match_id": match["match_id"],
                    "date": date,
                    "days_to_event": days_to_event,
                    "daily_sales": int(daily_sales),
                    "cumulative_sales": int(cumulative),
                    "sales_window": sales_window,
                    "marketing_period": "active" if match["marketing_start"] <= date <= match["marketing_end"] else "none",
                    "is_public_holiday": int(match["public_holiday"] and date == match["event_date"]),
                    "is_campaign_burst": int(abs((date - match["campaign_burst_date"]).days) <= 1),
                    "membership_on_sale": int(match["member_presale_start"] <= date <= match["member_presale_end"]),
                    "early_bird_active": int(match["early_bird_start"] <= date <= match["early_bird_end"]),
                    "school_holiday": int(match["school_holiday"]),
                    "source": "sample_simulated",
                }
            )

    return pd.DataFrame(rows)


def main() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260529)
    matches = build_matches(rng)
    daily_sales = build_daily_sales(matches, rng)

    matches.to_csv(DATA_DIR / "matches.csv", index=False)
    daily_sales.to_csv(DATA_DIR / "daily_sales.csv", index=False)
    matches.to_parquet(DATA_DIR / "matches.parquet", index=False)
    daily_sales.to_parquet(DATA_DIR / "daily_sales.parquet", index=False)
    print(f"Wrote {len(matches):,} matches and {len(daily_sales):,} daily rows to {DATA_DIR}")


if __name__ == "__main__":
    main()
