"""Future fixture assumptions for unreleased Adelaide Strikers seasons."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


DEFAULT_CAPACITY_BY_VENUE = {
    "Adelaide Oval": 53_500,
    "Karen Rolton Oval": 5_000,
    "Hurstville Oval": 9_000,
}


@dataclass(frozen=True)
class FutureFixtureConfig:
    planning_season: int = 2026
    planning_season_label: str = "2026/27"
    random_seed: int = 2026
    team: str = "Adelaide Strikers"


def generate_assumed_future_fixtures(
    historical_fixtures: pd.DataFrame,
    config: FutureFixtureConfig | None = None,
    random_seed: int | None = None,
) -> pd.DataFrame:
    """Create reproducible TBD fixtures from recent historical home patterns.

    These rows are marked as assumed and are intended for scenario planning only.
    They should be appended to model inputs for future forecasts but excluded from
    any historical training interpretation.
    """

    config = config or FutureFixtureConfig()
    seed = config.random_seed if random_seed is None else random_seed
    rng = np.random.default_rng(seed)
    fixtures = _clean_historical_fixtures(historical_fixtures)
    rows: list[dict[str, object]] = []

    for competition in ("BBL", "WBBL"):
        comp_history = fixtures[fixtures["competition"].eq(competition)].copy()
        if comp_history.empty:
            comp_history = _fallback_history(competition, config)

        latest_season = comp_history["season_start"].max()
        base = comp_history[comp_history["season_start"].eq(latest_season)].sort_values("match_date").copy()
        if base.empty:
            base = comp_history.sort_values("match_date").tail(5).copy()

        opponent_pool = (
            comp_history["opponent"].dropna().astype(str).loc[lambda s: s.str.len().gt(0)].unique().tolist()
        )
        if not opponent_pool:
            opponent_pool = _default_opponents(competition)
        chosen_opponents = _choose_opponents(rng, opponent_pool, len(base))

        for number, (_, base_row) in enumerate(base.iterrows(), start=1):
            target_date = _roll_date_to_planning_season(pd.to_datetime(base_row["match_date"]), config.planning_season)
            target_date = _closest_same_weekday(target_date, str(base_row.get("day_of_week") or target_date.day_name()))
            if competition == "BBL" and number == min(3, len(base)):
                target_date = pd.Timestamp(config.planning_season, 12, 31)

            opponent = chosen_opponents[number - 1]
            venue = str(base_row.get("venue") or _default_venue(competition))
            fixture_id = (
                f"{competition}_{config.planning_season}_NYE_ASSUMED"
                if competition == "BBL" and target_date.month == 12 and target_date.day == 31
                else f"{competition}_{config.planning_season}_ASSUMED_{number:02d}"
            )
            rows.append(
                {
                    "fixture_id": fixture_id,
                    "season": config.planning_season,
                    "season_label": config.planning_season_label,
                    "competition": competition,
                    "team": config.team,
                    "opponent": opponent,
                    "venue": venue,
                    "match_date": target_date.normalize(),
                    "match_time": str(base_row.get("match_time") or "19:15"),
                    "day_of_week": target_date.day_name(),
                    "is_home_match": True,
                    "fixture_label": _fixture_label(competition, opponent, target_date, True),
                    "capacity_total": _capacity_for_fixture(base_row, venue),
                    "fixture_status": "assumed",
                    "is_confirmed": False,
                    "is_assumed": True,
                    "assumption_confidence": "medium",
                    "notes": (
                        "Guaranteed/assumed NYE fixture generated from historical Strikers fixture pattern."
                        if competition == "BBL" and target_date.month == 12 and target_date.day == 31
                        else "Assumed fixture generated from historical Strikers fixture pattern. Replace once official fixture is available."
                    ),
                }
            )

    return pd.DataFrame(rows).sort_values(["competition", "match_date"]).reset_index(drop=True)


def _clean_historical_fixtures(fixtures: pd.DataFrame) -> pd.DataFrame:
    cleaned = fixtures.copy()
    if "match_date" not in cleaned.columns:
        cleaned["match_date"] = cleaned["event_date"] if "event_date" in cleaned.columns else pd.NaT
    cleaned["match_date"] = pd.to_datetime(cleaned["match_date"], errors="coerce")
    cleaned = cleaned.dropna(subset=["match_date"])
    if "competition" in cleaned:
        cleaned["competition"] = cleaned["competition"].fillna("BBL").astype(str).str.upper()
    else:
        cleaned["competition"] = "BBL"
    if "opponent" not in cleaned.columns:
        cleaned["opponent"] = None
    if "venue" not in cleaned.columns:
        cleaned["venue"] = None
    if "capacity_total" not in cleaned.columns:
        cleaned["capacity_total"] = np.nan
    cleaned["season_start"] = cleaned["season"] if "season" in cleaned else cleaned["match_date"].map(_season_start_from_date)
    cleaned["season_start"] = pd.to_numeric(cleaned["season_start"], errors="coerce").fillna(
        cleaned["match_date"].map(_season_start_from_date)
    )
    cleaned["day_of_week"] = cleaned["day_of_week"] if "day_of_week" in cleaned else cleaned["match_date"].dt.day_name()
    return cleaned


def _fallback_history(competition: str, config: FutureFixtureConfig) -> pd.DataFrame:
    if competition == "BBL":
        dates = pd.to_datetime(["2025-12-18", "2025-12-26", "2025-12-31", "2026-01-10", "2026-01-21"])
        opponents = ["Perth Scorchers", "Brisbane Heat", "Sydney Sixers", "Melbourne Stars", "Hobart Hurricanes"]
    else:
        dates = pd.to_datetime(["2025-10-12", "2025-10-19", "2025-10-28", "2025-11-04", "2025-11-13"])
        opponents = ["Sydney Thunder", "Brisbane Heat", "Perth Scorchers", "Melbourne Renegades", "Sydney Sixers"]
    return pd.DataFrame(
        {
            "season_start": config.planning_season - 1,
            "competition": competition,
            "opponent": opponents,
            "venue": _default_venue(competition),
            "match_date": dates,
            "day_of_week": dates.day_name(),
            "capacity_total": DEFAULT_CAPACITY_BY_VENUE[_default_venue(competition)],
        }
    )


def _season_start_from_date(date: pd.Timestamp) -> int:
    return int(date.year if date.month >= 7 else date.year - 1)


def _roll_date_to_planning_season(date: pd.Timestamp, planning_year: int) -> pd.Timestamp:
    target_year = planning_year if date.month >= 7 else planning_year + 1
    return pd.Timestamp(year=target_year, month=int(date.month), day=min(int(date.day), 28 if date.month == 2 else int(date.day)))


def _closest_same_weekday(date: pd.Timestamp, weekday_name: str) -> pd.Timestamp:
    weekday_lookup = {
        "Monday": 0,
        "Tuesday": 1,
        "Wednesday": 2,
        "Thursday": 3,
        "Friday": 4,
        "Saturday": 5,
        "Sunday": 6,
    }
    target = weekday_lookup.get(weekday_name, date.weekday())
    candidates = [date + pd.Timedelta(days=offset) for offset in range(-3, 4)]
    return min(candidates, key=lambda candidate: (candidate.weekday() != target, abs((candidate - date).days)))


def _choose_opponents(rng: np.random.Generator, opponents: list[str], count: int) -> list[str]:
    if len(opponents) >= count:
        return rng.choice(opponents, size=count, replace=False).tolist()
    return rng.choice(opponents, size=count, replace=True).tolist()


def _default_opponents(competition: str) -> list[str]:
    if competition == "BBL":
        return ["Perth Scorchers", "Sydney Sixers", "Brisbane Heat", "Melbourne Stars", "Hobart Hurricanes"]
    return ["Sydney Thunder", "Brisbane Heat", "Perth Scorchers", "Melbourne Renegades", "Sydney Sixers"]


def _default_venue(competition: str) -> str:
    return "Adelaide Oval" if competition == "BBL" else "Karen Rolton Oval"


def _capacity_for_fixture(row: pd.Series, venue: str) -> int | None:
    value = row.get("capacity_total")
    if pd.notna(value):
        try:
            return int(float(value))
        except (TypeError, ValueError):
            pass
    return DEFAULT_CAPACITY_BY_VENUE.get(venue)


def _fixture_label(competition: str, opponent: str, match_date: pd.Timestamp, assumed: bool) -> str:
    suffix = "TBD / assumed" if assumed else "confirmed"
    return f"{competition} v {opponent} | {match_date.strftime('%d %b %Y')} | {suffix}"
