"""Historical sales pace profiles built from comparable matches."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from preprocessing.features import completed_match_ids


@dataclass(frozen=True)
class PaceProfile:
    """Normalized historical pace profile."""

    frame: pd.DataFrame
    sample_size: int


class HistoricalPaceEngine:
    """Build weighted historical pace curves from many seasons of sales data."""

    def __init__(self, max_window_days: int = 120, min_matches: int = 3) -> None:
        self.max_window_days = max_window_days
        self.min_matches = min_matches
        self.matches: pd.DataFrame | None = None
        self.daily_sales: pd.DataFrame | None = None

    def fit(self, matches: pd.DataFrame, daily_sales: pd.DataFrame) -> "HistoricalPaceEngine":
        self.matches = matches.copy()
        self.daily_sales = daily_sales.copy()
        self.matches["match_id"] = self.matches["match_id"].astype(str)
        self.daily_sales["match_id"] = self.daily_sales["match_id"].astype(str)
        return self

    def get_profile(
        self,
        selected_match: pd.Series | dict | None = None,
        filters: dict[str, object] | None = None,
        reference_date: pd.Timestamp | None = None,
    ) -> PaceProfile:
        """Return a weighted normalized daily and cumulative pace profile."""

        self._ensure_fit()
        assert self.matches is not None
        assert self.daily_sales is not None

        candidate_matches = self._candidate_matches(filters, reference_date)
        if len(candidate_matches) < self.min_matches and filters:
            candidate_matches = self._broaden_candidates(filters, reference_date)
        if selected_match is not None and "match_id" in selected_match:
            candidate_matches = candidate_matches[
                candidate_matches["match_id"].astype(str) != str(selected_match["match_id"])
            ]

        if candidate_matches.empty:
            candidate_matches = self.matches[
                self.matches["match_id"].isin(completed_match_ids(self.matches, self.daily_sales, reference_date))
            ]

        weights = self._weights_for_selected_match(candidate_matches, selected_match)
        profile = self._weighted_profile(candidate_matches["match_id"].astype(str).tolist(), weights)
        return PaceProfile(frame=profile, sample_size=len(candidate_matches))

    def comparable_matches(
        self,
        selected_match: pd.Series | dict,
        limit: int = 8,
        filters: dict[str, object] | None = None,
        reference_date: pd.Timestamp | None = None,
    ) -> pd.DataFrame:
        """Return a ranked list of historical comparable matches."""

        self._ensure_fit()
        assert self.matches is not None
        assert self.daily_sales is not None

        candidates = self._candidate_matches(filters, reference_date)
        if len(candidates) < self.min_matches and filters:
            candidates = self._broaden_candidates(filters, reference_date)
        candidates = candidates[candidates["match_id"].astype(str) != str(selected_match["match_id"])]
        if candidates.empty:
            return pd.DataFrame()

        weights = self._weights_for_selected_match(candidates, selected_match)
        finals = (
            self.daily_sales.groupby("match_id", as_index=False)["cumulative_sales"]
            .max()
            .rename(columns={"cumulative_sales": "final_sales"})
        )
        ranked = candidates.assign(similarity=weights).merge(finals, on="match_id", how="left")
        return ranked.sort_values("similarity", ascending=False).head(limit).reset_index(drop=True)

    def _candidate_matches(
        self,
        filters: dict[str, object] | None,
        reference_date: pd.Timestamp | None,
    ) -> pd.DataFrame:
        assert self.matches is not None
        assert self.daily_sales is not None

        complete_ids = completed_match_ids(self.matches, self.daily_sales, reference_date)
        candidates = self.matches[self.matches["match_id"].isin(complete_ids)].copy()

        if filters:
            for column, value in filters.items():
                if value in (None, "All", "Any") or column not in candidates.columns:
                    continue
                if isinstance(value, (list, tuple, set)):
                    candidates = candidates[candidates[column].isin(value)]
                else:
                    candidates = candidates[candidates[column].eq(value)]

        return candidates.reset_index(drop=True)

    def _broaden_candidates(
        self,
        filters: dict[str, object],
        reference_date: pd.Timestamp | None,
    ) -> pd.DataFrame:
        reduced = dict(filters)
        if "season_label" in reduced:
            reduced.pop("season_label")
            candidates = self._candidate_matches(reduced, reference_date)
            if len(candidates) >= self.min_matches:
                return candidates

        if "competition" in reduced:
            candidates = self._candidate_matches({"competition": reduced["competition"]}, reference_date)
            if len(candidates) >= self.min_matches:
                return candidates

        return self._candidate_matches(None, reference_date)

    def _weights_for_selected_match(
        self,
        candidates: pd.DataFrame,
        selected_match: pd.Series | dict | None,
    ) -> np.ndarray:
        if candidates.empty:
            return np.array([])

        weights = np.ones(len(candidates), dtype=float)
        if selected_match is not None:
            selected = dict(selected_match)

            weights += np.where(candidates["venue"].eq(selected.get("venue")), 0.45, 0)
            weights += np.where(candidates["competition"].eq(selected.get("competition")), 0.25, 0)
            weights += np.where(candidates["event_category"].eq(selected.get("event_category")), 0.35, 0)
            weights += np.where(candidates["day_of_week"].eq(selected.get("day_of_week")), 0.15, 0)
            weights += np.where(candidates["opponent"].eq(selected.get("opponent")), 0.35, 0)

            strength_gap = (candidates["opponent_strength"] - float(selected.get("opponent_strength", 0))).abs()
            weights += np.clip(0.35 - strength_gap / 300, 0, 0.35)

            round_gap = (candidates["round"] - int(selected.get("round", 0))).abs()
            weights += np.clip(0.15 - round_gap / 50, 0, 0.15)

        season_min = candidates["season"].min()
        recency = 1 + (candidates["season"] - season_min) * 0.04
        weights = weights * recency.to_numpy(dtype=float)
        return weights / weights.sum()

    def _weighted_profile(self, match_ids: list[str], weights: np.ndarray) -> pd.DataFrame:
        assert self.daily_sales is not None

        days = np.arange(self.max_window_days, -1, -1)
        daily_matrix = []
        effective_weights = []

        for match_id, weight in zip(match_ids, weights):
            match_daily = self.daily_sales[self.daily_sales["match_id"].eq(match_id)]
            final_sales = match_daily["cumulative_sales"].max()
            if final_sales <= 0:
                continue

            series = (
                match_daily.groupby("days_to_event")["daily_sales"]
                .sum()
                .reindex(days, fill_value=0)
                .astype(float)
                / final_sales
            )
            daily_matrix.append(series.to_numpy())
            effective_weights.append(weight)

        if not daily_matrix:
            daily_share = np.ones(len(days), dtype=float) / len(days)
            sample_size = 0
        else:
            weight_vector = np.array(effective_weights, dtype=float)
            weight_vector = weight_vector / weight_vector.sum()
            daily_share = np.average(np.vstack(daily_matrix), axis=0, weights=weight_vector)
            sample_size = len(daily_matrix)

        daily_share = np.clip(daily_share, 0, None)
        if daily_share.sum() == 0:
            daily_share = np.ones(len(days), dtype=float) / len(days)
        else:
            daily_share = daily_share / daily_share.sum()

        cumulative_share = daily_share.cumsum()
        return pd.DataFrame(
            {
                "days_to_event": days,
                "historical_daily_share": daily_share,
                "historical_cumulative_share": cumulative_share,
                "historical_sample_size": sample_size,
            }
        )

    def _ensure_fit(self) -> None:
        if self.matches is None or self.daily_sales is None:
            raise RuntimeError("HistoricalPaceEngine.fit must be called before get_profile.")
