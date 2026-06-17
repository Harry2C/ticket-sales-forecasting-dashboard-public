"""Replaceable machine learning forecaster for ticket sales outcomes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from preprocessing.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, build_training_snapshots, enrich_sales_frame


@dataclass(frozen=True)
class ForecastResult:
    expected_final_sales: float
    lower_final_sales: float
    upper_final_sales: float
    confidence_level: float = 0.80


def _one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


class TicketSalesForecaster:
    """Random-forest baseline forecaster with tree-quantile confidence intervals."""

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.pipeline: Pipeline | None = None
        self.feature_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES
        self.training_rows = 0
        self.fallback_final_sales = 0.0

    def fit(
        self,
        matches: pd.DataFrame,
        daily_sales: pd.DataFrame,
        reference_date: pd.Timestamp | None = None,
    ) -> "TicketSalesForecaster":
        training = build_training_snapshots(matches, daily_sales, reference_date)
        self.training_rows = len(training)
        self.fallback_final_sales = float(training["final_sales"].median()) if not training.empty else 0.0

        if len(training) < 10:
            self.pipeline = None
            return self

        for column in self.feature_columns:
            if column not in training.columns:
                training[column] = 0 if column in NUMERIC_FEATURES else "Unknown"

        preprocessor = ColumnTransformer(
            transformers=[
                ("numeric", "passthrough", NUMERIC_FEATURES),
                ("categorical", _one_hot_encoder(), CATEGORICAL_FEATURES),
            ],
            remainder="drop",
        )
        model = RandomForestRegressor(
            n_estimators=180,
            min_samples_leaf=3,
            random_state=self.random_state,
            n_jobs=1,
        )
        self.pipeline = Pipeline([("preprocess", preprocessor), ("model", model)])
        self.pipeline.fit(training[self.feature_columns], training["final_sales"])
        return self

    def latest_snapshot(
        self,
        matches: pd.DataFrame,
        daily_sales: pd.DataFrame,
        match_id: str,
    ) -> pd.Series:
        enriched = enrich_sales_frame(matches, daily_sales)
        match_snapshots = enriched[enriched["match_id"].astype(str).eq(str(match_id))]
        if match_snapshots.empty:
            match = matches[matches["match_id"].astype(str).eq(str(match_id))].iloc[0]
            return self._empty_snapshot(match)
        return match_snapshots.sort_values("date").iloc[-1]

    def predict_snapshot(self, snapshot: pd.Series | dict) -> ForecastResult:
        row = pd.DataFrame([dict(snapshot)])
        current_sales = float(row.get("cumulative_sales", pd.Series([0])).iloc[0])

        for column in self.feature_columns:
            if column not in row.columns:
                row[column] = 0 if column in NUMERIC_FEATURES else "Unknown"

        if self.pipeline is None:
            expected = max(self.fallback_final_sales, current_sales)
            spread = max(expected * 0.12, 500)
            return ForecastResult(expected, max(current_sales, expected - spread), expected + spread)

        expected = float(self.pipeline.predict(row[self.feature_columns])[0])
        expected = max(expected, current_sales)

        transformed = self.pipeline.named_steps["preprocess"].transform(row[self.feature_columns])
        forest = self.pipeline.named_steps["model"]
        tree_predictions = np.array([tree.predict(transformed)[0] for tree in forest.estimators_])
        lower = max(current_sales, float(np.quantile(tree_predictions, 0.10)))
        upper = max(expected, float(np.quantile(tree_predictions, 0.90)))
        return ForecastResult(expected, lower, upper)

    def feature_importance(self, top_n: int = 12) -> pd.DataFrame:
        if self.pipeline is None:
            return pd.DataFrame(columns=["feature", "importance"])

        preprocessor = self.pipeline.named_steps["preprocess"]
        model = self.pipeline.named_steps["model"]
        feature_names = preprocessor.get_feature_names_out()
        cleaned_names = [name.replace("numeric__", "").replace("categorical__", "") for name in feature_names]
        importances = pd.DataFrame({"feature": cleaned_names, "importance": model.feature_importances_})
        return importances.sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)

    def _empty_snapshot(self, match: pd.Series) -> pd.Series:
        snapshot = dict(match)
        snapshot.update(
            {
                "date": pd.to_datetime(match["on_sale_date"]),
                "days_to_event": int((pd.to_datetime(match["event_date"]) - pd.to_datetime(match["on_sale_date"])).days),
                "days_on_sale": 0,
                "sales_progress": 0.0,
                "cumulative_sales": 0.0,
                "daily_sales": 0.0,
                "trailing_3_day_sales": 0.0,
                "trailing_7_day_sales": 0.0,
                "sales_acceleration": 0.0,
                "sales_window": "early-bird",
                "marketing_period": "none",
                "is_public_holiday": 0,
                "is_campaign_burst": 0,
                "membership_on_sale": 0,
                "early_bird_active": 1,
                "days_since_early_bird_start": 0,
                "event_month": pd.to_datetime(match["event_date"]).month,
                "is_weekend_event": int(match["day_of_week"] in ("Saturday", "Sunday")),
                "school_holiday": int(match.get("school_holiday", 0)),
                "double_header": int(match.get("double_header", 0)),
            }
        )
        return pd.Series(snapshot)


def build_forecast_curve(
    match: pd.Series | dict,
    actual_daily: pd.DataFrame,
    pace_profile: pd.DataFrame,
    forecast: ForecastResult,
) -> pd.DataFrame:
    """Turn a final-sales prediction into a daily cumulative forecast curve."""

    match_dict = dict(match)
    on_sale_date = pd.to_datetime(match_dict["on_sale_date"]).normalize()
    event_date = pd.to_datetime(match_dict["event_date"]).normalize()
    dates = pd.date_range(on_sale_date, event_date, freq="D")

    curve = pd.DataFrame({"date": dates})
    curve["days_to_event"] = (event_date - curve["date"]).dt.days
    curve = curve.merge(
        pace_profile[["days_to_event", "historical_daily_share"]],
        on="days_to_event",
        how="left",
    )
    curve["historical_daily_share"] = curve["historical_daily_share"].fillna(0).clip(lower=0)

    actual = actual_daily.sort_values("date").copy()
    if actual.empty:
        latest_date = on_sale_date - pd.Timedelta(days=1)
        current_sales = 0.0
        actual_cumulative = pd.DataFrame(columns=["date", "actual_cumulative"])
    else:
        latest_date = pd.to_datetime(actual["date"]).max().normalize()
        current_sales = float(actual.loc[actual["date"].eq(latest_date), "cumulative_sales"].max())
        actual_cumulative = actual[["date", "cumulative_sales"]].rename(columns={"cumulative_sales": "actual_cumulative"})

    curve = curve.merge(actual_cumulative, on="date", how="left")
    curve["actual_cumulative"] = pd.to_numeric(curve["actual_cumulative"], errors="coerce")

    future_mask = curve["date"] > latest_date
    future_weights = curve.loc[future_mask, "historical_daily_share"]
    if future_mask.any() and future_weights.sum() > 0:
        normalized_future = future_weights / future_weights.sum()
    elif future_mask.any():
        normalized_future = pd.Series(np.ones(future_mask.sum()) / future_mask.sum(), index=curve.index[future_mask])
    else:
        normalized_future = pd.Series(dtype=float)

    for label, final_value in [
        ("forecast_expected_cumulative", forecast.expected_final_sales),
        ("forecast_lower_cumulative", forecast.lower_final_sales),
        ("forecast_upper_cumulative", forecast.upper_final_sales),
    ]:
        curve[label] = curve["actual_cumulative"]
        remaining = max(float(final_value) - current_sales, 0)
        future_cumulative = current_sales + (normalized_future * remaining).cumsum()
        curve.loc[future_mask, label] = future_cumulative
        curve[label] = pd.to_numeric(curve[label], errors="coerce").ffill().fillna(0.0)

    curve["expected_forecast_daily"] = curve["forecast_expected_cumulative"].diff().fillna(
        curve["forecast_expected_cumulative"]
    )
    curve["uplift_10_cumulative"] = curve["forecast_expected_cumulative"] * 1.10
    return curve
