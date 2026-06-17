"""Plotly chart builders for the SACA / Adelaide Strikers planning dashboard."""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go


COLORS = {
    "ink": "#0F1728",
    "muted": "#67728A",
    "expected": "#005F9E",
    "actual": "#C9141B",
    "target": "#DAA907",
    "uplift": "#C9141B",
    "manual": "#0D7C66",
    "confidence": "rgba(0, 95, 158, 0.14)",
    "comparison": "#9AA3B2",
    "grid": "rgba(0, 95, 158, 0.10)",
    "panel": "rgba(255, 255, 255, 0.72)",
    "border": "rgba(15, 23, 40, 0.12)",
}


def format_number(value: float) -> str:
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:,.0f}"


def _base_layout(fig: go.Figure, y_title: str) -> go.Figure:
    fig.update_layout(
        title={"text": ""},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=COLORS["panel"],
        font={"family": "Aptos, Gill Sans, Trebuchet MS, sans-serif", "color": COLORS["ink"], "size": 13},
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
        margin={"l": 52, "r": 24, "t": 28, "b": 44},
        height=420,
        bargap=0.18,
        hoverlabel={"bgcolor": "rgba(15, 23, 40, 0.96)", "bordercolor": "rgba(255, 255, 255, 0.18)", "font": {"color": "#ffffff"}},
    )
    fig.update_xaxes(showgrid=False, title=None, linecolor=COLORS["border"], tickfont={"color": COLORS["ink"]})
    fig.update_yaxes(showgrid=True, gridcolor=COLORS["grid"], zeroline=False, title=y_title, linecolor=COLORS["border"], tickfont={"color": COLORS["ink"]})
    return fig


def cumulative_curve_figure(
    frame: pd.DataFrame,
    title: str,
    actual_col: str = "actual_cumulative",
    target_col: str = "target_cumulative",
    expected_col: str = "forecast_expected_cumulative",
    lower_col: str = "forecast_lower_cumulative",
    upper_col: str = "forecast_upper_cumulative",
    uplift_col: str = "uplift_10_cumulative",
    manual_col: str = "manual_target_cumulative",
) -> go.Figure:
    fig = go.Figure()
    if lower_col in frame and upper_col in frame:
        fig.add_trace(
            go.Scatter(
                x=frame["date"],
                y=frame[upper_col],
                mode="lines",
                line={"width": 0},
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=frame["date"],
                y=frame[lower_col],
                mode="lines",
                fill="tonexty",
                fillcolor=COLORS["confidence"],
                line={"width": 0},
                name="80% confidence range",
                hovertemplate="Lower %{y:,.0f}<extra></extra>",
            )
        )

    series_styles = [
        (actual_col, "Actual pace", COLORS["actual"], "solid", 3.6),
        (expected_col, "ML expected forecast", COLORS["expected"], "solid", 3.0),
        (target_col, "Planning target", COLORS["target"], "solid", 2.6),
        (uplift_col, "10% uplift target", COLORS["uplift"], "dot", 2.4),
        (manual_col, "Manual target", COLORS["manual"], "dashdot", 2.4),
    ]
    for column, name, color, dash, width in series_styles:
        if column in frame:
            fig.add_trace(
                go.Scatter(
                    x=frame["date"],
                    y=frame[column],
                    mode="lines",
                    name=name,
                    line={"color": color, "dash": dash, "width": width},
                    hovertemplate="%{x|%d %b %Y}<br>%{y:,.0f}<extra>" + name + "</extra>",
                )
            )
    fig = _base_layout(fig, "Cumulative tickets")
    fig.update_layout(meta={"title": title})
    return fig


def daily_sales_figure(frame: pd.DataFrame, title: str) -> go.Figure:
    fig = go.Figure()
    if "actual_daily" in frame:
        fig.add_trace(
            go.Bar(
                x=frame["date"],
                y=frame["actual_daily"],
                name="Actual daily sales",
                marker={"color": "rgba(201, 20, 27, 0.82)", "line": {"color": "rgba(15, 23, 40, 0.12)", "width": 0.6}},
                opacity=0.88,
                hovertemplate="%{x|%d %b %Y}<br>%{y:,.0f}<extra>Actual daily sales</extra>",
            )
        )
    if "required_daily" in frame:
        fig.add_trace(
            go.Scatter(
                x=frame["date"],
                y=frame["required_daily"],
                mode="lines",
                name="Required daily sales",
                line={"color": COLORS["target"], "width": 3},
                hovertemplate="%{x|%d %b %Y}<br>%{y:,.0f}<extra>Required daily sales</extra>",
            )
        )
    if "forecast_expected_daily" in frame:
        fig.add_trace(
            go.Scatter(
                x=frame["date"],
                y=frame["forecast_expected_daily"],
                mode="lines",
                name="ML expected daily",
                line={"color": COLORS["expected"], "width": 2.4, "dash": "solid"},
                hovertemplate="%{x|%d %b %Y}<br>%{y:,.0f}<extra>ML expected daily</extra>",
            )
        )
    fig = _base_layout(fig, "Daily tickets")
    fig.update_layout(meta={"title": title})
    return fig


def comparable_matches_figure(frame: pd.DataFrame) -> go.Figure:
    if frame.empty:
        return _base_layout(go.Figure(), "")

    display = frame.sort_values("final_sales", ascending=True).copy()
    fig = go.Figure(
        go.Bar(
            x=display["final_sales"],
            y=display["opponent"] + " | " + display["season_label"].astype(str),
            orientation="h",
            marker={"color": display["similarity"], "colorscale": [[0, "#D7DBE3"], [1, COLORS["expected"]]]},
            text=display["final_sales"].map(lambda value: f"{value:,.0f}"),
            textposition="outside",
            hovertemplate="Final sales %{x:,.0f}<br>Similarity %{marker.color:.2f}<extra></extra>",
        )
    )
    fig.update_layout(height=360, showlegend=False)
    return _base_layout(fig, "")


def feature_importance_figure(frame: pd.DataFrame) -> go.Figure:
    if frame.empty:
        return _base_layout(go.Figure(), "")

    display = frame.sort_values("importance", ascending=True)
    fig = go.Figure(
        go.Bar(
            x=display["importance"],
            y=display["feature"],
            orientation="h",
            marker={"color": COLORS["manual"]},
            hovertemplate="%{y}<br>Importance %{x:.3f}<extra></extra>",
        )
    )
    fig.update_layout(height=360, showlegend=False)
    return _base_layout(fig, "")


def season_comparison_figure(frame: pd.DataFrame, planning_season_label: str) -> go.Figure:
    fig = go.Figure()
    if frame.empty:
        return _base_layout(fig, "Cumulative tickets")

    for season_label, group in frame.groupby("season_label"):
        is_planning = season_label == planning_season_label
        fig.add_trace(
            go.Scatter(
                x=group["season_day"],
                y=group["actual_cumulative"],
                mode="lines",
                name=season_label,
                line={
                    "color": COLORS["actual"] if is_planning else COLORS["comparison"],
                    "width": 3.1 if is_planning else 2.0,
                    "dash": "solid" if is_planning else "dash",
                },
                hovertemplate="Day %{x}<br>%{y:,.0f}<extra>" + str(season_label) + "</extra>",
            )
        )
    fig = _base_layout(fig, "Cumulative tickets")
    fig.update_xaxes(title="Days since on-sale start", showgrid=False)
    return fig


def grouped_comparison_bar_figure(
    frame: pd.DataFrame,
    title: str,
    category_col: str = "analysis_group",
    actual_col: str = "current_paid_tickets",
    expected_col: str = "expected_paid_tickets_by_now",
    actual_label: str = "Actual by now",
    expected_label: str = "Expected by now",
) -> go.Figure:
    if frame.empty:
        return _base_layout(go.Figure(), "Tickets by now")

    display = (
        frame[[category_col, actual_col, expected_col]]
        .copy()
        .sort_values(expected_col, ascending=False)
        .head(12)
    )
    categories = display[category_col].astype(str).tolist()
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=categories,
            y=display[actual_col],
            name=actual_label,
            marker={"color": COLORS["actual"], "line": {"color": "rgba(15, 23, 40, 0.10)", "width": 0.5}},
            offsetgroup="actual",
            hovertemplate="%{x}<br>%{y:,.0f}<extra>" + actual_label + "</extra>",
        )
    )
    fig.add_trace(
        go.Bar(
            x=categories,
            y=display[expected_col],
            name=expected_label,
            marker={"color": COLORS["expected"], "line": {"color": "rgba(15, 23, 40, 0.10)", "width": 0.5}},
            offsetgroup="expected",
            hovertemplate="%{x}<br>%{y:,.0f}<extra>" + expected_label + "</extra>",
        )
    )
    fig = _base_layout(fig, "Tickets by now")
    fig.update_layout(meta={"title": title}, barmode="group", height=430)
    fig.update_xaxes(categoryorder="array", categoryarray=categories, tickangle=-24)
    return fig
