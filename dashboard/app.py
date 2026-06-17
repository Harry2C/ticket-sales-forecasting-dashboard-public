"""Streamlit dashboard for Two Circles client ticket sales planning."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dashboard.branding import (
    inject_css,
    metric_tile,
    mini_tile,
    render_alert_strip,
    render_brand_header,
    two_circles_wordmark,
)
from dashboard.config import (
    APP_TITLE,
    COMPETITION_OPTIONS,
    DEFAULT_HISTORICAL_SEASONS,
    PLANNING_SEASON_LABEL,
    SACA_LOGO_PATH,
    STRIKERS_LOGO_PATH,
)
from dashboard.services import (
    aggregate_selected_outputs,
    apply_transaction_filters,
    audience_summary,
    build_season_comparison_frame,
    current_sales_window,
    demographic_coverage,
    expected_purchase_index,
    filter_options,
    fixture_sales_summary,
    history_filters,
    insight_narrative,
    kpi_summary,
    marketing_recommendations,
    match_label,
    page_state_key,
    recommended_audiences,
    report_html,
    sales_window_summary,
    stale_snapshot_status,
)
from forecasting.historical_pace import HistoricalPaceEngine
from forecasting.ml_model import TicketSalesForecaster
from forecasting.targets import TargetAssumptions
from preprocessing.strikers_ingestion import (
    PROCESSED_DATA_DIR,
    RAW_DATA_DIR,
    StrikersDataBundle,
    build_forecasting_inputs,
    data_cache_signature,
    detect_source_files,
    expected_file_dataframe,
    load_real_strikers_data,
)
from utils.csv_utils import anonymise_identifier
from utils.charting import (
    COLORS,
    cumulative_curve_figure,
    daily_sales_figure,
    feature_importance_figure,
    format_number,
    grouped_comparison_bar_figure,
    season_comparison_figure,
)
from utils.data_loader import latest_data_date


FIXTURE_TARGETS_PATH = PROCESSED_DATA_DIR / "fixture_targets.csv"
CLIENT_VERSION_PATH = PROCESSED_DATA_DIR / "client_dashboard_version.json"
PROCESSED_OUTPUT_FILENAMES = (
    "transactions_normalised.csv",
    "fixtures_normalised.csv",
    "customers_normalised.csv",
    "matches.csv",
    "daily_sales.csv",
)
DUMMY_2627_SOURCE = "demo_26_27_dummy_august"
DUMMY_2627_START = pd.Timestamp("2026-08-01")
DUMMY_2627_END = pd.Timestamp("2026-08-31")

SETUP_FILE_COPY = {
    "bbl_tickets": {
        "title": "Historic sales - primary competition",
        "body": "Ticket transactions with order dates, fixture names, quantities, prices, and customer IDs.",
    },
    "wbbl_tickets": {
        "title": "Historic sales - secondary competition",
        "body": "Optional second competition or segment export, saved into the existing two-competition model slot.",
    },
    "bbl_fixtures": {
        "title": "Historic fixtures - primary competition",
        "body": "Fixture dates, venues, opponents, competition, and capacity fields for the historic sales base.",
    },
    "wbbl_fixtures": {
        "title": "Historic fixtures - secondary competition",
        "body": "Optional second competition fixture list for richer pace and benchmark modelling.",
    },
    "customers": {
        "title": "Customer info",
        "body": "Customer or fan records linked by GigyaUID, customer ID, member ID, contact ID, or equivalent.",
    },
    "future_fixtures": {
        "title": "Next season fixtures",
        "body": "Optional assumed or confirmed fixtures to score. Leave blank and the tool creates an assumed fixture set from historic timing.",
    },
}


NAV_SECTIONS = {
    "Pre On Sale": [
        "Fixture Forecasting",
        "Target Breakdown",
        "Historic Sales",
    ],
    "In Season Sales": [
        "Audience Insights",
        "Audience & Marketing Planner",
        "Reports",
        "Future Fixture Assumptions",
    ],
    "Data & Admin": [
        "Data Admin",
        "Client Setup",
    ],
}

NAV_ICONS = {
    "Pre On Sale": "◫",
    "In Season Sales": "◪",
    "Data & Admin": "⚙",
    "Fixture Forecasting": "↗",
    "Target Breakdown": "▥",
    "Historic Sales": "◔",
    "Audience Insights": "◎",
    "Audience & Marketing Planner": "⌁",
    "Reports": "▣",
    "Future Fixture Assumptions": "⇢",
    "Client Setup": "◉",
    "Data Admin": "⚙",
}

PAGES = [page for pages in NAV_SECTIONS.values() for page in pages]


st.set_page_config(
    page_title=APP_TITLE,
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_data(show_spinner="Loading Strikers data...")
def cached_real_data(
    raw_dir: str,
    mapping_overrides_json: str,
    random_seed: int,
    reprocess_token: int,
    raw_cache_token: str,
    processed_cache_token: str,
) -> StrikersDataBundle:
    overrides = json.loads(mapping_overrides_json or "{}")
    return load_real_strikers_data(raw_dir, mapping_overrides=overrides, random_seed=random_seed)


@st.cache_data(show_spinner=False)
def cached_ui_data_bundle(
    data_mode: str,
    matches: pd.DataFrame,
    daily_sales: pd.DataFrame,
    transactions: pd.DataFrame,
    fixtures: pd.DataFrame,
    customers: pd.DataFrame,
    future_fixtures: pd.DataFrame,
    file_status: pd.DataFrame,
    column_mappings: pd.DataFrame,
    validation_warnings: pd.DataFrame,
    metrics_json: str,
) -> StrikersDataBundle:
    metrics = json.loads(metrics_json or "{}")
    return _prepare_ui_data_bundle_uncached(
        StrikersDataBundle(
            data_mode=data_mode,
            matches=matches,
            daily_sales=daily_sales,
            transactions=transactions,
            fixtures=fixtures,
            customers=customers,
            future_fixtures=future_fixtures,
            file_status=file_status,
            column_mappings=column_mappings,
            validation_warnings=validation_warnings,
            metrics=metrics,
            loaded_sources={},
        )
    )


@st.cache_resource(show_spinner=False)
def cached_models(matches: pd.DataFrame, daily_sales: pd.DataFrame, reference_date: pd.Timestamp):
    pace_engine = HistoricalPaceEngine(max_window_days=150, min_matches=3).fit(matches, daily_sales)
    forecaster = TicketSalesForecaster().fit(matches, daily_sales, reference_date)
    return pace_engine, forecaster


@st.cache_data(show_spinner=False)
def cached_fixture_forecast_table(matches: pd.DataFrame, daily_sales: pd.DataFrame, reference_date: pd.Timestamp) -> pd.DataFrame:
    assumed_matches = matches[
        matches["season_label"].astype(str).eq(PLANNING_SEASON_LABEL)
        & matches.get("is_assumed", pd.Series(False, index=matches.index)).fillna(False).astype(bool)
    ].copy()
    if assumed_matches.empty:
        return pd.DataFrame()
    pace_engine = HistoricalPaceEngine(max_window_days=150, min_matches=3).fit(matches, daily_sales)
    forecaster = TicketSalesForecaster().fit(matches, daily_sales, reference_date)
    rows = []
    assumptions = TargetAssumptions()
    season_options = _season_options(matches)
    for _, match in assumed_matches.sort_values(["competition", "event_date"]).iterrows():
        filters = history_filters(str(match["competition"]), [season for season in DEFAULT_HISTORICAL_SEASONS if season in season_options])
        output = _build_single_output(
            match,
            matches,
            daily_sales,
            pace_engine,
            forecaster,
            assumptions,
            0.10,
            filters,
        )
        forecast = float(output["forecast"].expected_final_sales)
        rows.append(
            {
                "fixture_id": str(match["match_id"]),
                "fixture": str(match.get("fixture_label", match.get("match_id", ""))),
                "competition": str(match["competition"]),
                "date": pd.to_datetime(match["event_date"]).date(),
                "opponent": str(match.get("opponent", "TBD")),
                "status": str(match.get("fixture_status", "assumed")).title(),
                "forecasted_sales": round(forecast),
                "forecast_10pct_uplift": round(forecast * 1.10),
            }
        )
    return pd.DataFrame(rows)


def csv_bytes(frame: pd.DataFrame) -> bytes:
    """Return privacy-safe CSV bytes for currently displayed aggregated data."""

    return frame.to_csv(index=False).encode("utf-8")


def _download_csv(frame: pd.DataFrame, filename: str, key: str, label: str = "Download CSV") -> None:
    if frame is None or frame.empty:
        return
    st.download_button(
        label,
        data=csv_bytes(frame),
        file_name=filename,
        mime="text/csv",
        key=key,
    )


def _safe_file_part(value: object) -> str:
    text = str(value or "all").strip().lower()
    cleaned = "".join(char if char.isalnum() else "_" for char in text)
    return "_".join(part for part in cleaned.split("_") if part) or "all"


@st.cache_data(show_spinner="Building expected-by-now comparison...")
def cached_expected_index(
    transactions: pd.DataFrame,
    customers: pd.DataFrame,
    matches: pd.DataFrame,
    current_season_label: str,
    competition: str,
    as_at_date: pd.Timestamp,
    group_kind: str,
    baseline_seasons: tuple[str, ...],
    min_size: int,
    thresholds_json: str,
) -> pd.DataFrame:
    thresholds = json.loads(thresholds_json or "{}")
    return expected_purchase_index(
        transactions,
        customers,
        matches,
        current_season_label,
        competition=competition,
        as_at_date=as_at_date,
        group_kind=group_kind,
        baseline_seasons=baseline_seasons,
        min_size=min_size,
        thresholds=thresholds,
    )


def main() -> None:
    inject_css()
    _init_session_state()

    requested_page = str(st.session_state.get("page", ""))
    if requested_page == "Client Setup" or not st.session_state.get("client_setup_complete", False):
        _render_client_setup()
        return

    bundle, global_context = _load_app_context()
    if bundle.matches.empty:
        _render_app_header(bundle, global_context)
        _render_top_nav()
        st.error("No fixture model is available yet. Add client CSVs in Data Admin or data/raw/ and reprocess.")
        _render_data_admin(bundle)
        return

    reference_date = _reference_date(bundle)
    pace_engine, forecaster = cached_models(bundle.matches, bundle.daily_sales, reference_date)
    _render_app_header(bundle, global_context)
    page = _render_top_nav()

    if page == "Historic Sales":
        _render_historic_sales(bundle)
    elif page == "Fixture Forecasting":
        _render_fixture_forecasting_table(bundle, pace_engine, forecaster)
    elif page == "Target Breakdown":
        controls = _target_breakdown_controls(bundle)
        _render_target_breakdown(bundle, pace_engine, forecaster, controls)
    elif page == "Audience Insights":
        _render_audience_insights_tracking(bundle, pace_engine, forecaster, global_context)
    elif page == "Audience & Marketing Planner":
        controls = _inline_planning_controls(bundle, "audience_planner", include_customer_filters=False)
        context = _build_page_context(bundle, pace_engine, forecaster, controls)
        _render_audience_marketing_planner(bundle, controls, context, global_context)
    elif page == "Reports":
        controls = _inline_planning_controls(bundle, "reports", include_customer_filters=False)
        context = _build_page_context(bundle, pace_engine, forecaster, controls)
        _render_reports(context, controls, bundle, global_context)
    elif page == "Future Fixture Assumptions":
        _render_future_fixtures(bundle)
    elif page == "Data Admin":
        _render_data_admin(bundle)


def _render_client_setup() -> None:
    expected = expected_file_dataframe()
    detected = detect_source_files(RAW_DATA_DIR)
    expected = expected.assign(
        detected_file=expected["file_key"].map(lambda key: detected.get(str(key)).name if detected.get(str(key)) else ""),
    )
    required = expected[expected["required"].astype(bool)]
    required_ready = int(required["detected_file"].astype(str).ne("").sum())
    optional_ready = int(expected[~expected["required"].astype(bool)]["detected_file"].astype(str).ne("").sum())
    saved_version = _client_saved_version_summary()
    processed_ready = bool(saved_version["processed_ready"])

    st.markdown(
        f"""
        <section class="tc-setup-hero">
          <div class="tc-setup-content">
            <div class="tc-setup-kicker">Two Circles client model setup</div>
            <h1 class="tc-setup-title">Know fans best. Forecast the season next.</h1>
            <p class="tc-setup-copy">
              Upload historic sales, historic fixtures, customer information, and optional next-season fixtures.
              The app normalises the inputs, joins customers to ticket rows, builds the ML forecast model, and opens the planning dashboard against the next fixture set.
            </p>
            <div class="tc-setup-metrics">
              <div class="tc-setup-metric"><strong>{required_ready}/{len(required)}</strong><span>required file slots detected</span></div>
              <div class="tc-setup-metric"><strong>{"uploaded" if optional_ready else "auto"}</strong><span>next-season fixture source</span></div>
              <div class="tc-setup-metric"><strong>{"ready" if processed_ready else "fresh"}</strong><span>processed model cache</span></div>
            </div>
          </div>
          <div class="tc-loader-card">
            <div class="tc-loader-stage">
              <div class="tc-loader-rings"></div>
              <div class="tc-loader-wordmark">{two_circles_wordmark("tc-loader-logo")}</div>
            </div>
          </div>
        </section>
        <div class="tc-pipeline-strip">
          <div class="tc-pipeline-step"><strong>1. Upload</strong><span>Historic sales, fixtures, customer info, optional next-season assumptions.</span></div>
          <div class="tc-pipeline-step"><strong>2. Ingest</strong><span>Normalise columns, parse dates, map customers, validate joins.</span></div>
          <div class="tc-pipeline-step"><strong>3. Model</strong><span>Train historical pace and ML final-sales forecasts.</span></div>
          <div class="tc-pipeline-step"><strong>4. Predict</strong><span>Score uploaded fixtures or auto-created assumptions.</span></div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    _render_saved_version_panel(saved_version)

    st.markdown('<div class="section-kicker">Client data upload</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">Use the current raw files already detected, upload replacements below, or mix both. If no next-season fixture file is available, the tool creates assumed fixtures from historic timing and marks them as assumptions.</div>',
        unsafe_allow_html=True,
    )
    st.dataframe(
        expected[["label", "expected_file", "required", "detected_file", "kind", "competition"]],
        use_container_width=True,
        hide_index=True,
    )

    uploaded_files: dict[str, tuple[str, object]] = {}
    upload_cols = st.columns(2)
    for idx, row in expected.iterrows():
        file_key = str(row["file_key"])
        copy = SETUP_FILE_COPY.get(file_key, {})
        with upload_cols[idx % 2]:
            st.markdown(f'<div class="tc-setup-panel"><h3>{copy.get("title", row["label"])}</h3><p>{copy.get("body", "")}</p></div>', unsafe_allow_html=True)
            uploaded = st.file_uploader(
                str(row["expected_file"]),
                type=["csv"],
                key=f"setup_upload_{file_key}",
                help=f"Saved as {row['expected_file']} in data/raw/ for this local dashboard.",
            )
            if uploaded is not None:
                uploaded_files[file_key] = (str(row["expected_file"]), uploaded)

    action_cols = st.columns([1.2, 0.8, 2.2])
    run_clicked = action_cols[0].button("Ingest data and build model", type="primary", use_container_width=True)
    open_label = "Open saved dashboard" if processed_ready else "Open current dashboard"
    if action_cols[1].button(open_label, use_container_width=True, disabled=not processed_ready):
        st.session_state["client_setup_complete"] = True
        st.session_state["page"] = "Fixture Forecasting"
        st.session_state["nav_section"] = "Pre On Sale"
        st.session_state["nav_page"] = "Fixture Forecasting"
        st.rerun()

    if run_clicked:
        _run_client_setup_pipeline(uploaded_files)


def _processed_outputs_ready() -> bool:
    return all((PROCESSED_DATA_DIR / filename).exists() for filename in PROCESSED_OUTPUT_FILENAMES)


def _processed_outputs_timestamp() -> str:
    paths = [PROCESSED_DATA_DIR / filename for filename in PROCESSED_OUTPUT_FILENAMES]
    existing = [path for path in paths if path.exists()]
    if not existing:
        return "Unavailable"
    modified = max(path.stat().st_mtime for path in existing)
    return pd.Timestamp(modified, unit="s").strftime("%d %b %Y %H:%M")


def _read_client_version_manifest() -> dict[str, object]:
    if not CLIENT_VERSION_PATH.exists():
        return {}
    try:
        return json.loads(CLIENT_VERSION_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _client_saved_version_summary() -> dict[str, object]:
    targets = _load_fixture_targets()
    target_saved_at = _fixture_targets_last_saved()
    detected = detect_source_files(RAW_DATA_DIR)
    future_source = "Uploaded fixtures" if detected.get("future_fixtures") is not None else "Auto-created assumptions"
    manifest = _read_client_version_manifest()
    return {
        "processed_ready": _processed_outputs_ready(),
        "processed_updated_at": _processed_outputs_timestamp(),
        "target_count": len(targets),
        "target_saved_at": target_saved_at or "Not saved yet",
        "future_source": future_source,
        "manifest": manifest,
    }


def _render_saved_version_panel(saved_version: dict[str, object]) -> None:
    if not saved_version["processed_ready"]:
        st.info("No saved dashboard version yet. Upload data and build the model to create one.")
        return

    manifest = saved_version.get("manifest", {})
    st.markdown('<div class="section-kicker">Saved dashboard version</div>', unsafe_allow_html=True)
    cols = st.columns(4)
    cols[0].metric("Processed data", "Ready", str(saved_version["processed_updated_at"]))
    cols[1].metric("Saved fixture targets", f"{saved_version['target_count']:,}", str(saved_version["target_saved_at"]))
    cols[2].metric("Fixture source", str(saved_version["future_source"]), "Used for next-season scoring")
    cols[3].metric("Last model build", str(manifest.get("model_built_at", "Available")), f"{manifest.get('scored_fixtures', 0):,} scored fixtures")
    if int(saved_version["target_count"]) <= 0:
        st.warning("A saved data/model version is available, but fixture targets have not been saved yet.")


def _write_client_version_manifest(
    bundle: StrikersDataBundle | None = None,
    training_rows: int | None = None,
    scored_fixtures: int | None = None,
) -> None:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    existing = _read_client_version_manifest()
    targets = _load_fixture_targets()
    manifest = {
        **existing,
        "saved_at": pd.Timestamp.now().isoformat(),
        "processed_updated_at": _processed_outputs_timestamp(),
        "target_count": len(targets),
        "target_saved_at": _fixture_targets_last_saved() or "",
        "future_fixture_source": _client_saved_version_summary()["future_source"],
    }
    if bundle is not None:
        manifest.update(
            {
                "data_mode": bundle.data_mode,
                "ticket_rows": int(bundle.metrics.get("ticket_rows", 0)),
                "fixture_rows": int(bundle.metrics.get("fixture_rows", 0)),
                "customer_rows": int(bundle.metrics.get("customer_rows", 0)),
            }
        )
    if training_rows is not None:
        manifest["training_rows"] = int(training_rows)
        manifest["model_built_at"] = pd.Timestamp.now().strftime("%d %b %Y %H:%M")
    if scored_fixtures is not None:
        manifest["scored_fixtures"] = int(scored_fixtures)
    CLIENT_VERSION_PATH.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def _save_setup_uploads(uploaded_files: dict[str, tuple[str, object]]) -> int:
    if not uploaded_files:
        return 0
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for filename, uploaded in uploaded_files.values():
        (RAW_DATA_DIR / filename).write_bytes(uploaded.getbuffer())
    return len(uploaded_files)


def _clear_pipeline_caches() -> None:
    for cached_function in [
        cached_real_data,
        cached_ui_data_bundle,
        cached_models,
        cached_fixture_forecast_table,
        cached_expected_index,
    ]:
        cached_function.clear()


def _has_future_fixture_input(uploaded_files: dict[str, tuple[str, object]]) -> bool:
    if "future_fixtures" in uploaded_files:
        return True
    return detect_source_files(RAW_DATA_DIR).get("future_fixtures") is not None


def _run_client_setup_pipeline(uploaded_files: dict[str, tuple[str, object]]) -> None:
    with st.status("Building client forecast model...", expanded=True) as status:
        try:
            saved_count = _save_setup_uploads(uploaded_files)
            if saved_count:
                status.write(f"Saved {saved_count} uploaded CSV file(s) into data/raw/.")
            else:
                status.write("Using detected files from data/raw/ and the local processed cache where valid.")

            st.session_state["reprocess_token"] = st.session_state.get("reprocess_token", 0) + 1
            _clear_pipeline_caches()
            status.write("Starting ingestion and validation.")
            if _has_future_fixture_input(uploaded_files):
                status.write("Using uploaded next-season fixture assumptions for prediction.")
            else:
                status.write("No next-season fixture file found; creating assumed fixtures from historic timing.")

            progress_rows: list[str] = []

            def progress(message: str) -> None:
                progress_rows.append(message)
                status.write(message)

            bundle = load_real_strikers_data(
                RAW_DATA_DIR,
                mapping_overrides=st.session_state.get("mapping_overrides", {}),
                random_seed=int(st.session_state.get("fixture_seed", 2026)),
                write_processed=True,
                progress_callback=progress,
            )
            status.write("Preparing dashboard data model.")
            bundle = _prepare_ui_data_bundle(bundle)
            reference_date = _reference_date(bundle)
            status.write("Training historical pace engine and ML ticket-sales forecaster.")
            _, forecaster = cached_models(bundle.matches, bundle.daily_sales, reference_date)
            fixture_forecasts = cached_fixture_forecast_table(bundle.matches, bundle.daily_sales, reference_date)
            scored_count = len(fixture_forecasts)
            training_rows = int(getattr(forecaster, "training_rows", 0))
            _write_client_version_manifest(bundle, training_rows=training_rows, scored_fixtures=scored_count)
            status.update(
                label=f"Model ready: {training_rows:,} training snapshots, {scored_count:,} next-season fixture forecasts",
                state="complete",
            )
            st.session_state["client_setup_complete"] = True
            st.session_state["client_setup_summary"] = {
                "training_rows": training_rows,
                "scored_fixtures": scored_count,
                "progress": progress_rows,
            }
            st.session_state["page"] = "Fixture Forecasting"
            st.session_state["nav_section"] = "Pre On Sale"
            st.session_state["nav_page"] = "Fixture Forecasting"
            st.rerun()
        except Exception as exc:  # pragma: no cover - surfaced directly in Streamlit
            status.update(label="Data ingest needs attention", state="error")
            st.exception(exc)


def _load_app_context() -> tuple[StrikersDataBundle, dict[str, object]]:
    seed = int(st.session_state.get("fixture_seed", 2026))
    overrides_json = json.dumps(st.session_state.get("mapping_overrides", {}), sort_keys=True)
    raw_cache_token, processed_cache_token = data_cache_signature(RAW_DATA_DIR)
    bundle = cached_real_data(
        str(RAW_DATA_DIR),
        overrides_json,
        int(seed),
        int(st.session_state.get("reprocess_token", 0)),
        raw_cache_token,
        processed_cache_token,
    )
    bundle = _prepare_ui_data_bundle(bundle)
    latest_transaction_date = _reference_date(bundle)
    freshness = stale_snapshot_status(latest_transaction_date, today=pd.Timestamp.today())
    return bundle, {"latest_transaction_date": latest_transaction_date, "freshness": freshness}


def _prepare_ui_data_bundle(bundle: StrikersDataBundle) -> StrikersDataBundle:
    prepared = cached_ui_data_bundle(
        bundle.data_mode,
        bundle.matches,
        bundle.daily_sales,
        bundle.transactions,
        bundle.fixtures,
        bundle.customers,
        bundle.future_fixtures,
        bundle.file_status,
        bundle.column_mappings,
        bundle.validation_warnings,
        json.dumps(bundle.metrics, sort_keys=True, default=str),
    )
    prepared.loaded_sources = bundle.loaded_sources
    return prepared


def _prepare_ui_data_bundle_uncached(bundle: StrikersDataBundle) -> StrikersDataBundle:
    transactions = bundle.transactions.copy()
    customers = bundle.customers.copy()
    dummy_added = False

    if _should_add_dummy_2627_sales(transactions):
        dummy_transactions, dummy_customers = _generate_dummy_2627_sales(bundle)
        if not dummy_transactions.empty:
            transactions = pd.concat([transactions, dummy_transactions], ignore_index=True)
            customers = _merge_customer_rows(customers, dummy_customers)
            dummy_added = True

    transactions, demographic_metrics = _enrich_transactions_with_demographics(transactions, customers)
    matches = bundle.matches.copy()
    daily_sales = bundle.daily_sales.copy()
    if dummy_added:
        matches, daily_sales = build_forecasting_inputs(transactions, bundle.fixtures)

    metrics = dict(bundle.metrics)
    metrics.update(_ui_transaction_metrics(transactions, bundle.fixtures, customers))
    metrics.update(demographic_metrics)
    metrics["dummy_26_27_sales_active"] = bool(dummy_added)
    metrics["dummy_26_27_sales_label"] = (
        f"Synthetic August snapshot ({DUMMY_2627_START.date()} to {DUMMY_2627_END.date()})"
        if dummy_added
        else "Real or no 26/27 sales in current extract"
    )
    metrics["snapshot_as_at_date"] = _date_label(_latest_transaction_date(transactions))

    warnings = bundle.validation_warnings.copy()
    if dummy_added:
        warnings = pd.concat(
            [
                warnings,
                pd.DataFrame(
                    [
                        {
                            "severity": "info",
                            "area": "demo 26/27 snapshot",
                            "message": "Synthetic 26/27 sales from 1 Aug to 31 Aug 2026 are appended in-memory because no real 26/27 sales were found. Raw and processed real data are not overwritten.",
                        }
                    ]
                ),
            ],
            ignore_index=True,
        ).drop_duplicates()

    return StrikersDataBundle(
        data_mode=bundle.data_mode,
        matches=matches,
        daily_sales=daily_sales,
        transactions=transactions,
        fixtures=bundle.fixtures,
        customers=customers,
        future_fixtures=bundle.future_fixtures,
        file_status=bundle.file_status,
        column_mappings=bundle.column_mappings,
        validation_warnings=warnings,
        metrics=metrics,
        loaded_sources=bundle.loaded_sources,
    )


def _should_add_dummy_2627_sales(transactions: pd.DataFrame) -> bool:
    if transactions.empty or "season_label" not in transactions:
        return True
    planning = transactions[transactions["season_label"].astype(str).eq(PLANNING_SEASON_LABEL)].copy()
    if planning.empty:
        return True
    if "source_file" in planning:
        real_planning = planning[~planning["source_file"].astype(str).eq(DUMMY_2627_SOURCE)]
        return real_planning.empty
    return False


def _generate_dummy_2627_sales(bundle: StrikersDataBundle) -> tuple[pd.DataFrame, pd.DataFrame]:
    planning_matches = bundle.matches[bundle.matches["season_label"].astype(str).eq(PLANNING_SEASON_LABEL)].copy()
    if planning_matches.empty:
        return pd.DataFrame(), pd.DataFrame()
    planning_matches = planning_matches.sort_values(["competition", "event_date"]).reset_index(drop=True)
    rng = np.random.default_rng(20260831)
    customers = _dummy_customer_demographics(rng)
    customer_ids = customers["customer_id"].tolist()
    products = [
        {"ticket_type": "Adult", "ticket_class": "Gold", "price": 58.0, "share": 0.24, "family": False},
        {"ticket_type": "Family", "ticket_class": "Boundary Zone", "price": 36.0, "share": 0.25, "family": True},
        {"ticket_type": "Junior", "ticket_class": "General Admission", "price": 19.0, "share": 0.18, "family": True},
        {"ticket_type": "Concession", "ticket_class": "General Admission", "price": 31.0, "share": 0.16, "family": False},
        {"ticket_type": "Group", "ticket_class": "Premium Reserved", "price": 64.0, "share": 0.12, "family": False},
    ]
    rows: list[dict[str, object]] = []
    row_number = 1
    dates = pd.date_range(DUMMY_2627_START, DUMMY_2627_END, freq="D")
    for _, match in planning_matches.iterrows():
        competition = str(match.get("competition", "BBL"))
        match_id = str(match.get("match_id"))
        event_date = pd.to_datetime(match.get("event_date"), errors="coerce")
        is_nye = event_date.strftime("%m-%d") == "12-31" if pd.notna(event_date) else False
        comp_daily_rate = 0.07 if competition == "BBL" else 0.22
        base = 22 if competition == "BBL" else 8
        base *= 1.34 if is_nye else 1.0
        base *= 1.16 if str(match.get("fixture_status", "")).lower() == "assumed" else 1.0
        for date in dates:
            day = int((date - DUMMY_2627_START).days)
            launch = 2.55 if day <= 4 else 1.0
            late_uplift = 1.45 if day >= 24 else 1.0
            weekend = 1.12 if date.day_name() in {"Friday", "Saturday", "Sunday"} else 0.94
            mid_slowdown = 0.72 if 10 <= day <= 20 else 1.0
            daily_target = max(base * launch * late_uplift * weekend * mid_slowdown, 1.0)
            for product in products:
                product_bias = 1.0
                if product["ticket_type"] == "Family" and day < 10:
                    product_bias = 1.22
                if product["ticket_class"] == "Gold" and day >= 24:
                    product_bias = 1.18
                paid_tickets = int(max(0, rng.poisson(daily_target * float(product["share"]) * product_bias)))
                if paid_tickets:
                    customer_id = _dummy_gigya_for_row(rng, customer_ids, row_number)
                    demographics = _customer_demographic_lookup(customers, customer_id)
                    rows.append(
                        _dummy_transaction_row(
                            match,
                            date,
                            row_number,
                            customer_id,
                            demographics,
                            product["ticket_type"],
                            product["ticket_class"],
                            "Online" if row_number % 3 else "Member pre-sale",
                            paid_tickets,
                            paid_tickets,
                            0,
                            float(product["price"]),
                            bool(product["family"]),
                            False,
                        )
                    )
                    row_number += 1
            comp_tickets = int(max(0, rng.poisson(daily_target * comp_daily_rate)))
            if comp_tickets:
                customer_id = _dummy_gigya_for_row(rng, customer_ids, row_number, match_probability=0.48)
                demographics = _customer_demographic_lookup(customers, customer_id)
                rows.append(
                    _dummy_transaction_row(
                        match,
                        date,
                        row_number,
                        customer_id,
                        demographics,
                        "Complimentary",
                        "General Admission",
                        "Community allocation" if competition == "WBBL" else "Partner allocation",
                        comp_tickets,
                        0,
                        comp_tickets,
                        0.0,
                        True,
                        True,
                    )
                )
                row_number += 1

    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame, customers
    for column in bundle.transactions.columns:
        if column not in frame:
            frame[column] = None
    return frame[bundle.transactions.columns], customers


def _dummy_customer_demographics(rng: np.random.Generator, size: int = 1600) -> pd.DataFrame:
    ids = [f"dummy-gigya-{idx:05d}" for idx in range(1, size + 1)]
    age_bands = rng.choice(["18-30", "31-45", "46-60", "61+", "Family"], size=size, p=[0.24, 0.31, 0.21, 0.10, 0.14])
    postcodes = rng.choice(["5000", "5006", "5031", "5045", "5067", "5081", "5159"], size=size, p=[0.18, 0.14, 0.17, 0.14, 0.12, 0.13, 0.12])
    return pd.DataFrame(
        {
            "source_file": "demo_26_27_customer_demographics",
            "customer_id": ids,
            "anonymised_customer_id": [anonymise_identifier(value) for value in ids],
            "postcode": postcodes,
            "suburb": rng.choice(["Adelaide", "North Adelaide", "Unley", "Glenelg", "Norwood", "Prospect", "Flagstaff Hill"], size=size),
            "state": "SA",
            "age_band": age_bands,
            "gender": rng.choice(["Female", "Male", "Unknown"], size=size, p=[0.49, 0.47, 0.04]),
            "family_flag": pd.Series(age_bands).eq("Family").to_numpy(),
            "marketing_opt_in": rng.choice([True, False], size=size, p=[0.76, 0.24]),
            "email_opt_in": rng.choice([True, False], size=size, p=[0.70, 0.30]),
            "sms_opt_in": rng.choice([True, False], size=size, p=[0.42, 0.58]),
            "has_customer_record": True,
            "has_ticket_history": False,
        }
    )


def _dummy_gigya_for_row(
    rng: np.random.Generator,
    customer_ids: list[str],
    row_number: int,
    match_probability: float = 0.68,
) -> str | None:
    draw = float(rng.random())
    if draw <= match_probability:
        return str(rng.choice(customer_ids))
    if draw <= match_probability + 0.18:
        return f"dummy-unmatched-{row_number:05d}"
    return None


def _customer_demographic_lookup(customers: pd.DataFrame, customer_id: str | None) -> dict[str, object]:
    if not customer_id or customers.empty:
        return {}
    match = customers[customers["customer_id"].astype(str).eq(str(customer_id))]
    if match.empty:
        return {}
    return match.iloc[0].to_dict()


def _dummy_transaction_row(
    match: pd.Series,
    date: pd.Timestamp,
    row_number: int,
    customer_id: str | None,
    demographics: dict[str, object],
    ticket_type: str,
    ticket_class: str,
    sales_channel: str,
    tickets_sold: int,
    paid_tickets: int,
    comp_tickets: int,
    unit_price: float,
    family_flag: bool,
    is_comp: bool,
) -> dict[str, object]:
    match_id = str(match.get("match_id"))
    event_date = pd.to_datetime(match.get("event_date"), errors="coerce")
    transaction_id = f"dummy-2627-{row_number:06d}"
    gross = float(paid_tickets * unit_price)
    age_band = demographics.get("age_band")
    gender = demographics.get("gender")
    postcode = demographics.get("postcode")
    return {
        "source_file": DUMMY_2627_SOURCE,
        "source_row_number": row_number,
        "competition": str(match.get("competition", "BBL")),
        "transaction_id": transaction_id,
        "order_id": f"dummy-order-{row_number:06d}",
        "customer_id": customer_id,
        "anonymised_customer_id": anonymise_identifier(customer_id),
        "fixture_id": match_id,
        "fixture_name": str(match.get("fixture_label", match_id)),
        "event_name": str(match.get("fixture_label", match_id)),
        "team": "Adelaide Strikers",
        "opponent": str(match.get("opponent", "Opponent TBC")),
        "venue": str(match.get("venue", "Adelaide Oval")),
        "transaction_date": date,
        "match_date": event_date,
        "season_label": PLANNING_SEASON_LABEL,
        "season": 2026,
        "tickets_sold": float(tickets_sold),
        "paid_tickets_sold": float(paid_tickets),
        "comp_tickets_sold": float(comp_tickets),
        "gross_revenue": gross,
        "net_revenue": gross,
        "price_paid": float(unit_price),
        "ticket_type": ticket_type,
        "ticket_class": ticket_class,
        "price_type": ticket_type,
        "section": "GA" if "General" in ticket_class else ticket_class,
        "row": None,
        "seat": None,
        "sales_channel": sales_channel,
        "promo_code": "DUMMY2627",
        "purchaser_postcode": postcode,
        "purchaser_age_band": age_band,
        "purchaser_gender": gender,
        "purchaser_family_flag": bool(family_flag or demographics.get("family_flag", False)),
        "marketing_opt_in": bool(demographics.get("marketing_opt_in", False)) if demographics else False,
        "status": "Complimentary" if is_comp else "Paid",
        "is_refund": False,
        "is_comp": bool(is_comp),
        "sales_window": "early-sales",
        "customer_purchase_status": "Returning" if row_number % 4 else "New",
        "fixture_match_score": 1.0,
    }


def _merge_customer_rows(customers: pd.DataFrame, new_customers: pd.DataFrame) -> pd.DataFrame:
    if new_customers.empty:
        return customers.copy()
    if customers.empty:
        return new_customers.copy()
    columns = sorted(set(customers.columns) | set(new_customers.columns))
    left = customers.copy()
    right = new_customers.copy()
    for column in columns:
        if column not in left:
            left[column] = None
        if column not in right:
            right[column] = None
    frames = [frame[columns].dropna(axis=1, how="all") for frame in (left, right)]
    return pd.concat(frames, ignore_index=True, sort=False).drop_duplicates("customer_id", keep="first")


def _normalised_gigya(series: pd.Series) -> pd.Series:
    if series is None:
        return pd.Series(dtype=object)
    text = series.astype("string").str.strip()
    text = text.mask(text.str.lower().isin(["", "nan", "none", "null", "<na>"]))
    return text.str.lower()


def _enrich_transactions_with_demographics(
    transactions: pd.DataFrame,
    customers: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, object]]:
    if transactions.empty:
        frame = transactions.copy()
        return _add_demographic_filter_columns(frame), _empty_demographic_metrics(frame, customers)

    frame = transactions.copy()
    frame["_gigya_join_key"] = _normalised_gigya(frame.get("customer_id", pd.Series(index=frame.index, dtype=object)))
    customer_frame = customers.copy()
    if not customer_frame.empty and "customer_id" in customer_frame:
        customer_frame["_gigya_join_key"] = _normalised_gigya(customer_frame["customer_id"])
    else:
        customer_frame["_gigya_join_key"] = pd.Series(dtype=object)

    demo_cols = [
        column
        for column in [
            "_gigya_join_key",
            "age",
            "age_band",
            "gender",
            "postcode",
            "suburb",
            "state",
            "marketing_opt_in",
            "email_opt_in",
            "sms_opt_in",
            "family_flag",
        ]
        if column in customer_frame
    ]
    matched_rows = pd.Series(False, index=frame.index)
    if demo_cols and "_gigya_join_key" in demo_cols:
        lookup = customer_frame[demo_cols].dropna(subset=["_gigya_join_key"]).drop_duplicates("_gigya_join_key")
        matched_keys = set(lookup["_gigya_join_key"].dropna().astype(str))
        matched_rows = frame["_gigya_join_key"].astype(str).isin(matched_keys) & frame["_gigya_join_key"].notna()
        frame = frame.merge(lookup, on="_gigya_join_key", how="left", suffixes=("", "_customer"))

    fill_map = {
        "purchaser_age_band": "age_band",
        "purchaser_gender": "gender",
        "purchaser_postcode": "postcode",
        "purchaser_suburb": "suburb",
        "purchaser_state": "state",
    }
    for target, source in fill_map.items():
        customer_source = f"{source}_customer" if f"{source}_customer" in frame else source
        if target not in frame:
            frame[target] = None
        if customer_source in frame:
            frame[target] = _fill_missing_text(frame[target], frame[customer_source])

    for target, source in {"marketing_opt_in": "marketing_opt_in", "email_opt_in": "email_opt_in", "sms_opt_in": "sms_opt_in", "purchaser_family_flag": "family_flag"}.items():
        customer_source = f"{source}_customer" if f"{source}_customer" in frame else source
        if target not in frame:
            frame[target] = False
        if customer_source in frame:
            frame[target] = frame[target].where(frame[target].notna(), frame[customer_source])

    if "age" in frame and "age_filter" not in frame:
        frame["age_filter"] = frame["age"]
    frame = _add_demographic_filter_columns(frame)

    total_rows = int(len(frame))
    rows_with_id = int(frame["_gigya_join_key"].notna().sum())
    customer_rows_with_id = int(customer_frame["_gigya_join_key"].notna().sum()) if "_gigya_join_key" in customer_frame else 0
    matched_count = int(matched_rows.sum())
    metrics = {
        "ticketing_rows_with_gigyauid": rows_with_id,
        "customer_rows_with_gigyauid": customer_rows_with_id,
        "demographic_matched_ticket_rows": matched_count,
        "demographic_unmatched_ticket_rows": max(rows_with_id - matched_count, 0),
        "demographic_match_rate": float(matched_count / rows_with_id) if rows_with_id else 0.0,
        "age_coverage": _known_rate(frame["age_band_filter"]),
        "gender_coverage": _known_rate(frame["gender_filter"]),
        "postcode_coverage": _known_rate(frame["postcode_filter"]),
    }
    return frame.drop(columns=["_gigya_join_key"], errors="ignore"), metrics


def _fill_missing_text(primary: pd.Series, fallback: pd.Series) -> pd.Series:
    primary_text = primary.astype("string")
    missing = primary_text.isna() | primary_text.str.strip().str.lower().isin(["", "unknown", "unmatched", "none", "nan", "<na>"])
    return primary.where(~missing, fallback)


def _add_demographic_filter_columns(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["age_band_filter"] = _unknown_bucket(output.get("purchaser_age_band", pd.Series(index=output.index, dtype=object)))
    output["gender_filter"] = _unknown_bucket(output.get("purchaser_gender", pd.Series(index=output.index, dtype=object)))
    output["postcode_filter"] = _unknown_bucket(output.get("purchaser_postcode", pd.Series(index=output.index, dtype=object)))
    return output


def _unknown_bucket(series: pd.Series) -> pd.Series:
    values = series.astype("string").str.strip()
    values = values.mask(values.isna() | values.str.lower().isin(["", "nan", "none", "null", "<na>"]), "Unknown / unmatched")
    return values.astype(str)


def _known_rate(series: pd.Series) -> float:
    if series.empty:
        return 0.0
    known = ~series.astype(str).str.lower().isin(["unknown / unmatched", "unknown", "unmatched", "nan", "none", ""])
    return float(known.mean())


def _empty_demographic_metrics(transactions: pd.DataFrame, customers: pd.DataFrame) -> dict[str, object]:
    return {
        "ticketing_rows_with_gigyauid": 0,
        "customer_rows_with_gigyauid": int(customers["customer_id"].notna().sum()) if not customers.empty and "customer_id" in customers else 0,
        "demographic_matched_ticket_rows": 0,
        "demographic_unmatched_ticket_rows": 0,
        "demographic_match_rate": 0.0,
        "age_coverage": 0.0,
        "gender_coverage": 0.0,
        "postcode_coverage": 0.0,
    }


def _ui_transaction_metrics(transactions: pd.DataFrame, fixtures: pd.DataFrame, customers: pd.DataFrame) -> dict[str, object]:
    if transactions.empty:
        return {"ticket_rows": 0, "fixture_rows": len(fixtures), "customer_rows": len(customers)}
    return {
        "ticket_rows": int(len(transactions)),
        "fixture_rows": int(len(fixtures)),
        "customer_rows": int(len(customers)),
        "unique_customers": int(transactions["customer_id"].dropna().nunique()) if "customer_id" in transactions else 0,
        "paid_ticket_count": float(transactions["paid_tickets_sold"].sum()) if "paid_tickets_sold" in transactions else 0.0,
        "comp_ticket_count": float(transactions["comp_tickets_sold"].sum()) if "comp_tickets_sold" in transactions else 0.0,
        "refund_count": int(transactions["is_refund"].fillna(False).sum()) if "is_refund" in transactions else 0,
        "gross_revenue": float(transactions["gross_revenue"].sum()) if "gross_revenue" in transactions else 0.0,
        "net_revenue": float(transactions["net_revenue"].sum()) if "net_revenue" in transactions else 0.0,
    }


def _latest_transaction_date(transactions: pd.DataFrame) -> pd.Timestamp | None:
    if transactions.empty or "transaction_date" not in transactions:
        return None
    dates = pd.to_datetime(transactions["transaction_date"], errors="coerce").dropna()
    return dates.max().normalize() if not dates.empty else None


def _date_label(value: object) -> str:
    date = pd.to_datetime(value, errors="coerce")
    return "Unavailable" if pd.isna(date) else str(date.date())


def _render_app_header(bundle: StrikersDataBundle, global_context: dict[str, object]) -> None:
    planning_matches = bundle.matches[bundle.matches["season_label"].astype(str).eq(PLANNING_SEASON_LABEL)] if not bundle.matches.empty else pd.DataFrame()
    render_brand_header(
        SACA_LOGO_PATH,
        STRIKERS_LOGO_PATH,
        PLANNING_SEASON_LABEL,
        "BBL / WBBL",
        global_context["latest_transaction_date"],
        len(planning_matches),
        len(planning_matches),
    )


def _page_section(page: str) -> str:
    for section, pages in NAV_SECTIONS.items():
        if page in pages:
            return section
    return next(iter(NAV_SECTIONS))


def _default_page_for_section(section: str) -> str:
    pages = NAV_SECTIONS.get(section, PAGES)
    return pages[0] if pages else PAGES[0]


def _nav_button_label(label: str) -> str:
    icon = NAV_ICONS.get(label, "")
    return f"{icon} {label}" if icon else label


def _nav_column_widths(options: list[str]) -> list[float]:
    return [max(1.0, min(len(option) / 9.0, 3.4)) for option in options]


def _set_active_nav_section(section: str) -> None:
    st.session_state["nav_section"] = section
    page_history = dict(st.session_state.get("nav_page_history", {}))
    remembered_page = page_history.get(section, _default_page_for_section(section))
    if remembered_page not in NAV_SECTIONS.get(section, []):
        remembered_page = _default_page_for_section(section)
    st.session_state["nav_page"] = remembered_page
    st.session_state["page"] = remembered_page


def _set_active_nav_page(section: str, page: str) -> None:
    st.session_state["nav_section"] = section
    st.session_state["nav_page"] = page
    st.session_state["page"] = page
    page_history = dict(st.session_state.get("nav_page_history", {}))
    page_history[section] = page
    st.session_state["nav_page_history"] = page_history


def _render_top_nav() -> str:
    current_page = str(st.session_state.get("page", PAGES[0]))
    if current_page not in PAGES:
        current_page = PAGES[0]

    current_section = _page_section(current_page)
    st.session_state.setdefault("nav_section", current_section)
    if st.session_state.get("nav_section") not in NAV_SECTIONS:
        st.session_state["nav_section"] = current_section

    st.session_state.setdefault("nav_page_history", {})
    page_history = dict(st.session_state.get("nav_page_history", {}))
    page_history[current_section] = current_page
    st.session_state["nav_page_history"] = page_history

    selected_section = str(st.session_state.get("nav_section", current_section))
    st.markdown('<div class="section-nav-shell">', unsafe_allow_html=True)
    st.markdown('<div class="nav-caption">Workflow</div>', unsafe_allow_html=True)
    section_cols = st.columns(_nav_column_widths(list(NAV_SECTIONS.keys())), gap="small")
    for col, section in zip(section_cols, NAV_SECTIONS.keys()):
        with col:
            st.button(
                _nav_button_label(section),
                key=f"nav_section_{_safe_file_part(section)}",
                type="primary" if section == selected_section else "secondary",
                use_container_width=True,
                on_click=_set_active_nav_section,
                args=(section,),
            )
    st.markdown("</div>", unsafe_allow_html=True)

    selected_section = str(st.session_state.get("nav_section", current_section))
    section_pages = NAV_SECTIONS[str(selected_section)]
    remembered_page = page_history.get(str(selected_section), _default_page_for_section(str(selected_section)))
    if remembered_page not in section_pages:
        remembered_page = _default_page_for_section(str(selected_section))
    if st.session_state.get("nav_page") not in section_pages:
        st.session_state["nav_page"] = remembered_page

    selected_page = str(st.session_state.get("nav_page", remembered_page))
    st.markdown('<div class="page-nav-shell">', unsafe_allow_html=True)
    st.markdown(f'<div class="nav-caption">{selected_section}</div>', unsafe_allow_html=True)
    page_cols = st.columns(_nav_column_widths(section_pages), gap="small")
    for col, page in zip(page_cols, section_pages):
        with col:
            st.button(
                _nav_button_label(page),
                key=f"nav_page_{_safe_file_part(page)}",
                type="primary" if page == selected_page else "secondary",
                use_container_width=True,
                on_click=_set_active_nav_page,
                args=(str(selected_section), page),
            )
    st.markdown("</div>", unsafe_allow_html=True)

    selected_page = str(st.session_state.get("nav_page", remembered_page))
    st.session_state["page"] = str(selected_page)
    page_history[str(selected_section)] = str(selected_page)
    st.session_state["nav_page_history"] = page_history
    return str(selected_page)


def _inline_planning_controls(
    bundle: StrikersDataBundle,
    page_key: str,
    include_customer_filters: bool = False,
) -> dict[str, object]:
    st.markdown('<div class="filter-band">', unsafe_allow_html=True)
    top = st.columns([1.1, 0.9, 1.2, 1.0])
    season_label = top[0].selectbox(
        "Season",
        _season_options(bundle.matches),
        index=_season_options(bundle.matches).index(PLANNING_SEASON_LABEL)
        if PLANNING_SEASON_LABEL in _season_options(bundle.matches)
        else len(_season_options(bundle.matches)) - 1,
        key=page_state_key(page_key, "season"),
    )
    competition = top[1].selectbox("Competition", ["All"] + list(COMPETITION_OPTIONS), key=page_state_key(page_key, "competition"))
    include_assumed = top[2].checkbox(
        "Include assumed upcoming fixtures",
        value=season_label == PLANNING_SEASON_LABEL,
        key=page_state_key(page_key, "include_assumed"),
    )
    as_at_date = pd.Timestamp(
        top[3].date_input("As at snapshot", value=_reference_date(bundle).date(), key=page_state_key(page_key, "as_at_date"))
    )

    scoped_fixtures = _scoped_fixtures(bundle.fixtures, season_label, competition, [], include_assumed)
    fixture_label_col = scoped_fixtures["fixture_label"].fillna(scoped_fixtures["fixture_id"]).astype(str) if not scoped_fixtures.empty else pd.Series(dtype=str)
    fixture_options = ["Tournament total"] + fixture_label_col.tolist()
    fixture_labels = st.multiselect(
        "Fixture",
        fixture_options,
        default=["Tournament total"],
        key=page_state_key(page_key, "fixtures"),
    )
    fixture_ids: list[str] = []
    if fixture_labels and "Tournament total" not in fixture_labels and not scoped_fixtures.empty:
        label_to_id = dict(zip(fixture_label_col, scoped_fixtures["fixture_id"].astype(str)))
        fixture_ids = [label_to_id[label] for label in fixture_labels if label in label_to_id]

    audience_segment = "All"
    section = "All"
    sales_channel = "All"
    with st.expander("More filters", expanded=False):
        detail_cols = st.columns(4)
        ticket_type = detail_cols[0].selectbox("Ticket type", filter_options(bundle.transactions, "ticket_type"), key=page_state_key(page_key, "ticket_type"))
        ticket_class = detail_cols[1].selectbox("Ticket class", filter_options(bundle.transactions, "ticket_class"), key=page_state_key(page_key, "ticket_class"))
        ticket_status = detail_cols[2].selectbox(
            "Paid / comp",
            ["Paid + comps", "Paid tickets only", "Comps only", "Refunds / voids"],
            key=page_state_key(page_key, "ticket_status"),
        )
        sales_window = detail_cols[3].selectbox("Sales window", filter_options(bundle.transactions, "sales_window"), key=page_state_key(page_key, "sales_window"))

        if include_customer_filters:
            customer_cols = st.columns(4)
            audience_segment = customer_cols[0].selectbox(
                "Audience",
                ["All", "Families", "18-30s", "Marketable", "Returning purchasers"],
                key=page_state_key(page_key, "audience_segment"),
            )
            section = customer_cols[1].selectbox("Section", filter_options(bundle.transactions, "section"), key=page_state_key(page_key, "section"))
            sales_channel = customer_cols[2].selectbox("Sales channel", filter_options(bundle.transactions, "sales_channel"), key=page_state_key(page_key, "sales_channel"))
            customer_cols[3].selectbox("Customer ID basis", ["GigyaUID linked", "All customer IDs"], key=page_state_key(page_key, "customer_basis"))

    if st.button("Reset filters for this page", key=page_state_key(page_key, "reset_inline")):
        prefix = page_key.replace(" ", "_").lower()
        for key in list(st.session_state.keys()):
            if str(key).startswith(prefix):
                del st.session_state[key]
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    historical_labels = [value for value in _season_options(bundle.matches) if value != season_label]
    default_history = [label for label in DEFAULT_HISTORICAL_SEASONS if label in historical_labels] or historical_labels[-3:]
    return {
        "season_label": season_label,
        "competition": competition,
        "include_assumed": include_assumed,
        "fixture_ids": fixture_ids,
        "as_at_date": as_at_date,
        "audience_segment": audience_segment,
        "ticket_type": ticket_type,
        "ticket_class": ticket_class,
        "section": section,
        "sales_channel": sales_channel,
        "sales_window": sales_window,
        "ticket_status": ticket_status,
        "target_uplift_pct": 0.10,
        "early_bird_lift": 0.16,
        "member_presale_lift": 0.12,
        "marketing_lift": 0.18,
        "campaign_lift": 0.28,
        "final_week_lift": 0.22,
        "history_filters": {"season_label": default_history, **({"competition": competition} if competition != "All" else {})},
        "audience_group_by": "ticket_class",
    }


def _target_breakdown_controls(bundle: StrikersDataBundle) -> dict[str, object]:
    page_key = "target_breakdown"
    st.markdown('<div class="filter-band">', unsafe_allow_html=True)
    st.markdown('<div class="section-kicker">Planning scenario filters</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">Target Breakdown is a planning/projection page only. Targets are read from Fixture Forecasting and cannot be edited here.</div>',
        unsafe_allow_html=True,
    )
    top = st.columns([0.75, 1.05, 1.05, 1.35])
    competition = top[0].selectbox("Competition", ["All"] + list(COMPETITION_OPTIONS), key=page_state_key(page_key, "competition"))
    include_assumed = top[1].checkbox("Include assumed upcoming fixtures", value=True, key=page_state_key(page_key, "include_assumed"))
    target_modes = top[2].multiselect(
        "Target modes",
        ["Forecast", "Forecast +10%", "Stretch", "Base"],
        default=["Forecast", "Forecast +10%", "Stretch", "Base"],
        key=page_state_key(page_key, "target_modes"),
    )

    scoped_fixtures = _scoped_fixtures(bundle.fixtures, PLANNING_SEASON_LABEL, competition, [], include_assumed)
    fixture_label_col = scoped_fixtures["fixture_label"].fillna(scoped_fixtures["fixture_id"]).astype(str) if not scoped_fixtures.empty else pd.Series(dtype=str)
    fixture_options = ["Season total"] + fixture_label_col.tolist()
    selected_fixture = top[3].selectbox("Fixture", fixture_options, key=page_state_key(page_key, "fixture"))
    fixture_ids: list[str] = []
    selected_historical_fixture_id = ""
    selected_historical_choice = "None"
    selected_historical_fixture_label = "None"
    historical_choice_options = ["None", "Last season equivalent"]
    if selected_fixture != "Season total" and not scoped_fixtures.empty:
        label_to_id = dict(zip(fixture_label_col, scoped_fixtures["fixture_id"].astype(str)))
        fixture_ids = [label_to_id[selected_fixture]] if selected_fixture in label_to_id else []
        selected_match = scoped_fixtures[scoped_fixtures["fixture_id"].astype(str).isin(fixture_ids)].iloc[0] if fixture_ids else pd.Series(dtype=object)
        historical_options = _historical_fixture_options(bundle, competition, selected_match)
        if historical_options:
            historical_choice_options.extend(row["label"] for row in historical_options)
    selected_historical_choice = st.selectbox(
        "Historical comparison",
        historical_choice_options,
        key=page_state_key(page_key, "historical_choice"),
    )
    if selected_historical_choice not in {"None", "Last season equivalent"} and fixture_ids:
        selected_historical_fixture_label = selected_historical_choice
        historical_options = _historical_fixture_options(bundle, competition, selected_match)
        selected_historical_fixture_id = next(row["fixture_id"] for row in historical_options if row["label"] == selected_historical_choice)

    with st.expander("Segment filters", expanded=False):
        detail_cols = st.columns(3)
        ticket_type = detail_cols[0].selectbox("Ticket type", filter_options(bundle.transactions, "ticket_type"), key=page_state_key(page_key, "ticket_type"))
        ticket_class = detail_cols[1].selectbox("Ticket class", filter_options(bundle.transactions, "ticket_class"), key=page_state_key(page_key, "ticket_class"))
        ticket_status = detail_cols[2].selectbox(
            "Paid / comp",
            ["Paid + comps", "Paid tickets only", "Comps only", "Refunds / voids"],
            key=page_state_key(page_key, "ticket_status"),
        )

        demo_cols = st.columns(4)
        age_bands = demo_cols[0].multiselect("Age / age band", filter_options(bundle.transactions, "age_band_filter", include_all=False), key=page_state_key(page_key, "age_band"))
        genders = demo_cols[1].multiselect("Gender", filter_options(bundle.transactions, "gender_filter", include_all=False), key=page_state_key(page_key, "gender"))
        postcodes = demo_cols[2].multiselect("Postcode", filter_options(bundle.transactions, "postcode_filter", include_all=False), key=page_state_key(page_key, "postcode"))
        audience_segment = demo_cols[3].selectbox(
            "Audience segment",
            ["All", "Families", "18-30s", "Marketable", "Returning purchasers"],
            key=page_state_key(page_key, "audience_segment"),
        )

    if st.button("Reset filters for this page", key=page_state_key(page_key, "reset_inline")):
        prefix = page_key.replace(" ", "_").lower()
        for key in list(st.session_state.keys()):
            if str(key).startswith(prefix):
                del st.session_state[key]
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    historical_labels = [value for value in _season_options(bundle.matches) if value != PLANNING_SEASON_LABEL]
    default_history = [label for label in DEFAULT_HISTORICAL_SEASONS if label in historical_labels] or historical_labels[-3:]
    return {
        "season_label": PLANNING_SEASON_LABEL,
        "competition": competition,
        "include_assumed": include_assumed,
        "fixture_ids": fixture_ids,
        "fixture_choice": selected_fixture,
        "target_modes": target_modes or ["Forecast"],
        "historical_choice": selected_historical_choice,
        "historical_fixture_id": selected_historical_fixture_id,
        "historical_fixture_label": selected_historical_fixture_label,
        "as_at_date": _reference_date(bundle),
        "audience_segment": audience_segment,
        "ticket_type": ticket_type,
        "ticket_class": ticket_class,
        "section": "All",
        "sales_channel": "All",
        "sales_window": "All",
        "ticket_status": ticket_status,
        "age_band": age_bands,
        "gender": genders,
        "postcode": postcodes,
        "target_uplift_pct": 0.10,
        "early_bird_lift": 0.16,
        "member_presale_lift": 0.12,
        "marketing_lift": 0.18,
        "campaign_lift": 0.28,
        "final_week_lift": 0.22,
        "history_filters": {"season_label": default_history, **({"competition": competition} if competition != "All" else {})},
        "audience_group_by": "ticket_class",
    }


def _page_scope_controls(
    bundle: StrikersDataBundle,
    page_key: str,
    default_season: str,
    include_detail_filters: bool,
) -> dict[str, object]:
    st.sidebar.header("Page filters")
    _reset_page_filters(page_key)

    seasons = _season_options(bundle.matches)
    if default_season not in seasons:
        default_season = seasons[-1] if seasons else PLANNING_SEASON_LABEL
    season_label = st.sidebar.selectbox(
        "Season",
        seasons,
        index=seasons.index(default_season),
        key=page_state_key(page_key, "season"),
    )
    competition = st.sidebar.selectbox(
        "Competition",
        ["All"] + list(COMPETITION_OPTIONS),
        index=0,
        key=page_state_key(page_key, "competition"),
    )
    include_assumed = st.sidebar.checkbox(
        "Include assumed upcoming fixtures",
        value=season_label == PLANNING_SEASON_LABEL,
        key=page_state_key(page_key, "include_assumed"),
    )
    scoped_fixtures = _scoped_fixtures(bundle.fixtures, season_label, competition, [], include_assumed)
    fixture_label_col = scoped_fixtures["fixture_label"].fillna(scoped_fixtures["fixture_id"]).astype(str) if not scoped_fixtures.empty else pd.Series(dtype=str)
    fixture_options = ["Tournament total"] + fixture_label_col.tolist()
    selected_fixture_labels = st.sidebar.multiselect(
        "Fixture selector",
        fixture_options,
        default=["Tournament total"],
        key=page_state_key(page_key, "fixture_selector"),
    )
    fixture_ids: list[str] = []
    if selected_fixture_labels and "Tournament total" not in selected_fixture_labels and not scoped_fixtures.empty:
        label_to_id = dict(zip(fixture_label_col, scoped_fixtures["fixture_id"].astype(str)))
        fixture_ids = [label_to_id[label] for label in selected_fixture_labels if label in label_to_id]

    default_as_at = _reference_date(bundle)
    as_at_date = pd.Timestamp(
        st.sidebar.date_input(
            "As at uploaded snapshot date",
            value=default_as_at.date(),
            key=page_state_key(page_key, "as_at_date"),
        )
    )

    controls = {
        "season_label": season_label,
        "competition": competition,
        "include_assumed": include_assumed,
        "fixture_ids": fixture_ids,
        "as_at_date": as_at_date,
        "audience_segment": "All",
        "ticket_type": "All",
        "ticket_class": "All",
        "section": "All",
        "sales_channel": "All",
        "sales_window": "All",
        "ticket_status": "Paid + comps",
        "target_uplift_pct": 0.10,
        "early_bird_lift": 0.16,
        "member_presale_lift": 0.12,
        "marketing_lift": 0.18,
        "campaign_lift": 0.28,
        "final_week_lift": 0.22,
        "history_filters": _page_history_filters(bundle, page_key, competition, season_label),
        "audience_group_by": "ticket_class",
    }

    if include_detail_filters:
        st.sidebar.header("View filters")
        controls["audience_segment"] = st.sidebar.selectbox(
            "Audience segment",
            ["All", "Families", "18-30s", "Marketable", "Returning purchasers"],
            key=page_state_key(page_key, "audience_segment"),
        )
        controls["ticket_type"] = st.sidebar.selectbox("Ticket type", filter_options(bundle.transactions, "ticket_type"), key=page_state_key(page_key, "ticket_type"))
        controls["ticket_class"] = st.sidebar.selectbox("Ticket class", filter_options(bundle.transactions, "ticket_class"), key=page_state_key(page_key, "ticket_class"))
        controls["section"] = st.sidebar.selectbox("Section", filter_options(bundle.transactions, "section"), key=page_state_key(page_key, "section"))
        controls["sales_channel"] = st.sidebar.selectbox("Sales channel", filter_options(bundle.transactions, "sales_channel"), key=page_state_key(page_key, "sales_channel"))
        controls["sales_window"] = st.sidebar.selectbox("Sales window", filter_options(bundle.transactions, "sales_window"), key=page_state_key(page_key, "sales_window"))
        controls["ticket_status"] = st.sidebar.selectbox(
            "Paid / comp status",
            ["Paid + comps", "Paid tickets only", "Comps only", "Refunds / voids"],
            key=page_state_key(page_key, "ticket_status"),
        )

    st.sidebar.header("Targets")
    controls["target_uplift_pct"] = st.sidebar.slider("Uplift target %", -10, 35, 10, key=page_state_key(page_key, "target_uplift_pct")) / 100
    with st.sidebar.expander("Sales-window weights", expanded=False):
        controls["early_bird_lift"] = st.slider("Early sales weight %", 0, 40, 16, key=page_state_key(page_key, "early_bird_lift")) / 100
        controls["member_presale_lift"] = st.slider("Pre-sale weight %", 0, 40, 12, key=page_state_key(page_key, "member_presale_lift")) / 100
        controls["marketing_lift"] = st.slider("Campaign window weight %", 0, 60, 18, key=page_state_key(page_key, "marketing_lift")) / 100
        controls["campaign_lift"] = st.slider("Burst weight %", 0, 80, 28, key=page_state_key(page_key, "campaign_lift")) / 100
        controls["final_week_lift"] = st.slider("Match-week weight %", 0, 80, 22, key=page_state_key(page_key, "final_week_lift")) / 100

    return controls


def _page_history_filters(bundle: StrikersDataBundle, page_key: str, competition: str, season_label: str) -> dict[str, object]:
    historical_labels = [value for value in _season_options(bundle.matches) if value != season_label]
    default_history = [label for label in DEFAULT_HISTORICAL_SEASONS if label in historical_labels]
    with st.sidebar.expander("Advanced baseline settings", expanded=False):
        selected_history = st.multiselect(
            "Historical baseline seasons",
            historical_labels,
            default=default_history or historical_labels[-3:],
            key=page_state_key(page_key, "historical_seasons"),
        )
    if competition != "All":
        return history_filters(competition, selected_history)
    if selected_history:
        return {"season_label": selected_history}
    return {}


def _reset_page_filters(page_key: str) -> None:
    if st.sidebar.button("Reset filters for this page", key=page_state_key(page_key, "reset_filters")):
        prefix = f"{page_key}_"
        for key in list(st.session_state.keys()):
            if key.startswith(prefix):
                del st.session_state[key]
        st.rerun()


def _build_page_context(
    bundle: StrikersDataBundle,
    pace_engine: HistoricalPaceEngine,
    forecaster: TicketSalesForecaster,
    controls: dict[str, object],
) -> dict[str, object]:
    scoped_matches = _scoped_matches(
        bundle.matches,
        controls["season_label"],
        controls["competition"],
        controls["fixture_ids"],
        controls["include_assumed"],
    )
    scoped_fixtures = _scoped_fixtures(
        bundle.fixtures,
        controls["season_label"],
        controls["competition"],
        controls["fixture_ids"],
        controls["include_assumed"],
    )
    filtered_transactions = apply_transaction_filters(
        bundle.transactions,
        season_label=controls["season_label"],
        competition=controls["competition"],
        fixture_ids=controls["fixture_ids"],
        as_at_date=controls["as_at_date"],
        audience_segment=controls["audience_segment"],
        ticket_type=controls["ticket_type"],
        ticket_class=controls["ticket_class"],
        section=controls["section"],
        sales_channel=controls["sales_channel"],
        sales_window=controls["sales_window"],
        ticket_status=controls["ticket_status"],
    )
    filtered_transactions = _apply_demographic_filters(
        filtered_transactions,
        controls.get("age_band", []),
        controls.get("gender", []),
        controls.get("postcode", []),
    )
    assumptions = TargetAssumptions(
        early_bird_lift_pct=controls["early_bird_lift"],
        member_presale_lift_pct=controls["member_presale_lift"],
        marketing_lift_pct=controls["marketing_lift"],
        campaign_burst_lift_pct=controls["campaign_lift"],
        final_week_lift_pct=controls["final_week_lift"],
    )
    outputs = _build_forecast_outputs(
        scoped_matches,
        bundle.matches,
        bundle.daily_sales,
        pace_engine,
        forecaster,
        assumptions,
        controls["target_uplift_pct"],
        controls["history_filters"],
    )
    season_curve = aggregate_selected_outputs(outputs, len(outputs)) if outputs else pd.DataFrame()
    forecast_total = _last_value(season_curve, "forecast_expected_cumulative")
    target_total = _last_value(season_curve, "target_cumulative")
    kpis = kpi_summary(filtered_transactions, bundle.transactions, scoped_fixtures, forecast_total, target_total, controls["as_at_date"])
    fixture_frame = fixture_sales_summary(filtered_transactions, scoped_fixtures, scoped_matches, outputs)
    audience_frame = audience_summary(filtered_transactions, controls.get("audience_group_by", "ticket_class"))
    insights = insight_narrative(kpis, fixture_frame, audience_frame)
    return {
        "season_label": controls["season_label"],
        "competition": controls["competition"],
        "as_at_date": controls["as_at_date"],
        "history_filters": controls["history_filters"],
        "scoped_matches": scoped_matches,
        "scoped_fixtures": scoped_fixtures,
        "filtered_transactions": filtered_transactions,
        "assumptions": assumptions,
        "outputs": outputs,
        "season_curve": season_curve,
        "kpis": kpis,
        "fixture_frame": fixture_frame,
        "audience_frame": audience_frame,
        "insights": insights,
    }


def _apply_demographic_filters(
    transactions: pd.DataFrame,
    age_bands: list[str] | tuple[str, ...] | None,
    genders: list[str] | tuple[str, ...] | None,
    postcodes: list[str] | tuple[str, ...] | None,
) -> pd.DataFrame:
    frame = transactions.copy()
    for column, selected in {
        "age_band_filter": age_bands,
        "gender_filter": genders,
        "postcode_filter": postcodes,
    }.items():
        values = [str(value) for value in (selected or []) if str(value)]
        if values and column in frame:
            frame = frame[frame[column].fillna("Unknown / unmatched").astype(str).isin(values)]
    return frame.reset_index(drop=True)


def _expected_index(
    bundle: StrikersDataBundle,
    controls: dict[str, object],
    group_kind: str,
    min_size: int,
    thresholds: dict[str, float] | None = None,
) -> pd.DataFrame:
    transactions = apply_transaction_filters(
        bundle.transactions,
        season_label=None,
        competition=str(controls.get("competition", "All")),
        fixture_ids=[],
        as_at_date=None,
        audience_segment=str(controls.get("audience_segment", "All")),
        ticket_type=str(controls.get("ticket_type", "All")),
        ticket_class=str(controls.get("ticket_class", "All")),
        section=str(controls.get("section", "All")),
        sales_channel=str(controls.get("sales_channel", "All")),
        sales_window=str(controls.get("sales_window", "All")),
        ticket_status=str(controls.get("ticket_status", "Paid + comps")),
    )
    transactions = _apply_demographic_filters(
        transactions,
        controls.get("age_band", []),
        controls.get("gender", []),
        controls.get("postcode", []),
    )
    return cached_expected_index(
        transactions,
        bundle.customers,
        bundle.matches,
        str(controls["season_label"]),
        str(controls["competition"]),
        pd.Timestamp(controls["as_at_date"]),
        group_kind,
        tuple(_baseline_seasons_from_controls(controls.get("history_filters", {}))),
        int(min_size),
        json.dumps(thresholds or {}, sort_keys=True),
    )


def _render_header(
    bundle: StrikersDataBundle,
    controls: dict[str, object],
    scoped_matches: pd.DataFrame,
    reference_date: pd.Timestamp,
    kpis: dict[str, float],
) -> None:
    render_brand_header(
        SACA_LOGO_PATH,
        STRIKERS_LOGO_PATH,
        controls["season_label"],
        controls["competition"],
        reference_date,
        len(scoped_matches),
        len(scoped_matches),
    )
    status = "on track" if kpis["gap_to_target"] >= 0 else "at risk"
    render_alert_strip(
        f"Client CSV workflow. Selected view is {status}; forecast gap to target is {format_number(kpis['gap_to_target'])} paid tickets."
    )

    metric_cols = st.columns(6)
    metric_cols[0].markdown(metric_tile("Paid tickets", format_number(kpis["paid_tickets_sold"]), "Paid demand only"), unsafe_allow_html=True)
    metric_cols[1].markdown(metric_tile("Comps", format_number(kpis["comps_issued"]), "Included in attendance, separated from demand"), unsafe_allow_html=True)
    metric_cols[2].markdown(metric_tile("Gross revenue", _money(kpis["gross_revenue"]), f"ATP {_money(kpis['average_ticket_price'])}"), unsafe_allow_html=True)
    metric_cols[3].markdown(metric_tile("Forecast finish", format_number(kpis["forecast_total"]), "Model or historical fallback"), unsafe_allow_html=True)
    metric_cols[4].markdown(metric_tile("Gap to target", format_number(kpis["gap_to_target"]), "Forecast minus target"), unsafe_allow_html=True)
    metric_cols[5].markdown(metric_tile("Required per day", format_number(kpis["required_daily_run_rate"]), "To hit selected target"), unsafe_allow_html=True)


def _render_historic_sales(bundle: StrikersDataBundle) -> None:
    st.markdown('<div class="section-kicker">Historic Sales</div>', unsafe_allow_html=True)
    st.markdown('<div class="filter-band">', unsafe_allow_html=True)
    historical_seasons = _historical_season_options(bundle)
    default_season = _latest_historical_season(bundle) or (historical_seasons[-1] if historical_seasons else PLANNING_SEASON_LABEL)
    filters = st.columns([1, 0.8, 1.4, 0.8])
    season_label = filters[0].selectbox(
        "Season",
        historical_seasons or _season_options(bundle.matches),
        index=(historical_seasons or _season_options(bundle.matches)).index(default_season),
        key=page_state_key("historic_sales", "season"),
    )
    competition = filters[1].selectbox("BBL / WBBL", ["All"] + list(COMPETITION_OPTIONS), key=page_state_key("historic_sales", "competition"))
    scoped_fixtures = _scoped_fixtures(bundle.fixtures, season_label, competition, [], include_assumed=False)
    fixture_label_col = scoped_fixtures["fixture_label"].fillna(scoped_fixtures["fixture_id"]).astype(str) if not scoped_fixtures.empty else pd.Series(dtype=str)
    fixture_options = ["Season total"] + fixture_label_col.tolist()
    selected_fixture = filters[2].selectbox("Match", fixture_options, key=page_state_key("historic_sales", "fixture"))
    ticket_status = filters[3].selectbox("Paid / comp", ["Paid + comps", "Paid tickets only", "Comps only"], key=page_state_key("historic_sales", "ticket_status"))
    product_filters = st.columns(5)
    ticket_type = product_filters[0].selectbox("Ticket type", filter_options(bundle.transactions, "ticket_type"), key=page_state_key("historic_sales", "ticket_type"))
    ticket_class = product_filters[1].selectbox("Ticket class", filter_options(bundle.transactions, "ticket_class"), key=page_state_key("historic_sales", "ticket_class"))
    age_bands = product_filters[2].multiselect("Age / age band", filter_options(bundle.transactions, "age_band_filter", include_all=False), key=page_state_key("historic_sales", "age_band"))
    genders = product_filters[3].multiselect("Gender", filter_options(bundle.transactions, "gender_filter", include_all=False), key=page_state_key("historic_sales", "gender"))
    postcodes = product_filters[4].multiselect("Postcode", filter_options(bundle.transactions, "postcode_filter", include_all=False), key=page_state_key("historic_sales", "postcode"))
    if st.button("Reset filters for this page", key=page_state_key("historic_sales", "reset_filters")):
        for key in list(st.session_state.keys()):
            if str(key).startswith("historic_sales_"):
                del st.session_state[key]
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    fixture_ids: list[str] = []
    if selected_fixture != "Season total" and not scoped_fixtures.empty:
        label_to_id = dict(zip(fixture_label_col, scoped_fixtures["fixture_id"].astype(str)))
        fixture_ids = [label_to_id[selected_fixture]] if selected_fixture in label_to_id else []
    tx = apply_transaction_filters(
        bundle.transactions,
        season_label=season_label,
        competition=competition,
        fixture_ids=fixture_ids,
        ticket_status=ticket_status,
        ticket_type=ticket_type,
        ticket_class=ticket_class,
    )
    tx = _apply_demographic_filters(tx, age_bands, genders, postcodes)
    if any([age_bands, genders, postcodes]):
        st.info(
            f"Demographic filters retain unmatched rows as Unknown / unmatched. GigyaUID match rate: {bundle.metrics.get('demographic_match_rate', 0):.0%}; "
            f"age coverage {bundle.metrics.get('age_coverage', 0):.0%}, gender coverage {bundle.metrics.get('gender_coverage', 0):.0%}, postcode coverage {bundle.metrics.get('postcode_coverage', 0):.0%}."
        )
    kpis = kpi_summary(tx, bundle.transactions, scoped_fixtures)
    cols = st.columns(3)
    cols[0].markdown(metric_tile("Paid tickets", format_number(kpis["paid_tickets_sold"]), "Paid demand in selected historical scope"), unsafe_allow_html=True)
    cols[1].markdown(metric_tile("Comps", format_number(kpis["comps_issued"]), "Complimentary tickets in selected scope"), unsafe_allow_html=True)
    cols[2].markdown(metric_tile("Gross revenue", _money(kpis["gross_revenue"]), f"ATP {_money(kpis['average_ticket_price'])}"), unsafe_allow_html=True)

    curve = _transaction_sales_curve(tx, ticket_status)
    if curve.empty:
        st.info("No historical sales curve is available for this selection.")
    else:
        st.plotly_chart(cumulative_curve_figure(curve, "Historic cumulative sales curve"), use_container_width=True)
        _download_csv(
            curve,
            f"historic_sales_curve_{_safe_file_part(competition)}_{_safe_file_part(season_label)}.csv",
            page_state_key("historic_sales", "curve_csv"),
        )
        st.plotly_chart(daily_sales_figure(curve, "Historic daily sales"), use_container_width=True)
        _download_csv(
            curve[["date", "actual_daily", "actual_cumulative"]],
            f"historic_daily_sales_{_safe_file_part(competition)}_{_safe_file_part(season_label)}.csv",
            page_state_key("historic_sales", "daily_csv"),
        )

    fixture_frame = fixture_sales_summary(tx, scoped_fixtures, bundle.matches[bundle.matches["season_label"].astype(str).eq(str(season_label))], [])
    if not fixture_frame.empty:
        st.markdown('<div class="section-kicker">Fixture sales summary</div>', unsafe_allow_html=True)
        fixture_display = fixture_frame[["fixture_label", "competition", "match_date", "sold", "paid_sold", "comps", "gross_revenue", "capacity_sold_pct"]]
        st.dataframe(
            fixture_display,
            use_container_width=True,
            hide_index=True,
        )
        _download_csv(
            fixture_display,
            f"historic_fixture_summary_{_safe_file_part(competition)}_{_safe_file_part(season_label)}.csv",
            page_state_key("historic_sales", "fixture_summary_csv"),
        )


def _render_fixture_forecasting_table(
    bundle: StrikersDataBundle,
    pace_engine: HistoricalPaceEngine,
    forecaster: TicketSalesForecaster,
) -> None:
    st.markdown('<div class="section-kicker">Fixture Forecasting</div>', unsafe_allow_html=True)
    table = cached_fixture_forecast_table(bundle.matches, bundle.daily_sales, _reference_date(bundle))
    if table.empty:
        st.info("No assumed upcoming fixtures are available. Upload confirmed future fixture data or regenerate assumptions.")
        return
    target_state = st.session_state.setdefault("fixture_targets", _load_fixture_targets())
    table["saved_base_target"] = table["fixture_id"].map(lambda fixture_id: target_state.get(str(fixture_id), {}).get("base_target", np.nan))
    table["saved_stretch_target"] = table["fixture_id"].map(lambda fixture_id: target_state.get(str(fixture_id), {}).get("stretch_target", np.nan))
    table["target_status"] = table.apply(lambda row: "Saved" if str(row["fixture_id"]) in target_state else "Missing", axis=1)
    st.markdown(
        '<div class="section-note">Enter or edit base and stretch targets for each fixture, then click Save all targets. You can edit these targets again later.</div>',
        unsafe_allow_html=True,
    )
    display_table = table.drop(columns=["fixture_id"]).copy()
    st.dataframe(
        display_table,
        use_container_width=True,
        hide_index=True,
    )
    _download_csv(display_table, "fixture_forecast_targets_2026_27.csv", page_state_key("fixture_forecasting", "table_csv"))

    st.markdown('<div class="section-kicker">Fixture target entry</div>', unsafe_allow_html=True)
    saved_at = _fixture_targets_last_saved()
    st.caption(f"Saved targets last updated: {saved_at}" if saved_at else "Saved targets: not saved yet. Editing draft targets below.")
    draft_rows = _fixture_target_draft_table(table, target_state)
    with st.form("fixture_forecasting_targets_form", clear_on_submit=False):
        st.markdown('<div class="section-note">Editing draft targets. Changes are not applied to dependent dashboards until Save all targets is clicked.</div>', unsafe_allow_html=True)
        edited = st.data_editor(
            draft_rows,
            use_container_width=True,
            hide_index=True,
            disabled=["fixture_id", "fixture", "competition", "date", "opponent", "status", "forecasted_sales", "forecast_10pct_uplift", "saved_base_target", "saved_stretch_target"],
            column_config={
                "base_target": st.column_config.TextColumn("Base target"),
                "stretch_target": st.column_config.TextColumn("Stretch target"),
                "target_state": st.column_config.TextColumn("Saved / draft state"),
            },
            key=page_state_key("fixture_forecasting", "draft_target_editor"),
        )
        submitted = st.form_submit_button("Save all targets", type="primary")

    validation = validate_fixture_target_frame(edited)
    if not validation.empty:
        st.warning("Some draft target values need fixing before they can be saved.")
        st.dataframe(validation, use_container_width=True, hide_index=True)
        _download_csv(validation, "fixture_target_validation_errors.csv", page_state_key("fixture_forecasting", "validation_csv"))
    if submitted:
        if not validation.empty:
            st.error("Invalid targets were not saved. Fix the rows listed above and click Save all targets again.")
        else:
            next_state = _targets_from_editor_frame(edited)
            st.session_state["fixture_targets"] = next_state
            _save_fixture_targets(next_state)
            st.session_state["fixture_targets_saved_at"] = pd.Timestamp.now().strftime("%d %b %Y %H:%M")
            _write_client_version_manifest(bundle)
            st.success("Saved all fixture targets. These targets can still be edited later.")
            st.rerun()


def _render_current_season_tracker(
    context: dict[str, object],
    bundle: StrikersDataBundle,
    global_context: dict[str, object],
    controls: dict[str, object],
) -> None:
    season_curve = context["season_curve"]
    fixture_frame = context["fixture_frame"]
    kpis = context["kpis"]
    st.markdown('<div class="section-kicker">Current Season Tracker</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">Use this page for weekly Strikers planning and client check-ins after uploading the latest ticketing extract. Historical seasons are used as the benchmark, not the selected working dataset.</div>',
        unsafe_allow_html=True,
    )
    freshness = global_context.get("freshness", {})
    if freshness.get("is_stale"):
        st.warning(str(freshness.get("message", "The uploaded snapshot may be stale.")))
    else:
        st.info(str(freshness.get("message", "Current uploaded snapshot is available.")))

    current_rows = context["filtered_transactions"]
    if isinstance(current_rows, pd.DataFrame) and current_rows.empty:
        st.info("No transactions are present for this selected current/planning season in the uploaded snapshot yet. Forecasts, targets, and expected-by-now benchmarks are still available from historical data.")

    cols = st.columns(4)
    cols[0].markdown(mini_tile("Unique purchasers", format_number(kpis["unique_purchasers"])), unsafe_allow_html=True)
    cols[1].markdown(mini_tile("Average basket", f"{kpis['average_basket_size']:.1f}"), unsafe_allow_html=True)
    cols[2].markdown(mini_tile("Retention rate", f"{kpis['retention_rate']:.0%}"), unsafe_allow_html=True)
    capacity = "Unavailable" if pd.isna(kpis["capacity_sold_pct"]) else f"{kpis['capacity_sold_pct']:.1f}%"
    cols[3].markdown(mini_tile("Capacity sold", capacity), unsafe_allow_html=True)

    if isinstance(season_curve, pd.DataFrame) and not season_curve.empty:
        st.plotly_chart(cumulative_curve_figure(season_curve, "Current cumulative sales curve"), use_container_width=True)
        _download_csv(season_curve, "current_season_cumulative_curve_2026_27.csv", page_state_key("current_season", "curve_csv"))
        st.plotly_chart(daily_sales_figure(season_curve, "Daily sales and required run-rate"), use_container_width=True)
        _download_csv(season_curve, "current_season_daily_sales_2026_27.csv", page_state_key("current_season", "daily_csv"))
    else:
        st.info("No forecast curve is available for the selected current-season scope yet.")

    left, right = st.columns([1.25, 0.95])
    with left:
        st.markdown('<div class="section-kicker">Fixture performance</div>', unsafe_allow_html=True)
        if fixture_frame.empty:
            st.info("No fixtures match the selected filters.")
        else:
            fixture_display = fixture_frame.copy()
            fixture_display["index_vs_expected"] = fixture_display.apply(
                lambda row: _safe_display_index(row.get("paid_sold", 0), row.get("forecast", 0)),
                axis=1,
            )
            fixture_table = fixture_display[
                [
                    "fixture_label",
                    "competition",
                    "match_date",
                    "fixture_status",
                    "sold",
                    "paid_sold",
                    "comps",
                    "gross_revenue",
                    "target",
                    "forecast",
                    "gap",
                    "index_vs_expected",
                    "risk_status",
                ]
            ]
            st.dataframe(fixture_table, use_container_width=True, hide_index=True)
            _download_csv(fixture_table, "current_season_fixture_performance_2026_27.csv", page_state_key("current_season", "fixture_table_csv"))
    with right:
        st.markdown('<div class="section-kicker">Risk summary</div>', unsafe_allow_html=True)
        for insight in context["insights"]:
            st.write(f"- {insight}")
        audience_index = _expected_index(bundle, controls, "segment", 0)
        ticket_class_index = _expected_index(bundle, controls, "ticket_class", 0)
        _risk_bullets(audience_index, ticket_class_index)


def _render_fixture_forecasts(
    scoped_matches: pd.DataFrame,
    bundle: StrikersDataBundle,
    pace_engine: HistoricalPaceEngine,
    forecaster: TicketSalesForecaster,
    assumptions: TargetAssumptions,
    controls: dict[str, object],
) -> None:
    st.markdown('<div class="section-kicker">Fixture Forecasts</div>', unsafe_allow_html=True)
    if scoped_matches.empty:
        st.info("No fixtures are available for this scope.")
        return
    scoped_matches = scoped_matches.copy()
    scoped_matches["_label"] = [match_label(row) for _, row in scoped_matches.iterrows()]
    selected_label = st.selectbox("Choose fixture", scoped_matches["_label"].tolist())
    selected = scoped_matches[scoped_matches["_label"].eq(selected_label)].iloc[0]
    target_cols = st.columns(3)
    planning_target_override = target_cols[0].number_input("Manual planning target", min_value=0, value=int(selected["baseline_target"]), step=100)
    manual_target_override = target_cols[1].number_input("Manual scenario target", min_value=0, value=int(selected["manual_target"]), step=100)
    target_cols[2].metric("Fixture status", str(selected.get("fixture_status", "historical")).title())

    output = _build_single_output(
        selected,
        bundle.matches,
        bundle.daily_sales,
        pace_engine,
        forecaster,
        assumptions,
        controls["target_uplift_pct"],
        controls["history_filters"],
        float(planning_target_override),
        float(manual_target_override),
    )
    match_chart = output["match_chart"]
    forecast = output["forecast"]
    cols = st.columns(4)
    cols[0].metric("Actual paid to date", format_number(match_chart["actual_cumulative"].max()))
    cols[1].metric("Planning target", format_number(output["planning_target"]))
    cols[2].metric("Forecast finish", format_number(forecast.expected_final_sales))
    cols[3].metric("Forecast confidence", f"{format_number(forecast.lower_final_sales)} - {format_number(forecast.upper_final_sales)}")
    st.plotly_chart(cumulative_curve_figure(match_chart, "Fixture cumulative plan"), use_container_width=True)
    _download_csv(match_chart, f"fixture_forecast_curve_{_safe_file_part(selected['match_id'])}.csv", page_state_key("fixture_forecasts", "curve_csv"))
    st.plotly_chart(daily_sales_figure(match_chart, "Fixture daily run-rate"), use_container_width=True)
    _download_csv(match_chart, f"fixture_forecast_daily_{_safe_file_part(selected['match_id'])}.csv", page_state_key("fixture_forecasts", "daily_csv"))

    tx = bundle.transactions[bundle.transactions["fixture_id"].astype(str).eq(str(selected["match_id"]))]
    if not tx.empty:
        st.markdown('<div class="section-kicker">Drivers by ticket class</div>', unsafe_allow_html=True)
        driver_frame = audience_summary(tx, "ticket_class")
        st.dataframe(driver_frame, use_container_width=True, hide_index=True)
        _download_csv(driver_frame, f"fixture_forecast_ticket_class_{_safe_file_part(selected['match_id'])}.csv", page_state_key("fixture_forecasts", "drivers_csv"))
    if bool(selected.get("is_assumed", False)):
        st.warning("This is an assumed fixture. It is used for future planning and scenario modelling, not historical model training.")


def _render_target_breakdown(
    bundle: StrikersDataBundle,
    pace_engine: HistoricalPaceEngine,
    forecaster: TicketSalesForecaster,
    controls: dict[str, object],
) -> None:
    st.markdown('<div class="section-kicker">Target Breakdown</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">Planning/projection only. Base and stretch targets are read from Fixture Forecasting; actual 26/27 sales tracking sits in Audience Insights.</div>',
        unsafe_allow_html=True,
    )
    outputs_context = _target_breakdown_outputs(bundle, pace_engine, forecaster, controls)
    outputs = outputs_context["outputs"]
    if not outputs:
        st.info("No forecast outputs are available for this planning scope.")
        return

    selected_output = outputs_context["selected_output"]
    target_state = st.session_state.setdefault("fixture_targets", _load_fixture_targets())
    target_summary = _target_totals_for_outputs(outputs, target_state)
    missing_targets = target_summary["missing_fixture_labels"]
    if missing_targets:
        st.warning("Targets have not been set for one or more fixtures. Set targets in Fixture Forecasting.")
        if st.button("Go to Fixture Forecasting", key=page_state_key("target_breakdown", "go_fixture_forecasting")):
            st.session_state["page"] = "Fixture Forecasting"
            st.rerun()

    projection_target = float(target_summary["base_target"] or outputs_context["forecast_total"])
    projection = _segment_level_projection(bundle, controls, selected_output, projection_target, include_actuals=False)
    if projection["fallback_used"]:
        st.warning(str(projection["fallback_message"]))

    summary_cols = st.columns(5)
    summary_cols[0].markdown(mini_tile("Segment forecast", format_number(projection["forecast_final_tickets"])), unsafe_allow_html=True)
    summary_cols[1].markdown(mini_tile("Forecast +10%", format_number(projection["forecast_final_tickets"] * 1.10)), unsafe_allow_html=True)
    summary_cols[2].markdown(mini_tile("Saved base", _target_display(target_summary["base_target"])), unsafe_allow_html=True)
    summary_cols[3].markdown(mini_tile("Saved stretch", _target_display(target_summary["stretch_target"])), unsafe_allow_html=True)
    summary_cols[4].markdown(mini_tile("Target contribution", f"{projection['contribution_pct']:.1f}%"), unsafe_allow_html=True)

    chart_data = _target_breakdown_chart_data(
        bundle,
        controls,
        selected_output,
        projection,
        target_summary,
        int(outputs_context["max_offset"]),
    )
    if chart_data.empty:
        st.info("No target projection curve is available for this selection.")
    else:
        visible_cumulative = _visible_projection_columns(chart_data, controls["target_modes"], suffix="cumulative")
        visible_daily = _visible_projection_columns(chart_data, controls["target_modes"], suffix="daily")
        if controls.get("historical_fixture_id") and "historic_comparison_cumulative" in chart_data:
            visible_cumulative.append("historic_comparison_cumulative")
        if controls.get("historical_fixture_id") and "historic_comparison_daily" in chart_data:
            visible_daily.append("historic_comparison_daily")
        if str(controls.get("historical_choice", "None")) == "Last season equivalent":
            if "last_season_cumulative" in chart_data:
                visible_cumulative.append("last_season_cumulative")
            if "last_season_daily" in chart_data:
                visible_daily.append("last_season_daily")
        st.plotly_chart(_projection_figure(chart_data, visible_cumulative, "Cumulative sales projection", cumulative=True), use_container_width=True)
        _download_csv(chart_data[["date"] + visible_cumulative], "target_breakdown_cumulative_projection.csv", page_state_key("target_breakdown", "cumulative_csv"))
        st.plotly_chart(_projection_figure(chart_data, visible_daily, "Daily sales projection", cumulative=False), use_container_width=True)
        _download_csv(chart_data[["date"] + visible_daily], "target_breakdown_daily_projection.csv", page_state_key("target_breakdown", "daily_csv"))


def _render_audience_marketing_planner(
    bundle: StrikersDataBundle,
    controls: dict[str, object],
    context: dict[str, object],
    global_context: dict[str, object],
) -> None:
    st.markdown('<div class="section-kicker">Audience & Marketing Planner</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">Which audiences and ticket products are buying or not buying versus expectation at the same point in the sales cycle, and what should M&C do next?</div>',
        unsafe_allow_html=True,
    )
    current_window = current_sales_window(controls["as_at_date"], context["scoped_matches"].rename(columns={"event_date": "match_date"}))
    next_window = "match-week" if current_window == "match-week" else "next window: " + current_window.replace("-", " ")
    cols = st.columns(4)
    cols[0].markdown(mini_tile("Current uploaded snapshot", f"{global_context['latest_transaction_date']:%d %b %Y}"), unsafe_allow_html=True)
    cols[1].markdown(mini_tile("Current sales window", current_window.replace("-", " ").title()), unsafe_allow_html=True)
    cols[2].markdown(mini_tile("Upcoming context", next_window.title()), unsafe_allow_html=True)
    cols[3].markdown(mini_tile("Planning season", str(controls["season_label"])), unsafe_allow_html=True)

    with st.expander("How expected by now is calculated", expanded=False):
        st.write(
            "The planner compares actual paid tickets, revenue, and purchasers in the uploaded snapshot with historical sales that had reached the same point before the fixture. "
            "It aligns by days to fixture first, then uses inferred sales windows and blended historical contribution as a fallback. Late-window audiences are labelled Not due yet when history shows they normally buy later."
        )

    settings = st.columns(3)
    min_campaign_size = settings[0].slider("Minimum campaign audience size", 100, 10_000, 5_000, step=100, key=page_state_key("audience_planner", "min_campaign_size"))
    diagnostic_min_size = settings[1].slider("Minimum diagnostic size", 25, 1_000, 250, step=25, key=page_state_key("audience_planner", "diagnostic_min_size"))
    show_cols = settings[2].checkbox("Show detailed metric columns", value=False, key=page_state_key("audience_planner", "show_detail_columns"))

    with st.expander("Advanced status thresholds", expanded=False):
        ahead = st.slider("Ahead threshold", 1.00, 1.50, 1.10, step=0.01, key=page_state_key("audience_planner", "ahead_threshold"))
        on_track = st.slider("On track threshold", 0.80, 1.05, 0.95, step=0.01, key=page_state_key("audience_planner", "on_track_threshold"))
        watch = st.slider("Watch threshold", 0.70, 1.00, 0.85, step=0.01, key=page_state_key("audience_planner", "watch_threshold"))
        behind = st.slider("Behind threshold", 0.50, 0.90, 0.70, step=0.01, key=page_state_key("audience_planner", "behind_threshold"))
    thresholds = {"ahead": ahead, "on_track": on_track, "watch": watch, "behind": behind}
    baseline_seasons = _baseline_seasons_from_controls(controls["history_filters"])

    audience_index = _expected_index(bundle, controls, "segment", diagnostic_min_size, thresholds)
    ticket_type_index = _expected_index(bundle, controls, "ticket_type", diagnostic_min_size, thresholds)
    ticket_class_index = _expected_index(bundle, controls, "ticket_class", diagnostic_min_size, thresholds)
    combo_index = _expected_index(bundle, controls, "segment_product", diagnostic_min_size, thresholds)

    coverage = demographic_coverage(bundle.customers)
    low_coverage = coverage[coverage["coverage_pct"].lt(40)] if not coverage.empty else pd.DataFrame()
    if not low_coverage.empty:
        st.warning("Some demographic fields have low coverage. Audience recommendations remain aggregated and should avoid overclaiming demographic patterns.")

    tabs = st.tabs(["Audience segments", "Ticket types", "Ticket classes", "Segment x ticket product"])
    with tabs[0]:
        _render_index_tab(audience_index, "Audience segment actual vs expected", show_cols)
        recs = marketing_recommendations(audience_index, min_marketable_size=min_campaign_size)
        if not recs.empty:
            st.markdown('<div class="section-kicker">Recommended marketing actions</div>', unsafe_allow_html=True)
            rec_display = _recommendation_columns(recs)
            st.dataframe(rec_display, use_container_width=True, hide_index=True)
            _download_csv(rec_display, "audience_recommendations_2026_27.csv", page_state_key("audience_planner", "recommendations_csv"))
            st.success(recs.iloc[0]["rationale"] + " Recommended action: " + recs.iloc[0]["recommended_action"])
    with tabs[1]:
        _render_index_tab(ticket_type_index, "Ticket type actual vs expected", show_cols)
    with tabs[2]:
        _render_index_tab(ticket_class_index, "Ticket class actual vs expected", show_cols)
    with tabs[3]:
        _render_index_tab(combo_index, "Segment x ticket product actual vs expected", show_cols)

    st.markdown('<div class="section-kicker">Sales-window context</div>', unsafe_allow_html=True)
    window_frame = sales_window_summary(bundle.transactions[bundle.transactions["season_label"].ne(str(controls["season_label"]))])
    if window_frame.empty:
        st.info("No sales windows can be inferred from historical transactions.")
    else:
        st.dataframe(window_frame, use_container_width=True, hide_index=True)
        _download_csv(window_frame, "audience_planner_sales_window_context.csv", page_state_key("audience_planner", "sales_window_csv"))


def _render_audience_insights(transactions: pd.DataFrame, audience_frame: pd.DataFrame, controls: dict[str, object]) -> None:
    st.markdown('<div class="section-kicker">Audience Insights</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">Audience views are aggregated by default. Names, emails, phones, and addresses are not displayed.</div>',
        unsafe_allow_html=True,
    )
    if audience_frame.empty:
        st.info("No audience rows are available for the selected filters.")
        return
    st.dataframe(audience_frame, use_container_width=True, hide_index=True)
    _download_csv(audience_frame, "audience_insights_summary.csv", page_state_key("audience_insights", "summary_csv"))
    chart_frame = audience_frame.set_index(audience_frame.columns[0])[["paid_tickets", "comps"]]
    st.bar_chart(chart_frame)
    _download_csv(chart_frame.reset_index(), "audience_insights_paid_vs_comps_chart.csv", page_state_key("audience_insights", "chart_csv"))
    st.markdown('<div class="section-kicker">Automated flags</div>', unsafe_allow_html=True)
    for _, row in audience_frame.head(5).iterrows():
        st.write(
            f"- {row.iloc[0]}: {row['status']} with {row['paid_tickets']:,.0f} paid tickets, "
            f"{row['comps']:,.0f} comps, and {row['contribution_pct']:.1f}% contribution."
        )


def _render_audience_insights_planning(bundle: StrikersDataBundle) -> None:
    st.markdown('<div class="section-kicker">Audience Insights</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">Use historic audience behaviour to decide which segments should be targeted first as the new season goes on sale.</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="filter-band">', unsafe_allow_html=True)
    historical_seasons = _historical_season_options(bundle)
    default_season = _latest_historical_season(bundle) or (historical_seasons[-1] if historical_seasons else PLANNING_SEASON_LABEL)
    cols = st.columns([1, 0.8, 1.0, 1.0])
    baseline_season = cols[0].selectbox(
        "Historical season",
        historical_seasons,
        index=historical_seasons.index(default_season) if default_season in historical_seasons else len(historical_seasons) - 1,
        key=page_state_key("audience_insights_new", "baseline_season"),
    )
    competition = cols[1].selectbox("Competition", ["All"] + list(COMPETITION_OPTIONS), key=page_state_key("audience_insights_new", "competition"))
    segment_field = cols[2].selectbox(
        "Segment lens",
        ["ticket_class", "price_type", "purchaser_gender", "purchaser_postcode", "customer_purchase_status", "sales_window"],
        key=page_state_key("audience_insights_new", "segment_field"),
    )
    min_size = cols[3].number_input("Minimum segment tickets", min_value=10, max_value=10000, value=250, step=50, key=page_state_key("audience_insights_new", "min_size"))
    st.markdown("</div>", unsafe_allow_html=True)

    coverage = demographic_coverage(bundle.customers)
    if not coverage.empty:
        low = coverage[coverage["coverage_pct"].lt(40)]
        if not low.empty:
            st.warning("Demographic/postcode coverage is limited in the current customer extract, so demographic recommendations should be treated as directional.")

    planning_controls = {
        "season_label": PLANNING_SEASON_LABEL,
        "competition": competition,
        "as_at_date": _reference_date(bundle),
        "history_filters": {"season_label": [baseline_season], **({"competition": competition} if competition != "All" else {})},
    }
    current_index = _expected_index(bundle, planning_controls, "segment", int(min_size))
    if not current_index.empty:
        st.markdown('<div class="section-kicker">Planning-season segment status</div>', unsafe_allow_html=True)
        current_display = current_index[
            [
                "analysis_group",
                "status",
                "expected_paid_tickets_by_now",
                "expected_final_paid_tickets",
                "usual_purchase_window",
                "recommended_action",
                "rationale",
            ]
        ]
        st.dataframe(current_display, use_container_width=True, hide_index=True)
        _download_csv(current_display, "audience_insights_planning_segment_status_2026_27.csv", page_state_key("audience_insights_new", "segment_status_csv"))

    target_segments = _early_onsale_target_segments(bundle.transactions, bundle.customers, baseline_season, competition, segment_field, int(min_size))
    st.markdown('<div class="section-kicker">Start-of-season target segments</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">Segments below are based on customers who bought in the first month from 1 August in the selected historical season. If August sales are unavailable, the app falls back to the first 31 days of that season sales data.</div>',
        unsafe_allow_html=True,
    )
    if target_segments.empty:
        st.info("No early on-sale target segments are available for this selection.")
    else:
        st.dataframe(target_segments, use_container_width=True, hide_index=True)
        _download_csv(target_segments, "audience_insights_start_of_season_targets.csv", page_state_key("audience_insights_new", "target_segments_csv"))
        top = target_segments.iloc[0]
        st.success(
            f"First target: {top['segment']} delivered {top['early_paid_tickets']:,.0f} early paid tickets in {baseline_season}. "
            f"{top['recommendation']}"
        )


def _render_audience_insights_tracking(
    bundle: StrikersDataBundle,
    pace_engine: HistoricalPaceEngine,
    forecaster: TicketSalesForecaster,
    global_context: dict[str, object],
) -> None:
    st.markdown('<div class="section-kicker">Audience Insights</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">Actual-sales tracking for the 26/27 season. Use this page to compare the current uploaded snapshot with one historical benchmark and one selected target.</div>',
        unsafe_allow_html=True,
    )
    controls = _audience_insights_controls(bundle)
    actual_tx = _projection_transactions(
        bundle.transactions,
        controls,
        PLANNING_SEASON_LABEL,
        fixture_id=str(controls["fixture_ids"][0]) if controls.get("fixture_ids") else None,
        include_products=True,
        include_demographics=True,
        include_audience=True,
        as_at_date=pd.Timestamp(controls["as_at_date"]),
    )

    outputs_context = _target_breakdown_outputs(bundle, pace_engine, forecaster, controls)
    max_offset = int(outputs_context.get("max_offset", 170))
    actual_curve = _audience_actual_curve(actual_tx, controls, max_offset)
    historical_curve = _audience_historical_benchmark_curve(bundle, controls, max_offset)
    target_curve, target_note = _audience_target_curve(bundle, controls, outputs_context, max_offset)
    chart_data = _audience_tracking_chart_data(actual_curve, historical_curve, target_curve, max_offset)

    mode_label = "Demo/dummy 26/27 snapshot" if bundle.metrics.get("dummy_26_27_sales_active") else "Real uploaded 26/27 snapshot"
    freshness = global_context.get("freshness", {})
    status_cols = st.columns(4)
    status_cols[0].markdown(mini_tile("Actual data source", mode_label), unsafe_allow_html=True)
    status_cols[1].markdown(mini_tile("As at date", f"{pd.Timestamp(controls['as_at_date']):%d %b %Y}"), unsafe_allow_html=True)
    status_cols[2].markdown(mini_tile("Actual paid tickets", format_number(actual_tx["paid_tickets_sold"].sum() if "paid_tickets_sold" in actual_tx else 0)), unsafe_allow_html=True)
    status_cols[3].markdown(mini_tile("Actual comps", format_number(actual_tx["comp_tickets_sold"].sum() if "comp_tickets_sold" in actual_tx else 0)), unsafe_allow_html=True)
    if bundle.metrics.get("dummy_26_27_sales_active"):
        st.info(str(bundle.metrics.get("dummy_26_27_sales_label", "Synthetic 26/27 sales are active in demo mode.")))
    elif freshness.get("is_stale"):
        st.warning(str(freshness.get("message", "The uploaded snapshot may be stale.")))
    if target_note:
        st.warning(target_note)

    if chart_data.empty:
        st.info("No actual 26/27 chart data is available for this selection.")
    else:
        cumulative_cols = [column for column in ["actual_cumulative", "historical_cumulative", "target_cumulative"] if column in chart_data]
        daily_cols = [column for column in ["actual_daily", "historical_daily", "target_daily"] if column in chart_data]
        st.plotly_chart(_projection_figure(chart_data, cumulative_cols, "Audience Insights cumulative sales", cumulative=True), use_container_width=True)
        _download_csv(chart_data[["date"] + cumulative_cols], "audience_insights_cumulative_actual_vs_benchmark.csv", page_state_key("audience_insights", "cumulative_csv"))
        st.plotly_chart(_projection_figure(chart_data, daily_cols, "Audience Insights daily sales", cumulative=False), use_container_width=True)
        _download_csv(chart_data[["date"] + daily_cols], "audience_insights_daily_actual_vs_benchmark.csv", page_state_key("audience_insights", "daily_csv"))

    st.markdown('<div class="section-kicker">Actual-sales insight tables</div>', unsafe_allow_html=True)
    show_cols = st.checkbox("Show detailed metric columns", value=False, key=page_state_key("audience_insights", "show_detail_columns"))
    tabs = st.tabs(["Audience segments", "Ticket types", "Ticket classes", "Demographics", "Recommendations"])
    with tabs[0]:
        audience_index = _expected_index(bundle, controls, "segment", 0)
        _render_index_tab(audience_index, "Audience Insights segment performance", show_cols)
    with tabs[1]:
        ticket_type_index = _expected_index(bundle, controls, "ticket_type", 0)
        _render_index_tab(ticket_type_index, "Audience Insights ticket type performance", show_cols)
    with tabs[2]:
        ticket_class_index = _expected_index(bundle, controls, "ticket_class", 0)
        _render_index_tab(ticket_class_index, "Audience Insights ticket class performance", show_cols)
    with tabs[3]:
        demo_tables = {
            "Age band": ("age_band_filter", "audience_insights_age_band.csv"),
            "Gender": ("gender_filter", "audience_insights_gender.csv"),
            "Postcode": ("postcode_filter", "audience_insights_postcode.csv"),
        }
        for label, (column, filename) in demo_tables.items():
            table = audience_summary(actual_tx, column)
            st.markdown(f"**{label}**")
            if table.empty:
                st.info(f"No {label.lower()} rows are available for this filtered actual-sales view.")
            else:
                st.dataframe(table, use_container_width=True, hide_index=True)
                _download_csv(table, filename, page_state_key("audience_insights", f"{column}_csv"))
    with tabs[4]:
        rec_source = _expected_index(bundle, controls, "segment", 100)
        recs = marketing_recommendations(rec_source, min_marketable_size=1000)
        if recs.empty:
            st.info("No recommendation rows are available for this filtered actual-sales view.")
        else:
            rec_display = _recommendation_columns(recs)
            st.dataframe(rec_display, use_container_width=True, hide_index=True)
            _download_csv(rec_display, "audience_insights_recommendations.csv", page_state_key("audience_insights", "recommendations_csv"))


def _audience_insights_controls(bundle: StrikersDataBundle) -> dict[str, object]:
    page_key = "audience_insights"
    st.markdown('<div class="filter-band">', unsafe_allow_html=True)
    st.markdown('<div class="section-kicker">Actual-sales controls</div>', unsafe_allow_html=True)
    top = st.columns([0.7, 0.95, 0.95, 1.0])
    competition = top[0].selectbox("Competition", ["All"] + list(COMPETITION_OPTIONS), key=page_state_key(page_key, "competition"))
    include_assumed = top[1].checkbox("Include assumed upcoming fixtures", value=True, key=page_state_key(page_key, "include_assumed"))
    as_at_date = pd.Timestamp(
        top[2].date_input("As at uploaded snapshot", value=_planning_as_at_date(bundle).date(), key=page_state_key(page_key, "as_at_date"))
    )
    target_mode = top[3].selectbox("Target comparison", ["Forecast", "Forecast +10%", "Base", "Stretch"], key=page_state_key(page_key, "target_mode"))

    scoped_fixtures = _scoped_fixtures(bundle.fixtures, PLANNING_SEASON_LABEL, competition, [], include_assumed)
    fixture_label_col = scoped_fixtures["fixture_label"].fillna(scoped_fixtures["fixture_id"]).astype(str) if not scoped_fixtures.empty else pd.Series(dtype=str)
    fixture_options = ["Season total"] + fixture_label_col.tolist()
    selected_fixture = st.selectbox("Fixture", fixture_options, key=page_state_key(page_key, "fixture"))
    fixture_ids: list[str] = []
    historical_fixture_id = ""
    historical_options = ["Blended historical average", "Last season equivalent"]
    label_to_id: dict[str, str] = {}
    if selected_fixture != "Season total" and not scoped_fixtures.empty:
        label_to_id = dict(zip(fixture_label_col, scoped_fixtures["fixture_id"].astype(str)))
        fixture_ids = [label_to_id[selected_fixture]] if selected_fixture in label_to_id else []
        selected_match = scoped_fixtures[scoped_fixtures["fixture_id"].astype(str).isin(fixture_ids)].iloc[0] if fixture_ids else pd.Series(dtype=object)
        for row in _historical_fixture_options(bundle, competition, selected_match):
            historical_options.append(row["label"])
    historical_choice = st.selectbox("Historical comparison", historical_options, key=page_state_key(page_key, "historical_choice"))
    if historical_choice not in {"Blended historical average", "Last season equivalent"} and selected_fixture != "Season total":
        selected_match = scoped_fixtures[scoped_fixtures["fixture_id"].astype(str).isin(fixture_ids)].iloc[0] if fixture_ids else pd.Series(dtype=object)
        options = _historical_fixture_options(bundle, competition, selected_match)
        historical_fixture_id = next((row["fixture_id"] for row in options if row["label"] == historical_choice), "")

    with st.expander("Segment filters", expanded=False):
        product_cols = st.columns(3)
        ticket_type = product_cols[0].selectbox("Ticket type", filter_options(bundle.transactions, "ticket_type"), key=page_state_key(page_key, "ticket_type"))
        ticket_class = product_cols[1].selectbox("Ticket class", filter_options(bundle.transactions, "ticket_class"), key=page_state_key(page_key, "ticket_class"))
        ticket_status = product_cols[2].selectbox(
            "Paid / comp",
            ["Paid + comps", "Paid tickets only", "Comps only", "Refunds / voids"],
            key=page_state_key(page_key, "ticket_status"),
        )
        demo_cols = st.columns(4)
        age_bands = demo_cols[0].multiselect("Age / age band", filter_options(bundle.transactions, "age_band_filter", include_all=False), key=page_state_key(page_key, "age_band"))
        genders = demo_cols[1].multiselect("Gender", filter_options(bundle.transactions, "gender_filter", include_all=False), key=page_state_key(page_key, "gender"))
        postcodes = demo_cols[2].multiselect("Postcode", filter_options(bundle.transactions, "postcode_filter", include_all=False), key=page_state_key(page_key, "postcode"))
        audience_segment = demo_cols[3].selectbox(
            "Audience segment",
            ["All", "Families", "18-30s", "Marketable", "Returning purchasers"],
            key=page_state_key(page_key, "audience_segment"),
        )

    if st.button("Reset filters for this page", key=page_state_key(page_key, "reset_inline")):
        for key in list(st.session_state.keys()):
            if str(key).startswith(f"{page_key}_"):
                del st.session_state[key]
        st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

    historical_labels = _historical_season_options(bundle)
    if historical_choice == "Last season equivalent":
        selected_history = historical_labels[-1:] if historical_labels else []
    else:
        selected_history = [label for label in DEFAULT_HISTORICAL_SEASONS if label in historical_labels] or historical_labels[-3:]
    return {
        "season_label": PLANNING_SEASON_LABEL,
        "competition": competition,
        "include_assumed": include_assumed,
        "fixture_ids": fixture_ids,
        "fixture_choice": selected_fixture,
        "target_modes": [target_mode],
        "target_mode": target_mode,
        "historical_choice": historical_choice,
        "historical_fixture_id": historical_fixture_id,
        "historical_fixture_label": historical_choice,
        "as_at_date": as_at_date,
        "audience_segment": audience_segment,
        "ticket_type": ticket_type,
        "ticket_class": ticket_class,
        "section": "All",
        "sales_channel": "All",
        "sales_window": "All",
        "ticket_status": ticket_status,
        "age_band": age_bands,
        "gender": genders,
        "postcode": postcodes,
        "target_uplift_pct": 0.10,
        "early_bird_lift": 0.16,
        "member_presale_lift": 0.12,
        "marketing_lift": 0.18,
        "campaign_lift": 0.28,
        "final_week_lift": 0.22,
        "history_filters": {"season_label": selected_history, **({"competition": competition} if competition != "All" else {})},
        "audience_group_by": "ticket_class",
    }


def _render_recommended_audiences(bundle: StrikersDataBundle, controls: dict[str, object]) -> None:
    st.markdown('<div class="section-kicker">Recommended Audiences</div>', unsafe_allow_html=True)
    min_size = st.slider("Minimum audience size", min_value=100, max_value=10_000, value=5_000, step=100)
    recommendations = recommended_audiences(
        bundle.customers,
        bundle.transactions,
        PLANNING_SEASON_LABEL,
        min_size=min_size,
        as_at_date=controls["as_at_date"],
    )
    if recommendations.empty:
        st.info("No marketable audience recommendations are available yet. Load customer data with gigyauid/customer IDs and opt-in fields to unlock this page.")
        return
    st.dataframe(recommendations, use_container_width=True, hide_index=True)
    _download_csv(recommendations, "recommended_audiences_2026_27.csv", page_state_key("recommended_audiences", "table_csv"))
    top = recommendations.iloc[0]
    st.success(
        f"Recommended next audience: {top['segment']} with estimated upside of {top['estimated_ticket_upside']:,.0f} tickets. "
        f"{top['rationale']}"
    )


def _render_sales_windows(transactions: pd.DataFrame) -> None:
    st.markdown('<div class="section-kicker">Sales Windows / Campaign Planner</div>', unsafe_allow_html=True)
    window_frame = sales_window_summary(transactions)
    if window_frame.empty:
        st.info("No sales windows can be inferred from the selected transactions.")
        return
    edited = st.data_editor(window_frame, use_container_width=True, hide_index=True)
    _download_csv(edited, "sales_window_plan_current_view.csv", page_state_key("sales_windows", "table_csv"))
    st.bar_chart(window_frame.set_index("sales_window")[["paid_tickets", "comps"]])
    _download_csv(window_frame, "sales_window_paid_vs_comps_chart.csv", page_state_key("sales_windows", "chart_csv"))
    if st.button("Save sales window plan locally"):
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        edited.to_csv(PROCESSED_DATA_DIR / "sales_window_plan.csv", index=False)
        st.success(f"Saved to {PROCESSED_DATA_DIR / 'sales_window_plan.csv'}")


def _render_reports(
    context: dict[str, object],
    controls: dict[str, object],
    bundle: StrikersDataBundle,
    global_context: dict[str, object],
) -> None:
    st.markdown('<div class="section-kicker">Reports</div>', unsafe_allow_html=True)
    report_type = st.selectbox(
        "Report type",
        [
            "Weekly current-season update",
            "Audience opportunity summary",
            "Early bird performance summary",
            "Ticket product performance summary",
            "Fixture risk summary",
            "NYE fixture planning summary",
        ],
        key=page_state_key("reports", "report_type"),
    )
    assumed_count = int(bundle.fixtures.get("is_assumed", pd.Series(dtype=bool)).fillna(False).sum()) if not bundle.fixtures.empty else 0
    assumed_note = f"{assumed_count} future fixtures are currently marked TBD / assumed." if assumed_count else "No assumed fixtures are included in the selected scope."
    audience_index = _expected_index(bundle, controls, "segment", 250)
    ticket_type_index = _expected_index(bundle, controls, "ticket_type", 250)
    ticket_class_index = _expected_index(bundle, controls, "ticket_class", 250)
    snapshot_note = f"Current uploaded snapshot latest transaction date: {global_context['latest_transaction_date']:%d %b %Y}. {global_context['freshness']['message']}"
    html = report_html(
        report_type,
        context["kpis"],
        context["fixture_frame"],
        context["insights"],
        assumed_note,
        audience_index=audience_index,
        ticket_type_index=ticket_type_index,
        ticket_class_index=ticket_class_index,
        snapshot_note=snapshot_note,
    )
    st.markdown(html, unsafe_allow_html=True)
    st.download_button("Download HTML report", data=html, file_name=f"{report_type.lower().replace(' ', '_')}.html", mime="text/html")
    if not context["fixture_frame"].empty:
        st.download_button(
            "Download fixture summary CSV",
            data=context["fixture_frame"].to_csv(index=False),
            file_name="fixture_summary.csv",
            mime="text/csv",
        )
    if not audience_index.empty:
        st.download_button(
            "Download audience opportunity CSV",
            data=audience_index.to_csv(index=False),
            file_name="audience_opportunity_summary.csv",
            mime="text/csv",
        )
    if not ticket_type_index.empty:
        _download_csv(ticket_type_index, "ticket_type_performance_report.csv", page_state_key("reports", "ticket_type_csv"))
    if not ticket_class_index.empty:
        _download_csv(ticket_class_index, "ticket_class_performance_report.csv", page_state_key("reports", "ticket_class_csv"))


def _render_historical_overview(
    bundle: StrikersDataBundle,
    pace_engine: HistoricalPaceEngine,
    forecaster: TicketSalesForecaster,
    reference_date: pd.Timestamp,
    global_context: dict[str, object],
) -> None:
    page_key = "historical_overview"
    st.markdown('<div class="section-kicker">Historical Overview / Data QA</div>', unsafe_allow_html=True)
    st.markdown('<div class="filter-band">', unsafe_allow_html=True)
    historical_seasons = [season for season in _season_options(bundle.matches) if season != PLANNING_SEASON_LABEL]
    default_season = historical_seasons[-1] if historical_seasons else "All historical"
    cols = st.columns([1.1, 0.8, 0.9])
    season_choice = cols[0].selectbox(
        "Historical season",
        ["All historical"] + historical_seasons,
        index=(["All historical"] + historical_seasons).index(default_season) if default_season in ["All historical"] + historical_seasons else 0,
        key=page_state_key(page_key, "season"),
    )
    competition = cols[1].selectbox("Competition", ["All"] + list(COMPETITION_OPTIONS), index=0, key=page_state_key(page_key, "competition"))
    ticket_status = cols[2].selectbox(
        "Paid / comp status",
        ["Paid + comps", "Paid tickets only", "Comps only", "Refunds / voids"],
        key=page_state_key(page_key, "ticket_status"),
    )
    st.markdown("</div>", unsafe_allow_html=True)
    season_filter = None if season_choice == "All historical" else season_choice
    tx = apply_transaction_filters(
        bundle.transactions,
        season_label=season_filter,
        competition=competition,
        ticket_status=ticket_status,
    )
    fixtures = bundle.fixtures.copy()
    if season_filter:
        fixtures = fixtures[fixtures["season_label"].astype(str).eq(str(season_filter))]
    else:
        fixtures = fixtures[fixtures["season_label"].astype(str).ne(PLANNING_SEASON_LABEL)]
    if competition != "All" and not fixtures.empty:
        fixtures = fixtures[fixtures["competition"].astype(str).eq(str(competition))]
    matches = bundle.matches.copy()
    if season_filter:
        matches = matches[matches["season_label"].astype(str).eq(str(season_filter))]
    else:
        matches = matches[matches["season_label"].astype(str).ne(PLANNING_SEASON_LABEL)]
    if competition != "All" and not matches.empty:
        matches = matches[matches["competition"].astype(str).eq(str(competition))]

    kpis = kpi_summary(tx, bundle.transactions, fixtures, as_at_date=global_context["latest_transaction_date"])
    _render_header(
        bundle,
        {"season_label": season_choice, "competition": competition},
        matches,
        reference_date,
        kpis,
    )
    st.markdown(
        '<div class="section-note">Use this page to validate historical ticketing data and understand prior-season sales curves. Planning and campaign recommendations sit in Fixture Forecasting, Target Breakdown, and Audience & Marketing Planner.</div>',
        unsafe_allow_html=True,
    )
    cols = st.columns(5)
    cols[0].markdown(mini_tile("Ticket rows", f"{len(tx):,}"), unsafe_allow_html=True)
    cols[1].markdown(mini_tile("Paid tickets", format_number(kpis["paid_tickets_sold"])), unsafe_allow_html=True)
    cols[2].markdown(mini_tile("Comps", format_number(kpis["comps_issued"])), unsafe_allow_html=True)
    cols[3].markdown(mini_tile("Gross revenue", _money(kpis["gross_revenue"])), unsafe_allow_html=True)
    cols[4].markdown(mini_tile("ATP", _money(kpis["average_ticket_price"])), unsafe_allow_html=True)

    st.markdown('<div class="section-kicker">Historical cumulative curves</div>', unsafe_allow_html=True)
    comparison_seasons = historical_seasons[-4:] if season_choice == "All historical" else [season_choice]
    if competition == "All":
        tabs = st.tabs(["BBL", "WBBL"])
        for tab, comp in zip(tabs, COMPETITION_OPTIONS):
            with tab:
                comparison = build_season_comparison_frame(bundle.matches, bundle.daily_sales, comp, comparison_seasons)
                if comparison.empty:
                    st.info(f"No historical curve is available for {comp}.")
                else:
                    st.plotly_chart(season_comparison_figure(comparison, PLANNING_SEASON_LABEL), use_container_width=True)
                    _download_csv(comparison, f"historical_overview_{_safe_file_part(comp)}_season_comparison.csv", page_state_key("historical_overview", f"{comp}_comparison_csv"))
    else:
        comparison = build_season_comparison_frame(bundle.matches, bundle.daily_sales, competition, comparison_seasons)
        if comparison.empty:
            st.info("No historical curve is available for this selection.")
        else:
            st.plotly_chart(season_comparison_figure(comparison, PLANNING_SEASON_LABEL), use_container_width=True)
            _download_csv(comparison, f"historical_overview_{_safe_file_part(competition)}_season_comparison.csv", page_state_key("historical_overview", "comparison_csv"))

    fixture_frame = fixture_sales_summary(tx, fixtures, matches, [])
    st.markdown('<div class="section-kicker">Fixture-level QA</div>', unsafe_allow_html=True)
    if fixture_frame.empty:
        st.info("No fixture rows are available for the selected historical scope.")
    else:
        fixture_display = fixture_frame[
                [
                    "fixture_label",
                    "competition",
                    "match_date",
                    "sold",
                    "paid_sold",
                    "comps",
                    "gross_revenue",
                    "capacity_sold_pct",
                ]
        ]
        st.dataframe(fixture_display, use_container_width=True, hide_index=True)
        _download_csv(fixture_display, "historical_overview_fixture_qa.csv", page_state_key("historical_overview", "fixture_qa_csv"))
    if not bundle.validation_warnings.empty:
        st.markdown('<div class="section-kicker">Data confidence warnings</div>', unsafe_allow_html=True)
        st.dataframe(bundle.validation_warnings, use_container_width=True, hide_index=True)
        _download_csv(bundle.validation_warnings, "historical_overview_validation_warnings.csv", page_state_key("historical_overview", "warnings_csv"))


def _render_index_tab(frame: pd.DataFrame, title: str, show_detail_columns: bool) -> None:
    if frame.empty:
        st.info("No expected-by-now rows are available for this view.")
        return
    chart_frame = (
        frame.sort_values("expected_paid_tickets_by_now", ascending=False)
        .head(12)
        .reset_index(drop=True)
    )
    st.plotly_chart(
        grouped_comparison_bar_figure(
            chart_frame,
            title,
            category_col="analysis_group",
            actual_col="current_paid_tickets",
            expected_col="expected_paid_tickets_by_now",
        ),
        use_container_width=True,
    )
    _download_csv(
        chart_frame[["analysis_group", "current_paid_tickets", "expected_paid_tickets_by_now"]],
        f"{_safe_file_part(title)}_chart.csv",
        page_state_key("index_tab", f"{_safe_file_part(title)}_chart_csv"),
    )
    compact_cols = [
        "analysis_group",
        "status",
        "current_paid_tickets",
        "current_comps",
        "expected_paid_tickets_by_now",
        "ticket_index",
        "paid_ticket_gap",
        "expected_revenue_by_now",
        "revenue_gap",
        "usual_purchase_window",
        "current_sales_window",
        "recommended_action",
        "rationale",
    ]
    detail_cols = [
        "analysis_group",
        "definition",
        "eligible_audience_size",
        "marketable_audience_size",
        "current_unique_purchasers",
        "expected_purchasers_by_now",
        "actual_purchase_rate",
        "expected_purchase_rate",
        "purchase_rate_index",
        "current_average_basket_size",
        "current_average_ticket_price",
        "confidence",
    ]
    columns = compact_cols + detail_cols if show_detail_columns else compact_cols
    available = [column for column in columns if column in frame]
    display = frame[available]
    st.dataframe(display, use_container_width=True, hide_index=True)
    _download_csv(display, f"{_safe_file_part(title)}.csv", page_state_key("index_tab", f"{_safe_file_part(title)}_table_csv"))


def _recommendation_columns(frame: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "analysis_group",
        "definition",
        "marketable_audience_size",
        "current_unique_purchasers",
        "expected_purchasers_by_now",
        "actual_purchase_rate",
        "expected_purchase_rate",
        "purchase_rate_index",
        "current_paid_tickets",
        "expected_paid_tickets_by_now",
        "paid_ticket_gap",
        "revenue_gap",
        "usual_purchase_window",
        "current_sales_window",
        "recommended_action",
        "suggested_message_angle",
        "suggested_timing",
        "suggested_ticket_product",
        "confidence",
        "rationale",
        "marketable_note",
    ]
    return frame[[column for column in columns if column in frame]]


def _risk_bullets(audience_index: pd.DataFrame, ticket_class_index: pd.DataFrame) -> None:
    if audience_index.empty and ticket_class_index.empty:
        st.info("No audience or ticket-class index rows are available yet.")
        return
    if not audience_index.empty:
        risky = audience_index[audience_index["status"].isin(["At risk", "Behind", "Watch"])]
        if not risky.empty:
            row = risky.sort_values("paid_ticket_gap").iloc[0]
            st.write(
                f"- Top audience watch item: {row['analysis_group']} is {abs(row['paid_ticket_gap']):,.0f} paid tickets behind expected by now "
                f"(index {row['ticket_index']:.2f})."
            )
        not_due = audience_index[audience_index["status"].eq("Not due yet")]
        if not not_due.empty:
            row = not_due.iloc[0]
            st.write(f"- Not due yet: {row['analysis_group']} usually buys in {row['usual_purchase_window']}; prepare later-window activity.")
    if not ticket_class_index.empty:
        class_risk = ticket_class_index[ticket_class_index["status"].isin(["At risk", "Behind", "Watch"])]
        if not class_risk.empty:
            row = class_risk.sort_values("paid_ticket_gap").iloc[0]
            st.write(
                f"- Ticket class watch item: {row['analysis_group']} is {abs(row['paid_ticket_gap']):,.0f} paid tickets behind expected by now."
            )
        class_upside = ticket_class_index[ticket_class_index["status"].eq("Ahead")]
        if not class_upside.empty:
            row = class_upside.sort_values("ticket_index", ascending=False).iloc[0]
            st.write(f"- Positive trend: {row['analysis_group']} is ahead of expected demand with an index of {row['ticket_index']:.2f}.")


def _baseline_seasons_from_controls(history_filter_values: object) -> list[str]:
    if not isinstance(history_filter_values, dict):
        return []
    values = history_filter_values.get("season_label", [])
    if isinstance(values, str):
        return [values]
    return [str(value) for value in values]


def _safe_display_index(actual: object, expected: object) -> float:
    expected_value = float(expected or 0)
    actual_value = float(actual or 0)
    if expected_value <= 0:
        return 0.0
    return actual_value / expected_value


def _render_future_fixtures(bundle: StrikersDataBundle) -> None:
    st.markdown('<div class="section-kicker">Future Fixture Assumptions</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">Upcoming fixture rows come from uploaded next-season assumptions when supplied. Otherwise they are generated from historical home fixture timing and marked as assumed until confirmed.</div>',
        unsafe_allow_html=True,
    )
    if bundle.future_fixtures.empty:
        st.info("No future fixture assumptions are available.")
        return
    edited = st.data_editor(bundle.future_fixtures, use_container_width=True, hide_index=True)
    _download_csv(edited, "future_fixture_assumptions_current_view.csv", page_state_key("future_fixtures", "table_csv"))
    if st.button("Save edited fixture assumptions locally"):
        PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        edited.to_csv(PROCESSED_DATA_DIR / "future_fixture_assumptions.csv", index=False)
        st.success(f"Saved to {PROCESSED_DATA_DIR / 'future_fixture_assumptions.csv'}")
    st.warning("Assumed fixtures are excluded from historical training interpretation and used only for future forecasts, targets, and scenario modelling.")


def _render_data_admin(bundle: StrikersDataBundle) -> None:
    st.markdown('<div class="section-kicker">Data Admin</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-note">Place private client CSV snapshots in data/raw/ or upload them here. The app reads exported ticketing extracts, not a live API feed. Raw and processed private data folders are ignored by git.</div>',
        unsafe_allow_html=True,
    )
    if bundle.metrics.get("processed_cache_used"):
        st.info(
            "Fast startup path active: the dashboard loaded normalized files from data/processed. "
            "Use Reprocess real data after changing raw CSV extracts or mapping overrides."
        )
    st.dataframe(bundle.file_status, use_container_width=True, hide_index=True)
    _download_csv(bundle.file_status, "data_admin_detected_files.csv", page_state_key("data_admin", "file_status_csv"))

    upload_cols = st.columns(2)
    expected = expected_file_dataframe()
    uploaded_files = {}
    for idx, row in expected.iterrows():
        with upload_cols[idx % 2]:
            uploaded = st.file_uploader(row["expected_file"], type=["csv"], key=f"upload_{row['file_key']}")
            if uploaded is not None:
                uploaded_files[row["expected_file"]] = uploaded
    if uploaded_files and st.button("Save uploaded CSVs to data/raw/"):
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        for filename, uploaded in uploaded_files.items():
            (RAW_DATA_DIR / filename).write_bytes(uploaded.getbuffer())
        st.session_state["reprocess_token"] = st.session_state.get("reprocess_token", 0) + 1
        cached_real_data.clear()
        st.success("Saved uploads. Reprocess real data to refresh the dashboard.")

    cols = st.columns(4)
    cols[0].metric("Ticket rows", f"{bundle.metrics.get('ticket_rows', 0):,}")
    cols[1].metric("Unique customers", f"{bundle.metrics.get('unique_customers', 0):,}")
    cols[2].metric("Customer join rate", f"{bundle.metrics.get('customer_join_rate', 0):.0%}")
    cols[3].metric("Fixture join rate", f"{bundle.metrics.get('fixture_join_rate', 0):.0%}")
    cols = st.columns(4)
    cols[0].metric("Paid tickets", format_number(bundle.metrics.get("paid_ticket_count", 0)))
    cols[1].metric("Comps", format_number(bundle.metrics.get("comp_ticket_count", 0)))
    cols[2].metric("Refund / void rows", f"{bundle.metrics.get('refund_count', 0):,}")
    cols[3].metric("Gross revenue", _money(bundle.metrics.get("gross_revenue", 0)))
    latest_transaction_date = _reference_date(bundle)
    freshness = stale_snapshot_status(latest_transaction_date, today=pd.Timestamp.today())
    cols = st.columns(4)
    cols[0].metric("Latest transaction date", f"{latest_transaction_date:%d %b %Y}")
    cols[1].metric("Extract age", "Unavailable" if freshness["age_days"] is None else f"{freshness['age_days']} days")
    cols[2].metric("Ticket count", format_number(bundle.metrics.get("paid_ticket_count", 0) + bundle.metrics.get("comp_ticket_count", 0)))
    cols[3].metric("Data freshness", "Stale" if freshness["is_stale"] else "Current")
    if bundle.metrics.get("processed_cache_used"):
        st.caption(
            f"Processed cache built: {bundle.metrics.get('processed_cache_timestamp', 'Unavailable')} | "
            f"Latest raw snapshot seen: {bundle.metrics.get('raw_snapshot_timestamp', 'Unavailable')}"
        )
    if freshness["is_stale"]:
        st.warning(str(freshness["message"]))

    st.markdown('<div class="section-kicker">GigyaUID demographic link status</div>', unsafe_allow_html=True)
    demo_cols = st.columns(4)
    demo_cols[0].metric("Ticket rows with GigyaUID", f"{bundle.metrics.get('ticketing_rows_with_gigyauid', 0):,}")
    demo_cols[1].metric("Customer rows with GigyaUID", f"{bundle.metrics.get('customer_rows_with_gigyauid', 0):,}")
    demo_cols[2].metric("Matched ticket rows", f"{bundle.metrics.get('demographic_matched_ticket_rows', 0):,}")
    demo_cols[3].metric("Demographic match rate", f"{bundle.metrics.get('demographic_match_rate', 0):.0%}")
    coverage_cols = st.columns(4)
    coverage_cols[0].metric("Age coverage", f"{bundle.metrics.get('age_coverage', 0):.0%}")
    coverage_cols[1].metric("Gender coverage", f"{bundle.metrics.get('gender_coverage', 0):.0%}")
    coverage_cols[2].metric("Postcode coverage", f"{bundle.metrics.get('postcode_coverage', 0):.0%}")
    coverage_cols[3].metric("26/27 snapshot", "Dummy Aug 2026" if bundle.metrics.get("dummy_26_27_sales_active") else "Real / unavailable")
    st.caption(str(bundle.metrics.get("dummy_26_27_sales_label", "")))

    coverage = demographic_coverage(bundle.customers)
    if not coverage.empty:
        st.markdown('<div class="section-kicker">Demographic and contact coverage</div>', unsafe_allow_html=True)
        st.dataframe(coverage, use_container_width=True, hide_index=True)
        _download_csv(coverage, "data_admin_demographic_coverage.csv", page_state_key("data_admin", "coverage_csv"))

    st.markdown('<div class="section-kicker">Suggested column mapping</div>', unsafe_allow_html=True)
    if bundle.column_mappings.empty:
        if bundle.metrics.get("processed_cache_used"):
            st.info("Suggested raw-source mappings are hidden while using the processed startup cache. Click Reprocess real data to inspect raw mappings and previews.")
        else:
            st.info("No real source mappings are available yet.")
    else:
        selected_key = st.selectbox("Mapping file", sorted(bundle.column_mappings["file_key"].unique()))
        mapping_frame = bundle.column_mappings[bundle.column_mappings["file_key"].eq(selected_key)][["field", "source_column"]].copy()
        edited = st.data_editor(mapping_frame, use_container_width=True, hide_index=True)
        _download_csv(edited, f"data_admin_mapping_{_safe_file_part(selected_key)}.csv", page_state_key("data_admin", "mapping_csv"))
        if st.button("Apply mapping override"):
            overrides = st.session_state.setdefault("mapping_overrides", {})
            overrides[selected_key] = dict(zip(edited["field"], edited["source_column"]))
            st.session_state["reprocess_token"] = st.session_state.get("reprocess_token", 0) + 1
            cached_real_data.clear()
            st.success("Mapping override stored for this session. Reprocess real data to apply it.")

    st.markdown('<div class="section-kicker">Validation warnings</div>', unsafe_allow_html=True)
    if bundle.validation_warnings.empty:
        st.success("No validation warnings for the current mode.")
    else:
        st.dataframe(bundle.validation_warnings, use_container_width=True, hide_index=True)
        _download_csv(bundle.validation_warnings, "data_admin_validation_warnings.csv", page_state_key("data_admin", "validation_csv"))

    if bundle.loaded_sources:
        st.markdown('<div class="section-kicker">Raw column preview</div>', unsafe_allow_html=True)
        loaded_keys = [key for key, source in bundle.loaded_sources.items() if source.csv is not None]
        if loaded_keys:
            selected_source = st.selectbox("Preview source", loaded_keys)
            profile = bundle.loaded_sources[selected_source].csv.profile
            st.dataframe(profile, use_container_width=True, hide_index=True)
            _download_csv(profile, f"data_admin_raw_column_profile_{_safe_file_part(selected_source)}.csv", page_state_key("data_admin", "profile_csv"))
    elif bundle.metrics.get("processed_cache_used"):
        st.markdown('<div class="section-kicker">Raw column preview</div>', unsafe_allow_html=True)
        st.info("Raw column previews are skipped during the fast processed-data startup path. Reprocess real data to inspect raw columns and profile values.")

    if st.button("Reprocess real data"):
        st.session_state["reprocess_token"] = st.session_state.get("reprocess_token", 0) + 1
        cached_real_data.clear()
        st.success("Real data cache cleared. Change a control or rerun the app to reload.")


def _build_forecast_outputs(
    selected_matches: pd.DataFrame,
    matches: pd.DataFrame,
    daily_sales: pd.DataFrame,
    pace_engine: HistoricalPaceEngine,
    forecaster: TicketSalesForecaster,
    assumptions: TargetAssumptions,
    target_uplift_pct: float,
    selected_history_filters: dict[str, object],
) -> list[dict]:
    outputs = []
    for _, match in selected_matches.sort_values("event_date").iterrows():
        outputs.append(
            _build_single_output(
                match,
                matches,
                daily_sales,
                pace_engine,
                forecaster,
                assumptions,
                target_uplift_pct,
                selected_history_filters,
            )
        )
    return outputs


def _build_single_output(
    match: pd.Series,
    matches: pd.DataFrame,
    daily_sales: pd.DataFrame,
    pace_engine: HistoricalPaceEngine,
    forecaster: TicketSalesForecaster,
    assumptions: TargetAssumptions,
    target_uplift_pct: float,
    selected_history_filters: dict[str, object],
    planning_target_override: float | None = None,
    manual_target_override: float | None = None,
) -> dict:
    from dashboard.services import build_match_output

    return build_match_output(
        match,
        matches,
        daily_sales,
        pace_engine,
        forecaster,
        assumptions,
        target_uplift_pct,
        history_filter_values=selected_history_filters,
        planning_target_override=planning_target_override,
        manual_target_override=manual_target_override,
    )


def _scoped_matches(
    matches: pd.DataFrame,
    season_label: str,
    competition: str,
    fixture_ids: list[str],
    include_assumed: bool,
) -> pd.DataFrame:
    frame = matches[matches["season_label"].astype(str).eq(str(season_label))].copy()
    if competition != "All":
        frame = frame[frame["competition"].astype(str).eq(str(competition))]
    if fixture_ids:
        frame = frame[frame["match_id"].astype(str).isin([str(value) for value in fixture_ids])]
    if not include_assumed and "is_assumed" in frame:
        frame = frame[~frame["is_assumed"].fillna(False).astype(bool)]
    return frame.sort_values("event_date").reset_index(drop=True)


def _scoped_fixtures(
    fixtures: pd.DataFrame,
    season_label: str,
    competition: str,
    fixture_ids: list[str],
    include_assumed: bool,
) -> pd.DataFrame:
    if fixtures.empty:
        return fixtures.copy()
    frame = fixtures[fixtures["season_label"].astype(str).eq(str(season_label))].copy()
    if competition != "All":
        frame = frame[frame["competition"].astype(str).eq(str(competition))]
    if fixture_ids:
        frame = frame[frame["fixture_id"].astype(str).isin([str(value) for value in fixture_ids])]
    if not include_assumed and "is_assumed" in frame:
        frame = frame[~frame["is_assumed"].fillna(False).astype(bool)]
    return frame.sort_values("match_date").reset_index(drop=True)


def _historical_season_options(bundle: StrikersDataBundle) -> list[str]:
    seasons = [season for season in _season_options(bundle.matches) if season != PLANNING_SEASON_LABEL]
    return seasons or _season_options(bundle.matches)


def _latest_historical_season(bundle: StrikersDataBundle) -> str | None:
    if bundle.transactions.empty or "season_label" not in bundle.transactions:
        values = _historical_season_options(bundle)
        return values[-1] if values else None
    values = sorted(
        season
        for season in bundle.transactions["season_label"].dropna().astype(str).unique().tolist()
        if season != PLANNING_SEASON_LABEL
    )
    return values[-1] if values else None


def _transaction_sales_curve(transactions: pd.DataFrame, ticket_status: str) -> pd.DataFrame:
    if transactions.empty or "transaction_date" not in transactions:
        return pd.DataFrame()
    value_column = _ticket_value_column(ticket_status)
    frame = transactions.copy()
    frame["date"] = pd.to_datetime(frame["transaction_date"], errors="coerce").dt.normalize()
    frame[value_column] = pd.to_numeric(frame[value_column], errors="coerce").fillna(0)
    frame = frame.dropna(subset=["date"])
    if frame.empty:
        return pd.DataFrame()
    daily = frame.groupby("date", as_index=False)[value_column].sum().sort_values("date")
    daily = daily.rename(columns={value_column: "actual_daily"})
    daily["actual_cumulative"] = daily["actual_daily"].cumsum()
    return daily


def _ticket_value_column(ticket_status: str) -> str:
    if ticket_status == "Paid tickets only":
        return "paid_tickets_sold"
    if ticket_status == "Comps only":
        return "comp_tickets_sold"
    return "tickets_sold"


def _planning_as_at_date(bundle: StrikersDataBundle) -> pd.Timestamp:
    if not bundle.transactions.empty and "season_label" in bundle.transactions:
        planning = bundle.transactions[bundle.transactions["season_label"].astype(str).eq(PLANNING_SEASON_LABEL)].copy()
        if not planning.empty and "transaction_date" in planning:
            dates = pd.to_datetime(planning["transaction_date"], errors="coerce").dropna()
            if not dates.empty:
                return dates.max().normalize()
    if bundle.metrics.get("dummy_26_27_sales_active"):
        return DUMMY_2627_END
    return _reference_date(bundle)


def _fixture_target_values(output: dict) -> dict[str, float]:
    fixture_id = str(output["match_id"])
    forecast = float(output["forecast"].expected_final_sales)
    stored = st.session_state.setdefault("fixture_targets", _load_fixture_targets()).get(fixture_id, {})
    return {
        "base_target": float(stored.get("base_target", forecast)),
        "stretch_target": float(stored.get("stretch_target", forecast * 1.10)),
    }


def _store_fixture_target(fixture_id: str, base_target: float, stretch_target: float) -> None:
    state = st.session_state.setdefault("fixture_targets", {})
    state[str(fixture_id)] = {"base_target": float(base_target), "stretch_target": float(stretch_target)}
    st.session_state["fixture_targets"] = state
    _save_fixture_targets(state)


def _load_fixture_targets(path: Path = FIXTURE_TARGETS_PATH) -> dict[str, dict[str, float]]:
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    required = {"fixture_id", "base_target", "stretch_target"}
    if frame.empty or not required.issubset(frame.columns):
        return {}
    return {
        str(row.fixture_id): {"base_target": float(row.base_target), "stretch_target": float(row.stretch_target)}
        for row in frame.itertuples(index=False)
        if pd.notna(row.fixture_id)
    }


def _save_fixture_targets(targets: dict[str, dict[str, float]], path: Path = FIXTURE_TARGETS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    saved_at = pd.Timestamp.now().isoformat()
    rows = [
        {
            "fixture_id": fixture_id,
            "base_target": values.get("base_target", 0),
            "stretch_target": values.get("stretch_target", 0),
            "saved_at": values.get("saved_at", saved_at),
        }
        for fixture_id, values in sorted(targets.items())
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def _fixture_targets_last_saved(path: Path = FIXTURE_TARGETS_PATH) -> str | None:
    if "fixture_targets_saved_at" in st.session_state:
        return str(st.session_state["fixture_targets_saved_at"])
    if not path.exists():
        return None
    modified = pd.Timestamp(path.stat().st_mtime, unit="s")
    return modified.strftime("%d %b %Y %H:%M")


def _fixture_target_draft_table(table: pd.DataFrame, target_state: dict[str, dict[str, float]]) -> pd.DataFrame:
    rows = []
    for _, row in table.iterrows():
        fixture_id = str(row["fixture_id"])
        saved = target_state.get(fixture_id, {})
        base = saved.get("base_target", row["forecasted_sales"])
        stretch = saved.get("stretch_target", row["forecast_10pct_uplift"])
        rows.append(
            {
                "fixture_id": fixture_id,
                "fixture": row["fixture"],
                "competition": row["competition"],
                "date": row["date"],
                "opponent": row["opponent"],
                "status": row["status"],
                "forecasted_sales": row["forecasted_sales"],
                "forecast_10pct_uplift": row["forecast_10pct_uplift"],
                "saved_base_target": saved.get("base_target", np.nan),
                "saved_stretch_target": saved.get("stretch_target", np.nan),
                "base_target": f"{float(base):.0f}",
                "stretch_target": f"{float(stretch):.0f}",
                "target_state": "Saved targets" if saved else "Missing target",
            }
        )
    return pd.DataFrame(rows)


def validate_fixture_target_frame(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for idx, row in frame.iterrows():
        base = _parse_target_input(row.get("base_target"))
        stretch = _parse_target_input(row.get("stretch_target"))
        errors = validate_fixture_targets(base, stretch)
        if errors:
            rows.append(
                {
                    "fixture": row.get("fixture", row.get("fixture_id", idx)),
                    "base_target": row.get("base_target"),
                    "stretch_target": row.get("stretch_target"),
                    "issue": "; ".join(errors),
                }
            )
    return pd.DataFrame(rows)


def _targets_from_editor_frame(frame: pd.DataFrame) -> dict[str, dict[str, float]]:
    saved_at = pd.Timestamp.now().isoformat()
    targets: dict[str, dict[str, float]] = {}
    for _, row in frame.iterrows():
        fixture_id = str(row["fixture_id"])
        targets[fixture_id] = {
            "base_target": float(_parse_target_input(row.get("base_target")) or 0),
            "stretch_target": float(_parse_target_input(row.get("stretch_target")) or 0),
            "saved_at": saved_at,
        }
    return targets


def _parse_target_input(value: object) -> float | None:
    try:
        text = str(value).replace(",", "").replace("$", "").strip()
        if not text:
            return None
        return float(text)
    except (TypeError, ValueError):
        return None


def validate_fixture_targets(base_target: float | None, stretch_target: float | None) -> list[str]:
    errors: list[str] = []
    if base_target is None:
        errors.append("Base target must be numeric.")
    elif base_target < 0:
        errors.append("Base target must be non-negative.")
    if stretch_target is None:
        errors.append("Stretch target must be numeric.")
    elif stretch_target < 0:
        errors.append("Stretch target must be non-negative.")
    if base_target is not None and stretch_target is not None and stretch_target < base_target:
        errors.append("Stretch target should be greater than or equal to the base target.")
    return errors


def _fixture_target_status(
    fixture_id: str,
    base_target: float | None,
    stretch_target: float | None,
    target_state: dict[str, dict[str, float]],
) -> str:
    saved = target_state.get(str(fixture_id))
    if saved is None:
        return "Missing"
    if base_target is None or stretch_target is None:
        return "Draft / unsaved"
    if abs(float(saved.get("base_target", 0)) - float(base_target)) > 0.01:
        return "Draft / unsaved"
    if abs(float(saved.get("stretch_target", 0)) - float(stretch_target)) > 0.01:
        return "Draft / unsaved"
    return "Saved"


def _target_chart_for_selected_mode(match_chart: pd.DataFrame, selected_target: float) -> pd.DataFrame:
    chart = match_chart.copy()
    if chart.empty:
        return chart
    if "manual_target_cumulative" in chart and chart["manual_target_cumulative"].max() > 0:
        original_total = float(chart["manual_target_cumulative"].iloc[-1])
        ratio = float(selected_target) / original_total if original_total else 1.0
        chart["manual_target_cumulative"] = chart["manual_target_cumulative"] * ratio
    elif "target_cumulative" in chart and chart["target_cumulative"].max() > 0:
        original_total = float(chart["target_cumulative"].iloc[-1])
        ratio = float(selected_target) / original_total if original_total else 1.0
        chart["manual_target_cumulative"] = chart["target_cumulative"] * ratio
    return chart


def _target_breakdown_outputs(
    bundle: StrikersDataBundle,
    pace_engine: HistoricalPaceEngine,
    forecaster: TicketSalesForecaster,
    controls: dict[str, object],
) -> dict[str, object]:
    scoped_matches = _scoped_matches(bundle.matches, PLANNING_SEASON_LABEL, controls["competition"], controls["fixture_ids"], controls["include_assumed"])
    assumptions = TargetAssumptions()
    outputs = _build_forecast_outputs(
        scoped_matches,
        bundle.matches,
        bundle.daily_sales,
        pace_engine,
        forecaster,
        assumptions,
        0.10,
        controls["history_filters"],
    )
    if not outputs:
        return {"outputs": [], "selected_output": None, "forecast_total": 0.0, "max_offset": 30}
    if controls["fixture_ids"] and len(outputs) == 1:
        selected_output = outputs[0]
        curve = selected_output["match_chart"]
        forecast_total = float(selected_output["forecast"].expected_final_sales)
        event_date = pd.to_datetime(selected_output["match"].get("event_date"), errors="coerce")
    else:
        curve = aggregate_selected_outputs(outputs, len(outputs))
        forecast_total = _last_value(curve, "forecast_expected_cumulative")
        event_dates = pd.to_datetime(scoped_matches["event_date"], errors="coerce").dropna()
        event_date = event_dates.max() if not event_dates.empty else DUMMY_2627_START + pd.Timedelta(days=170)
        selected_output = {
            "match_id": "season_total",
            "match": pd.Series({"event_date": event_date, "competition": controls["competition"]}),
            "match_chart": curve,
            "forecast": None,
        }
    max_offset = max(int((pd.to_datetime(event_date).normalize() - DUMMY_2627_START).days), 30) if pd.notna(event_date) else 170
    return {"outputs": outputs, "selected_output": selected_output, "forecast_total": forecast_total, "max_offset": max_offset}


def _target_totals_for_outputs(outputs: list[dict], target_state: dict[str, dict[str, float]]) -> dict[str, object]:
    base_total = 0.0
    stretch_total = 0.0
    missing: list[str] = []
    for output in outputs:
        fixture_id = str(output["match_id"])
        saved = target_state.get(fixture_id)
        if saved:
            base_total += float(saved.get("base_target", 0) or 0)
            stretch_total += float(saved.get("stretch_target", 0) or 0)
        else:
            missing.append(match_label(output["match"]))
    return {
        "base_target": base_total if not missing else np.nan,
        "stretch_target": stretch_total if not missing else np.nan,
        "missing_fixture_labels": missing,
    }


def _target_display(value: object) -> str:
    return "Missing" if pd.isna(value) else format_number(float(value))


def _historical_fixture_options(bundle: StrikersDataBundle, competition: str, selected_match: pd.Series) -> list[dict[str, str]]:
    historical = bundle.fixtures[bundle.fixtures["season_label"].astype(str).ne(PLANNING_SEASON_LABEL)].copy()
    if competition != "All":
        historical = historical[historical["competition"].astype(str).eq(str(competition))]
    if historical.empty:
        return []
    selected_date = pd.to_datetime(selected_match.get("match_date"), errors="coerce")
    historical["match_date"] = pd.to_datetime(historical["match_date"], errors="coerce")
    if pd.notna(selected_date):
        selected_month_day = selected_date.dayofyear
        historical["_similarity"] = (historical["match_date"].dt.dayofyear - selected_month_day).abs()
    else:
        historical["_similarity"] = 999
    historical = historical.sort_values(["_similarity", "season_label", "match_date"]).head(25)
    return [
        {
            "fixture_id": str(row.fixture_id),
            "label": f"{row.competition} v {row.opponent} | {pd.to_datetime(row.match_date).strftime('%d %b %Y')} | {row.season_label}",
        }
        for row in historical.itertuples(index=False)
    ]


def _last_season_equivalent_fixture_id(
    bundle: StrikersDataBundle,
    competition: str,
    selected_match: pd.Series,
    latest_season: str,
) -> str:
    historical = bundle.fixtures[bundle.fixtures["season_label"].astype(str).eq(str(latest_season))].copy()
    if competition != "All":
        historical = historical[historical["competition"].astype(str).eq(str(competition))]
    if historical.empty:
        return ""

    selected_date = pd.to_datetime(selected_match.get("event_date", selected_match.get("match_date")), errors="coerce")
    selected_opponent = str(selected_match.get("opponent", "") or "")
    historical["match_date"] = pd.to_datetime(historical["match_date"], errors="coerce")
    historical["_opponent_match"] = historical["opponent"].fillna("").astype(str).eq(selected_opponent)
    if pd.notna(selected_date):
        historical["_similarity"] = (historical["match_date"].dt.dayofyear - selected_date.dayofyear).abs()
    else:
        historical["_similarity"] = 999
    historical = historical.sort_values(["_opponent_match", "_similarity", "match_date"], ascending=[False, True, True])
    return str(historical.iloc[0]["fixture_id"]) if not historical.empty else ""


def _target_breakdown_last_season_curve(
    bundle: StrikersDataBundle,
    controls: dict[str, object],
    selected_output: dict,
    max_offset: int,
) -> pd.DataFrame:
    latest_season = _latest_historical_season(bundle)
    if latest_season is None:
        return pd.DataFrame()

    if str(selected_output.get("match_id")) == "season_total":
        latest_history = bundle.transactions[bundle.transactions["season_label"].astype(str).eq(str(latest_season))].copy()
        historical = _projection_transactions(
            latest_history,
            controls,
            latest_season,
            fixture_id=None,
            include_products=True,
            include_demographics=True,
            include_audience=True,
            as_at_date=None,
        )
        curve = _aligned_historical_curve(
            historical,
            str(controls.get("ticket_status", "Paid + comps")),
            max_offset,
            blend=False,
        )
        return curve.rename(
            columns={
                "historical_daily": "last_season_daily",
                "historical_cumulative": "last_season_cumulative",
            }
        )

    fixture_id = _last_season_equivalent_fixture_id(
        bundle,
        str(controls.get("competition", "All")),
        selected_output.get("match", pd.Series(dtype=object)),
        latest_season,
    )
    if not fixture_id:
        return pd.DataFrame()
    curve = _historical_comparison_curve(bundle.transactions, controls, fixture_id, max_offset)
    if curve.empty:
        return pd.DataFrame()
    return curve.rename(
        columns={
            "historic_comparison_daily": "last_season_daily",
            "historic_comparison_cumulative": "last_season_cumulative",
        }
    )


def _target_breakdown_chart_data(
    bundle: StrikersDataBundle,
    controls: dict[str, object],
    selected_output: dict,
    projection: dict[str, object],
    target_summary: dict[str, object],
    max_offset: int,
) -> pd.DataFrame:
    curve = projection["curve"].copy()
    if curve.empty:
        return curve
    frame = curve[["date", "forecast_expected_daily", "forecast_expected_cumulative"]].copy()
    forecast_total = float(projection.get("forecast_final_tickets", 0) or _last_value(frame, "forecast_expected_cumulative"))
    frame["forecast_plus_10_daily"] = frame["forecast_expected_daily"] * 1.10
    frame["forecast_plus_10_cumulative"] = frame["forecast_plus_10_daily"].cumsum()
    for label, total_col, daily_col, cumulative_col in [
        ("Base", "base_target", "base_daily", "base_cumulative"),
        ("Stretch", "stretch_target", "stretch_daily", "stretch_cumulative"),
    ]:
        total = target_summary.get(total_col)
        if pd.notna(total) and forecast_total > 0:
            segment_total = float(total) * float(projection.get("contribution_pct", 100.0)) / 100
            ratio = segment_total / forecast_total if forecast_total else 0
            frame[daily_col] = frame["forecast_expected_daily"] * ratio
            frame[cumulative_col] = frame[daily_col].cumsum()
    if controls.get("historical_fixture_id"):
        historic = _historical_comparison_curve(bundle.transactions, controls, str(controls["historical_fixture_id"]), max_offset)
        if not historic.empty:
            frame = frame.merge(historic, on="date", how="left")
            frame["historic_comparison_daily"] = frame["historic_comparison_daily"].fillna(0)
            frame["historic_comparison_cumulative"] = frame["historic_comparison_cumulative"].ffill().fillna(0)
    elif str(controls.get("historical_choice", "None")) == "Last season equivalent":
        last_season = _target_breakdown_last_season_curve(bundle, controls, selected_output, max_offset)
        if not last_season.empty:
            frame = frame.merge(last_season, on="date", how="left")
            frame["last_season_daily"] = frame["last_season_daily"].fillna(0)
            frame["last_season_cumulative"] = frame["last_season_cumulative"].ffill().fillna(0)
    return frame


def _historical_comparison_curve(
    transactions: pd.DataFrame,
    controls: dict[str, object],
    fixture_id: str,
    max_offset: int,
) -> pd.DataFrame:
    historical = _projection_transactions(
        transactions,
        controls,
        None,
        fixture_id=fixture_id,
        include_products=True,
        include_demographics=True,
        include_audience=True,
        as_at_date=None,
    )
    if historical.empty:
        return pd.DataFrame()
    value_column = _ticket_value_column(str(controls.get("ticket_status", "Paid + comps")))
    historical = historical.copy()
    historical["transaction_date"] = pd.to_datetime(historical["transaction_date"], errors="coerce")
    historical = historical.dropna(subset=["transaction_date"])
    historical["season_start"] = historical["season_label"].map(_season_start_year)
    historical = historical.dropna(subset=["season_start"])
    if value_column not in historical:
        historical[value_column] = 0.0
    historical[value_column] = pd.to_numeric(historical[value_column], errors="coerce").fillna(0)
    historical["offset"] = (
        historical["transaction_date"].dt.normalize()
        - pd.to_datetime(historical["season_start"].astype(int).astype(str) + "-08-01")
    ).dt.days.clip(lower=0, upper=max_offset)
    daily = historical.groupby("offset", as_index=False)[value_column].sum().rename(columns={value_column: "historic_comparison_daily"})
    offsets = pd.DataFrame({"offset": range(0, max_offset + 1)})
    daily = offsets.merge(daily, on="offset", how="left").fillna(0)
    daily["date"] = DUMMY_2627_START + pd.to_timedelta(daily["offset"], unit="D")
    daily["historic_comparison_cumulative"] = daily["historic_comparison_daily"].cumsum()
    return daily[["date", "historic_comparison_daily", "historic_comparison_cumulative"]]


def _visible_projection_columns(frame: pd.DataFrame, target_modes: list[str], suffix: str) -> list[str]:
    lookup = {
        "Forecast": f"forecast_expected_{suffix}",
        "Forecast +10%": f"forecast_plus_10_{suffix}",
        "Stretch": f"stretch_{suffix}",
        "Base": f"base_{suffix}",
    }
    return [column for mode, column in lookup.items() if mode in target_modes and column in frame]


def _projection_figure(frame: pd.DataFrame, columns: list[str], title: str, cumulative: bool) -> go.Figure:
    labels = {
        "forecast_expected_cumulative": "Forecast",
        "forecast_plus_10_cumulative": "Forecast +10%",
        "base_cumulative": "Base",
        "stretch_cumulative": "Stretch",
        "historic_comparison_cumulative": "Historic comparison",
        "last_season_cumulative": "Last season",
        "forecast_expected_daily": "Forecast",
        "forecast_plus_10_daily": "Forecast +10%",
        "base_daily": "Base",
        "stretch_daily": "Stretch",
        "historic_comparison_daily": "Historic comparison",
        "last_season_daily": "Last season",
        "actual_cumulative": "Actual 26/27",
        "actual_daily": "Actual 26/27",
        "historical_cumulative": "Historical comparison",
        "historical_daily": "Historical comparison",
        "target_cumulative": "Target",
        "target_daily": "Target",
    }
    colors = {
        "Forecast": COLORS["expected"],
        "Forecast +10%": COLORS["uplift"],
        "Base": COLORS["target"],
        "Stretch": COLORS["manual"],
        "Historic comparison": COLORS["comparison"],
        "Last season": COLORS["comparison"],
        "Actual 26/27": COLORS["actual"],
        "Historical comparison": COLORS["comparison"],
        "Target": COLORS["target"],
    }
    fig = go.Figure()
    for column in columns:
        if column not in frame:
            continue
        label = labels.get(column, column.replace("_", " ").title())
        mode = "lines+markers" if not cumulative and label == "Actual 26/27" else "lines"
        line_width = 3.4 if label in {"Actual 26/27", "Forecast"} else 2.5
        dash = "dash" if label in {"Historic comparison", "Historical comparison", "Last season"} else "solid"
        fig.add_trace(
            go.Scatter(
                x=frame["date"],
                y=frame[column],
                mode=mode,
                name=label,
                line={"width": line_width, "color": colors.get(label, COLORS["ink"]), "dash": dash, "shape": "linear"},
                marker={"size": 5.5, "color": colors.get(label, COLORS["ink"])} if mode == "lines+markers" else None,
                hovertemplate="%{x|%d %b %Y}<br>%{y:,.0f}<extra>" + label + "</extra>",
            )
        )
    fig.update_layout(
        title={"text": ""},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor=COLORS["panel"],
        hovermode="x unified",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
        margin={"l": 52, "r": 24, "t": 28, "b": 44},
        height=420,
        meta={"title": title},
        hoverlabel={"bgcolor": "rgba(15, 23, 40, 0.96)", "bordercolor": "rgba(255, 255, 255, 0.18)", "font": {"color": "#ffffff"}},
    )
    fig.update_xaxes(showgrid=False, title=None, linecolor=COLORS["border"], tickfont={"color": COLORS["ink"]})
    fig.update_yaxes(
        showgrid=True,
        gridcolor=COLORS["grid"],
        zeroline=False,
        title="Cumulative tickets" if cumulative else "Daily tickets",
        linecolor=COLORS["border"],
        tickfont={"color": COLORS["ink"]},
    )
    return fig


def _audience_actual_curve(actual_tx: pd.DataFrame, controls: dict[str, object], max_offset: int) -> pd.DataFrame:
    value_column = _ticket_value_column(str(controls.get("ticket_status", "Paid + comps")))
    if actual_tx.empty or "transaction_date" not in actual_tx or value_column not in actual_tx:
        return pd.DataFrame(columns=["date", "actual_daily", "actual_cumulative"])
    frame = actual_tx.copy()
    frame["date"] = pd.to_datetime(frame["transaction_date"], errors="coerce").dt.normalize()
    frame = frame.dropna(subset=["date"])
    if frame.empty:
        return pd.DataFrame(columns=["date", "actual_daily", "actual_cumulative"])
    frame[value_column] = pd.to_numeric(frame[value_column], errors="coerce").fillna(0)
    daily = frame.dropna(subset=["date"]).groupby("date", as_index=False)[value_column].sum().rename(columns={value_column: "actual_daily"})
    observed_dates = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
    output = pd.DataFrame({"date": observed_dates}).merge(daily, on="date", how="left")
    output["actual_daily"] = output["actual_daily"].fillna(0.0)
    output["actual_cumulative"] = output["actual_daily"].cumsum()
    return output


def _audience_historical_benchmark_curve(bundle: StrikersDataBundle, controls: dict[str, object], max_offset: int) -> pd.DataFrame:
    if controls.get("historical_fixture_id"):
        fixture_curve = _historical_comparison_curve(bundle.transactions, controls, str(controls["historical_fixture_id"]), max_offset)
        if fixture_curve.empty:
            return pd.DataFrame()
        return fixture_curve.rename(
            columns={
                "historic_comparison_daily": "historical_daily",
                "historic_comparison_cumulative": "historical_cumulative",
            }
        )

    history_seasons = _baseline_seasons_from_controls(controls.get("history_filters", {}))
    historical_base = bundle.transactions[bundle.transactions["season_label"].astype(str).isin(history_seasons)].copy()
    historical = _projection_transactions(
        historical_base,
        controls,
        None,
        fixture_id=None,
        include_products=True,
        include_demographics=True,
        include_audience=True,
        as_at_date=None,
    )
    return _aligned_historical_curve(
        historical,
        str(controls.get("ticket_status", "Paid + comps")),
        max_offset,
        blend=str(controls.get("historical_choice", "Blended historical average")) != "Last season equivalent",
    )


def _aligned_historical_curve(historical: pd.DataFrame, ticket_status: str, max_offset: int, blend: bool = True) -> pd.DataFrame:
    dates = pd.DataFrame({"offset": range(0, max_offset + 1)})
    if historical.empty or "transaction_date" not in historical or "season_label" not in historical:
        dates["date"] = DUMMY_2627_START + pd.to_timedelta(dates["offset"], unit="D")
        dates["historical_daily"] = 0.0
        dates["historical_cumulative"] = 0.0
        return dates[["date", "historical_daily", "historical_cumulative"]]
    value_column = _ticket_value_column(ticket_status)
    frame = historical.copy()
    frame["transaction_date"] = pd.to_datetime(frame["transaction_date"], errors="coerce")
    frame = frame.dropna(subset=["transaction_date"])
    frame["season_start"] = frame["season_label"].map(_season_start_year)
    frame = frame.dropna(subset=["season_start"])
    if value_column not in frame:
        frame[value_column] = 0.0
    frame[value_column] = pd.to_numeric(frame[value_column], errors="coerce").fillna(0)
    frame["offset"] = (
        frame["transaction_date"].dt.normalize()
        - pd.to_datetime(frame["season_start"].astype(int).astype(str) + "-08-01")
    ).dt.days.clip(lower=0, upper=max_offset)
    daily_by_season = frame.groupby(["season_label", "offset"], as_index=False)[value_column].sum().rename(columns={value_column: "historical_daily"})
    if daily_by_season.empty:
        return pd.DataFrame()
    if not blend:
        latest = sorted(daily_by_season["season_label"].dropna().astype(str).unique().tolist())[-1]
        daily = daily_by_season[daily_by_season["season_label"].astype(str).eq(latest)].groupby("offset", as_index=False)["historical_daily"].sum()
    else:
        daily = daily_by_season.groupby("offset", as_index=False)["historical_daily"].mean()
    output = dates.merge(daily, on="offset", how="left").fillna({"historical_daily": 0.0})
    output["date"] = DUMMY_2627_START + pd.to_timedelta(output["offset"], unit="D")
    output["historical_cumulative"] = output["historical_daily"].cumsum()
    return output[["date", "historical_daily", "historical_cumulative"]]


def _audience_target_curve(
    bundle: StrikersDataBundle,
    controls: dict[str, object],
    outputs_context: dict[str, object],
    max_offset: int,
) -> tuple[pd.DataFrame, str]:
    outputs = outputs_context.get("outputs", [])
    if not outputs:
        return pd.DataFrame(), "No target comparison is available because no forecast outputs exist for this scope."
    selected_output = outputs_context["selected_output"]
    target_state = st.session_state.setdefault("fixture_targets", _load_fixture_targets())
    target_summary = _target_totals_for_outputs(outputs, target_state)
    projection_target = float(target_summary["base_target"]) if pd.notna(target_summary["base_target"]) else float(outputs_context.get("forecast_total", 0.0))
    projection = _segment_level_projection(bundle, controls, selected_output, projection_target, include_actuals=False)
    chart_data = _target_breakdown_chart_data(bundle, controls, outputs_context["selected_output"], projection, target_summary, max_offset)
    if chart_data.empty:
        return pd.DataFrame(), "No target curve is available for this segment."
    target_mode = str(controls.get("target_mode", "Forecast"))
    daily_col = _visible_projection_columns(chart_data, [target_mode], suffix="daily")
    cumulative_col = _visible_projection_columns(chart_data, [target_mode], suffix="cumulative")
    if not daily_col or not cumulative_col:
        missing = ", ".join(target_summary.get("missing_fixture_labels", [])[:3])
        suffix = f" Missing examples: {missing}." if missing else ""
        return pd.DataFrame(), f"{target_mode} target is unavailable for this scope. Set targets in Fixture Forecasting if using Base or Stretch.{suffix}"
    return (
        chart_data[["date", daily_col[0], cumulative_col[0]]].rename(columns={daily_col[0]: "target_daily", cumulative_col[0]: "target_cumulative"}),
        "",
    )


def _audience_tracking_chart_data(
    actual_curve: pd.DataFrame,
    historical_curve: pd.DataFrame,
    target_curve: pd.DataFrame,
    max_offset: int,
) -> pd.DataFrame:
    if actual_curve is not None and not actual_curve.empty and "date" in actual_curve:
        frame = actual_curve[["date"]].drop_duplicates().sort_values("date").reset_index(drop=True)
    else:
        frame = pd.DataFrame({"date": DUMMY_2627_START + pd.to_timedelta(range(0, max_offset + 1), unit="D")})
    for candidate in [actual_curve, historical_curve, target_curve]:
        if candidate is not None and not candidate.empty and "date" in candidate:
            frame = frame.merge(candidate, on="date", how="left")
    for column in ["actual_daily", "historical_daily", "target_daily"]:
        if column in frame:
            frame[column] = frame[column].fillna(0.0)
    for column in ["actual_cumulative", "historical_cumulative", "target_cumulative"]:
        if column in frame:
            frame[column] = frame[column].ffill().fillna(0.0)
    return frame


def _segment_level_projection(
    bundle: StrikersDataBundle,
    controls: dict[str, object],
    selected_output: dict,
    selected_target: float,
    min_historical_tickets: int = 100,
    include_actuals: bool = True,
) -> dict[str, object]:
    selected_match_id = str(selected_output["match_id"])
    as_at = pd.to_datetime(controls["as_at_date"]).normalize()
    history_seasons = _baseline_seasons_from_controls(controls.get("history_filters", {}))
    historical_base = bundle.transactions[bundle.transactions["season_label"].astype(str).isin(history_seasons)].copy()
    current_tx = (
        _projection_transactions(
            bundle.transactions,
            controls,
            PLANNING_SEASON_LABEL,
            fixture_id=selected_match_id if selected_match_id != "season_total" else None,
            include_products=True,
            include_demographics=True,
            include_audience=True,
            as_at_date=as_at,
        )
        if include_actuals
        else pd.DataFrame()
    )
    fallback_levels = [
        ("selected filters", True, True, True, str(_selected_filter_label(controls))),
        ("product-level fallback", True, False, True, "demographic filters removed"),
        ("ticket product fallback", True, False, False, "demographic and audience filters removed"),
        ("competition-level fallback", False, False, False, "ticket product, audience, and demographic filters removed"),
        ("all-sales fallback", False, False, False, "all competitions used"),
    ]

    chosen_history = pd.DataFrame()
    chosen_level = fallback_levels[0]
    value_column = _ticket_value_column(str(controls.get("ticket_status", "Paid + comps")))
    for level in fallback_levels:
        label, include_products, include_demographics, include_audience, _ = level
        competition_override = "All" if label == "all-sales fallback" else None
        candidate = _projection_transactions(
            historical_base,
            controls,
            None,
            fixture_id=None,
            include_products=include_products,
            include_demographics=include_demographics,
            include_audience=include_audience,
            as_at_date=None,
            competition_override=competition_override,
        )
        sample = float(candidate.get(value_column, pd.Series(dtype=float)).sum()) if not candidate.empty else 0.0
        if sample >= min_historical_tickets or label == "all-sales fallback":
            chosen_history = candidate
            chosen_level = level
            break

    curve, forecast_final, expected_by_now, contribution_pct, revenue_per_ticket = _build_segment_projection_curve(
        chosen_history,
        historical_base,
        current_tx,
        selected_output,
        selected_target,
        as_at,
        value_column,
    )
    actual_by_now = float(current_tx[value_column].sum()) if not current_tx.empty and value_column in current_tx else 0.0
    sample_size = float(chosen_history[value_column].sum()) if not chosen_history.empty and value_column in chosen_history else 0.0
    fallback_used = chosen_level[0] != "selected filters" or sample_size < min_historical_tickets
    fallback_message = (
        f"Low historical sample for the exact selected group. Using {chosen_level[0]} ({chosen_level[4]}). "
        f"Historical sample is {sample_size:,.0f} selected tickets; minimum is {min_historical_tickets:,.0f}."
    )
    return {
        "curve": curve,
        "forecast_final_tickets": float(forecast_final),
        "forecast_final_revenue": float(forecast_final * revenue_per_ticket),
        "expected_by_now": float(expected_by_now),
        "actual_by_now": actual_by_now,
        "contribution_pct": float(contribution_pct),
        "fallback_used": bool(fallback_used),
        "fallback_message": fallback_message,
        "sample_size": sample_size,
    }


def _projection_transactions(
    transactions: pd.DataFrame,
    controls: dict[str, object],
    season_label: str | None,
    fixture_id: str | None,
    include_products: bool,
    include_demographics: bool,
    include_audience: bool,
    as_at_date: pd.Timestamp | None,
    competition_override: str | None = None,
) -> pd.DataFrame:
    if transactions.empty:
        return transactions.copy()
    fixture_ids = [fixture_id] if fixture_id else []
    frame = apply_transaction_filters(
        transactions,
        season_label=season_label,
        competition=competition_override if competition_override is not None else str(controls.get("competition", "All")),
        fixture_ids=fixture_ids,
        as_at_date=as_at_date,
        audience_segment=str(controls.get("audience_segment", "All")) if include_audience else "All",
        ticket_type=str(controls.get("ticket_type", "All")) if include_products else "All",
        ticket_class=str(controls.get("ticket_class", "All")) if include_products else "All",
        sales_window=str(controls.get("sales_window", "All")),
        ticket_status=str(controls.get("ticket_status", "Paid + comps")),
    )
    if include_demographics:
        frame = _apply_demographic_filters(frame, controls.get("age_band", []), controls.get("gender", []), controls.get("postcode", []))
    return frame


def _smoothed_projection_daily_share(offsets: pd.Series, historical_paid: pd.Series, horizon_end: pd.Timestamp) -> pd.Series:
    paid = pd.to_numeric(historical_paid, errors="coerce").fillna(0).clip(lower=0)
    if paid.sum() <= 0:
        return pd.Series(np.ones(len(paid)) / max(len(paid), 1), index=paid.index)

    smoothed = paid.rolling(window=7, center=True, min_periods=1).mean()
    dates = DUMMY_2627_START + pd.to_timedelta(pd.to_numeric(offsets, errors="coerce").fillna(0), unit="D")
    days_to_event = (pd.to_datetime(horizon_end).normalize() - dates).dt.days.clip(lower=0)

    campaign_tail = np.where(days_to_event.between(7, 21), (21 - days_to_event) / 14, 0.0)
    final_week = np.where(days_to_event.between(0, 7), (8 - days_to_event) / 8, 0.0)
    matchday = np.where(days_to_event.eq(0), 0.35, np.where(days_to_event.eq(1), 0.18, 0.0))

    weights = smoothed * (1 + 0.10 * campaign_tail + 0.18 * final_week + matchday)
    weight_total = float(weights.sum())
    if weight_total <= 0:
        return pd.Series(np.ones(len(weights)) / max(len(weights), 1), index=weights.index)
    return pd.Series(weights / weight_total, index=weights.index)


def _build_segment_projection_curve(
    historical_tx: pd.DataFrame,
    historical_total_tx: pd.DataFrame,
    current_tx: pd.DataFrame,
    selected_output: dict,
    selected_target: float,
    as_at: pd.Timestamp,
    value_column: str = "paid_tickets_sold",
) -> tuple[pd.DataFrame, float, float, float, float]:
    match = selected_output["match"]
    event_date = pd.to_datetime(match.get("event_date"), errors="coerce")
    horizon_end = event_date if pd.notna(event_date) else DUMMY_2627_START + pd.Timedelta(days=170)
    max_offset = max(int((horizon_end.normalize() - DUMMY_2627_START).days), 30)
    forecast_final = 0.0
    revenue_per_ticket = 0.0
    contribution_pct = 100.0

    if historical_tx.empty:
        base_chart = selected_output["match_chart"].copy()
        if base_chart.empty:
            return pd.DataFrame(), 0.0, 0.0, 0.0, 0.0
        curve = base_chart[["date", "forecast_expected_daily", "forecast_expected_cumulative"]].copy()
        forecast_final = _last_value(curve, "forecast_expected_cumulative")
        curve["target_daily"] = curve["forecast_expected_daily"]
        curve["target_cumulative"] = curve["target_daily"].cumsum()
        curve = _merge_current_actuals(curve, current_tx, value_column)
        curve["required_daily"] = _required_daily_for_curve(curve, float(selected_target), as_at)
        curve["forecast_revenue"] = curve["forecast_expected_daily"] * revenue_per_ticket
        curve["sales_window"] = curve["date"].map(_projection_sales_window)
        return curve, forecast_final, _expected_at_date(curve, as_at), contribution_pct, revenue_per_ticket

    historical = historical_tx.copy()
    historical["transaction_date"] = pd.to_datetime(historical["transaction_date"], errors="coerce")
    historical = historical.dropna(subset=["transaction_date"])
    historical["season_start"] = historical["season_label"].map(_season_start_year)
    historical = historical.dropna(subset=["season_start"])
    historical["offset"] = (
        historical["transaction_date"].dt.normalize()
        - pd.to_datetime(historical["season_start"].astype(int).astype(str) + "-08-01")
    ).dt.days.clip(lower=0, upper=max_offset)
    if value_column not in historical:
        historical[value_column] = 0.0
    historical[value_column] = pd.to_numeric(historical[value_column], errors="coerce").fillna(0)
    historical["gross_revenue"] = pd.to_numeric(historical["gross_revenue"], errors="coerce").fillna(0)
    final_by_season = historical.groupby("season_label")[value_column].sum()
    forecast_final = float(final_by_season.mean()) if not final_by_season.empty else 0.0
    total_tickets = float(historical[value_column].sum())
    revenue_per_ticket = float(historical["gross_revenue"].sum() / total_tickets) if total_tickets else 0.0
    daily = historical.groupby("offset", as_index=False).agg(
        historical_paid=(value_column, "sum"),
        historical_revenue=("gross_revenue", "sum"),
    )
    offsets = pd.DataFrame({"offset": range(0, max_offset + 1)})
    daily = offsets.merge(daily, on="offset", how="left").fillna(0)
    daily["daily_share"] = _smoothed_projection_daily_share(daily["offset"], daily["historical_paid"], horizon_end)

    total_comp_history = _total_historical_paid_for_contribution(historical_total_tx, selected_output, value_column)
    contribution_pct = float(forecast_final / total_comp_history * 100) if total_comp_history else 100.0
    target_total = float(selected_target * contribution_pct / 100)
    curve = pd.DataFrame(
        {
            "date": DUMMY_2627_START + pd.to_timedelta(daily["offset"], unit="D"),
            "forecast_expected_daily": daily["daily_share"] * forecast_final,
            "target_daily": daily["daily_share"] * target_total,
        }
    )
    curve["forecast_expected_cumulative"] = curve["forecast_expected_daily"].cumsum()
    curve["target_cumulative"] = curve["target_daily"].cumsum()
    curve["forecast_revenue"] = curve["forecast_expected_daily"] * revenue_per_ticket
    curve = _merge_current_actuals(curve, current_tx, value_column)
    curve["required_daily"] = _required_daily_for_curve(curve, target_total, as_at)
    curve["sales_window"] = curve["date"].map(_projection_sales_window)
    return curve, forecast_final, _expected_at_date(curve, as_at), contribution_pct, revenue_per_ticket


def _merge_current_actuals(curve: pd.DataFrame, current_tx: pd.DataFrame, value_column: str = "paid_tickets_sold") -> pd.DataFrame:
    output = curve.copy()
    if current_tx.empty:
        output["actual_daily"] = 0.0
        output["actual_cumulative"] = 0.0
        return output
    actual = current_tx.copy()
    actual["date"] = pd.to_datetime(actual["transaction_date"], errors="coerce").dt.normalize()
    if value_column not in actual:
        actual[value_column] = 0.0
    actual[value_column] = pd.to_numeric(actual[value_column], errors="coerce").fillna(0)
    actual = actual.dropna(subset=["date"]).groupby("date", as_index=False)[value_column].sum().rename(columns={value_column: "actual_daily"})
    output = output.merge(actual, on="date", how="left")
    output["actual_daily"] = output["actual_daily"].fillna(0.0)
    output["actual_cumulative"] = output["actual_daily"].cumsum()
    return output


def _required_daily_for_curve(curve: pd.DataFrame, target_total: float, as_at: pd.Timestamp) -> pd.Series:
    if curve.empty:
        return pd.Series(dtype=float)
    actual_to_date = float(curve.loc[pd.to_datetime(curve["date"]) <= as_at, "actual_daily"].sum()) if "actual_daily" in curve else 0.0
    remaining_mask = pd.to_datetime(curve["date"]) > as_at
    remaining_days = int(remaining_mask.sum())
    required = max(float(target_total) - actual_to_date, 0) / remaining_days if remaining_days else 0.0
    return pd.Series(np.where(remaining_mask, required, 0.0), index=curve.index)


def _expected_at_date(curve: pd.DataFrame, as_at: pd.Timestamp) -> float:
    if curve.empty or "forecast_expected_cumulative" not in curve:
        return 0.0
    eligible = curve[pd.to_datetime(curve["date"]) <= as_at]
    return float(eligible["forecast_expected_cumulative"].iloc[-1]) if not eligible.empty else 0.0


def _projection_sales_window(date: object) -> str:
    parsed = pd.to_datetime(date, errors="coerce")
    if pd.isna(parsed):
        return "unknown"
    offset = int((parsed.normalize() - DUMMY_2627_START).days)
    if offset <= 31:
        return "early-sales"
    if offset <= 75:
        return "general-sale"
    if offset <= 120:
        return "campaign-window"
    return "match-week"


def _season_start_year(value: object) -> int | float:
    text = str(value or "")
    try:
        return int(text.split("/")[0])
    except (TypeError, ValueError):
        return np.nan


def _total_historical_paid_for_contribution(historical_tx: pd.DataFrame, selected_output: dict, value_column: str = "paid_tickets_sold") -> float:
    if historical_tx.empty:
        return 0.0
    competition = str(selected_output["match"].get("competition", "All"))
    frame = historical_tx.copy()
    if competition != "All" and "competition" in frame:
        frame = frame[frame["competition"].astype(str).eq(competition)]
    if value_column not in frame:
        return 0.0
    frame[value_column] = pd.to_numeric(frame[value_column], errors="coerce").fillna(0)
    by_season = frame.groupby("season_label")[value_column].sum()
    return float(by_season.mean()) if not by_season.empty else 0.0


def _selected_filter_label(controls: dict[str, object]) -> str:
    parts = [
        controls.get("ticket_type", "All"),
        controls.get("ticket_class", "All"),
        ", ".join(controls.get("age_band", []) or []),
        ", ".join(controls.get("gender", []) or []),
        ", ".join(controls.get("postcode", []) or []),
        controls.get("audience_segment", "All"),
    ]
    selected = [str(part) for part in parts if str(part) and str(part) != "All"]
    return " + ".join(selected) if selected else "all selected filters"


def _early_onsale_target_segments(
    transactions: pd.DataFrame,
    customers: pd.DataFrame,
    season_label: str,
    competition: str,
    segment_field: str,
    min_size: int,
) -> pd.DataFrame:
    if transactions.empty or segment_field not in transactions:
        return pd.DataFrame()
    frame = transactions[transactions["season_label"].astype(str).eq(str(season_label))].copy()
    if competition != "All":
        frame = frame[frame["competition"].astype(str).eq(str(competition))]
    if frame.empty:
        return pd.DataFrame()
    frame["transaction_date"] = pd.to_datetime(frame["transaction_date"], errors="coerce")
    august_start = pd.Timestamp(year=int(frame["transaction_date"].dt.year.dropna().min() or pd.Timestamp.today().year), month=8, day=1)
    august_mask = frame["transaction_date"].dt.month.eq(8)
    early = frame[august_mask & frame["transaction_date"].between(august_start, august_start + pd.Timedelta(days=30))].copy()
    if early.empty:
        first_date = frame["transaction_date"].dropna().min()
        early = frame[frame["transaction_date"].between(first_date, first_date + pd.Timedelta(days=30))].copy() if pd.notna(first_date) else pd.DataFrame()
    if early.empty:
        return pd.DataFrame()
    early[segment_field] = early[segment_field].fillna("Unknown").astype(str)
    grouped = early.groupby(segment_field).agg(
        early_paid_tickets=("paid_tickets_sold", "sum"),
        early_comps=("comp_tickets_sold", "sum"),
        early_revenue=("gross_revenue", "sum"),
        early_customers=("customer_id", "nunique"),
        orders=("order_id", "nunique"),
    ).reset_index().rename(columns={segment_field: "segment"})
    grouped = grouped[grouped["early_paid_tickets"].ge(min_size)].copy()
    if grouped.empty:
        return grouped
    grouped["segment_lens"] = segment_field
    grouped["early_atp"] = grouped["early_revenue"] / grouped["early_paid_tickets"].replace(0, pd.NA)
    grouped["early_basket_size"] = grouped["early_paid_tickets"] / grouped["orders"].replace(0, pd.NA)
    total_paid = max(float(grouped["early_paid_tickets"].sum()), 1)
    grouped["early_share_pct"] = grouped["early_paid_tickets"] / total_paid * 100
    grouped["recommendation"] = grouped.apply(_early_segment_recommendation, axis=1)
    return grouped[
        [
            "segment_lens",
            "segment",
            "early_paid_tickets",
            "early_customers",
            "early_revenue",
            "early_atp",
            "early_basket_size",
            "early_share_pct",
            "recommendation",
        ]
    ].sort_values("early_paid_tickets", ascending=False).reset_index(drop=True)


def _early_segment_recommendation(row: pd.Series) -> str:
    segment = str(row.get("segment", "")).lower()
    if "family" in segment or "boundary" in segment:
        return "Use family-value, fixture-date clarity, and early-access urgency in the opening sales window."
    if "gold" in segment or "platinum" in segment or "premium" in segment:
        return "Use premium availability, seat quality, and scarcity messaging rather than discount framing."
    if "general" in segment or "ga" in segment or "hill" in segment:
        return "Use simple group attendance, social proof, and easy entry messaging for early on-sale."
    if "female" in segment or "male" in segment:
        return "Use broad season-opening creative, then refine with product and fixture preference where available."
    return "Prioritise clear fixture choice, value reassurance, and early-window urgency."


def _season_options(matches: pd.DataFrame) -> list[str]:
    if matches.empty or "season_label" not in matches:
        return [PLANNING_SEASON_LABEL]
    values = sorted(matches["season_label"].dropna().astype(str).unique().tolist())
    return values or [PLANNING_SEASON_LABEL]


def _reference_date(bundle: StrikersDataBundle) -> pd.Timestamp:
    if not bundle.daily_sales.empty:
        return latest_data_date(bundle.daily_sales)
    if not bundle.transactions.empty and "transaction_date" in bundle.transactions:
        parsed = pd.to_datetime(bundle.transactions["transaction_date"], errors="coerce").dropna()
        if not parsed.empty:
            return parsed.max().normalize()
    return pd.Timestamp.today().normalize()


def _last_value(frame: pd.DataFrame, column: str) -> float:
    if frame.empty or column not in frame:
        return 0.0
    return float(frame[column].iloc[-1])


def _money(value: float) -> str:
    return f"${float(value):,.0f}"


def _init_session_state() -> None:
    st.session_state.setdefault("mapping_overrides", {})
    st.session_state.setdefault("reprocess_token", 0)
    st.session_state.setdefault("fixture_seed", 2026)
    st.session_state.setdefault("fixture_targets", _load_fixture_targets())
    st.session_state.setdefault("client_setup_complete", False)


if __name__ == "__main__":
    main()
