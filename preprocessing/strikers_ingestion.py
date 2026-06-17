"""Real Adelaide Strikers CSV ingestion and analytics model assembly."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from shutil import copyfileobj
from typing import Iterable
from zlib import crc32
import re
import zipfile

import numpy as np
import pandas as pd

from dashboard.config import APP_DATA_DIR, PLANNING_SEASON, PLANNING_SEASON_LABEL, PROJECT_ROOT
from forecasting.future_fixtures import FutureFixtureConfig, generate_assumed_future_fixtures
from utils.csv_utils import (
    CsvLoadResult,
    anonymise_identifier,
    clean_text,
    extract_date_from_text,
    normalise_token,
    parse_bool_series,
    parse_date_series,
    parse_number_series,
    read_csv_robust,
)
from utils.data_loader import DEFAULT_DATA_DIR, load_dataset


RAW_DATA_DIR = APP_DATA_DIR / "raw"
PROCESSED_DATA_DIR = APP_DATA_DIR / "processed"
DEMO_DATA_DIR = PROJECT_ROOT / "data" / "demo"
BUNDLED_RAW_ARCHIVE = PROJECT_ROOT / "data" / "bootstrap" / "strikers-raw-data.zip"


@dataclass(frozen=True)
class ExpectedFile:
    key: str
    label: str
    canonical_name: str
    kind: str
    competition: str | None
    aliases: tuple[str, ...]
    required: bool = True


@dataclass
class LoadedSource:
    spec: ExpectedFile
    path: Path | None
    csv: CsvLoadResult | None = None
    missing: bool = False
    mapping: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)


@dataclass
class StrikersDataBundle:
    data_mode: str
    matches: pd.DataFrame
    daily_sales: pd.DataFrame
    transactions: pd.DataFrame
    fixtures: pd.DataFrame
    customers: pd.DataFrame
    future_fixtures: pd.DataFrame
    file_status: pd.DataFrame
    column_mappings: pd.DataFrame
    validation_warnings: pd.DataFrame
    metrics: dict[str, object]
    loaded_sources: dict[str, LoadedSource] = field(default_factory=dict)


EXPECTED_FILES: tuple[ExpectedFile, ...] = (
    ExpectedFile(
        "bbl_tickets",
        "BBL ticket transactions",
        "Strikers All Tickets.csv",
        "tickets",
        "BBL",
        (
            "Strikers All Tickets",
            "Strikers All Tickets.csv",
            "Historical Ticket Sales",
            "Historical Ticket Sales.csv",
            "Client Historical Sales",
            "Client Historical Sales.csv",
        ),
    ),
    ExpectedFile(
        "wbbl_tickets",
        "WBBL ticket transactions",
        "Strikers All Tickets WBBL.csv",
        "tickets",
        "WBBL",
        (
            "Strikers All Tickets WBBL",
            "Strikers All Tickets WBBL.csv",
            "Historical Ticket Sales WBBL",
            "Historical Ticket Sales WBBL.csv",
        ),
    ),
    ExpectedFile(
        "bbl_fixtures",
        "BBL fixtures",
        "Strikers BBL Fixtures.csv",
        "fixtures",
        "BBL",
        (
            "Strikers BBL Fixtures",
            "Strikers BBL Fixtures.csv",
            "Historic Fixtures",
            "Historic Fixtures.csv",
            "Historical Fixtures",
            "Historical Fixtures.csv",
            "Client Historical Fixtures",
            "Client Historical Fixtures.csv",
        ),
    ),
    ExpectedFile(
        "wbbl_fixtures",
        "WBBL fixtures",
        "Strikers WBBL Fixtures.csv",
        "fixtures",
        "WBBL",
        (
            "Strikers WBBL Fixtures",
            "Strikers WBBL Fixtures.csv",
            "Historic Fixtures WBBL",
            "Historic Fixtures WBBL.csv",
            "Historical Fixtures WBBL",
            "Historical Fixtures WBBL.csv",
        ),
    ),
    ExpectedFile(
        "customers",
        "Customer data",
        "Strikers Customer Data.csv",
        "customers",
        None,
        (
            "Strikers Customer Data",
            "Strikers Customer Data.csv",
            "Customer Info",
            "Customer Info.csv",
            "Customer Data",
            "Customer Data.csv",
            "Client Customer Data",
            "Client Customer Data.csv",
        ),
    ),
    ExpectedFile(
        "future_fixtures",
        "Next season fixture assumptions",
        "Client Next Season Fixtures.csv",
        "future_fixtures",
        None,
        (
            "Client Next Season Fixtures",
            "Client Next Season Fixtures.csv",
            "Next Season Fixtures",
            "Next Season Fixtures.csv",
            "Future Fixtures",
            "Future Fixtures.csv",
            "Fixture Assumptions",
            "Fixture Assumptions.csv",
        ),
        False,
    ),
)


COLUMN_SYNONYMS: dict[str, tuple[str, ...]] = {
    "competition": ("competition", "comp", "league", "tournament", "event_type", "event category"),
    "transaction_id": ("transaction_id", "transaction", "txn_id", "sale_id", "ticket_id", "barcode"),
    "order_id": ("order_id", "order_number", "booking_id", "booking_number", "purchase_id", "invoice_id"),
    "customer_id": (
        "gigyauid",
        "gigya_uid",
        "gigya uid",
        "gigyaid",
        "gigya_id",
        "customer_gigya_uid",
        "customer gigya uid",
        "patron_gigyauid",
        "patron gigyauid",
        "gigy_uid",
        "gigy_id",
        "customer_id",
        "account_id",
        "patron_id",
        "client_id",
        "constituent_id",
        "member_id",
        "contact_id",
        "email_hash",
    ),
    "fixture_id": ("fixture_id", "event_id", "match_id", "manifest_id"),
    "fixture_name": ("fixture_name", "event_name", "event", "match_name", "performance_name", "product_name"),
    "event_name": ("event_name", "event", "fixture_name", "match_name", "performance"),
    "team": ("team", "home_team", "club"),
    "opponent": ("opponent", "away_team", "opposition", "visiting_team"),
    "venue": ("venue", "stadium", "facility", "location"),
    "match_date": ("match_date", "event_date", "fixture_date", "game_date", "date"),
    "match_time": ("match_time", "event_time", "time", "start_time"),
    "season": ("season", "season_label", "season_name", "year", "season_year"),
    "transaction_date": (
        "transaction_date",
        "order_date",
        "purchase_date",
        "paid_date",
        "created_date",
        "booking_date",
        "transaction_timestamp",
        "sale_date",
    ),
    "tickets_sold": ("tickets_sold", "quantity", "qty", "ticket_count", "seats", "seat_count", "number_of_tickets"),
    "gross_revenue": ("gross_revenue", "gross_amount", "total_price", "total", "item_total", "paid_amount", "amount"),
    "net_revenue": ("net_revenue", "net_amount", "net", "revenue", "sales_value"),
    "price_paid": ("price_paid", "ticket_price", "price", "unit_price", "amount_paid"),
    "ticket_type": ("ticket_type", "ticket_category", "sales_type", "market_type", "ticket_group"),
    "ticket_class": ("ticket_class", "price_category", "class", "area", "seat_category", "product"),
    "price_type": ("price_type", "price_code", "price_level", "concession", "ticket_description"),
    "section": ("section", "stand", "bay", "block"),
    "row": ("row", "seat_row"),
    "seat": ("seat", "seat_number"),
    "sales_channel": ("sales_channel", "channel", "seller", "source", "platform"),
    "promo_code": ("promo_code", "promotion", "discount_code", "offer_code"),
    "status": ("status", "order_status", "ticket_status", "transaction_status"),
    "postcode": ("postcode", "post_code", "zip", "purchaser_postcode"),
    "suburb": ("suburb", "city", "town"),
    "state": ("state", "region", "province"),
    "age": ("age", "customer_age", "patron_age"),
    "age_band": ("age_band", "age_range", "age_group"),
    "date_of_birth": ("date_of_birth", "birth_date", "dob", "date_of_birth__c"),
    "gender": ("gender", "sex"),
    "family_flag": ("family_flag", "family", "has_children", "child_flag"),
    "marketing_opt_in": ("marketing_opt_in", "marketing", "email_opt_in", "marketable", "opt_in"),
    "email_opt_in": ("email_opt_in", "email_marketing", "email_permission"),
    "sms_opt_in": ("sms_opt_in", "sms_marketing", "mobile_permission"),
    "capacity_total": ("capacity_total", "capacity", "venue_capacity", "attendance_capacity"),
    "notes": ("notes", "comment", "comments"),
}


TICKET_POSITIONAL_FIELDS = {
    "transaction_id": "column_001",
    "customer_id": "column_002",
    "fixture_name": "column_004",
    "event_name": "column_004",
    "tickets_sold": "column_022",
    "team": "column_006",
    "opponent": "column_007",
    "gender": "column_008",
    "season": "column_010",
    "season_label": "column_011",
    "venue": "column_012",
    "state": "column_013",
    "fixture_label": "column_014",
    "transaction_date": "column_015",
    "ticket_type": "column_016",
    "ticket_class": "column_017",
    "price_type": "column_018",
    "price_paid": "column_021",
    "gross_revenue": "column_021",
    "net_revenue": "column_021",
    "section": "column_023",
    "order_id": "column_024",
    "sales_channel": "column_025",
}

FIXTURE_POSITIONAL_FIELDS = {
    "season": "column_001",
    "venue": "column_002",
    "team": "column_003",
    "opponent": "column_004",
    "match_date": "column_005",
}

CUSTOMER_POSITIONAL_FIELDS = {
    "customer_id": "column_001",
    "gigyauid": "column_001",
}


def load_demo_strikers_data() -> StrikersDataBundle:
    """Load synthetic demo data and wrap it in the same analytics contract as real data."""

    data_dir = DEMO_DATA_DIR if (DEMO_DATA_DIR / "matches.csv").exists() else DEFAULT_DATA_DIR
    matches, daily_sales = load_dataset(data_dir)
    fixtures = _fixtures_from_matches(matches, fixture_status="assumed")
    transactions = _demo_transactions_from_daily(matches, daily_sales)
    transactions = _annotate_customer_purchase_status(transactions)
    customers = _customers_from_transactions(pd.DataFrame(), transactions)
    future_fixtures = fixtures[fixtures["season_label"].eq(PLANNING_SEASON_LABEL)].copy()
    metrics = _metrics(transactions, fixtures, customers)
    warnings = pd.DataFrame(
        [
            {
                "severity": "info",
                "area": "demo",
                "message": "Synthetic demo data is loaded. It is not real SACA or Strikers customer data.",
            }
        ]
    )
    return StrikersDataBundle(
        data_mode="Demo data",
        matches=matches,
        daily_sales=daily_sales,
        transactions=transactions,
        fixtures=fixtures,
        customers=customers,
        future_fixtures=future_fixtures,
        file_status=_empty_file_status(),
        column_mappings=pd.DataFrame(columns=["file_key", "field", "source_column", "mapping_type"]),
        validation_warnings=warnings,
        metrics=metrics,
        loaded_sources={},
    )


def load_real_strikers_data(
    raw_dir: str | Path | None = None,
    mapping_overrides: dict[str, dict[str, str]] | None = None,
    random_seed: int = 2026,
    write_processed: bool = True,
    sample_rows: int | None = None,
    progress_callback=None,
) -> StrikersDataBundle:
    """Detect, load, validate, and normalize the five real Strikers CSV exports."""

    root = Path(raw_dir) if raw_dir else RAW_DATA_DIR
    mapping_overrides = mapping_overrides or {}
    _maybe_extract_bundled_raw_archive(root, progress_callback=progress_callback)
    if write_processed and not mapping_overrides and sample_rows is None and _can_use_processed_cache(root):
        _progress(progress_callback, "Loading cached processed Strikers outputs")
        return _load_processed_real_data(root)
    sources = _load_sources(root, mapping_overrides, sample_rows=sample_rows, progress_callback=progress_callback)

    ticket_frames: list[pd.DataFrame] = []
    fixture_frames: list[pd.DataFrame] = []
    future_fixture_frames: list[pd.DataFrame] = []
    customers = pd.DataFrame()
    warnings: list[dict[str, str]] = []
    mapping_rows: list[dict[str, str]] = []

    for source in sources.values():
        warnings.extend(_warning_rows(source.spec.key, source.warnings))
        if source.missing or source.csv is None:
            continue
        mapping_rows.extend(
            {"file_key": source.spec.key, "field": field, "source_column": column, "mapping_type": "suggested"}
            for field, column in sorted(source.mapping.items())
        )
        if source.spec.kind == "tickets":
            _progress(progress_callback, f"Normalising {source.spec.label}")
            ticket_frames.append(_normalise_ticket_file(source))
        elif source.spec.kind == "fixtures":
            _progress(progress_callback, f"Normalising {source.spec.label}")
            fixture_frames.append(_normalise_fixture_file(source))
        elif source.spec.kind == "future_fixtures":
            _progress(progress_callback, f"Normalising {source.spec.label}")
            future_fixture_frames.append(_normalise_future_fixture_file(source))
        elif source.spec.kind == "customers":
            _progress(progress_callback, f"Normalising {source.spec.label}")
            customers = _normalise_customer_file(source)

    _progress(progress_callback, "Combining transaction and fixture tables")
    transactions = pd.concat(ticket_frames, ignore_index=True) if ticket_frames else _empty_transactions()
    transactions = _annotate_customer_purchase_status(transactions)
    fixtures = pd.concat(fixture_frames, ignore_index=True) if fixture_frames else pd.DataFrame()
    fixtures = _prepare_fixtures(fixtures)

    if fixtures.empty and not transactions.empty:
        warnings.append(
            {
                "severity": "warning",
                "area": "fixtures",
                "message": "No fixture CSVs were loaded; fixtures were inferred from ticket event names, opponents, venues, and dates.",
            }
        )
        fixtures = _infer_fixtures_from_transactions(transactions)

    _progress(progress_callback, "Matching ticket rows to fixtures")
    transactions, fixtures, match_warning_rows = _match_transactions_to_fixtures(transactions, fixtures)
    warnings.extend(match_warning_rows)
    _progress(progress_callback, "Building customer aggregates")
    customers = _customers_from_transactions(customers, transactions)

    if future_fixture_frames:
        _progress(progress_callback, "Using uploaded next-season fixture assumptions")
        future_fixtures = _prepare_fixtures(pd.concat(future_fixture_frames, ignore_index=True))
    else:
        _progress(progress_callback, "Generating future fixture assumptions")
        future_fixtures = generate_assumed_future_fixtures(
            fixtures[fixtures["fixture_status"].ne("assumed")] if not fixtures.empty else fixtures,
            FutureFixtureConfig(planning_season=PLANNING_SEASON, planning_season_label=PLANNING_SEASON_LABEL),
            random_seed=random_seed,
        )
    all_fixtures = _merge_fixtures(fixtures, future_fixtures)
    _progress(progress_callback, "Building forecasting inputs")
    matches, daily_sales = build_forecasting_inputs(transactions, all_fixtures)

    metrics = _metrics(transactions, all_fixtures, customers)
    warnings.extend(_validation_warnings(transactions, all_fixtures, customers, metrics))
    file_status = _file_status_frame(sources)
    mapping_frame = pd.DataFrame(mapping_rows, columns=["file_key", "field", "source_column", "mapping_type"])
    warning_frame = pd.DataFrame(warnings, columns=["severity", "area", "message"]).drop_duplicates()

    if write_processed:
        _progress(progress_callback, "Writing processed outputs")
        _write_processed_outputs(transactions, all_fixtures, customers, matches, daily_sales)

    return StrikersDataBundle(
        data_mode="Client uploaded data",
        matches=matches,
        daily_sales=daily_sales,
        transactions=transactions,
        fixtures=all_fixtures,
        customers=customers,
        future_fixtures=future_fixtures,
        file_status=file_status,
        column_mappings=mapping_frame,
        validation_warnings=warning_frame,
        metrics=metrics,
        loaded_sources=sources,
    )


def _load_processed_real_data(raw_dir: Path) -> StrikersDataBundle:
    transactions = _read_processed_frame(PROCESSED_DATA_DIR / "transactions_normalised.csv", parse_dates=["transaction_date", "match_date"])
    fixtures = _read_processed_frame(PROCESSED_DATA_DIR / "fixtures_normalised.csv", parse_dates=["match_date"])
    customers = _read_processed_frame(PROCESSED_DATA_DIR / "customers_normalised.csv")
    matches = _read_processed_frame(PROCESSED_DATA_DIR / "matches.csv", parse_dates=["event_date", "on_sale_date", "early_bird_start", "early_bird_end", "member_presale_start", "member_presale_end", "general_public_start", "marketing_start", "marketing_end", "campaign_burst_date"])
    daily_sales = _read_processed_frame(PROCESSED_DATA_DIR / "daily_sales.csv", parse_dates=["date"])
    future_fixtures = fixtures[fixtures.get("fixture_status", pd.Series(dtype=object)).astype(str).eq("assumed")].copy() if not fixtures.empty else pd.DataFrame()
    metrics = _metrics(transactions, fixtures, customers)
    metrics["data_load_path"] = "processed-cache"
    metrics["processed_cache_used"] = True
    metrics["processed_cache_timestamp"] = _cache_timestamp_label(_latest_mtime(_processed_output_paths()))
    metrics["raw_snapshot_timestamp"] = _cache_timestamp_label(_latest_mtime(_raw_snapshot_paths(raw_dir)))
    warnings = _validation_warnings(transactions, fixtures, customers, metrics)
    warnings.append(
        {
            "severity": "info",
            "area": "processing",
            "message": "Loaded normalized outputs from data/processed for faster startup. Use Reprocess real data in Data Admin after replacing raw CSV extracts or changing mappings.",
        }
    )
    return StrikersDataBundle(
        data_mode="Client uploaded data",
        matches=matches,
        daily_sales=daily_sales,
        transactions=transactions,
        fixtures=fixtures,
        customers=customers,
        future_fixtures=future_fixtures,
        file_status=_processed_file_status_frame(raw_dir),
        column_mappings=pd.DataFrame(columns=["file_key", "field", "source_column", "mapping_type"]),
        validation_warnings=pd.DataFrame(warnings, columns=["severity", "area", "message"]).drop_duplicates(),
        metrics=metrics,
        loaded_sources={},
    )


def expected_file_dataframe() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "file_key": spec.key,
                "expected_file": spec.canonical_name,
                "label": spec.label,
                "kind": spec.kind,
                "competition": spec.competition or "All",
                "required": spec.required,
            }
            for spec in EXPECTED_FILES
        ]
    )


def data_cache_signature(raw_dir: str | Path | None = None) -> tuple[str, str]:
    root = Path(raw_dir) if raw_dir else RAW_DATA_DIR
    return (
        _cache_timestamp_label(_latest_mtime(_raw_snapshot_paths(root))),
        _cache_timestamp_label(_latest_mtime(_processed_output_paths())),
    )


def _processed_output_paths() -> list[Path]:
    return [
        PROCESSED_DATA_DIR / "transactions_normalised.csv",
        PROCESSED_DATA_DIR / "fixtures_normalised.csv",
        PROCESSED_DATA_DIR / "customers_normalised.csv",
        PROCESSED_DATA_DIR / "matches.csv",
        PROCESSED_DATA_DIR / "daily_sales.csv",
    ]


def _raw_snapshot_paths(raw_dir: Path) -> list[Path]:
    paths = [path for path in detect_source_files(raw_dir).values() if path is not None] if raw_dir.exists() else []
    if _is_project_raw_dir(raw_dir) and BUNDLED_RAW_ARCHIVE.exists():
        paths.append(BUNDLED_RAW_ARCHIVE)
    return paths


def _can_use_processed_cache(raw_dir: Path) -> bool:
    processed_paths = _processed_output_paths()
    if not all(path.exists() for path in processed_paths):
        return False
    raw_paths = _raw_snapshot_paths(raw_dir)
    if not raw_paths:
        return False
    latest_processed = _latest_mtime(processed_paths)
    latest_raw = _latest_mtime(raw_paths)
    return latest_processed is not None and latest_raw is not None and latest_processed >= latest_raw


def _latest_mtime(paths: Iterable[Path]) -> pd.Timestamp | None:
    existing = [path for path in paths if path.exists()]
    if not existing:
        return None
    latest = max(path.stat().st_mtime for path in existing)
    return pd.Timestamp(latest, unit="s")


def _cache_timestamp_label(value: pd.Timestamp | None) -> str:
    if value is None or pd.isna(value):
        return "Unavailable"
    return value.strftime("%d %b %Y %H:%M")


def _read_processed_frame(path: Path, parse_dates: Iterable[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    parse_cols = [column for column in (parse_dates or [])]
    frame = pd.read_csv(path, parse_dates=parse_cols or None)
    for column in parse_cols:
        if column in frame:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
    return frame


def _processed_file_status_frame(raw_dir: Path) -> pd.DataFrame:
    rows = []
    detected = detect_source_files(raw_dir)
    for spec in EXPECTED_FILES:
        matched = detected.get(spec.key)
        rows.append(
            {
                "file_key": spec.key,
                "expected_file": spec.canonical_name,
                "detected_file": matched.name if matched else "",
                "status": "loaded from processed cache" if matched else ("missing" if spec.required else "optional"),
                "rows": np.nan,
                "columns": np.nan,
                "encoding": "",
                "header_present": np.nan,
                "required": spec.required,
            }
        )
    return pd.DataFrame(rows)


def detect_source_files(raw_dir: str | Path | None = None) -> dict[str, Path | None]:
    root = Path(raw_dir) if raw_dir else RAW_DATA_DIR
    files = list(root.glob("*.csv")) if root.exists() else []
    lookup = {_normalised_filename(path): path for path in files}
    detected: dict[str, Path | None] = {}
    for spec in EXPECTED_FILES:
        detected[spec.key] = None
        for alias in spec.aliases:
            candidate = lookup.get(_normalised_filename(alias))
            if candidate is not None:
                detected[spec.key] = candidate
                break
    return detected


def _is_project_raw_dir(raw_dir: Path) -> bool:
    return raw_dir.resolve() == RAW_DATA_DIR.resolve()


def _maybe_extract_bundled_raw_archive(raw_dir: Path, progress_callback=None) -> None:
    if not _is_project_raw_dir(raw_dir) or not BUNDLED_RAW_ARCHIVE.exists():
        return

    detected = detect_source_files(raw_dir)
    missing_specs = [spec for spec in EXPECTED_FILES if detected.get(spec.key) is None]
    if not missing_specs:
        return

    raw_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(BUNDLED_RAW_ARCHIVE) as archive:
        member_lookup = {
            Path(member).name: member
            for member in archive.namelist()
            if member and not member.endswith("/")
        }
        extracted_any = False
        for spec in missing_specs:
            member = member_lookup.get(spec.canonical_name)
            if member is None:
                continue
            target = raw_dir / spec.canonical_name
            if target.exists():
                continue
            if not extracted_any:
                _progress(progress_callback, f"Extracting bundled private archive: {BUNDLED_RAW_ARCHIVE.name}")
                extracted_any = True
            with archive.open(member) as source, target.open("wb") as destination:
                copyfileobj(source, destination)


def build_forecasting_inputs(transactions: pd.DataFrame, fixtures: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return the match/daily frames expected by the existing forecast modules."""

    if fixtures.empty:
        return _empty_matches(), _empty_daily_sales()

    matches = fixtures.copy()
    matches["match_id"] = matches["fixture_id"].astype(str)
    matches["season"] = pd.to_numeric(matches.get("season"), errors="coerce").fillna(
        matches["season_label"].map(_season_start_from_label)
    )
    matches["season"] = matches["season"].fillna(PLANNING_SEASON).astype(int)
    matches["event_date"] = pd.to_datetime(matches["match_date"], errors="coerce")
    matches["on_sale_date"] = _fixture_on_sale_dates(matches, transactions)
    matches["early_bird_start"] = matches["on_sale_date"]
    matches["early_bird_end"] = matches["on_sale_date"] + pd.Timedelta(days=20)
    matches["member_presale_start"] = matches["early_bird_end"] + pd.Timedelta(days=7)
    matches["member_presale_end"] = matches["member_presale_start"] + pd.Timedelta(days=6)
    matches["general_public_start"] = matches["member_presale_end"] + pd.Timedelta(days=1)
    matches["marketing_start"] = matches["event_date"] - pd.Timedelta(days=28)
    matches["marketing_end"] = matches["event_date"] - pd.Timedelta(days=10)
    matches["campaign_burst_date"] = matches["event_date"] - pd.Timedelta(days=7)
    matches["day_of_week"] = matches["event_date"].dt.day_name()
    matches["round"] = matches.groupby(["season_label", "competition"]).cumcount() + 1
    matches["home_match_number"] = matches["round"]
    matches["event_category"] = np.select(
        [
            matches["event_date"].dt.strftime("%m-%d").eq("12-31"),
            matches["competition"].eq("WBBL"),
            matches["opponent"].astype(str).str.contains("Sixers|Scorchers|Heat", case=False, na=False),
        ],
        ["new-year", "family", "marquee"],
        default="standard",
    )
    matches["campaign_theme"] = np.where(matches["event_category"].eq("new-year"), "New Year's Eve Bash", "Fixture push")
    matches["opponent_strength"] = 80
    matches["historical_attendance"] = matches["capacity_total"].fillna(0) * 0.55
    matches["ticket_price"] = _average_ticket_price_by_fixture(transactions, matches)
    matches["membership_base"] = 0
    matches["prior_season_performance"] = 0
    matches["weather_temp_c"] = np.where(matches["competition"].eq("BBL"), 28, 23)
    matches["public_holiday"] = matches["event_date"].dt.strftime("%m-%d").isin({"12-31", "01-26"}).astype(int)
    matches["school_holiday"] = matches["event_date"].dt.month.isin([12, 1]).astype(int)
    matches["double_header"] = 0
    matches["finals_contention"] = 0
    matches["planning_season"] = matches["season_label"].eq(PLANNING_SEASON_LABEL).astype(int)

    final_paid = (
        transactions.groupby("fixture_id", as_index=False)["paid_tickets_sold"].sum()
        if not transactions.empty
        else pd.DataFrame(columns=["fixture_id", "paid_tickets_sold"])
    )
    matches = matches.merge(final_paid, on="fixture_id", how="left")
    capacity_floor = matches["capacity_total"].fillna(0) * np.where(matches["competition"].eq("BBL"), 0.58, 0.46)
    matches["baseline_target"] = np.maximum(matches["paid_tickets_sold"].fillna(0) * 1.08, capacity_floor).round(-2)
    matches["baseline_target"] = matches["baseline_target"].mask(matches["baseline_target"].lt(1), 5_000)
    matches["manual_target"] = (matches["baseline_target"] * 1.08).round(-2)
    matches = matches.drop(columns=["paid_tickets_sold"], errors="ignore")

    daily_sales = _daily_sales_from_transactions(transactions, matches)
    return _match_columns(matches), daily_sales


def _load_sources(
    root: Path,
    mapping_overrides: dict[str, dict[str, str]],
    sample_rows: int | None = None,
    progress_callback=None,
) -> dict[str, LoadedSource]:
    detected = detect_source_files(root)
    sources: dict[str, LoadedSource] = {}
    for spec in EXPECTED_FILES:
        path = detected.get(spec.key)
        if path is None:
            warnings = [f"Missing expected file: {spec.canonical_name}"] if spec.required else []
            sources[spec.key] = LoadedSource(spec=spec, path=None, missing=True, warnings=warnings)
            continue
        _progress(progress_callback, f"Reading {path.name}")
        csv = read_csv_robust(path, nrows=sample_rows)
        _progress(progress_callback, f"Loaded {path.name}: {len(csv.frame):,} rows x {len(csv.frame.columns):,} columns")
        mapping = suggest_column_mapping(csv.frame, spec.kind, csv.header_present)
        mapping.update({field: column for field, column in mapping_overrides.get(spec.key, {}).items() if column})
        source = LoadedSource(spec=spec, path=path, csv=csv, mapping=mapping, warnings=list(csv.warnings))
        _warn_unmapped_required(source)
        sources[spec.key] = source
    return sources


def _progress(progress_callback, message: str) -> None:
    if progress_callback is not None:
        progress_callback(message)


def suggest_column_mapping(frame: pd.DataFrame, kind: str, header_present: bool = True) -> dict[str, str]:
    mapping: dict[str, str] = {}
    normalized_columns = {normalise_token(column): column for column in frame.columns}

    for field, synonyms in COLUMN_SYNONYMS.items():
        for synonym in synonyms:
            token = normalise_token(synonym)
            if token in normalized_columns:
                mapping[field] = normalized_columns[token]
                break
        if field in mapping:
            continue
        for column_token, column in normalized_columns.items():
            if any(column_token == normalise_token(s) or normalise_token(s) in column_token for s in synonyms):
                mapping[field] = column
                break

    if not header_present:
        positional = {
            "tickets": TICKET_POSITIONAL_FIELDS,
            "fixtures": FIXTURE_POSITIONAL_FIELDS,
            "customers": CUSTOMER_POSITIONAL_FIELDS,
        }.get(kind, {})
        for field, column in positional.items():
            if column in frame.columns:
                mapping.setdefault(field, column)
    return mapping


def _normalise_ticket_file(source: LoadedSource) -> pd.DataFrame:
    assert source.csv is not None
    frame = source.csv.frame.copy()
    mapping = source.mapping
    index = frame.index
    out = pd.DataFrame(index=index)
    out["source_file"] = source.path.name if source.path else source.spec.canonical_name
    out["source_row_number"] = index + 1
    out["competition"] = _text(frame, mapping, "competition").fillna(source.spec.competition or _competition_from_filename(source.spec.canonical_name))
    out["transaction_id"] = _text(frame, mapping, "transaction_id")
    out["order_id"] = _text(frame, mapping, "order_id")
    out["customer_id"] = _text(frame, mapping, "customer_id")
    if "column_019" in frame.columns:
        out["customer_id"] = out["customer_id"].where(out["customer_id"].notna(), frame["column_019"].map(clean_text))
    out["anonymised_customer_id"] = out["customer_id"].map(anonymise_identifier)
    out["fixture_id"] = _text(frame, mapping, "fixture_id")
    out["fixture_name"] = _coalesce_text(frame, mapping, ["fixture_name", "fixture_label", "event_name"])
    out["event_name"] = _coalesce_text(frame, mapping, ["event_name", "fixture_name", "fixture_label"])
    out["team"] = _text(frame, mapping, "team").fillna("Adelaide Strikers")
    out["opponent"] = _text(frame, mapping, "opponent")
    out["venue"] = _text(frame, mapping, "venue")
    out["transaction_date"] = parse_date_series(_series(frame, mapping, "transaction_date"))
    out["match_date"] = parse_date_series(_series(frame, mapping, "match_date"))
    missing_match_date = out["match_date"].isna()
    if missing_match_date.any():
        inferred = extract_date_from_text(out.loc[missing_match_date, "event_name"].fillna(out.loc[missing_match_date, "fixture_name"]))
        out.loc[missing_match_date, "match_date"] = inferred
    out["season_label"] = _coalesce_text(frame, mapping, ["season_label", "season"]).map(_normalise_season_label)
    out["season_label"] = out["season_label"].fillna(out["match_date"].map(_season_label_from_date))
    out["season_label"] = out["season_label"].fillna(out["transaction_date"].map(_season_label_from_date))
    out["season"] = out["season_label"].map(_season_start_from_label).fillna(PLANNING_SEASON)
    out["tickets_sold"] = parse_number_series(_series(frame, mapping, "tickets_sold"), default=1.0)
    out["tickets_sold"] = out["tickets_sold"].mask(out["tickets_sold"].eq(0), 1.0)
    gross = parse_number_series(_series(frame, mapping, "gross_revenue"), default=np.nan)
    net = parse_number_series(_series(frame, mapping, "net_revenue"), default=np.nan)
    price = parse_number_series(_series(frame, mapping, "price_paid"), default=np.nan)
    out["gross_revenue"] = gross.fillna(net).fillna(price * out["tickets_sold"].abs()).fillna(0.0)
    out["net_revenue"] = net.fillna(out["gross_revenue"])
    out["price_paid"] = price.fillna(
        out["gross_revenue"].abs() / out["tickets_sold"].abs().replace(0, np.nan)
    ).fillna(0.0)
    out["ticket_type"] = _text(frame, mapping, "ticket_type")
    out["ticket_class"] = _text(frame, mapping, "ticket_class")
    out["price_type"] = _text(frame, mapping, "price_type")
    out["section"] = _text(frame, mapping, "section")
    out["row"] = _text(frame, mapping, "row")
    out["seat"] = _text(frame, mapping, "seat")
    out["sales_channel"] = _text(frame, mapping, "sales_channel")
    out["promo_code"] = _text(frame, mapping, "promo_code")
    out["purchaser_postcode"] = _text(frame, mapping, "postcode")
    out["purchaser_age_band"] = _text(frame, mapping, "age_band")
    out["purchaser_gender"] = _coalesce_text(frame, mapping, ["gender"])
    out["purchaser_family_flag"] = parse_bool_series(_series(frame, mapping, "family_flag"))
    out["marketing_opt_in"] = parse_bool_series(_series(frame, mapping, "marketing_opt_in"))
    out["status"] = _text(frame, mapping, "status")
    combined_text = (
        out[["ticket_type", "ticket_class", "price_type", "status"]]
        .fillna("")
        .agg(" ".join, axis=1)
        .str.lower()
    )
    out["is_refund"] = (
        out["tickets_sold"].lt(0)
        | out["gross_revenue"].lt(0)
        | combined_text.str.contains("refund|cancel|void|reversal", regex=True)
    )
    out["is_comp"] = (
        out["gross_revenue"].abs().le(0.01)
        | combined_text.str.contains("comp|complimentary|free|contingency", regex=True)
    ) & ~out["is_refund"]
    out["paid_tickets_sold"] = out["tickets_sold"].where(~out["is_comp"], 0).clip(lower=-999_999)
    out["comp_tickets_sold"] = out["tickets_sold"].where(out["is_comp"], 0).clip(lower=0)
    out["sales_window"] = _infer_sales_window(out["transaction_date"], out["match_date"])
    out["customer_id"] = out["customer_id"].map(clean_text)
    out["transaction_id"] = out["transaction_id"].where(out["transaction_id"].notna(), _stable_ids(out, "txn"))
    out["order_id"] = out["order_id"].where(out["order_id"].notna(), out["transaction_id"])
    return out.reset_index(drop=True)


def _normalise_fixture_file(source: LoadedSource) -> pd.DataFrame:
    assert source.csv is not None
    frame = source.csv.frame.copy()
    mapping = source.mapping
    out = pd.DataFrame(index=frame.index)
    out["source_file"] = source.path.name if source.path else source.spec.canonical_name
    out["competition"] = _text(frame, mapping, "competition").fillna(source.spec.competition or _competition_from_filename(source.spec.canonical_name))
    out["fixture_id"] = _text(frame, mapping, "fixture_id")
    out["season_label"] = _coalesce_text(frame, mapping, ["season_label", "season"]).map(_normalise_season_label)
    out["team"] = _text(frame, mapping, "team").fillna("Adelaide Strikers")
    out["opponent"] = _text(frame, mapping, "opponent")
    out["venue"] = _text(frame, mapping, "venue")
    out["match_date"] = parse_date_series(_series(frame, mapping, "match_date"))
    out["match_time"] = _text(frame, mapping, "match_time").fillna("")
    out["day_of_week"] = out["match_date"].dt.day_name()
    out["is_home_match"] = True
    out["capacity_total"] = parse_number_series(_series(frame, mapping, "capacity_total"), default=np.nan)
    out["notes"] = _text(frame, mapping, "notes")
    out["season_label"] = out["season_label"].fillna(out["match_date"].map(_season_label_from_date))
    out["season"] = out["season_label"].map(_season_start_from_label)
    out["fixture_status"] = "historical"
    out["is_confirmed"] = True
    out["is_assumed"] = False
    out["fixture_label"] = out.apply(_fixture_label_from_row, axis=1)
    out["fixture_id"] = out["fixture_id"].where(out["fixture_id"].notna(), _fixture_ids(out))
    return out.dropna(subset=["match_date"]).reset_index(drop=True)


def _normalise_future_fixture_file(source: LoadedSource) -> pd.DataFrame:
    out = _normalise_fixture_file(source)
    if out.empty:
        return out
    out["season_label"] = out["season_label"].fillna(PLANNING_SEASON_LABEL)
    out["season"] = pd.to_numeric(out["season"], errors="coerce").fillna(PLANNING_SEASON)
    out["fixture_status"] = "assumed"
    out["is_confirmed"] = False
    out["is_assumed"] = True
    out["notes"] = out["notes"].fillna("Uploaded next-season fixture assumption.")
    out["fixture_label"] = out.apply(_fixture_label_from_row, axis=1)
    return out.reset_index(drop=True)


def _normalise_customer_file(source: LoadedSource) -> pd.DataFrame:
    assert source.csv is not None
    frame = source.csv.frame.copy()
    mapping = source.mapping
    out = pd.DataFrame(index=frame.index)
    out["source_file"] = source.path.name if source.path else source.spec.canonical_name
    out["customer_id"] = _text(frame, mapping, "customer_id")
    out["anonymised_customer_id"] = out["customer_id"].map(anonymise_identifier)
    out["postcode"] = _text(frame, mapping, "postcode")
    out["suburb"] = _text(frame, mapping, "suburb")
    out["state"] = _text(frame, mapping, "state")
    out["age"] = parse_number_series(_series(frame, mapping, "age"), default=np.nan)
    dob = parse_date_series(_series(frame, mapping, "date_of_birth"))
    age_from_dob = ((pd.Timestamp("2026-08-31") - dob).dt.days / 365.25).where(dob.notna())
    out["age"] = out["age"].fillna(age_from_dob).round()
    out["age_band"] = _text(frame, mapping, "age_band")
    out["age_band"] = out["age_band"].fillna(out["age"].map(_age_band_from_age))
    out["gender"] = _text(frame, mapping, "gender")
    out["family_flag"] = parse_bool_series(_series(frame, mapping, "family_flag"))
    out["marketing_opt_in"] = parse_bool_series(_series(frame, mapping, "marketing_opt_in"))
    out["email_opt_in"] = parse_bool_series(_series(frame, mapping, "email_opt_in"))
    out["sms_opt_in"] = parse_bool_series(_series(frame, mapping, "sms_opt_in"))
    out["has_customer_record"] = True
    out = out.dropna(subset=["customer_id"]).drop_duplicates(subset=["customer_id"]).reset_index(drop=True)
    return out


def _prepare_fixtures(fixtures: pd.DataFrame) -> pd.DataFrame:
    if fixtures.empty:
        return fixtures
    prepared = fixtures.copy()
    prepared["match_date"] = pd.to_datetime(prepared["match_date"], errors="coerce")
    prepared = prepared.dropna(subset=["match_date"])
    prepared["season_label"] = prepared["season_label"].fillna(prepared["match_date"].map(_season_label_from_date))
    prepared["season"] = prepared["season"].fillna(prepared["season_label"].map(_season_start_from_label))
    prepared["fixture_label"] = prepared.apply(_fixture_label_from_row, axis=1)
    prepared["fixture_id"] = prepared["fixture_id"].where(prepared["fixture_id"].notna(), _fixture_ids(prepared))
    prepared = prepared.drop_duplicates(subset=["fixture_id"]).reset_index(drop=True)
    return prepared


def _infer_fixtures_from_transactions(transactions: pd.DataFrame) -> pd.DataFrame:
    if transactions.empty:
        return pd.DataFrame()
    group_cols = ["competition", "season_label", "opponent", "venue", "match_date"]
    grouped = transactions.dropna(subset=["match_date"]).groupby(group_cols, dropna=False).agg(
        team=("team", "first"),
        fixture_name=("fixture_name", "first"),
    )
    inferred = grouped.reset_index()
    inferred["season"] = inferred["season_label"].map(_season_start_from_label)
    inferred["match_time"] = ""
    inferred["day_of_week"] = pd.to_datetime(inferred["match_date"]).dt.day_name()
    inferred["is_home_match"] = True
    inferred["capacity_total"] = np.nan
    inferred["notes"] = "Inferred from ticket export because fixture CSV was unavailable."
    inferred["fixture_status"] = "historical"
    inferred["is_confirmed"] = False
    inferred["is_assumed"] = False
    inferred["fixture_label"] = inferred.apply(_fixture_label_from_row, axis=1)
    inferred["fixture_id"] = _fixture_ids(inferred)
    return inferred


def _match_transactions_to_fixtures(transactions: pd.DataFrame, fixtures: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, str]]]:
    warnings: list[dict[str, str]] = []
    if transactions.empty or fixtures.empty:
        return transactions, fixtures, warnings

    tx = transactions.copy()
    fx = fixtures.copy()
    tx["_exact_key"] = _fixture_match_key(tx, include_date=True)
    fx["_exact_key"] = _fixture_match_key(fx, include_date=True)
    exact = fx[["_exact_key", "fixture_id"]].drop_duplicates("_exact_key").rename(columns={"fixture_id": "_matched_fixture_id"})
    tx = tx.merge(exact, on="_exact_key", how="left")
    tx["fixture_match_score"] = np.where(tx["_matched_fixture_id"].notna(), 1.0, 0.0)
    tx["fixture_id"] = tx["fixture_id"].where(tx["fixture_id"].notna(), tx["_matched_fixture_id"])

    missing = tx["fixture_id"].isna()
    if missing.any():
        tx.loc[missing, "_soft_key"] = _fixture_match_key(tx.loc[missing], include_date=False)
        fx["_soft_key"] = _fixture_match_key(fx, include_date=False)
        soft_candidates = fx.groupby("_soft_key").agg(
            _matched_fixture_id=("fixture_id", "first"),
            _candidate_count=("fixture_id", "nunique"),
        ).reset_index()
        tx = tx.merge(soft_candidates, on="_soft_key", how="left", suffixes=("", "_soft"))
        soft_mask = tx["fixture_id"].isna() & tx["_matched_fixture_id_soft"].notna() & tx["_candidate_count"].eq(1)
        tx.loc[soft_mask, "fixture_id"] = tx.loc[soft_mask, "_matched_fixture_id_soft"]
        tx.loc[soft_mask, "fixture_match_score"] = 0.72

    match_rate = tx["fixture_id"].notna().mean() if len(tx) else 0
    if match_rate < 0.85:
        warnings.append(
            {
                "severity": "warning",
                "area": "fixture matching",
                "message": f"Fixture join rate is {match_rate:.0%}. Check event names, dates, opponents, and venue mappings.",
            }
        )

    tx = tx.drop(columns=[column for column in tx.columns if column.startswith("_")], errors="ignore")
    fx = fx.drop(columns=[column for column in fx.columns if column.startswith("_")], errors="ignore")
    return tx, fx, warnings


def _customers_from_transactions(customers: pd.DataFrame, transactions: pd.DataFrame) -> pd.DataFrame:
    if transactions.empty:
        return customers

    tx = transactions.dropna(subset=["customer_id"]).copy()
    if tx.empty:
        return customers

    tx["transaction_date"] = pd.to_datetime(tx["transaction_date"], errors="coerce")
    tx["usual_purchase_month"] = tx["transaction_date"].dt.month_name()
    tx["usual_purchase_window"] = tx["sales_window"].fillna("unknown")
    aggregates = tx.groupby("customer_id").agg(
        first_purchase_date=("transaction_date", "min"),
        last_purchase_date=("transaction_date", "max"),
        seasons_purchased_count=("season_label", "nunique"),
        lifetime_tickets=("tickets_sold", "sum"),
        lifetime_revenue=("gross_revenue", "sum"),
        family_flag=("purchaser_family_flag", "max"),
        marketing_opt_in=("marketing_opt_in", "max"),
    ).reset_index()
    mode_specs = {
        "preferred_ticket_type": "ticket_type",
        "preferred_ticket_class": "ticket_class",
        "usual_purchase_month": "usual_purchase_month",
        "usual_purchase_window": "usual_purchase_window",
        "postcode": "purchaser_postcode",
        "age_band": "purchaser_age_band",
        "gender": "purchaser_gender",
    }
    for output_column, source_column in mode_specs.items():
        aggregates = aggregates.merge(
            _fast_group_mode(tx, "customer_id", source_column, output_column),
            on="customer_id",
            how="left",
        )
    aggregates["anonymised_customer_id"] = aggregates["customer_id"].map(anonymise_identifier)
    aggregates["has_ticket_history"] = True

    if customers.empty:
        aggregates["has_customer_record"] = False
        return aggregates

    merged = customers.merge(aggregates, on=["customer_id", "anonymised_customer_id"], how="outer", suffixes=("", "_from_sales"))
    merged["has_customer_record"] = merged["has_customer_record"].eq(True)
    merged["has_ticket_history"] = merged["has_ticket_history"].eq(True)
    for column in ["postcode", "age_band", "gender", "family_flag", "marketing_opt_in"]:
        sales_column = f"{column}_from_sales"
        if sales_column in merged.columns:
            merged[column] = merged[column].where(merged[column].notna(), merged[sales_column])
            merged = merged.drop(columns=[sales_column])
    return merged


def _annotate_customer_purchase_status(transactions: pd.DataFrame) -> pd.DataFrame:
    if transactions.empty or "customer_id" not in transactions:
        return transactions
    annotated = transactions.copy()
    valid = annotated.dropna(subset=["customer_id", "season_label", "transaction_date"]).copy()
    if valid.empty:
        annotated["customer_purchase_status"] = "Unknown"
        return annotated
    valid["transaction_date"] = pd.to_datetime(valid["transaction_date"], errors="coerce")
    first = valid.sort_values("transaction_date").groupby("customer_id")["season_label"].first()
    annotated["_first_purchase_season"] = annotated["customer_id"].map(first)
    annotated["customer_purchase_status"] = np.where(
        annotated["season_label"].astype(str).eq(annotated["_first_purchase_season"].astype(str)),
        "New",
        "Returning",
    )
    return annotated.drop(columns=["_first_purchase_season"])


def _daily_sales_from_transactions(transactions: pd.DataFrame, matches: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "match_id",
        "date",
        "days_to_event",
        "daily_sales",
        "cumulative_sales",
        "sales_window",
        "marketing_period",
        "is_public_holiday",
        "is_campaign_burst",
        "membership_on_sale",
        "early_bird_active",
        "school_holiday",
        "source",
    ]
    if transactions.empty:
        return pd.DataFrame(columns=columns)

    tx = transactions.dropna(subset=["fixture_id", "transaction_date"]).copy()
    if tx.empty:
        return pd.DataFrame(columns=columns)
    tx["date"] = pd.to_datetime(tx["transaction_date"]).dt.normalize()
    grouped = tx.groupby(["fixture_id", "date", "sales_window"], dropna=False).agg(
        daily_sales=("paid_tickets_sold", "sum"),
        total_daily_tickets=("tickets_sold", "sum"),
        daily_comps=("comp_tickets_sold", "sum"),
        daily_revenue=("gross_revenue", "sum"),
    ).reset_index()
    grouped = grouped.merge(matches[["match_id", "event_date", "early_bird_start", "early_bird_end", "member_presale_start", "member_presale_end", "marketing_start", "marketing_end", "campaign_burst_date", "school_holiday"]], left_on="fixture_id", right_on="match_id", how="left")
    grouped["days_to_event"] = (pd.to_datetime(grouped["event_date"]) - pd.to_datetime(grouped["date"])).dt.days
    grouped = grouped.sort_values(["match_id", "date"])
    grouped["cumulative_sales"] = grouped.groupby("match_id")["daily_sales"].cumsum()
    grouped["marketing_period"] = np.where(grouped["date"].between(grouped["marketing_start"], grouped["marketing_end"]), "active", "none")
    grouped["is_public_holiday"] = pd.to_datetime(grouped["event_date"]).dt.strftime("%m-%d").isin({"12-31", "01-26"}).astype(int)
    grouped["is_campaign_burst"] = (pd.to_datetime(grouped["date"]) - pd.to_datetime(grouped["campaign_burst_date"])).abs().dt.days.le(1).astype(int)
    grouped["membership_on_sale"] = grouped["date"].between(grouped["member_presale_start"], grouped["member_presale_end"]).astype(int)
    grouped["early_bird_active"] = grouped["date"].between(grouped["early_bird_start"], grouped["early_bird_end"]).astype(int)
    grouped["source"] = "real_csv_normalised"
    grouped = grouped.rename(columns={"fixture_id": "fixture_id_source"})
    return grouped[columns].reset_index(drop=True)


def _demo_transactions_from_daily(matches: pd.DataFrame, daily_sales: pd.DataFrame) -> pd.DataFrame:
    joined = daily_sales.merge(matches, on="match_id", how="left", suffixes=("", "_match"))
    rows: list[pd.DataFrame] = []
    for is_comp, label, rate in [(False, "Paid", 0.0), (True, "Comp", 0.06)]:
        frame = joined.copy()
        comp_rate = np.where(frame["competition"].eq("WBBL"), 0.16, 0.05) if is_comp else 0.0
        tickets = (frame["daily_sales"] * comp_rate).round().astype(int) if is_comp else (frame["daily_sales"] * (1 - np.where(frame["competition"].eq("WBBL"), 0.16, 0.05))).round().astype(int)
        frame = frame[tickets.gt(0)].copy()
        frame["tickets_sold"] = tickets[tickets.gt(0)].to_numpy()
        frame["paid_tickets_sold"] = np.where(is_comp, 0, frame["tickets_sold"])
        frame["comp_tickets_sold"] = np.where(is_comp, frame["tickets_sold"], 0)
        frame["is_comp"] = is_comp
        frame["ticket_type"] = np.where(frame["competition"].eq("WBBL"), "Family / community", "Public")
        frame["ticket_class"] = np.where(frame["competition"].eq("WBBL"), "General Admission", "The Hill")
        frame["price_type"] = label
        frame["price_paid"] = np.where(is_comp, 0, frame["ticket_price"])
        frame["gross_revenue"] = frame["paid_tickets_sold"] * frame["price_paid"]
        frame["net_revenue"] = frame["gross_revenue"]
        frame["customer_id"] = [
            f"DEMO-{row.competition}-{row.match_id}-{pd.to_datetime(row.date).strftime('%Y%m%d')}-{label}"
            for row in frame.itertuples()
        ]
        frame["anonymised_customer_id"] = frame["customer_id"].map(anonymise_identifier)
        frame["transaction_id"] = frame["customer_id"]
        frame["order_id"] = frame["customer_id"]
        frame["fixture_id"] = frame["match_id"]
        frame["fixture_name"] = frame["fixture_label"] if "fixture_label" in frame else frame["opponent"]
        frame["event_name"] = frame["fixture_name"]
        frame["transaction_date"] = frame["date"]
        frame["match_date"] = frame["event_date"]
        frame["source_file"] = "synthetic_demo"
        frame["source_row_number"] = np.arange(len(frame)) + 1
        frame["section"] = np.where(frame["competition"].eq("WBBL"), "GA", "HILL")
        frame["sales_channel"] = "Demo"
        frame["is_refund"] = False
        frame["purchaser_postcode"] = np.where(frame["competition"].eq("BBL"), "5000", "5031")
        frame["purchaser_age_band"] = np.where(frame["competition"].eq("WBBL"), "Family", "25-39")
        frame["purchaser_gender"] = "Unknown"
        frame["purchaser_family_flag"] = frame["competition"].eq("WBBL")
        frame["marketing_opt_in"] = True
        rows.append(frame)
    if not rows:
        return _empty_transactions()
    transactions = pd.concat(rows, ignore_index=True)
    schema = _empty_transactions().columns.tolist()
    for column in schema:
        if column not in transactions:
            transactions[column] = 1.0 if column == "fixture_match_score" else None
    return transactions[schema]


def _fixtures_from_matches(matches: pd.DataFrame, fixture_status: str = "historical") -> pd.DataFrame:
    fixtures = matches.rename(columns={"match_id": "fixture_id", "event_date": "match_date"}).copy()
    fixtures["fixture_label"] = fixtures.apply(_fixture_label_from_row, axis=1)
    fixtures["capacity_total"] = fixtures.get("capacity_total", np.nan)
    fixtures["fixture_status"] = fixture_status
    fixtures["is_confirmed"] = fixture_status != "assumed"
    fixtures["is_assumed"] = fixture_status == "assumed"
    fixtures["notes"] = np.where(fixtures["is_assumed"], "Synthetic demo planning fixture.", "Synthetic demo historical fixture.")
    return fixtures[
        [
            "fixture_id",
            "season",
            "season_label",
            "competition",
            "team",
            "opponent",
            "venue",
            "match_date",
            "day_of_week",
            "fixture_label",
            "capacity_total",
            "fixture_status",
            "is_confirmed",
            "is_assumed",
            "notes",
        ]
    ].copy()


def _merge_fixtures(fixtures: pd.DataFrame, future_fixtures: pd.DataFrame) -> pd.DataFrame:
    if fixtures.empty:
        return future_fixtures.copy()
    existing_planning = fixtures[
        fixtures["season_label"].eq(PLANNING_SEASON_LABEL) & fixtures["fixture_status"].isin(["confirmed", "historical"])
    ]
    if not existing_planning.empty:
        future_to_add = future_fixtures[~future_fixtures["competition"].isin(existing_planning["competition"].unique())]
    else:
        future_to_add = future_fixtures
    combined = pd.concat([fixtures, future_to_add], ignore_index=True)
    return combined.drop_duplicates(subset=["fixture_id"]).reset_index(drop=True)


def _metrics(transactions: pd.DataFrame, fixtures: pd.DataFrame, customers: pd.DataFrame) -> dict[str, object]:
    if transactions.empty:
        return {
            "ticket_rows": 0,
            "fixture_rows": len(fixtures),
            "customer_rows": len(customers),
            "unique_customers": 0,
            "paid_ticket_count": 0,
            "comp_ticket_count": 0,
            "refund_count": 0,
            "gross_revenue": 0.0,
            "net_revenue": 0.0,
            "customer_join_rate": 0.0,
            "fixture_join_rate": 0.0,
            "date_range": "Unavailable",
        }
    if "has_customer_record" in customers:
        source_customers = customers[customers["has_customer_record"].fillna(False)]
    else:
        source_customers = customers
    customer_ids = set(source_customers["customer_id"].dropna().astype(str)) if "customer_id" in source_customers else set()
    tx_customers = transactions["customer_id"].dropna().astype(str)
    return {
        "ticket_rows": int(len(transactions)),
        "fixture_rows": int(len(fixtures)),
        "customer_rows": int(len(customers)),
        "unique_customers": int(transactions["customer_id"].dropna().nunique()),
        "paid_ticket_count": float(transactions["paid_tickets_sold"].sum()),
        "comp_ticket_count": float(transactions["comp_tickets_sold"].sum()),
        "refund_count": int(transactions["is_refund"].sum()),
        "gross_revenue": float(transactions["gross_revenue"].sum()),
        "net_revenue": float(transactions["net_revenue"].sum()),
        "customer_join_rate": float(tx_customers.isin(customer_ids).mean()) if len(tx_customers) and customer_ids else 0.0,
        "fixture_join_rate": float(transactions["fixture_id"].notna().mean()),
        "date_range": _date_range_label(transactions["transaction_date"]),
        "competitions": ", ".join(sorted(transactions["competition"].dropna().unique())),
        "seasons": ", ".join(sorted(transactions["season_label"].dropna().astype(str).unique())),
    }


def _validation_warnings(
    transactions: pd.DataFrame,
    fixtures: pd.DataFrame,
    customers: pd.DataFrame,
    metrics: dict[str, object],
) -> list[dict[str, str]]:
    warnings: list[dict[str, str]] = []
    if transactions.empty:
        warnings.append({"severity": "warning", "area": "tickets", "message": "No ticket transaction rows are available."})
        return warnings
    if transactions["transaction_date"].isna().mean() > 0.05:
        warnings.append({"severity": "warning", "area": "dates", "message": "More than 5% of ticket rows have invalid transaction dates."})
    if transactions["match_date"].isna().mean() > 0.05:
        warnings.append({"severity": "warning", "area": "dates", "message": "More than 5% of ticket rows have no match date. Fixture matching may be weaker."})
    if metrics.get("customer_join_rate", 0) < 0.5 and not customers.empty:
        warnings.append({"severity": "warning", "area": "customers", "message": f"Customer join rate is {metrics['customer_join_rate']:.0%}. Confirm the customer ID/gigyauid mapping."})
    if metrics.get("fixture_join_rate", 0) < 0.85:
        warnings.append({"severity": "warning", "area": "fixtures", "message": f"Fixture join rate is {metrics['fixture_join_rate']:.0%}. Some sales may be excluded from fixture-level forecasting."})
    if transactions.duplicated(subset=["transaction_id", "order_id", "fixture_id", "customer_id"]).mean() > 0.02:
        warnings.append({"severity": "info", "area": "duplicates", "message": "Possible duplicate transaction/order/customer rows detected. This may be normal if the export is seat-level."})
    if fixtures["capacity_total"].isna().all():
        warnings.append({"severity": "info", "area": "capacity", "message": "Fixture capacity was not found. Capacity sold metrics will use manual/default assumptions where available."})
    return warnings


def _file_status_frame(sources: dict[str, LoadedSource]) -> pd.DataFrame:
    rows = []
    for source in sources.values():
        rows.append(
            {
                "file_key": source.spec.key,
                "expected_file": source.spec.canonical_name,
                "detected_file": source.path.name if source.path else "",
                "status": "missing" if source.missing and source.spec.required else ("optional" if source.missing else "loaded"),
                "rows": 0 if source.csv is None else len(source.csv.frame),
                "columns": 0 if source.csv is None else len(source.csv.frame.columns),
                "encoding": "" if source.csv is None else source.csv.encoding,
                "header_present": False if source.csv is None else source.csv.header_present,
                "required": source.spec.required,
            }
        )
    return pd.DataFrame(rows)


def _empty_file_status() -> pd.DataFrame:
    frame = expected_file_dataframe()
    frame["detected_file"] = ""
    frame["status"] = "demo"
    frame["rows"] = 0
    frame["columns"] = 0
    frame["encoding"] = ""
    frame["header_present"] = False
    frame["required"] = frame.get("required", True)
    return frame


def _warning_rows(area: str, warnings: Iterable[str]) -> list[dict[str, str]]:
    return [{"severity": "warning", "area": area, "message": warning} for warning in warnings]


def _warn_unmapped_required(source: LoadedSource) -> None:
    required = {
        "tickets": ("customer_id", "transaction_date", "tickets_sold", "gross_revenue"),
        "fixtures": ("opponent", "venue", "match_date"),
        "customers": ("customer_id",),
    }.get(source.spec.kind, ())
    for field in required:
        if field not in source.mapping:
            source.warnings.append(f"Could not map required field '{field}'.")


def _write_processed_outputs(
    transactions: pd.DataFrame,
    fixtures: pd.DataFrame,
    customers: pd.DataFrame,
    matches: pd.DataFrame,
    daily_sales: pd.DataFrame,
) -> None:
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    transactions.to_csv(PROCESSED_DATA_DIR / "transactions_normalised.csv", index=False)
    fixtures.to_csv(PROCESSED_DATA_DIR / "fixtures_normalised.csv", index=False)
    customers.to_csv(PROCESSED_DATA_DIR / "customers_normalised.csv", index=False)
    matches.to_csv(PROCESSED_DATA_DIR / "matches.csv", index=False)
    daily_sales.to_csv(PROCESSED_DATA_DIR / "daily_sales.csv", index=False)


def _series(frame: pd.DataFrame, mapping: dict[str, str], field: str) -> pd.Series:
    column = mapping.get(field)
    if column and column in frame.columns:
        return frame[column]
    return pd.Series([None] * len(frame), index=frame.index, dtype=object)


def _text(frame: pd.DataFrame, mapping: dict[str, str], field: str) -> pd.Series:
    return _series(frame, mapping, field).map(clean_text)


def _coalesce_text(frame: pd.DataFrame, mapping: dict[str, str], fields: list[str]) -> pd.Series:
    output = pd.Series([None] * len(frame), index=frame.index, dtype=object)
    for field in fields:
        values = _text(frame, mapping, field)
        output = output.where(output.notna(), values)
    return output


def _stable_ids(frame: pd.DataFrame, prefix: str) -> pd.Series:
    return pd.Series(
        [
            f"{prefix}_{crc32('|'.join(map(str, row)).encode('utf-8')) & 0xFFFFFFFF:08x}_{idx}"
            for idx, row in enumerate(frame.fillna("").astype(str).to_numpy())
        ],
        index=frame.index,
    )


def _fixture_ids(fixtures: pd.DataFrame) -> pd.Series:
    def make_id(row: pd.Series) -> str:
        raw = "|".join(
            [
                str(row.get("competition") or "BBL"),
                str(row.get("season_label") or ""),
                str(pd.to_datetime(row.get("match_date")).date() if pd.notna(row.get("match_date")) else ""),
                str(row.get("opponent") or "unknown"),
                str(row.get("venue") or "unknown"),
            ]
        )
        return f"{normalise_token(row.get('competition') or 'BBL').upper()}_{crc32(raw.encode('utf-8')) & 0xFFFFFFFF:08x}"

    return fixtures.apply(make_id, axis=1)


def _fixture_match_key(frame: pd.DataFrame, include_date: bool) -> pd.Series:
    parts = [
        frame.get("competition", "").astype(str).map(normalise_token),
        frame.get("season_label", "").astype(str).map(normalise_token),
        frame.get("opponent", "").astype(str).map(normalise_token),
        frame.get("venue", "").astype(str).map(normalise_token),
    ]
    if include_date:
        date = pd.to_datetime(frame.get("match_date"), errors="coerce").dt.strftime("%Y%m%d").fillna("")
        parts.append(date)
    return pd.Series(["|".join(values) for values in zip(*parts)], index=frame.index)


def _fixture_label_from_row(row: pd.Series) -> str:
    date = pd.to_datetime(row.get("match_date"), errors="coerce")
    date_label = date.strftime("%d %b %Y") if pd.notna(date) else "date TBC"
    status = "TBD / assumed" if bool(row.get("is_assumed", False)) else str(row.get("fixture_status") or "historical")
    return f"{row.get('competition', 'BBL')} v {row.get('opponent', 'Opponent TBC')} | {date_label} | {status}"


def _normalise_season_label(value: object) -> str | None:
    text = clean_text(value)
    if not text:
        return None
    match = re.search(r"(\d{2,4})\s*[-/]\s*(\d{2,4})", text)
    if not match:
        year_match = re.search(r"(20\d{2})", text)
        if year_match:
            start = int(year_match.group(1))
            return f"{start}/{str(start + 1)[-2:]}"
        return text
    start_raw, end_raw = match.groups()
    start = int(start_raw)
    if start < 100:
        start += 2000
    end = int(end_raw[-2:])
    return f"{start}/{end:02d}"


def _season_label_from_date(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    date = pd.to_datetime(value, errors="coerce")
    if pd.isna(date):
        return None
    start = date.year if date.month >= 7 else date.year - 1
    return f"{start}/{str(start + 1)[-2:]}"


def _season_start_from_label(value: object) -> int | float:
    text = clean_text(value)
    if not text:
        return np.nan
    match = re.search(r"(20\d{2})", text)
    return int(match.group(1)) if match else np.nan


def _competition_from_filename(name: str) -> str:
    return "WBBL" if "wbbl" in normalise_token(name) else "BBL"


def _normalised_filename(path_or_name: str | Path) -> str:
    path = Path(path_or_name)
    name = path.stem if path.suffix else str(path_or_name)
    return normalise_token(name)


def _infer_sales_window(transaction_date: pd.Series, match_date: pd.Series) -> pd.Series:
    tx = pd.to_datetime(transaction_date, errors="coerce")
    match = pd.to_datetime(match_date, errors="coerce")
    days_to_match = (match - tx).dt.days
    return pd.Series(
        np.select(
            [
                days_to_match.le(7),
                days_to_match.between(8, 28),
                days_to_match.between(29, 75),
                days_to_match.gt(75),
            ],
            ["match-week", "campaign-window", "general-sale", "early-sales"],
            default="unknown",
        ),
        index=transaction_date.index,
    )


def _fixture_on_sale_dates(matches: pd.DataFrame, transactions: pd.DataFrame) -> pd.Series:
    default_offsets = np.where(matches["competition"].eq("WBBL"), 90, 120)
    defaults = pd.to_datetime(matches["event_date"]) - pd.to_timedelta(default_offsets, unit="D")
    if transactions.empty:
        return pd.Series(defaults, index=matches.index)
    first_dates = transactions.groupby("fixture_id")["transaction_date"].min()
    return matches["fixture_id"].map(first_dates).fillna(pd.Series(defaults, index=matches.index))


def _average_ticket_price_by_fixture(transactions: pd.DataFrame, matches: pd.DataFrame) -> pd.Series:
    if transactions.empty:
        return pd.Series(np.where(matches["competition"].eq("BBL"), 49.0, 29.0), index=matches.index)
    paid = transactions[transactions["paid_tickets_sold"].gt(0)].copy()
    if paid.empty:
        return pd.Series(np.where(matches["competition"].eq("BBL"), 49.0, 29.0), index=matches.index)
    paid["atp"] = paid["gross_revenue"] / paid["paid_tickets_sold"].replace(0, np.nan)
    lookup = paid.groupby("fixture_id")["atp"].median()
    fallback = np.where(matches["competition"].eq("BBL"), 49.0, 29.0)
    return matches["fixture_id"].map(lookup).fillna(pd.Series(fallback, index=matches.index)).clip(lower=0)


def _date_range_label(series: pd.Series) -> str:
    parsed = pd.to_datetime(series, errors="coerce").dropna()
    if parsed.empty:
        return "Unavailable"
    return f"{parsed.min().date()} to {parsed.max().date()}"


def _age_band_from_age(value: object) -> str | None:
    age = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(age):
        return None
    if age < 18:
        return "Under 18"
    if age <= 30:
        return "18-30"
    if age <= 45:
        return "31-45"
    if age <= 60:
        return "46-60"
    return "61+"


def _mode_or_unknown(series: pd.Series) -> object:
    values = series.dropna()
    if values.empty:
        return "Unknown"
    mode = values.mode()
    return mode.iloc[0] if not mode.empty else values.iloc[0]


def _fast_group_mode(frame: pd.DataFrame, group_col: str, value_col: str, output_col: str) -> pd.DataFrame:
    if value_col not in frame:
        return pd.DataFrame({group_col: [], output_col: []})
    values = frame[[group_col, value_col]].dropna().copy()
    if values.empty:
        return pd.DataFrame({group_col: [], output_col: []})
    values[value_col] = values[value_col].astype(str)
    counts = values.groupby([group_col, value_col], sort=False).size().reset_index(name="_count")
    counts = counts.sort_values([group_col, "_count"], ascending=[True, False])
    return counts.drop_duplicates(group_col)[[group_col, value_col]].rename(columns={value_col: output_col})


def _match_columns(matches: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "match_id",
        "season",
        "season_label",
        "competition",
        "team",
        "round",
        "home_match_number",
        "opponent",
        "venue",
        "event_date",
        "on_sale_date",
        "early_bird_start",
        "early_bird_end",
        "member_presale_start",
        "member_presale_end",
        "general_public_start",
        "day_of_week",
        "event_category",
        "campaign_theme",
        "opponent_strength",
        "historical_attendance",
        "ticket_price",
        "membership_base",
        "prior_season_performance",
        "weather_temp_c",
        "marketing_start",
        "marketing_end",
        "campaign_burst_date",
        "public_holiday",
        "school_holiday",
        "double_header",
        "finals_contention",
        "baseline_target",
        "manual_target",
        "planning_season",
        "fixture_status",
        "is_assumed",
        "capacity_total",
    ]
    for column in columns:
        if column not in matches:
            matches[column] = 0 if column not in {"fixture_status", "is_assumed"} else ("historical" if column == "fixture_status" else False)
    return matches[columns].sort_values(["season", "competition", "event_date"]).reset_index(drop=True)


def _empty_matches() -> pd.DataFrame:
    return pd.DataFrame(columns=_match_columns(pd.DataFrame()).columns)


def _empty_daily_sales() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "match_id",
            "date",
            "days_to_event",
            "daily_sales",
            "cumulative_sales",
            "sales_window",
            "marketing_period",
            "is_public_holiday",
            "is_campaign_burst",
            "membership_on_sale",
            "early_bird_active",
            "school_holiday",
            "source",
        ]
    )


def _empty_transactions() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "source_file",
            "source_row_number",
            "competition",
            "transaction_id",
            "order_id",
            "customer_id",
            "anonymised_customer_id",
            "fixture_id",
            "fixture_name",
            "event_name",
            "team",
            "opponent",
            "venue",
            "transaction_date",
            "match_date",
            "season_label",
            "season",
            "tickets_sold",
            "paid_tickets_sold",
            "comp_tickets_sold",
            "gross_revenue",
            "net_revenue",
            "price_paid",
            "ticket_type",
            "ticket_class",
            "price_type",
            "section",
            "row",
            "seat",
            "sales_channel",
            "promo_code",
            "purchaser_postcode",
            "purchaser_age_band",
            "purchaser_gender",
            "purchaser_family_flag",
            "marketing_opt_in",
            "status",
            "is_refund",
            "is_comp",
            "sales_window",
            "customer_purchase_status",
            "fixture_match_score",
        ]
    )
