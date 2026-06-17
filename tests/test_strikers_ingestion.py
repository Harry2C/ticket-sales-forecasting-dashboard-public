from __future__ import annotations

import zipfile

import pandas as pd

import preprocessing.strikers_ingestion as ingestion
from forecasting.future_fixtures import FutureFixtureConfig, generate_assumed_future_fixtures
from forecasting.targets import generate_target_curve
from preprocessing.strikers_ingestion import load_real_strikers_data, suggest_column_mapping
from utils.csv_utils import parse_date_series, read_csv_robust


HEADERLESS_BBL_TICKET = (
    '37835249,gigya-test-001,1794,Y24-25 - STR v SIX - BBL - SA - 15/01/2025,1,Adelaide Strikers,'
    'Sydney Sixers,Male,BBL T20,Y24-25,Y24-25,Adelaide Oval,SA,Y24-25 - BBL - BBL14,'
    '2025-01-15,Public,The Hill - Family Zone,GATE SALES - Adult,'
    '95A98864-218E-4F9A-ACAC-1650F405CAC0,1.0,28,1,SFAMILYGA35217941,'
    '"20250115,158573-e1783",Ticketek,1,1'
)


def test_headerless_ticket_export_uses_positional_mapping(tmp_path):
    path = tmp_path / "Strikers All Tickets.csv"
    path.write_text(HEADERLESS_BBL_TICKET + "\n", encoding="utf-8-sig")

    loaded = read_csv_robust(path)
    mapping = suggest_column_mapping(loaded.frame, "tickets", loaded.header_present)

    assert not loaded.header_present
    assert mapping["customer_id"] == "column_002"
    assert mapping["gross_revenue"] == "column_021"
    assert mapping["tickets_sold"] == "column_022"


def test_customer_mapping_recognises_gigya_uid_and_dob_variations():
    frame = pd.DataFrame(
        {
            "Gigya UID": ["abc"],
            "DOB": ["01/01/1995"],
            "Post Code": ["0500"],
            "Gender": ["Female"],
        }
    )

    mapping = suggest_column_mapping(frame, "customers", header_present=True)

    assert mapping["customer_id"] == "Gigya UID"
    assert mapping["date_of_birth"] == "DOB"
    assert mapping["postcode"] == "Post Code"


def test_real_loader_normalises_headerless_exports_and_joins_gigyauid(tmp_path):
    (tmp_path / "Strikers All Tickets.csv").write_text(HEADERLESS_BBL_TICKET + "\n", encoding="utf-8-sig")
    (tmp_path / "Strikers BBL Fixtures.csv").write_text(
        "Y24-25,Adelaide Oval,Adelaide Strikers,Sydney Sixers,2025-01-15\n",
        encoding="utf-8-sig",
    )
    (tmp_path / "Strikers Customer Data.csv").write_text(
        "gigya-test-001,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL\n",
        encoding="utf-8-sig",
    )

    bundle = load_real_strikers_data(tmp_path, write_processed=False)
    transaction = bundle.transactions.iloc[0]

    assert transaction["competition"] == "BBL"
    assert transaction["customer_id"] == "gigya-test-001"
    assert transaction["anonymised_customer_id"]
    assert transaction["fixture_id"] == bundle.fixtures.loc[bundle.fixtures["fixture_status"].eq("historical"), "fixture_id"].iloc[0]
    assert bundle.metrics["customer_join_rate"] == 1.0
    assert bundle.metrics["fixture_join_rate"] == 1.0
    assert bundle.daily_sales["daily_sales"].iloc[0] == 1


def test_future_fixture_assumptions_include_reproducible_nye_fixture():
    historical = pd.DataFrame(
        {
            "fixture_id": ["B1", "B2", "W1"],
            "season": [2025, 2025, 2025],
            "season_label": ["2025/26", "2025/26", "2025/26"],
            "competition": ["BBL", "BBL", "WBBL"],
            "team": ["Adelaide Strikers", "Adelaide Strikers", "Adelaide Strikers"],
            "opponent": ["Sydney Sixers", "Perth Scorchers", "Brisbane Heat"],
            "venue": ["Adelaide Oval", "Adelaide Oval", "Karen Rolton Oval"],
            "match_date": pd.to_datetime(["2025-12-18", "2026-01-10", "2025-10-19"]),
            "capacity_total": [53500, 53500, 5000],
        }
    )

    first = generate_assumed_future_fixtures(historical, FutureFixtureConfig(random_seed=7), random_seed=7)
    second = generate_assumed_future_fixtures(historical, FutureFixtureConfig(random_seed=7), random_seed=7)
    nye = first[first["fixture_id"].str.contains("NYE")]

    assert not nye.empty
    assert pd.Timestamp("2026-12-31") in set(pd.to_datetime(nye["match_date"]))
    assert first["opponent"].tolist() == second["opponent"].tolist()
    assert first["fixture_status"].eq("assumed").all()


def test_iso_fixture_dates_are_not_dayfirst_swapped():
    parsed = parse_date_series(pd.Series(["2019-12-01", "01/12/2019"]))

    assert parsed.iloc[0] == pd.Timestamp("2019-12-01")
    assert parsed.iloc[1] == pd.Timestamp("2019-12-01")


def test_target_curve_handles_on_sale_after_event_without_crashing():
    match = {
        "on_sale_date": pd.Timestamp("2026-12-31"),
        "event_date": pd.Timestamp("2026-12-01"),
        "marketing_start": pd.Timestamp("2026-11-01"),
        "marketing_end": pd.Timestamp("2026-11-20"),
        "campaign_burst_date": pd.Timestamp("2026-11-25"),
    }
    pace = pd.DataFrame({"days_to_event": [0], "historical_daily_share": [1.0]})

    curve = generate_target_curve(match, pace, total_sales_target=10_000)

    assert len(curve) == 1
    assert curve["target_cumulative"].iloc[-1] == 10_000


def test_real_loader_can_reuse_processed_cache(tmp_path, monkeypatch):
    processed_dir = tmp_path / "processed"
    monkeypatch.setattr(ingestion, "PROCESSED_DATA_DIR", processed_dir)

    (tmp_path / "Strikers All Tickets.csv").write_text(HEADERLESS_BBL_TICKET + "\n", encoding="utf-8-sig")
    (tmp_path / "Strikers BBL Fixtures.csv").write_text(
        "Y24-25,Adelaide Oval,Adelaide Strikers,Sydney Sixers,2025-01-15\n",
        encoding="utf-8-sig",
    )
    (tmp_path / "Strikers Customer Data.csv").write_text(
        "gigya-test-001,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL\n",
        encoding="utf-8-sig",
    )

    first = load_real_strikers_data(tmp_path, write_processed=True)
    second = load_real_strikers_data(tmp_path, write_processed=True)

    assert not first.metrics.get("processed_cache_used", False)
    assert second.metrics.get("processed_cache_used", False)
    assert (processed_dir / "transactions_normalised.csv").exists()
    assert len(second.transactions) == len(first.transactions)


def test_real_loader_can_extract_private_archive_when_project_raw_dir_is_empty(tmp_path, monkeypatch):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    archive_path = tmp_path / "strikers-raw-data.zip"
    customer_row = "gigya-test-001,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL\n"

    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("Strikers All Tickets.csv", HEADERLESS_BBL_TICKET + "\n")
        archive.writestr("Strikers All Tickets WBBL.csv", HEADERLESS_BBL_TICKET + "\n")
        archive.writestr("Strikers BBL Fixtures.csv", "Y24-25,Adelaide Oval,Adelaide Strikers,Sydney Sixers,2025-01-15\n")
        archive.writestr("Strikers WBBL Fixtures.csv", "Y24-25,Karen Rolton Oval,Adelaide Strikers,Brisbane Heat,2025-01-19\n")
        archive.writestr("Strikers Customer Data.csv", customer_row)

    monkeypatch.setattr(ingestion, "RAW_DATA_DIR", raw_dir)
    monkeypatch.setattr(ingestion, "BUNDLED_RAW_ARCHIVE", archive_path)

    bundle = load_real_strikers_data(raw_dir, write_processed=False)

    assert (raw_dir / "Strikers All Tickets.csv").exists()
    assert (raw_dir / "Strikers Customer Data.csv").exists()
    assert not bundle.transactions.empty
    assert bundle.loaded_sources["bbl_tickets"].path == raw_dir / "Strikers All Tickets.csv"
    assert bundle.loaded_sources["customers"].path == raw_dir / "Strikers Customer Data.csv"


def test_real_loader_without_uploaded_files_still_builds_assumed_planning_fixtures(tmp_path):
    bundle = load_real_strikers_data(tmp_path, write_processed=False)

    assert bundle.transactions.empty
    assert not bundle.future_fixtures.empty
    assert bundle.fixtures["fixture_status"].eq("assumed").all()
    assert set(bundle.fixtures["competition"].unique()) == {"BBL", "WBBL"}


def test_loader_uses_uploaded_next_season_fixture_assumptions(tmp_path):
    (tmp_path / "Strikers All Tickets.csv").write_text(HEADERLESS_BBL_TICKET + "\n", encoding="utf-8-sig")
    (tmp_path / "Strikers BBL Fixtures.csv").write_text(
        "Y24-25,Adelaide Oval,Adelaide Strikers,Sydney Sixers,2025-01-15\n",
        encoding="utf-8-sig",
    )
    (tmp_path / "Strikers Customer Data.csv").write_text(
        "gigya-test-001,NULL,NULL,NULL,NULL,NULL,NULL,NULL,NULL\n",
        encoding="utf-8-sig",
    )
    (tmp_path / "Client Next Season Fixtures.csv").write_text(
        "competition,season,venue,team,opponent,match_date\n"
        "BBL,2026/27,Adelaide Oval,Adelaide Strikers,Melbourne Stars,2026-12-22\n",
        encoding="utf-8-sig",
    )

    bundle = load_real_strikers_data(tmp_path, write_processed=False)

    assert "future_fixtures" in bundle.loaded_sources
    assert bundle.loaded_sources["future_fixtures"].path == tmp_path / "Client Next Season Fixtures.csv"
    assert not bundle.future_fixtures.empty
    assert bundle.future_fixtures["opponent"].tolist() == ["Melbourne Stars"]
    assert bundle.future_fixtures["fixture_status"].eq("assumed").all()
    assert pd.Timestamp("2026-12-22") in set(pd.to_datetime(bundle.matches["event_date"]))
