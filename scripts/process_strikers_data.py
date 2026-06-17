"""Process private Adelaide Strikers raw CSVs into local normalized outputs."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocessing.strikers_ingestion import PROCESSED_DATA_DIR, RAW_DATA_DIR, load_real_strikers_data


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Validate and normalize Strikers ticket, fixture, and customer CSVs from data/raw/."
    )
    parser.add_argument(
        "--raw-dir",
        default=str(RAW_DATA_DIR),
        help="Directory containing the five private Strikers CSV files.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2026,
        help="Stable random seed for assumed future fixture opponents.",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Validate only; do not write normalized files to data/processed/.",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=None,
        help="Read only the first N rows of each CSV for a quick mapping/validation smoke test.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return a non-zero exit code when critical ticket or fixture data is missing.",
    )
    args = parser.parse_args()

    def progress(message: str) -> None:
        print(f"[process] {message}", flush=True)

    bundle = load_real_strikers_data(
        args.raw_dir,
        random_seed=args.seed,
        write_processed=not args.no_write,
        sample_rows=args.sample_rows,
        progress_callback=progress,
    )

    print("\nDetected files")
    print(bundle.file_status.to_string(index=False))

    print("\nSafe processing summary")
    for label, key in [
        ("Ticket rows", "ticket_rows"),
        ("Fixture rows", "fixture_rows"),
        ("Customer rows", "customer_rows"),
        ("Unique customers", "unique_customers"),
        ("Paid tickets", "paid_ticket_count"),
        ("Comp tickets", "comp_ticket_count"),
        ("Refund / void rows", "refund_count"),
        ("Gross revenue", "gross_revenue"),
        ("Net revenue", "net_revenue"),
        ("Customer join rate", "customer_join_rate"),
        ("Fixture join rate", "fixture_join_rate"),
        ("Date range", "date_range"),
        ("Competitions", "competitions"),
        ("Seasons", "seasons"),
    ]:
        value = bundle.metrics.get(key, "Unavailable")
        if key.endswith("_rate") and isinstance(value, float):
            value = f"{value:.1%}"
        elif isinstance(value, float):
            value = f"{value:,.0f}"
        print(f"- {label}: {value}")

    print("\nValidation warnings")
    if bundle.validation_warnings.empty:
        print("- None")
    else:
        for row in bundle.validation_warnings.itertuples(index=False):
            print(f"- [{row.severity}] {row.area}: {row.message}")

    if not args.no_write:
        print("\nWrote normalized outputs")
        for filename in [
            "transactions_normalised.csv",
            "fixtures_normalised.csv",
            "customers_normalised.csv",
            "matches.csv",
            "daily_sales.csv",
        ]:
            print(f"- {PROCESSED_DATA_DIR / filename}")

    if args.strict and (bundle.transactions.empty or bundle.fixtures.empty):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
