# Data Layout

Sample data is committed under `data/sample` so the dashboard runs from a clean clone.

This iteration ships with Strikers-specific sample data:

- planning season: `2026/27`
- historical seasons: `2023/24`, `2024/25`, `2025/26`
- competitions: `BBL`, `WBBL`

Supported production inputs:

- `matches.csv` or `matches.parquet`
- `daily_sales.csv` or `daily_sales.parquet`

Important match-level fields now include:

- `season_label`
- `competition`
- `early_bird_start`
- `early_bird_end`
- `member_presale_start`
- `member_presale_end`
- `general_public_start`

Important daily fields include:

- `sales_window`
- `early_bird_active`
- `membership_on_sale`
- `school_holiday`

The loader reads CSV by default for quick local startup and supports parquet by setting `TICKET_DASHBOARD_PREFER_PARQUET=1`. Drop raw exports into `data/raw/` and write cleaned modelling-ready files to `data/processed/`; both directories are ignored by git to keep large private datasets out of the repository.
## Data folders

- `data/demo/` contains synthetic demo data that is safe to commit and lets the dashboard run from a clean clone.
- `data/raw/` is for the five private Strikers CSV exports. It is ignored by git except for a placeholder.
- `data/processed/` is for locally generated normalized copies and saved planning assumptions. It is ignored by git except for a placeholder.
- `data/sample/` is retained as a legacy fallback for older prototype data.

Do not commit real SACA, Strikers, Ticketek/Ticketmaster, or customer-level exports.
