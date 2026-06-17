# Two Circles Ticket Sales Forecasting Command Centre

A Streamlit ticket-selling decision tool for Two Circles client planning. The command centre starts with a branded client setup screen, ingests uploaded historic ticket sales, historic fixture, customer, and optional next-season fixture CSV snapshots, then trains the forecasting model and opens the dashboard tabs for planning, targets, audience insight, and reporting.

This public deployment copy is sanitized: it contains no CSV, parquet, zip, or spreadsheet data files. Uploaded data is saved only to the app runtime while the Streamlit session/environment remains available.

The fixed planning season is currently `2026/27`. The current sample and private data defaults still use `BBL` and `WBBL`, but the first-run flow is designed as a reusable client upload path.

## What The App Does

- Detects the five real Strikers CSV files from `data/raw/` without committing private data.
- Handles UTF-8 BOM, Excel-style CSVs, headerless exports, duplicate headers, Australian dates, money strings, booleans, blanks, and common Ticketek/Ticketmaster-style column names.
- Uses `GigyaUID`/customer GUID-style IDs as the individual identifier where present, joins demographic data onto ticket rows where possible, and displays anonymised IDs or aggregate audiences by default.
- Normalises BBL and WBBL tickets into one transaction model, fixtures into one fixture model, and customers into one customer model.
- Links ticket rows to fixtures and customers and reports join quality.
- Generates editable `2026/27` assumed future fixtures until official fixtures are available, including a guaranteed BBL NYE fixture on 31 December.
- Provides pages for Fixture Forecasting, Target Breakdown, Audience Insights, Audience & Marketing Planner, Reports, Future Fixture Assumptions, Historic Sales, and Data Admin.
- Opens on a Two Circles branded setup/loading screen where users can upload client data and run ingestion plus model training before entering the dashboard.
- Keeps filters page-scoped so changing a historical QA selection does not change target building or audience planning.
- Keeps navigation inside the same Streamlit app window using internal page tabs.
- Exports the underlying filtered data for charts and tables as CSV.
- Adds an in-memory dummy `2026/27` August sales snapshot from `1 August 2026` to `31 August 2026` when no real `2026/27` ticket sales are present, so Audience Insights and other actual-sales views can respond before official current-season extracts exist.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run The Dashboard

```bash
streamlit run dashboard/app.py --server.port 8502
```

If port `8502` is in use, choose another port:

```bash
streamlit run dashboard/app.py --server.port 8503
```

## Client CSV Workflow

Place the five private CSVs in:

```text
data/raw/
```

Expected file names:

- `Strikers All Tickets.csv`
- `Strikers All Tickets WBBL.csv`
- `Strikers BBL Fixtures.csv`
- `Strikers WBBL Fixtures.csv`
- `Strikers Customer Data.csv`

Close stem variations are also recognised, for example `Strikers All Tickets WBBL` without the `.csv` shown in a file picker.

Do not commit these files. `data/raw/*` and `data/processed/*` are ignored by git, with only `.gitkeep` placeholders retained.

For private-repo collaboration, you can also commit a bundled archive at:

```text
data/bootstrap/strikers-raw-data.zip
```

When the five CSVs are missing from `data/raw/` and that private archive exists, the app extracts the files into `data/raw/` automatically on first run. Keep this archive in private repositories only.

The app describes these files as uploaded snapshots or latest extracts. It is not connected to a true live ticketing API. Upload or replace the CSVs and reprocess when a new ticketing extract is available.

## Client Setup Screen

On first load, the app shows a Two Circles branded setup screen. Use it to upload or reuse:

- Historic ticket sales CSVs.
- Historic fixture CSVs.
- Customer information CSV.
- Optional `Client Next Season Fixtures.csv`.

Click `Ingest data and build model` to save uploaded CSVs into `data/raw/`, normalise the files, join customer records, build forecasting inputs, train the historical pace and ML forecaster, and score next-season fixtures. If the optional next-season fixture file is omitted, the app keeps using generated fixture assumptions.

The local Streamlit upload limit is configured to `250 MB` in `.streamlit/config.toml`.

After a model has been built, the setup screen can be reopened from `Data & Admin > Client Setup`. The app shows a local saved dashboard version whenever processed data exists, and includes the count/timestamp of saved fixture targets when targets have been entered. Click `Open saved dashboard` to return to the planning tabs without re-uploading data.

Process and validate the raw files from the command line with:

```bash
python scripts/process_strikers_data.py
```

That command writes normalized local outputs to `data/processed/`:

- `transactions_normalised.csv`
- `fixtures_normalised.csv`
- `customers_normalised.csv`
- `matches.csv`
- `daily_sales.csv`

When these processed outputs are newer than the raw CSVs, the Streamlit app uses them as a fast startup path. That avoids re-reading and normalizing large raw extracts on every page load. Use `Reprocess real data` in `Data Admin` after replacing raw extracts or changing mapping overrides.

Use validation-only mode when you want the same checks without writing outputs:

```bash
python scripts/process_strikers_data.py --no-write
```

Use a fast smoke test on very large exports with:

```bash
python scripts/process_strikers_data.py --no-write --sample-rows 5000
```

## App Workflow

The navigation is ordered around the clearest jobs:

1. `Fixture Forecasting` shows all assumed `2026/27` fixtures with forecasted sales, 10% uplift, and fixture-specific Base and Stretch target entry. Users can edit every fixture target first, then click `Save all targets` once. Targets are saved locally to `data/processed/fixture_targets.csv` and can be edited later.
2. `Target Breakdown` is a `2026/27` planning/projection page only. It does not edit targets and does not show actual sales. It has two grouped filter sections, then two charts: cumulative sales projection and daily sales projection.
3. `Audience Insights` is the actual-sales tracking page. It uses real uploaded `2026/27` ticket rows where present, otherwise the dummy August snapshot. Charts show Actual `26/27` plus one historical comparison and one selected target.
4. `Audience & Marketing Planner` merges recommended audiences with sales-window context and campaign planning. It shows audience segments, ticket types, ticket classes, and segment x ticket product combinations against expected purchase rates at the same point in the sales cycle.
5. `Reports` creates one-page summaries with fixture risk, audience and ticket-product index metrics, paid/comps separation, snapshot notes, and recommended actions.
6. `Future Fixture Assumptions` edits the generated future fixture set.
7. `Historic Sales` validates historical ticket totals, paid tickets, comps, revenue, and sales curves. It defaults to the most recent historical season and supports filters for season, competition, fixture, paid/comps status, ticket type, ticket class, age/age band, gender, and postcode.
8. `Data Admin` manages file detection, uploads, mapping, validation, and freshness checks.

### Page-Scoped Filters

Each page owns its own Streamlit session-state keys, for example `historic_sales_season`, `target_breakdown_competition`, and `audience_planner_competition`. This prevents filters leaking across workflows. Every major page includes `Reset filters for this page` where there are page-level controls.

Global state is deliberately narrow: page selection, assumed fixture seed, mapping overrides, reprocess token, and the detected uploaded-snapshot status.

### CSV Exports

Charts export the aggregated dataframe used to draw the chart, not a PNG. Tables export the currently filtered table data. Exports are privacy-safe by default and avoid raw customer identifiers unless a page is already deliberately showing a safe aggregated table.

### Performance And Framework Suitability

The app remains a Streamlit implementation. Streamlit is still suitable for this prototype-to-working-dashboard phase because the team can iterate quickly in Python, reuse pandas/forecasting code directly, and run the tool locally with private CSV snapshots.

This iteration reduces rerun cost by caching real CSV ingestion, cleaned UI datasets, model resources, expected-by-now calculations, and assumed-fixture forecast tables. It also prefers normalized `data/processed/` outputs when they are fresher than the raw CSVs, so startup does not need to rebuild the full dataset every time. Expensive target changes now happen behind explicit form actions where practical, especially `Save all targets` in Fixture Forecasting.

The main performance risks are large CSV reloads, repeated transformation of raw files, forecast/model table rebuilds, and Streamlit reruns after every widget interaction. If performance becomes a blocker after heavier real extracts, sensible alternatives are:

- Plotly Dash for tighter callback control while staying in Python.
- React/Next.js with a Python/FastAPI backend for a more responsive production UX.
- Evidence.dev, Observable Framework, or a static/precomputed reporting layer where most analysis can be generated ahead of meetings.

### Demo Data

There is no separate dashboard demo mode in this iteration because the working tool is intended to read uploaded Strikers snapshots. Synthetic/demo data remains in `data/demo/` for development and tests only, and is clearly separate from private `data/raw/` and `data/processed/` files.

If no real `2026/27` ticket sales are found, the app appends an in-memory dummy August snapshot covering `1 August 2026` to `31 August 2026`. This gives Audience Insights and other actual-sales views one month of realistic movement. The dummy rows include BBL/WBBL sales, paid tickets, comps, ticket types, ticket classes, sales channels, matched/unmatched/missing GigyaUIDs, and demographic variation. Real uploaded data is never overwritten by this dummy snapshot, and real `2026/27` ticket rows take precedence when present.

## Data Admin

Use the `Data Admin` page to:

- See detected and missing files.
- Upload any of the expected CSVs into `data/raw/`.
- Preview raw column profiles.
- Review suggested field mappings.
- Apply session-level manual mapping overrides.
- See validation warnings, row counts, date ranges, competitions, seasons, unique customers, fixture join rate, customer join rate, paid tickets, comps, refunds, gross revenue, and net revenue.
- See GigyaUID join status, demographic match rate, demographic field coverage, current snapshot/as-at date, and whether the app is using the dummy August `2026/27` snapshot.
- Download validation, mapping, coverage, file-status, and raw-profile tables as CSV.
- Reprocess real data.

## Column Mapping

The ingestion layer supports both named-column files and the observed headerless Strikers export shape.

Named-column matching recognises likely variants such as:

- `GigyaUID`, `Gigya UID`, `gigya_uid`, `gigyaId`, `gigya_id`, customer Gigya UID, patron GigyaUID, account ID, patron ID, member ID, contact ID, email hash -> `customer_id`
- order date, purchase date, paid date, booking date -> `transaction_date`
- quantity, seats, ticket count -> `tickets_sold`
- gross amount, paid amount, total price, item total -> `gross_revenue`
- event, fixture, match name -> `fixture_name`
- age, customer age, DOB, date of birth -> `age` / `age_band`

Ticket transactions and customer demographic rows are linked on cleaned GigyaUID/customer IDs. IDs are trimmed and matched case-insensitively. Unmatched ticket rows remain in the dataset and appear as `Unknown / unmatched` in demographic filters.

For headerless ticket exports, the positional fallback maps the customer individual key from ticket column 2, which matches `Strikers Customer Data.csv` column 1 in the current files. It falls back to the GUID-like ticket/account column only when column 2 is blank. Fixture exports with five columns are treated as season, venue, team, opponent, and match date.

## Future Fixture Assumptions

The app generates assumed `2026/27` BBL and WBBL fixtures from recent historical fixture timing.

Logic:

- Uses the latest comparable historical home fixture pattern by competition.
- Preserves general time of year and similar day-of-week where practical.
- Forces a BBL fixture on `31 December 2026` and labels it as NYE assumed.
- Assigns random placeholder opponents from historical opponent pools using a stable seed.
- Marks rows as `fixture_status = assumed`.
- Keeps assumed fixtures out of historical training interpretation.

Edit or save assumptions on the `Future Fixture Assumptions` page. When official fixture CSVs are available, place them in `data/raw/` and reprocess real data.

## Forecasting And Targets

The existing forecasting stack is preserved:

- `forecasting/historical_pace.py` builds weighted historical sales pace curves by days to event.
- `forecasting/ml_model.py` trains a random-forest baseline where enough historical snapshots exist and falls back to transparent historical estimates when data is thin.
- `forecasting/targets.py` creates target curves with heavier weights in sales windows such as early sales, pre-sale, campaign bursts, and match week.

Forecasts for assumed fixtures are labelled as assumption-led scenario outputs, not confirmed fixture forecasts.

`Fixture Forecasting` requires fixture-level Base and Stretch target entry. Each fixture starts with model forecast and 10% uplift defaults, but targets are marked `Missing` until saved. Draft edits are made inside one target-entry form, stretch must be greater than or equal to base, and saved values persist locally in `data/processed/fixture_targets.csv` only after `Save all targets` is clicked.

`Target Breakdown` has no season filter and analyses `2026/27` only. It reads Base and Stretch targets from Fixture Forecasting as read-only inputs. The page shows planning lines only: Forecast, Forecast +10%, Base, Stretch, and a historic comparison fixture where a single fixture is selected. When segment filters are applied, the page builds a segment-level projection curve from matching historical sales behaviour. If the selected group has too little history, it transparently falls back to product-level, competition-level, or all-sales curves and warns the user.

`Audience Insights` is where actual `26/27` sales are analysed. It uses the latest uploaded `26/27` extract where present and the dummy August snapshot otherwise. Its charts show a maximum of three lines: Actual `26/27`, one historical comparison, and one target comparison. Segment, ticket type, ticket class, demographic, recommendation, and over/under-indexing tables sit underneath the charts.

## Expected Purchase Rates And Indexing

The `Audience Insights`, `Audience & Marketing Planner`, and `Reports` use a shared expected-by-now calculation.

For each audience segment, ticket type, or ticket class, the app calculates:

- current paid tickets, comps, total tickets, revenue, unique purchasers, average basket size, and average ticket price
- expected paid tickets, total tickets, revenue, and purchasers by now
- expected final contribution
- ticket index, revenue index, purchaser index, and purchase-rate index
- paid-ticket gap and revenue gap
- status: `Ahead`, `On track`, `Watch`, `Behind`, `At risk`, or `Not due yet`

The preferred alignment is days to fixture. For a future fixture, the app asks: at this many days before match day, how much had comparable historical groups usually sold? If fixture-level timing is not sufficient, inferred sales windows and blended historical contribution provide fallback context.

Threshold defaults:

- `Ahead`: index >= 1.10
- `On track`: 0.95 to 1.10
- `Watch`: 0.85 to 0.95
- `Behind`: 0.70 to 0.85
- `At risk`: below 0.70

Late-buying audiences are protected from false churn flags. If a group usually buys in campaign or match-week windows and the current snapshot is still early, the app labels the group `Not due yet` and recommends monitoring or preparing a later-window campaign rather than treating the group as lost.

## Audience Segments And Privacy

The app builds aggregated segments where the data supports them, including returning purchasers, new purchasers, families, 18-30s where age data exists, Gold buyers, Boundary Zone buyers, General Admission buyers, premium buyers, usual early-window buyers, usual late-window buyers, NYE purchasers, WBBL comp recipients, paid-ticket-only groups, and comp-heavy groups.

Demographic coverage is shown in `Data Admin`. If age, family, postcode, or opt-in coverage is low, the app treats those findings as directional. Names, emails, phone numbers, and postal addresses are not displayed. Customer behaviour is surfaced at aggregated segment level by default.

## Sales Windows

Sales windows are now context for audience planning rather than a standalone top-level workflow. The planner infers windows from transaction dates and fixture dates where explicit sales-window files are not available. Common inferred windows include early sales, general sale, campaign window, and match week.

Window context is used to explain whether a segment is due now, not due yet, or should be prioritised in the current window. Manual sales-window configuration can be expanded later if repeatable campaign calendars are required.

## Implemented Now

- Real Strikers data loading from private CSVs.
- Robust CSV loading and file detection.
- Headerless Strikers export positional mapping.
- Unified transactions, fixtures, customers, matches, and daily sales tables.
- Customer anonymisation and `gigyauid` as the individual ID where present.
- GigyaUID demographic linking, unmatched-row retention, match-rate reporting, and age/gender/postcode coverage.
- Fixture/customer join-rate reporting.
- Paid ticket, comp, refund, revenue, ATP, basket-size, capacity, and purchaser KPIs.
- Fixture forecast curves, fixture-specific Base/Stretch target saving, uplift targets, and required run-rate.
- Historic Sales with local product and demographic filters plus CSV exports.
- Target Breakdown for `2026/27`, including grouped planning/segment filters, read-only saved target display, cumulative and daily projection charts, and fallback messaging for low sample sizes.
- Audience Insights actual-sales tracking for `2026/27`, with Actual plus one historical benchmark plus one target line, and useful aggregated tables underneath.
- In-memory dummy `2026/27` August sales data where no real current-season sales are present.
- Historic Sales / Data QA with local filters, historical curves, and CSV exports.
- Audience & Marketing Planner with expected-by-now indexing by audience segment, ticket type, ticket class, and segment x ticket product.
- Marketing recommendations that cite index, gap, usual purchase window, and suggested action.
- Sales-window inference as recommendation context.
- One-page HTML report generation and CSV downloads with audience and ticket-product index metrics.
- Future fixture assumptions with forced NYE.
- Tests for forecasting, target curves, ingestion, schema mapping, GigyaUID demographic linking, fixture target persistence, dummy `2026/27` sales, segment-level projections, fixture matching, future fixture assumptions, page-scoped state keys, expected purchase indexes, not-due-yet logic, stale snapshots, and report content.

## Scaffolded / Next Priorities

- Manual mapping overrides are session-level in Streamlit; persist them to a local config file if repeatable production mappings are needed.
- Campaign annotations and saved sales-window libraries are intentionally light. The main workflow is audience and ticket-product action planning.
- Recommended audiences are aggregate expected-by-now heuristics, not a trained propensity model.
- Report exports currently support HTML and CSV. PowerPoint/PDF export can be added later.
- Forecast accuracy metrics are available in the modelling layer conceptually but need a dedicated backtesting page once enough real historical seasons are loaded.
- Customer data is intentionally privacy-preserving; any future marketable-audience export should be deliberate, permissioned, and clearly labelled.

## Tests

```bash
pytest
```

In restricted sandboxes, pytest may warn that it cannot write `.pytest_cache`; that does not affect test results.

## Project Structure

```text
dashboard/
  app.py                  Streamlit entrypoint and page routing
  branding.py             Brand CSS and header helpers
  config.py               Planning-season and asset constants
  services.py             Dashboard analytics/view-model helpers
forecasting/
  future_fixtures.py      Assumed fixture generation
  historical_pace.py      Historical comparable-match pace engine
  ml_model.py             Replaceable ML forecaster
  season.py               Season aggregation
  targets.py              Target curve generation
preprocessing/
  strikers_ingestion.py   Real-data ingestion and normalisation
  features.py             Feature engineering
utils/
  csv_utils.py            Excel/CSV parsing, profiling, and type helpers
  charting.py             Plotly chart builders
  data_loader.py          Demo CSV/parquet loaders
data/
  demo/                   Legacy synthetic developer/test data, not used by the dashboard
  raw/                    Private real CSVs, ignored by git
  processed/              Local normalised outputs, ignored by git
scripts/
  generate_sample_data.py Legacy sample-data generator
tests/
  test_forecasting.py
  test_strikers_ingestion.py
```
