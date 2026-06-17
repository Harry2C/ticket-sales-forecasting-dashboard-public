# Free Hosted Deployment

This app is prepared for Streamlit Community Cloud, which is the simplest free
option for a shareable Streamlit dashboard with file uploads.

This public deployment copy intentionally contains no CSV, parquet, zip, or
spreadsheet data files. Users upload their own files at runtime.

## Deploy On Streamlit Community Cloud

1. Go to https://share.streamlit.io/deploy.
2. Connect the GitHub account that can access this repository.
3. Select repository `Harry2C/ticket-sales-forecasting-dashboard-public`.
4. Select branch `main`.
5. Set the main file path to `streamlit_app.py`.
6. Deploy the app and use the generated `streamlit.app` URL as the shareable link.

## Runtime Notes

- Uploaded CSVs and generated model outputs are written to `/tmp/ticket-dashboard-data`
  when the app is launched through `streamlit_app.py`.
- The free Community Cloud filesystem should be treated as runtime storage, not
  permanent storage. A redeploy or app restart can reset uploaded data and saved
  targets.
- A public free app should be used with demo or non-confidential data only.
  Make the app private, or add authenticated user-specific storage, before
  uploading sensitive client data.
- For durable private client data, use a paid host with persistent disk or add an
  external store such as S3, Snowflake, or a database.
- The upload limit is configured in `.streamlit/config.toml` as `250 MB`.
- Python is pinned in `runtime.txt` so Streamlit Cloud uses a version compatible
  with the pinned analytics dependencies.

## Local Equivalent

The existing local command still uses the repository `data/` folder:

```bash
streamlit run dashboard/app.py --server.port 8502
```

To mimic hosted runtime storage locally:

```bash
streamlit run streamlit_app.py --server.port 8502
```
