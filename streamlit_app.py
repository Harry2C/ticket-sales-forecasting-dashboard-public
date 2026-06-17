"""Streamlit Community Cloud entrypoint.

Community Cloud launches apps from the repository root. This wrapper sets a
writable runtime data directory before importing the dashboard, so uploaded CSVs
and generated model outputs are not written into the checked-out source tree.
"""

from __future__ import annotations

import os


os.environ.setdefault("TICKET_DASHBOARD_DATA_DIR", "/tmp/ticket-dashboard-data")

from dashboard.app import main


main()
