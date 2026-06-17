"""Dashboard-wide configuration and brand constants."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_DATA_DIR = Path(os.environ.get("TICKET_DASHBOARD_DATA_DIR", PROJECT_ROOT / "data")).expanduser()
ASSETS_DIR = PROJECT_ROOT / "dashboard" / "assets"

ORGANISATION_NAME = "Two Circles"
ORGANISATION_SHORT = "Two Circles"
TEAM_NAME = "Client Growth"
APP_TITLE = "Ticket Sales Forecasting Command Centre"
PLANNING_SEASON = 2026
PLANNING_SEASON_LABEL = "2026/27"
COMPETITION_OPTIONS = ("BBL", "WBBL")
DEFAULT_HISTORICAL_SEASONS = ("2023/24", "2024/25", "2025/26")

SACA_LOGO_PATH = ASSETS_DIR / "saca.svg"
STRIKERS_LOGO_PATH = ASSETS_DIR / "adelaide-strikers.svg"
AKKURAT_LIGHT_PATH = ASSETS_DIR / "fonts" / "Akkurat-Light.ttf"
AKKURAT_REGULAR_PATH = ASSETS_DIR / "fonts" / "Akkurat.ttf"
AKKURAT_BOLD_PATH = ASSETS_DIR / "fonts" / "Akkurat-Bold.ttf"
TWO_CIRCLES_FULL_WHITE_PATH = ASSETS_DIR / "logos" / "two-circles-full-white.svg"
TWO_CIRCLES_FULL_BLACK_PATH = ASSETS_DIR / "logos" / "two-circles-full-black.svg"
TWO_CIRCLES_FULL_COLOR_PATH = ASSETS_DIR / "logos" / "two-circles-full-color.svg"
TWO_CIRCLES_SYMBOL_WHITE_PATH = ASSETS_DIR / "logos" / "two-circles-symbol-white.svg"
TWO_CIRCLES_SYMBOL_NAVY_PATH = ASSETS_DIR / "logos" / "two-circles-symbol-navy.svg"
