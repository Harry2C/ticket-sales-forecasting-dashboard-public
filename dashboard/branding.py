"""Branding helpers for the Streamlit dashboard."""

from __future__ import annotations

import base64
from pathlib import Path

import pandas as pd
import streamlit as st

from dashboard.config import (
    AKKURAT_BOLD_PATH,
    AKKURAT_LIGHT_PATH,
    AKKURAT_REGULAR_PATH,
    APP_TITLE,
    ORGANISATION_NAME,
    TEAM_NAME,
    TWO_CIRCLES_FULL_BLACK_PATH,
    TWO_CIRCLES_FULL_COLOR_PATH,
    TWO_CIRCLES_FULL_WHITE_PATH,
)


def _svg_data_uri(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:image/svg+xml;base64,{encoded}"


def _font_data_uri(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
    return f"data:font/ttf;base64,{encoded}"


def inject_css() -> None:
    font_css = f"""
        <style>
        @font-face {{
            font-family: "Akkurat";
            src: url("{_font_data_uri(AKKURAT_LIGHT_PATH)}") format("truetype");
            font-weight: 300;
            font-style: normal;
            font-display: swap;
        }}
        @font-face {{
            font-family: "Akkurat";
            src: url("{_font_data_uri(AKKURAT_REGULAR_PATH)}") format("truetype");
            font-weight: 400;
            font-style: normal;
            font-display: swap;
        }}
        @font-face {{
            font-family: "Akkurat";
            src: url("{_font_data_uri(AKKURAT_BOLD_PATH)}") format("truetype");
            font-weight: 700;
            font-style: normal;
            font-display: swap;
        }}
        </style>
        """
    st.markdown(
        font_css
        + """
        <style>
        :root {
            --ink: #101114;
            --ink-soft: #5d626a;
            --paper: #f4f5f2;
            --panel: rgba(255, 255, 255, 0.92);
            --line: rgba(16, 17, 20, 0.12);
            --navy: #00677f;
            --gold: #19b8d8;
            --red: #101114;
            --teal: #14a8c6;
            --success: #00886f;
            --warn: #b76200;
            --tc-blue: #10b8db;
            --tc-blue-deep: #00677f;
            --tc-black: #101114;
            --tc-white: #ffffff;
        }
        .stApp {
            background:
                linear-gradient(135deg, rgba(16,17,20,0.04) 0 18%, transparent 18% 100%),
                repeating-linear-gradient(90deg, rgba(16,17,20,0.035) 0 1px, transparent 1px 56px),
                linear-gradient(180deg, #ffffff 0%, #f5f7f7 58%, #eef8fb 100%);
            color: var(--ink);
            font-family: "Akkurat", "Helvetica Neue", sans-serif;
            font-weight: 300;
        }
        .stApp p,
        .stApp div,
        .stApp span,
        .stApp label,
        .stApp input,
        .stApp textarea,
        .stApp select {
            font-family: "Akkurat", "Helvetica Neue", sans-serif;
        }
        .stApp strong,
        .stApp b,
        .stApp button,
        h1, h2, h3, h4 {
            font-family: "Akkurat", "Helvetica Neue", sans-serif;
            font-weight: 700;
        }
        .block-container {
            max-width: none;
            width: 100%;
            padding-top: 0.9rem;
            padding-left: 1.35rem;
            padding-right: 1.35rem;
            padding-bottom: 3rem;
        }
        [data-testid="stSidebar"] {
            display: none;
        }
        [data-testid="collapsedControl"] {
            display: none;
        }
        h1, h2, h3, h4 {
            letter-spacing: 0;
            color: var(--ink);
        }
        .brand-shell {
            position: relative;
            overflow: hidden;
            border: 1px solid var(--line);
            border-radius: 8px;
            padding: 18px 22px 16px;
            background:
                linear-gradient(90deg, rgba(16,17,20,0.98) 0%, rgba(16,17,20,0.96) 58%, rgba(0,103,127,0.94) 100%);
            box-shadow: 0 18px 42px rgba(16, 17, 20, 0.16);
            color: white;
        }
        .brand-shell::after {
            content: "";
            position: absolute;
            inset: auto -8% -42% auto;
            width: min(44vw, 520px);
            aspect-ratio: 1;
            border: 18px solid rgba(16,184,219,0.22);
            border-radius: 999px;
        }
        .brand-grid {
            display: grid;
            grid-template-columns: minmax(0, 1fr) auto;
            gap: 18px;
            position: relative;
            z-index: 1;
            align-items: center;
        }
        .brand-kicker {
            color: rgba(255,255,255,0.72);
            font-size: 0.78rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0;
            margin-bottom: 0.55rem;
        }
        .brand-title {
            font-family: "Akkurat", "Helvetica Neue", sans-serif;
            font-size: clamp(2.05rem, 3vw, 3.2rem);
            font-weight: 800;
            line-height: 1.02;
            margin: 0;
            color: #fff;
        }
        .brand-pills {
            display: flex;
            flex-wrap: wrap;
            gap: 0.6rem;
            margin-top: 0.85rem;
        }
        .brand-pill {
            display: inline-flex;
            align-items: center;
            gap: 0.45rem;
            border-radius: 8px;
            border: 1px solid rgba(255, 255, 255, 0.18);
            background: rgba(255,255,255,0.10);
            padding: 0.48rem 0.82rem;
            font-size: 0.82rem;
            color: rgba(255,255,255,0.88);
        }
        .logo-row {
            display: flex;
            align-items: center;
            gap: 0.7rem;
            flex-wrap: wrap;
            justify-content: flex-end;
        }
        .logo-card,
        .tc-wordmark-card {
            display: flex;
            align-items: center;
            justify-content: center;
            min-height: 104px;
            min-width: 218px;
            border-radius: 8px;
            background: rgba(255,255,255,0.96);
            border: 1px solid rgba(255,255,255,0.18);
            padding: 0.9rem 1rem;
        }
        .logo-card img {
            max-height: 48px;
            width: auto;
            display: block;
        }
        .tc-logo-wrap {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 100%;
        }
        .tc-logo-img {
            display: block;
            width: min(100%, 198px);
            height: auto;
        }
        .section-nav-shell,
        .page-nav-shell {
            margin: 12px 0 0;
        }
        .page-nav-shell {
            margin-bottom: 18px;
            border-bottom: 1px solid rgba(15,23,40,0.12);
        }
        .nav-caption {
            color: var(--ink-soft);
            font-size: 0.72rem;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0;
            margin-bottom: 0.45rem;
        }
        .section-nav-shell .stButton,
        .page-nav-shell .stButton {
            width: 100%;
        }
        .section-nav-shell .stButton > button,
        .page-nav-shell .stButton > button {
            width: 100%;
            min-height: 48px;
            justify-content: flex-start;
            text-align: left;
            font-weight: 780;
            font-size: 0.97rem;
            padding: 0.2rem 0.92rem;
            white-space: normal;
            line-height: 1.2;
            transition: background 120ms ease, border-color 120ms ease, box-shadow 120ms ease, transform 120ms ease;
        }
        .section-nav-shell .stButton > button[kind="secondary"] {
            border: 1px solid rgba(15,23,40,0.12);
            border-radius: 10px;
            background: rgba(255,255,255,0.74);
            color: var(--ink);
            box-shadow: 0 10px 24px rgba(15, 23, 40, 0.04);
        }
        .section-nav-shell .stButton > button[kind="secondary"]:hover {
            border-color: rgba(0,95,158,0.28);
            background: rgba(255,255,255,0.94);
        }
        .section-nav-shell .stButton > button[kind="primary"] {
            border: 1px solid rgba(0,95,158,0.30);
            border-radius: 10px;
            background: linear-gradient(180deg, rgba(255,255,255,0.98), rgba(16,184,219,0.14));
            color: var(--ink);
            box-shadow: inset 0 -3px 0 var(--gold);
        }
        .page-nav-shell .stButton > button[kind="secondary"] {
            border: 1px solid rgba(15,23,40,0.12);
            border-bottom: 0;
            border-radius: 10px 10px 0 0;
            background: rgba(255,255,255,0.72);
            color: var(--ink);
        }
        .page-nav-shell .stButton > button[kind="secondary"]:hover {
            border-color: rgba(0,95,158,0.34);
            background: rgba(255,255,255,0.92);
        }
        .page-nav-shell .stButton > button[kind="primary"] {
            border: 1px solid rgba(0,95,158,0.42);
            border-bottom: 0;
            border-radius: 10px 10px 0 0;
            background: #ffffff;
            color: var(--ink);
            box-shadow: inset 0 -3px 0 var(--red);
            transform: translateY(1px);
        }
        .filter-band {
            border: 1px solid var(--line);
            border-radius: 8px;
            background: rgba(255,255,255,0.72);
            padding: 10px 12px 6px;
            margin: 8px 0 12px;
        }
        .filter-band [data-testid="stWidgetLabel"] p {
            font-size: 0.82rem;
            font-weight: 760;
        }
        .filter-band [data-testid="stExpander"] {
            margin-top: 0.4rem;
        }
        .filter-band [data-testid="stExpander"] details {
            border: 1px solid var(--line);
            border-radius: 8px;
            background: rgba(255,255,255,0.64);
        }
        .filter-band [data-testid="stExpander"] summary p {
            font-size: 0.84rem;
            font-weight: 780;
        }
        .alert-strip {
            margin: 14px 0 8px;
            border-radius: 8px;
            border: 1px solid rgba(15, 23, 40, 0.10);
            padding: 13px 16px;
            font-weight: 680;
            background: linear-gradient(90deg, rgba(0, 95, 158, 0.08), rgba(201, 20, 27, 0.08));
            color: var(--ink);
        }
        .date-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 12px;
            margin: 12px 0 8px;
        }
        .mini-tile {
            border: 1px solid var(--line);
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.84);
            padding: 14px 15px 12px;
            min-height: 88px;
        }
        .mini-label {
            color: var(--ink-soft);
            font-size: 0.76rem;
            text-transform: uppercase;
            letter-spacing: 0;
            font-weight: 760;
        }
        .mini-value {
            margin-top: 8px;
            color: var(--ink);
            font-size: 1.08rem;
            font-weight: 780;
        }
        .metric-card {
            border: 1px solid var(--line);
            border-radius: 8px;
            background: rgba(255, 255, 255, 0.88);
            padding: 15px 16px 13px;
            min-height: 104px;
        }
        .metric-label {
            color: var(--ink-soft);
            font-size: 0.76rem;
            text-transform: uppercase;
            letter-spacing: 0;
            font-weight: 760;
        }
        .metric-value {
            color: var(--ink);
            font-size: 1.95rem;
            font-weight: 860;
            line-height: 1.08;
            margin-top: 8px;
        }
        .metric-note {
            color: var(--ink-soft);
            font-size: 0.88rem;
            margin-top: 5px;
        }
        .section-kicker {
            margin: 16px 0 4px;
            color: var(--ink);
            font-size: 1.06rem;
            font-weight: 840;
        }
        .section-note {
            color: var(--ink-soft);
            font-size: 0.88rem;
            margin-bottom: 6px;
        }
        div[data-testid="stTabs"] [data-baseweb="tab-list"] {
            gap: 0.5rem;
            border-bottom: 1px solid rgba(15,23,40,0.12);
        }
        div[data-testid="stTabs"] [data-baseweb="tab"] {
            border: 1px solid rgba(15,23,40,0.12);
            border-bottom: 0;
            border-radius: 8px 8px 0 0;
            background: rgba(255,255,255,0.70);
            min-height: 44px;
            padding: 0.12rem 0.84rem;
            color: var(--ink);
            font-weight: 780;
            transition: background 120ms ease, border-color 120ms ease;
        }
        div[data-testid="stTabs"] [data-baseweb="tab"]:hover {
            border-color: rgba(0,95,158,0.34);
            background: rgba(255,255,255,0.90);
        }
        div[data-testid="stTabs"] [aria-selected="true"] {
            background: #ffffff;
            border-color: rgba(0,95,158,0.42);
            box-shadow: inset 0 -3px 0 var(--red);
        }
        div[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
            display: none;
        }
        [data-testid="stMetric"] {
            border: 1px solid var(--line);
            border-radius: 8px;
            background: rgba(255,255,255,0.88);
            padding: 12px;
        }
        .tc-setup-hero {
            position: relative;
            overflow: hidden;
            min-height: 430px;
            border-radius: 8px;
            border: 1px solid rgba(16,17,20,0.14);
            background:
                linear-gradient(90deg, rgba(16,17,20,0.96) 0%, rgba(16,17,20,0.94) 52%, rgba(0,103,127,0.92) 100%),
                repeating-linear-gradient(0deg, rgba(255,255,255,0.08) 0 1px, transparent 1px 42px);
            color: white;
            padding: clamp(1.4rem, 3vw, 3rem);
            display: grid;
            grid-template-columns: minmax(0, 1.1fr) minmax(260px, 0.74fr);
            gap: 2rem;
            align-items: center;
            box-shadow: 0 24px 60px rgba(16,17,20,0.18);
        }
        .tc-setup-hero::before,
        .tc-setup-hero::after {
            content: "";
            position: absolute;
            border-radius: 999px;
            pointer-events: none;
        }
        .tc-setup-hero::before {
            inset: -16% auto auto 54%;
            width: min(42vw, 520px);
            aspect-ratio: 1;
            border: 18px solid rgba(16,184,219,0.30);
            animation: tc-ring-pulse 4.8s ease-in-out infinite;
        }
        .tc-setup-hero::after {
            inset: auto 7% -34% auto;
            width: min(28vw, 360px);
            aspect-ratio: 1;
            border: 13px solid rgba(255,255,255,0.16);
            animation: tc-ring-pulse 4.8s ease-in-out infinite reverse;
        }
        .tc-setup-content {
            position: relative;
            z-index: 1;
        }
        .tc-setup-kicker {
            font-size: 0.84rem;
            font-weight: 850;
            text-transform: uppercase;
            letter-spacing: 0;
            color: rgba(255,255,255,0.72);
            margin-bottom: 0.85rem;
        }
        .tc-setup-title {
            margin: 0;
            max-width: 860px;
            color: #fff;
            font-family: "Akkurat", "Helvetica Neue", sans-serif;
            font-size: clamp(2.75rem, 5vw, 5.8rem);
            font-weight: 800;
            line-height: 0.94;
            text-transform: uppercase;
            letter-spacing: 0;
        }
        .tc-setup-copy {
            max-width: 680px;
            color: rgba(255,255,255,0.78);
            font-size: 1.04rem;
            line-height: 1.55;
            margin: 1.15rem 0 0;
        }
        .tc-loader-card {
            position: relative;
            z-index: 1;
            border-radius: 8px;
            border: 1px solid rgba(255,255,255,0.18);
            background: rgba(255,255,255,0.94);
            color: var(--tc-black);
            padding: 1.5rem;
            overflow: hidden;
        }
        .tc-loader-card::after {
            content: "";
            position: absolute;
            inset: 0;
            background: linear-gradient(105deg, transparent 0 34%, rgba(16,184,219,0.24) 44%, transparent 54% 100%);
            transform: translateX(-100%);
            animation: tc-scan 2.6s ease-in-out infinite;
        }
        .tc-loader-stage {
            position: relative;
            z-index: 1;
            display: grid;
            place-items: center;
            min-height: 260px;
            border: 1px solid rgba(16,17,20,0.10);
            border-radius: 8px;
            background:
                radial-gradient(circle at center, rgba(16,184,219,0.12), transparent 50%),
                #ffffff;
        }
        .tc-loader-rings {
            position: absolute;
            width: 188px;
            aspect-ratio: 1;
            border-radius: 999px;
            border: 2px solid rgba(16,17,20,0.12);
            animation: tc-loader-spin 8s linear infinite;
        }
        .tc-loader-rings::before,
        .tc-loader-rings::after {
            content: "";
            position: absolute;
            border-radius: 999px;
            border: 2px solid rgba(16,184,219,0.58);
        }
        .tc-loader-rings::before {
            inset: 26px;
            border-left-color: transparent;
        }
        .tc-loader-rings::after {
            inset: -22px;
            border-right-color: transparent;
            border-bottom-color: rgba(16,17,20,0.18);
        }
        .tc-loader-wordmark {
            position: relative;
            z-index: 2;
            transform: scale(1.08);
            animation: tc-logo-breathe 3.2s ease-in-out infinite;
        }
        .tc-loader-logo .tc-logo-img {
            width: min(72%, 226px);
        }
        .tc-setup-metrics {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.7rem;
            margin-top: 1.25rem;
        }
        .tc-setup-metric {
            border: 1px solid rgba(255,255,255,0.16);
            border-radius: 8px;
            background: rgba(255,255,255,0.09);
            padding: 0.85rem 0.9rem;
        }
        .tc-setup-metric strong {
            display: block;
            font-size: 1.35rem;
            line-height: 1.05;
            color: #fff;
        }
        .tc-setup-metric span {
            display: block;
            margin-top: 0.28rem;
            color: rgba(255,255,255,0.72);
            font-size: 0.8rem;
        }
        .tc-upload-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: 1rem;
            margin-top: 1rem;
        }
        .tc-setup-panel {
            border: 1px solid rgba(16,17,20,0.12);
            border-radius: 8px;
            background: rgba(255,255,255,0.90);
            padding: 1rem;
        }
        .tc-setup-panel h3 {
            margin: 0 0 0.45rem;
            font-size: 1.05rem;
            letter-spacing: 0;
        }
        .tc-setup-panel p {
            color: var(--ink-soft);
            margin: 0 0 0.9rem;
            font-size: 0.9rem;
        }
        .tc-pipeline-strip {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 0.75rem;
            margin: 1rem 0 0.5rem;
        }
        .tc-pipeline-step {
            border: 1px solid rgba(16,17,20,0.12);
            border-radius: 8px;
            background: #fff;
            padding: 0.85rem 0.9rem;
            min-height: 86px;
        }
        .tc-pipeline-step strong {
            display: block;
            color: var(--ink);
            font-size: 0.92rem;
        }
        .tc-pipeline-step span {
            display: block;
            color: var(--ink-soft);
            font-size: 0.8rem;
            margin-top: 0.28rem;
        }
        @keyframes tc-scan {
            0%, 42% { transform: translateX(-105%); }
            72%, 100% { transform: translateX(105%); }
        }
        @keyframes tc-loader-spin {
            to { transform: rotate(360deg); }
        }
        @keyframes tc-logo-breathe {
            0%, 100% { transform: scale(1.02); filter: drop-shadow(0 8px 18px rgba(16,17,20,0.10)); }
            50% { transform: scale(1.08); filter: drop-shadow(0 14px 28px rgba(16,184,219,0.22)); }
        }
        @keyframes tc-ring-pulse {
            0%, 100% { transform: scale(1); opacity: 0.72; }
            50% { transform: scale(1.05); opacity: 0.42; }
        }
        @media (max-width: 980px) {
            .brand-grid {
                grid-template-columns: 1fr;
            }
            .logo-row {
                justify-content: flex-start;
            }
            .date-grid {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
            .tc-setup-hero,
            .tc-upload-grid,
            .tc-pipeline-strip {
                grid-template-columns: 1fr;
            }
        }
        @media (max-width: 620px) {
            .date-grid {
                grid-template-columns: 1fr;
            }
            .brand-shell {
                padding: 18px;
            }
            .brand-title {
                font-size: 2.05rem;
            }
            .tc-setup-hero {
                min-height: 0;
            }
            .tc-setup-metrics {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def metric_tile(label: str, value: str, note: str) -> str:
    return f"""
    <div class="metric-card">
      <div class="metric-label">{label}</div>
      <div class="metric-value">{value}</div>
      <div class="metric-note">{note}</div>
    </div>
    """


def mini_tile(label: str, value: str) -> str:
    return f"""
    <div class="mini-tile">
      <div class="mini-label">{label}</div>
      <div class="mini-value">{value}</div>
    </div>
    """


def two_circles_wordmark(extra_class: str = "", variant: str = "black") -> str:
    logo_path = {
        "white": TWO_CIRCLES_FULL_WHITE_PATH,
        "color": TWO_CIRCLES_FULL_COLOR_PATH,
        "black": TWO_CIRCLES_FULL_BLACK_PATH,
    }.get(variant, TWO_CIRCLES_FULL_BLACK_PATH)
    class_attr = f"tc-logo-wrap {extra_class}".strip()
    return f"""
    <span class="{class_attr}" aria-label="Two Circles">
      <img class="tc-logo-img" src="{_svg_data_uri(logo_path)}" alt="Two Circles" />
    </span>
    """


def render_brand_header(
    saca_logo_path: Path,
    strikers_logo_path: Path,
    planning_season_label: str,
    competition: str,
    snapshot_date: pd.Timestamp,
    home_match_count: int,
    total_matches: int,
) -> None:
    st.markdown(
        f"""
        <div class="brand-shell">
          <div class="brand-grid">
            <div>
              <div class="brand-kicker">{ORGANISATION_NAME} | {TEAM_NAME} | KNOW FANS BEST</div>
              <h1 class="brand-title">{APP_TITLE}</h1>
              <div class="brand-pills">
                <div class="brand-pill"><strong>Planning season</strong> {planning_season_label}</div>
                <div class="brand-pill"><strong>Competition</strong> {competition}</div>
                <div class="brand-pill"><strong>Snapshot</strong> {snapshot_date.strftime('%d %b %Y')}</div>
                <div class="brand-pill"><strong>Season model</strong> {home_match_count} of {total_matches} home matches</div>
              </div>
            </div>
            <div class="logo-row">
              <div class="tc-wordmark-card">{two_circles_wordmark()}</div>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_alert_strip(message: str) -> None:
    st.markdown(f'<div class="alert-strip">{message}</div>', unsafe_allow_html=True)
