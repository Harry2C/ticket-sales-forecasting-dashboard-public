"""Robust CSV helpers for Excel-style Strikers exports."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
import re
from typing import Iterable
import warnings

import pandas as pd


COMMON_ENCODINGS = ("utf-8-sig", "utf-8", "cp1252", "latin1")
NULL_TOKENS = {"", "null", "none", "nan", "nat", "#n/a", "n/a", "-"}


@dataclass(frozen=True)
class CsvLoadResult:
    path: Path
    frame: pd.DataFrame
    encoding: str
    header_present: bool
    raw_columns: list[str]
    cleaned_columns: list[str]
    warnings: list[str]
    profile: pd.DataFrame


def normalise_token(value: object) -> str:
    """Return a stable lowercase token for fuzzy filename and column matching."""

    text = "" if value is None else str(value)
    text = text.replace("\ufeff", "").strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    return re.sub(r"_+", "_", text).strip("_")


def clean_column_names(columns: Iterable[object]) -> list[str]:
    """Clean headers and suffix duplicates while preserving column order."""

    seen: dict[str, int] = {}
    cleaned: list[str] = []
    for idx, column in enumerate(columns, start=1):
        token = normalise_token(column) or f"column_{idx:03d}"
        seen[token] = seen.get(token, 0) + 1
        cleaned.append(token if seen[token] == 1 else f"{token}_{seen[token]}")
    return cleaned


def clean_text(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).replace("\ufeff", "").strip()
    if text.lower() in NULL_TOKENS:
        return None
    return text


def anonymise_identifier(value: object, salt: str = "adelaide-strikers") -> str | None:
    text = clean_text(value)
    if not text:
        return None
    return sha256(f"{salt}:{text}".encode("utf-8")).hexdigest()[:16]


def parse_number_series(series: pd.Series | None, default: float = 0.0) -> pd.Series:
    """Parse Australian/Excel currency, quantity, and percentage-like strings."""

    if series is None:
        return pd.Series(dtype=float)
    text = series.astype(str).str.strip()
    negative = text.str.match(r"^\(.*\)$", na=False)
    text = text.str.replace(r"[\$,]", "", regex=True)
    text = text.str.replace("%", "", regex=False)
    text = text.str.replace("(", "-", regex=False).str.replace(")", "", regex=False)
    text = text.mask(text.str.lower().isin(NULL_TOKENS), "")
    values = pd.to_numeric(text, errors="coerce")
    values = values.mask(negative & values.gt(0), -values)
    return values.fillna(default)


def parse_bool_series(series: pd.Series | None, default: bool = False) -> pd.Series:
    if series is None:
        return pd.Series(dtype=bool)
    text = series.astype(str).str.strip().str.lower()
    truthy = {"true", "t", "yes", "y", "1", "opt in", "opt-in", "subscribed"}
    falsey = {"false", "f", "no", "n", "0", "opt out", "opt-out", "unsubscribed"}
    parsed = pd.Series(default, index=series.index, dtype=bool)
    parsed.loc[text.isin(truthy)] = True
    parsed.loc[text.isin(falsey)] = False
    return parsed


def parse_date_series(series: pd.Series | None, dayfirst: bool = True) -> pd.Series:
    """Parse mixed Excel/Australian date strings without raising."""

    if series is None:
        return pd.Series(dtype="datetime64[ns]")
    cleaned = series.astype(str).str.strip().replace({token: None for token in NULL_TOKENS})
    parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")

    iso_mask = cleaned.fillna("").str.match(r"^\d{4}-\d{1,2}-\d{1,2}(?:\D|$)")
    if iso_mask.any():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            parsed.loc[iso_mask] = pd.to_datetime(cleaned.loc[iso_mask], errors="coerce", dayfirst=False)

    remaining = parsed.isna() & cleaned.notna()
    if remaining.any():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            parsed.loc[remaining] = pd.to_datetime(cleaned.loc[remaining], errors="coerce", dayfirst=dayfirst)

    missing = parsed.isna() & cleaned.notna()
    if missing.any():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            fallback = pd.to_datetime(cleaned[missing], errors="coerce", dayfirst=not dayfirst)
        parsed.loc[missing] = fallback
    return parsed


def extract_date_from_text(series: pd.Series | None) -> pd.Series:
    if series is None:
        return pd.Series(dtype="datetime64[ns]")
    text = series.astype(str)
    patterns = [
        r"(\d{1,2}/\d{1,2}/\d{2,4})",
        r"(\d{4}-\d{1,2}-\d{1,2})",
        r"(\d{1,2}-\d{1,2}-\d{2,4})",
    ]
    extracted = pd.Series(index=series.index, dtype=object)
    for pattern in patterns:
        mask = extracted.isna()
        found = text[mask].str.extract(pattern, expand=False)
        extracted.loc[mask] = found
    return parse_date_series(extracted)


def read_csv_robust(path: str | Path, nrows: int | None = None, profile_rows: int = 5_000) -> CsvLoadResult:
    """Read CSVs exported from Excel/Ticketek with encoding and header fallbacks."""

    csv_path = Path(path)
    warnings: list[str] = []
    last_error: Exception | None = None

    for encoding in COMMON_ENCODINGS:
        try:
            sample = pd.read_csv(
                csv_path,
                encoding=encoding,
                header=None,
                nrows=8,
                dtype=str,
                keep_default_na=False,
                on_bad_lines="warn",
            )
            header_present = _looks_like_header(sample.iloc[0].tolist() if not sample.empty else [])
            read_kwargs = {
                "encoding": encoding,
                "dtype": str,
                "keep_default_na": False,
                "on_bad_lines": "warn",
                "low_memory": False,
            }
            if nrows is not None:
                read_kwargs["nrows"] = nrows
            if header_present:
                frame = pd.read_csv(csv_path, header=0, **read_kwargs)
                raw_columns = [str(column).replace("\ufeff", "").strip() for column in frame.columns]
                frame.columns = clean_column_names(raw_columns)
            else:
                frame = pd.read_csv(csv_path, header=None, **read_kwargs)
                raw_columns = [f"column_{idx:03d}" for idx in range(1, len(frame.columns) + 1)]
                frame.columns = raw_columns
                warnings.append("No header row detected; positional Strikers export mapping will be used.")

            frame = _drop_blank_rows(frame)
            cleaned_columns = list(frame.columns)
            profile = profile_columns(frame, sample_rows=profile_rows)
            return CsvLoadResult(csv_path, frame, encoding, header_present, raw_columns, cleaned_columns, warnings, profile)
        except Exception as exc:  # pragma: no cover - exercised by fallback encodings in real exports.
            last_error = exc

    raise ValueError(f"Could not read CSV {csv_path}: {last_error}")


def profile_columns(frame: pd.DataFrame, max_examples: int = 3, sample_rows: int = 5_000) -> pd.DataFrame:
    sample_frame = frame.head(sample_rows)
    rows: list[dict[str, object]] = []
    for column in sample_frame.columns:
        series = sample_frame[column].map(clean_text)
        non_null = series.dropna()
        examples = non_null.astype(str).head(max_examples).tolist()
        rows.append(
            {
                "column": column,
                "non_blank": int(non_null.shape[0]),
                "blank": int(series.isna().sum()),
                "example_values": ", ".join(value[:48] for value in examples),
                "inferred_type": infer_column_type(non_null),
            }
        )
    return pd.DataFrame(rows)


def infer_column_type(series: pd.Series) -> str:
    if series.empty:
        return "empty"
    sample = series.astype(str).head(100)
    numeric_rate = pd.to_numeric(
        sample.str.replace(r"[\$,()%]", "", regex=True).str.replace(",", "", regex=False),
        errors="coerce",
    ).notna().mean()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        date_rate = pd.to_datetime(sample, errors="coerce", dayfirst=True).notna().mean()
    if date_rate >= 0.75:
        return "date"
    if numeric_rate >= 0.85:
        return "number"
    if sample.str.lower().isin({"yes", "no", "y", "n", "true", "false", "1", "0"}).mean() >= 0.85:
        return "boolean"
    return "text"


def _drop_blank_rows(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = frame.replace(r"^\s*$", pd.NA, regex=True)
    return cleaned.dropna(how="all").reset_index(drop=True)


def _looks_like_header(values: list[object]) -> bool:
    if not values:
        return False
    tokens = [normalise_token(value) for value in values if normalise_token(value)]
    if not tokens:
        return False

    header_terms = {
        "account",
        "age",
        "amount",
        "channel",
        "class",
        "client",
        "contact",
        "customer",
        "date",
        "email",
        "event",
        "fixture",
        "gender",
        "id",
        "match",
        "member",
        "name",
        "opponent",
        "order",
        "paid",
        "postcode",
        "price",
        "quantity",
        "revenue",
        "season",
        "section",
        "ticket",
        "venue",
    }
    term_hits = sum(any(term in token.split("_") or token.endswith(f"_{term}") for term in header_terms) for token in tokens)
    short_label_rate = sum(len(token) <= 32 and not re.search(r"\d{4}|\d{1,2}_\d{1,2}", token) for token in tokens) / len(tokens)
    return term_hits >= 2 and short_label_rate >= 0.6
