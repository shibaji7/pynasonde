"""Canonical echo DataFrame schema for Digisonde ionogram scaling.

The Digisonde binary parsers expose format-specific column names.  RSF/SBF
tables use raw DPS-style names such as ``frequency_reading`` and ``height``;
MMM tables use ``frequency_mhz`` and ``range_km``.  The scaler algorithms should
not care which file format produced the table, so this module normalizes those
parser outputs to one internal schema.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

CANONICAL_COLUMNS = (
    "time",
    "frequency_mhz",
    "height_km",
    "amplitude_db",
    "mode",
    "mpa_db",
    "doppler_bin",
    "doppler_hz",
    "phase",
    "azimuth",
    "source_format",
)

REQUIRED_CANONICAL_COLUMNS = (
    "frequency_mhz",
    "height_km",
    "amplitude_db",
    "mode",
)


@dataclass(frozen=True)
class EchoSchema:
    """Column mapping from one parser output schema to canonical scaler names."""

    source_format: str
    frequency_mhz: str | None = None
    frequency_hz: str | None = None
    height_km: str | None = None
    amplitude_db: str | None = None
    mode: str | None = None
    time: str | None = None
    mpa_db: str | None = None
    doppler_bin: str | None = None
    doppler_hz: str | None = None
    phase: str | None = None
    azimuth: str | None = None


SCHEMA_MAP: dict[str, EchoSchema] = {
    "canonical": EchoSchema(
        source_format="canonical",
        frequency_mhz="frequency_mhz",
        height_km="height_km",
        amplitude_db="amplitude_db",
        mode="mode",
        time="time",
        mpa_db="mpa_db",
        doppler_bin="doppler_bin",
        doppler_hz="doppler_hz",
        phase="phase",
        azimuth="azimuth",
    ),
    "rsf": EchoSchema(
        source_format="rsf",
        frequency_hz="frequency_reading",
        height_km="height",
        amplitude_db="amplitude",
        mode="pol",
        time="date",
        mpa_db="mpa",
        doppler_bin="dop_num",
        phase="phase",
        azimuth="azimuth",
    ),
    "sbf": EchoSchema(
        source_format="sbf",
        frequency_hz="frequency_reading",
        height_km="height",
        amplitude_db="amplitude",
        mode="pol",
        time="date",
        mpa_db="mpa",
        doppler_bin="dop_num",
        phase="phase",
        azimuth="azimuth",
    ),
    "mmm": EchoSchema(
        source_format="mmm",
        frequency_mhz="frequency_mhz",
        height_km="range_km",
        amplitude_db="amplitude_dB",
        mode="polarization",
        time="datetime",
        mpa_db="mpa_dB",
        doppler_bin="doppler_channel",
        doppler_hz="doppler_hz",
    ),
}


def detect_echo_schema(df: pd.DataFrame) -> str:
    """Infer the parser schema for a Digisonde echo DataFrame.

    Args:
        df: Parser output or already-normalized echo DataFrame.

    Returns:
        One of ``"canonical"``, ``"rsf"``, ``"sbf"``, or ``"mmm"``. RSF and
        SBF are intentionally indistinguishable from columns alone; the detector
        returns ``"rsf"`` for that shared schema.

    Raises:
        ValueError: If the DataFrame columns do not match a supported schema.
    """
    cols = set(df.columns)
    if {"frequency_mhz", "height_km", "amplitude_db", "mode"} <= cols:
        return "canonical"
    if {"frequency_mhz", "range_km", "amplitude_dB", "polarization"} <= cols:
        return "mmm"
    if {"frequency_reading", "height", "amplitude", "pol"} <= cols:
        return "rsf"
    raise ValueError(
        "Could not infer Digisonde echo schema. Expected RSF/SBF columns "
        "(frequency_reading, height, amplitude, pol), MMM columns "
        "(frequency_mhz, range_km, amplitude_dB, polarization), or canonical "
        "columns (frequency_mhz, height_km, amplitude_db, mode)."
    )


def normalize_mode_values(values: pd.Series) -> pd.Series:
    """Normalize polarization/mode labels to ``O``/``X`` where possible."""
    out = values.astype("string").str.strip().str.upper()
    out = out.str.replace("-MODE", "", regex=False)
    out = out.str.replace("MODE", "", regex=False)
    out = out.str.strip()
    first = out.str[0]
    out = out.where(first.isna() | ~first.isin(["O", "X"]), first)
    return out.astype(object)


def normalize_echo_dataframe(
    df: pd.DataFrame,
    *,
    source_format: str = "auto",
    mode: str | None = None,
    col_map: Mapping[str, str] | None = None,
    copy: bool = True,
) -> pd.DataFrame:
    """Return a scaler-ready DataFrame with canonical echo columns.

    Args:
        df: Echo DataFrame from RSF, SBF, MMM, or an already-canonical table.
        source_format: ``"auto"``, ``"rsf"``, ``"sbf"``, ``"mmm"``, or
            ``"canonical"``.
        mode: Optional mode filter, usually ``"O"`` or ``"X"``.
        col_map: Optional source-column overrides keyed by canonical names.
            Useful for custom parser outputs. Supported keys match
            :class:`EchoSchema` fields.
        copy: If True, copy the input before adding canonical columns.

    Returns:
        A DataFrame containing the original columns plus canonical columns.

    Raises:
        KeyError: If required source columns are missing.
        ValueError: If the source format cannot be inferred or is unsupported.
    """
    if source_format == "auto":
        source_format = detect_echo_schema(df)
    source_format = source_format.lower()
    if source_format not in SCHEMA_MAP:
        raise ValueError(
            f"Unsupported source_format={source_format!r}; expected one of "
            f"{sorted(SCHEMA_MAP)} or 'auto'."
        )

    schema = SCHEMA_MAP[source_format]
    overrides = dict(col_map or {})
    out = df.copy() if copy else df

    def source(name: str) -> str | None:
        return overrides.get(name, getattr(schema, name))

    freq_mhz_col = source("frequency_mhz")
    freq_hz_col = source("frequency_hz")
    if freq_mhz_col and freq_mhz_col in out.columns:
        out["frequency_mhz"] = pd.to_numeric(out[freq_mhz_col], errors="coerce")
    elif freq_hz_col and freq_hz_col in out.columns:
        out["frequency_mhz"] = pd.to_numeric(out[freq_hz_col], errors="coerce") / 1e6
    else:
        raise KeyError(
            "DataFrame must contain a frequency column in MHz or Hz for "
            f"source_format={source_format!r}."
        )

    required_sources = {
        "height_km": source("height_km"),
        "amplitude_db": source("amplitude_db"),
        "mode": source("mode"),
    }
    missing = [
        src for src in required_sources.values() if src and src not in out.columns
    ]
    if missing:
        raise KeyError(f"Missing required echo columns: {missing}")

    out["height_km"] = pd.to_numeric(
        out[required_sources["height_km"]], errors="coerce"
    )
    out["amplitude_db"] = pd.to_numeric(
        out[required_sources["amplitude_db"]], errors="coerce"
    )
    out["mode"] = normalize_mode_values(out[required_sources["mode"]])

    optional_numeric = {
        "mpa_db": source("mpa_db"),
        "doppler_bin": source("doppler_bin"),
        "doppler_hz": source("doppler_hz"),
        "phase": source("phase"),
        "azimuth": source("azimuth"),
    }
    for target, src in optional_numeric.items():
        if src and src in out.columns:
            out[target] = pd.to_numeric(out[src], errors="coerce")
        elif target not in out.columns:
            out[target] = np.nan

    time_col = source("time")
    if time_col and time_col in out.columns:
        out["time"] = pd.to_datetime(out[time_col], errors="coerce", utc=True)
    elif "time" not in out.columns:
        out["time"] = pd.NaT

    out["source_format"] = source_format
    out = out.dropna(subset=["frequency_mhz", "height_km", "amplitude_db"])

    if mode is not None:
        mode_norm = normalize_mode_values(pd.Series([mode])).iloc[0]
        out = out[out["mode"] == mode_norm].copy()

    return out.reset_index(drop=True)
