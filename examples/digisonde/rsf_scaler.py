"""Batch-scale RSF ionograms with the generic Digisonde scaler.

This example demonstrates the DataFrame-native scaler workflow:

1. Read one or more RSF files with ``RsfExtractor``.
2. Convert each parsed file to a pandas DataFrame.
3. Run the generic Viterbi scaler on O-mode echoes.
4. Save a scaled-parameter CSV and a Viterbi diagnostic figure.

The bundled example uses small WS833 RSF files under ``examples/data``.  To run
on a campaign directory, set ``RSF_SCALER_GLOB`` before execution, for example:

    RSF_SCALER_GLOB="/path/to/day/*.RSF" python examples/digisonde/rsf_scaler.py

Outputs:
    tmp/rsf_scaled_example.csv
    docs/examples/figures/rsf_scaler_viterbi.png
"""

from __future__ import annotations

import glob
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from joblib import Parallel, delayed
from loguru import logger

from pynasonde.digisonde.parsers.rsf import RsfExtractor
from pynasonde.digisonde.scaler import (
    normalize_echo_dataframe,
    scale_dataframe_with_preset,
)

DATA_GLOB = os.environ.get("RSF_SCALER_GLOB", "examples/data/*.RSF")
N_JOBS = int(os.environ.get("RSF_SCALER_N_JOBS", "1"))
OUT_CSV = Path("tmp/rsf_scaled_example.csv")
OUT_FIG = Path("docs/examples/figures/rsf_scaler_viterbi.png")
STATION_LAT_DEG = float(os.environ.get("RSF_SCALER_LAT_DEG", "33.72"))
STATION_LON_DEG = float(os.environ.get("RSF_SCALER_LON_DEG", "-106.74"))
TUNING_PRESET = os.environ.get("RSF_SCALER_TUNING_PRESET", "balanced")
E_LAYER_MAX_SZA = float(os.environ.get("RSF_SCALER_E_LAYER_MAX_SZA", "100.0"))


def _process_one_file(fpath: str) -> dict | None:
    """Parse and scale one RSF file."""
    try:
        ext = RsfExtractor(
            fpath,
            extract_time_from_name=True,
            extract_stn_from_name=True,
        )
        ext.extract()
        raw_df = ext.to_pandas()
        if raw_df.empty:
            logger.warning(f"Skipping empty RSF file: {fpath}")
            return None

        normalized_all = normalize_echo_dataframe(raw_df)

        result, details = scale_dataframe_with_preset(
            raw_df,
            time=getattr(ext, "date", None),
            lat_deg=STATION_LAT_DEG,
            lon_deg=STATION_LON_DEG,
            tuning_preset=TUNING_PRESET,
            e_layer_max_sza=E_LAYER_MAX_SZA,
            mode="O",
            return_details=True,
        )

        row = result.to_series()
        row["time"] = getattr(ext, "date", pd.NaT)
        row["source_file"] = Path(fpath).name
        row["n_raw_echoes"] = len(raw_df)
        row["n_cleaned_echoes_E"] = len(
            details["per_layer"].get("E", {}).get("cleaned", pd.DataFrame())
        )
        row["n_cleaned_echoes_F2"] = len(
            details["per_layer"].get("F2", {}).get("cleaned", pd.DataFrame())
        )
        row["station"] = Path(fpath).name.split("_")[0]
        row["sza_deg"] = details["preset"].sza_deg
        row["sza_regime"] = details["preset"].regime
        row["tuning_preset"] = details["preset"].tuning_preset
        row["e_layer_enabled"] = details["preset"].enabled["E"]
        for key, value in details["preset"].metadata.items():
            row[key] = value

        return {
            "row": row.to_dict(),
            "normalized": normalized_all,
            "cleaned": details["cleaned"],
            "traces": details["traces"],
        }
    except Exception as exc:
        logger.warning(f"Failed to scale {fpath}: {exc}")
        return None


def _suppress_plot_rfi_columns(
    df: pd.DataFrame,
    *,
    freq_bin: float = 0.05,
    min_count: int = 80,
    min_span_km: float = 260.0,
) -> pd.DataFrame:
    """Remove visually dominant vertical columns from the example plot only."""
    if df.empty:
        return df.copy()

    work = df.copy()
    fbin = (work["frequency_mhz"] / freq_bin).round() * freq_bin
    stats = (
        work.assign(_fbin=fbin)
        .groupby("_fbin")["height_km"]
        .agg(n="size", span=lambda x: float(x.max() - x.min()))
    )
    rfi_bins = stats[(stats["n"] >= min_count) & (stats["span"] >= min_span_km)].index
    if len(rfi_bins) == 0:
        return work
    return work[~fbin.isin(rfi_bins)].reset_index(drop=True)


def _plot_viterbi_ionogram(
    results: list[dict],
    scaled: pd.DataFrame,
    out_path: Path,
) -> None:
    """Create a raw O-mode/Viterbi and raw O/X diagnostic figure."""
    if not results:
        raise ValueError("No scaler results available for plotting.")

    first = max(
        results,
        key=lambda item: len(item["traces"].get("E", pd.DataFrame())),
    )
    norm = first["normalized"]
    traces = first["traces"]
    time_label = pd.to_datetime(first["row"]["time"]).strftime("%Y-%m-%d %H:%M UT")

    fig, (ax0, ax1) = plt.subplots(
        1,
        2,
        figsize=(11.5, 5.2),
        sharey=True,
        constrained_layout=True,
    )

    # Left: raw O/X positive-power echoes; traces come from layer-specific filtering.
    mode_colors = {"O": "#005AB5", "X": "#DC3220"}
    raw_modes = norm[norm["mode"].isin(mode_colors) & (norm["amplitude_db"] > 0)].copy()
    if raw_modes.empty:
        raw_modes = norm[norm["mode"].isin(mode_colors)].copy()

    for mode, color in mode_colors.items():
        mode_df = raw_modes[raw_modes["mode"] == mode]
        if mode_df.empty:
            continue
        ax0.scatter(
            mode_df["frequency_mhz"],
            mode_df["height_km"],
            s=0.45,
            color=color,
            linewidths=0,
            rasterized=True,
            label=f"{mode}-mode raw",
        )

    trace_colors = {"E": "#F0E442", "F2": "#D55E00", "F1": "#56B4E9", "Es": "#009E73"}
    for layer, trace in traces.items():
        if trace.empty:
            continue
        ax0.plot(
            trace["frequency_mhz"],
            trace["height_km"],
            color=trace_colors.get(layer, "white"),
            lw=2.4,
            label=f"Viterbi {layer}",
        )

    ax0.set_title("Raw O/X modes + Viterbi")
    ax0.set_xlabel("Frequency (MHz)")
    ax0.set_ylabel("Virtual height (km)")
    ax0.set_xlim(1.0, min(13.0, max(2.0, raw_modes["frequency_mhz"].max() + 0.5)))
    ax0.set_ylim(80.0, min(560.0, max(250.0, raw_modes["height_km"].quantile(0.995))))
    ax0.grid(True, alpha=0.25)
    ax0.legend(loc="upper right", fontsize=8)

    # Right: raw O/X mode context without Viterbi overlays.
    for mode, color in mode_colors.items():
        mode_df = raw_modes[raw_modes["mode"] == mode]
        if mode_df.empty:
            continue
        ax1.scatter(
            mode_df["frequency_mhz"],
            mode_df["height_km"],
            s=0.45,
            color=color,
            linewidths=0,
            rasterized=True,
            label=f"{mode}-mode",
        )
    ax1.set_title("Raw O/X mode context")
    ax1.set_xlabel("Frequency (MHz)")
    ax1.set_xlim(ax0.get_xlim())
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="upper right", fontsize=8)

    fig.suptitle(
        f"RSF raw ionogram and layer-specific Viterbi traces\n{time_label}",
        fontweight="bold",
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved figure: {out_path}")


if __name__ == "__main__":
    files = sorted(glob.glob(DATA_GLOB))
    if not files:
        raise FileNotFoundError(
            f"No RSF files matched {DATA_GLOB!r}. Set RSF_SCALER_GLOB to a "
            "directory containing Digisonde .RSF files."
        )

    logger.info(f"Processing {len(files)} RSF files with N_JOBS={N_JOBS}")
    results = Parallel(n_jobs=N_JOBS, backend="loky", verbose=5)(
        delayed(_process_one_file)(fpath) for fpath in files
    )
    results = [result for result in results if result is not None]

    scaled = pd.DataFrame([result["row"] for result in results])
    if not scaled.empty and "time" in scaled.columns:
        scaled = scaled.sort_values("time").reset_index(drop=True)

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    scaled.to_csv(OUT_CSV, index=False, float_format="%g")
    logger.info(f"Saved {len(scaled)} scaled rows: {OUT_CSV}")

    _plot_viterbi_ionogram(results, scaled, OUT_FIG)
