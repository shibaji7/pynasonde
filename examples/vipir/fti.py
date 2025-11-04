"""Generate frequency–time interval plots from VIPIR NGI ionogram archives.

This example demonstrates how to:

1. Discover and load a collection of NGI files with :class:`DataSource`.
2. Collapse the ionogram cubes into a long-form dataframe suitable for plotting.
3. Render stacked interval plots with :class:`Ionogram` and persist the figure for MkDocs.

Before running the script update the paths inside ``__main__`` to point at your
own NGI dataset. The defaults reference the Speed Demon 2022 campaign layout.
"""

from __future__ import annotations

import datetime as dt
import os
import shutil
from collections.abc import Sequence
from pathlib import Path

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from loguru import logger

from pynasonde.digisonde.digi_utils import setsize
from pynasonde.vipir.ngi.plotlib import Ionogram
from pynasonde.vipir.ngi.source import DataSource

font_size = 15
setsize(font_size)


def generate_fti_profiles(
    folder: str,
    fig_file_name: str | None = None,
    fig_title: str | None = None,
    stn: str = "",
    frequency_bands: Sequence[tuple[float, float]] | None = None,
    date: dt.datetime | None = None,
) -> pd.DataFrame:
    """Render stacked O-mode frequency–time interval plots for the requested bands.

    Parameters
    ----------
    folder
        Directory containing the NGI files to be ingested.
    fig_file_name
        Optional destination for the output figure. Defaults to
        ``docs/examples/figures/fti.png``.
    fig_title
        Optional figure title. When omitted the station identifier (if supplied)
        and the observation date are combined with the frequency window.
    stn
        Station identifier used when building auto-generated titles.
    frequency_bands
        Iterable of (f_min, f_max) tuples describing the frequency windows (MHz)
        to extract and plot. Each tuple produces one subplot stacked vertically.

    Returns
    -------
    pandas.DataFrame
        Flattened RTI dataframe with ``time``, ``range``, mode-specific
        power/noise columns, and metadata describing the contributing
        frequency band(s).
    """
    bands = list(
        frequency_bands or [(3.5, 4.5)]
    )  # Maintain band ordering to drive subplot stacking.
    if not bands:
        raise ValueError(
            "frequency_bands must include at least one (f_min, f_max) interval"
        )
    for band in bands:
        if len(band) != 2:
            raise ValueError(
                "Each entry in frequency_bands must contain exactly two elements"
            )

    logger.info(f"Loading NGI datasets from {folder}")
    ds = DataSource(source_folder=folder)
    ds.load_data_sets(
        0, -1, n_jobs=20
    )  # Pull every NGI cube for the folder using modest parallelism.

    mode = "O"
    rti = pd.DataFrame()
    for dataset in ds.datasets:
        # Assemble a timestamp for the snapshot and project the ionogram cube onto
        # frequency × height grids so the values can be flattened into a DataFrame.
        time = dt.datetime(
            dataset.year,
            dataset.month,
            dataset.day,
            dataset.hour,
            dataset.minute,
            dataset.second,
        )
        logger.info(f"Processing snapshot at {time:%Y-%m-%d %H:%M:%S}")
        frequency, range_gate = np.meshgrid(
            dataset.Frequency, dataset.Range, indexing="ij"
        )
        noise, _ = np.meshgrid(
            getattr(dataset, f"{mode}_mode_noise"), dataset.Range, indexing="ij"
        )
        # Flatten the gridded parameters so they can be stored in a DataFrame.
        frame = pd.DataFrame(
            {
                "frequency": frequency.ravel() / 1e3,  # convert to MHz
                "range": range_gate.ravel(),  # already in km
                f"{mode}_mode_power": getattr(dataset, f"{mode}_mode_power").ravel(),
                f"{mode}_mode_noise": noise.ravel(),
            }
        )
        frame["time"] = time
        # Restrict to the shared altitude/frequency envelope covered by the requested bands.
        frame = frame[
            (frame.range <= 400)
            & (frame.range >= 50)
            & (frame.frequency >= np.min(np.array(bands)))
            & (frame.frequency <= np.max(np.array(bands)))
        ]
        rti = pd.concat([rti, frame], ignore_index=True)

    if rti.empty:
        logger.warning("No RTI samples found in the input datasets")
        return rti

    fig_file = Path(fig_file_name or "docs/examples/figures/fti.png")
    fig_file.parent.mkdir(parents=True, exist_ok=True)
    obs_start = pd.to_datetime(rti.time.min()).to_pydatetime()
    obs_end = pd.to_datetime(rti.time.max()).to_pydatetime()
    if obs_start == obs_end:
        obs_end = obs_start + dt.timedelta(minutes=1)
    if fig_title is None:
        station_label = f"{stn} / " if stn else ""
        bands_text = " + ".join(f"{b[0]:.2f}-{b[1]:.2f}" for b in bands)
        fig_title = f"{station_label}{obs_start:%d %b %Y}, " f"$f_0$=[{bands_text}] MHz"

    # One subplot per band; the class handles positioning and shared sizing.
    ionogram = Ionogram(
        fig_title="", nrows=len(bands), ncols=1, figsize=(6, 3), font_size=font_size
    )
    if fig_title:
        ionogram.fig.suptitle(
            fig_title, fontsize=ionogram.font_size + 2, y=0.99, x=0.02, ha="left"
        )
        ionogram.fig.subplots_adjust(top=0.92)

    xdate_lims = [obs_start, obs_end]
    if date is not None:
        # When a specific day is supplied, pin the x-axis to a full 24-hour window.
        xdate_lims = [date, date + dt.timedelta(hours=24)]

    band_frames: list[pd.DataFrame] = []
    for idx, (f_min, f_max) in enumerate(bands):
        band_df = rti[(rti.frequency >= f_min) & (rti.frequency <= f_max)].copy()
        band_label = f"({chr(97+idx)}) {f_min:.2f}-{f_max:.2f} MHz"
        if band_df.empty:
            logger.warning(
                "No RTI samples found for frequency band {} within {}",
                band_label,
                folder,
            )
            ax = ionogram._add_axis(del_ticks=False)
            ax.set_xlim(xdate_lims)
            ax.set_ylim(50, 400)
            ax.set_xlabel("Time, UT" if idx == len(bands) - 1 else "")
            ax.set_ylabel("Virtual Height, km")
            ax.text(
                0.5,
                0.5,
                "No data available",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontdict={"size": ionogram.font_size},
            )
            ax.text(
                0.99,
                1.02,
                band_label,
                ha="right",
                va="bottom",
                transform=ax.transAxes,
                fontdict={"size": ionogram.font_size},
            )
            continue

        band_df = band_df.assign(
            frequency_band_min=f_min,
            frequency_band_max=f_max,
            band_label=band_label,
        )
        band_frames.append(band_df)

        axis = ionogram.add_interval_plots(
            band_df,
            mode,
            xlabel="Time, UT" if idx == len(bands) - 1 else "",
            ylabel="Virtual Height, km",
            ylim=[50, 400],
            add_cbar=idx == len(bands) - 1,
            cbar_label="O-mode Power, dB",
            cmap="Spectral",
            noise_scale=1,
            date_format=r"$%H^{%M}$",
            del_ticks=False,
            xtick_locator=mdates.HourLocator(interval=6),
            xdate_lims=xdate_lims,
        )
        axis.set_ylim(50, 400)
        # Add a panel-level tag so readers can cross-reference the band in text.
        axis.text(
            0.99,
            1.02,
            band_label,
            ha="right",
            va="bottom",
            transform=axis.transAxes,
            fontdict={"size": ionogram.font_size},
        )

    if not band_frames:
        ionogram.close()
        return pd.DataFrame()

    ionogram.fig.subplots_adjust(hspace=0.25)
    ionogram.save(fig_file)
    ionogram.close()
    logger.info(f"Saved FTI figure to {fig_file}")
    return pd.concat(band_frames, ignore_index=True)


if __name__ == "__main__":
    # Example configuration for the Speed Demon 2022 campaign; adjust as needed.
    data_root = Path(
        os.environ.get(
            "VIPIR_SPEED_DEMON_ROOT", "/media/chakras4/ERAU/SpeedDemon/WI937/individual"
        )
    )
    temp_root = Path("/tmp/vipir_fti")
    stn = "WI937"

    for doy in range(234, 235):
        date = dt.datetime(2022, 1, 1) + dt.timedelta(days=doy - 1)
        src = data_root / "2022" / f"{doy}" / "ionogram"
        tmp = temp_root / f"{doy}" / "ionogram"
        shutil.rmtree(tmp.parent, ignore_errors=True)
        shutil.copytree(src, tmp)
        try:
            title = None  # f"Speed Demon / {date:%Y-%m-%d}"
            generate_fti_profiles(
                folder=str(tmp),
                fig_file_name=f"docs/examples/figures/fti.{stn}.{date:%Yj}.png",
                fig_title=title,
                stn=stn,
                frequency_bands=[
                    (2.0, 3.5),
                    (4.0, 6.0),
                ],
                date=date,
            )
        finally:
            shutil.rmtree(tmp.parent, ignore_errors=True)
