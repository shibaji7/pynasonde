"""Generate frequency–time interval plots from VIPIR NGI ionogram archives.

This example demonstrates how to:

1. Discover and load a collection of NGI files with :class:`DataSource`.
2. Collapse the ionogram cubes into a long-form dataframe suitable for plotting.
3. Render an interval plot with :class:`Ionogram` and persist the figure for MkDocs.

Before running the script update the paths inside ``__main__`` to point at your
own NGI dataset. The defaults reference the Speed Demon 2022 campaign layout.
"""

from __future__ import annotations

import datetime as dt
import os
import shutil
from pathlib import Path

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from loguru import logger

from pynasonde.vipir.ngi.plotlib import Ionogram
from pynasonde.vipir.ngi.source import DataSource


def generate_fti_profiles(
    folder: str,
    fig_file_name: str | None = None,
    fig_title: str | None = None,
    stn: str = "",
    flim: tuple[float, float] = (3.5, 4.5),
    date: dt.datetime | None = None,
) -> pd.DataFrame:
    """Render an O-mode frequency–time interval plot for a folder of NGI files.

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
    flim
        Two-element tuple describing the lower/upper frequency bounds in MHz.

    Returns
    -------
    pandas.DataFrame
        Flattened RTI dataframe with ``time``, ``range``, and mode-specific
        power/noise columns suitable for further analysis.
    """
    if len(flim) != 2:
        raise ValueError("flim must contain exactly two elements: (f_min, f_max)")

    logger.info(f"Loading NGI datasets from {folder}")
    ds = DataSource(source_folder=folder)
    ds.load_data_sets(0, -1)

    mode = "O"
    rti = pd.DataFrame()
    for dataset in ds.datasets:
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
        f_min, f_max = flim
        frame = frame[(frame.frequency >= f_min) & (frame.frequency <= f_max)]
        rti = pd.concat([rti, frame], ignore_index=True)

    if rti.empty:
        logger.warning("No RTI samples found in the requested frequency window")
        return rti

    fig_file = Path(fig_file_name or "docs/examples/figures/fti.png")
    fig_file.parent.mkdir(parents=True, exist_ok=True)
    obs_start = pd.to_datetime(rti.time.min()).to_pydatetime()
    obs_end = pd.to_datetime(rti.time.max()).to_pydatetime()
    if obs_start == obs_end:
        obs_end = obs_start + dt.timedelta(minutes=1)
    if fig_title is None:
        station_label = f"{stn} / " if stn else ""
        fig_title = (
            f"{station_label}{obs_start:%d %b %Y}, "
            f"$f_0$=[{flim[0]:.2f}-{flim[1]:.2f}] MHz"
        )

    ionogram = Ionogram(fig_title=fig_title, nrows=1, ncols=1, figsize=(6, 3))
    axis = ionogram.add_interval_plots(
        rti,
        mode,
        xlabel="Time, UT",
        ylabel="Virtual Height, km",
        ylim=[50, 400],
        add_cbar=True,
        cbar_label="O-mode Power, dB",
        cmap="Spectral",
        noise_scale=1,
        date_format=r"$%H^{%M}$",
        del_ticks=False,
        xtick_locator=mdates.HourLocator(interval=6),
        xdate_lims=[obs_start, obs_end],
    )
    axis.set_xlim(date, date + dt.timedelta(hours=24))
    axis.set_ylim(50, 400)
    axis.text(0.95, 1.05, "", ha="right", va="center", transform=axis.transAxes)
    ionogram.save(fig_file)
    ionogram.close()
    logger.info(f"Saved FTI figure to {fig_file}")
    return rti


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
                flim=(2, 3.5),
                date=date,
            )
        finally:
            shutil.rmtree(tmp.parent, ignore_errors=True)
