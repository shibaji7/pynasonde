import datetime as dt
import glob
import os
from dataclasses import dataclass
from typing import List

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger
from matplotlib.dates import DateFormatter

import pynasonde.ngi.utils as utils


@dataclass
class Dataset:
    URSI: str = ""
    StationName: str = ""
    year: int = 1970  # UTC
    daynumber: int = 1  # UTC
    month: int = 1  # UTC
    day: int = 1  # UTC
    hour: int = 0  # UTC
    minute: int = 0  # UTC
    second: int = 0  # UTC
    epoch: np.datetime64 = 0  # UTC
    latitude: float = 0.0  # degree_north
    longitude: float = 0.0  # degree_east
    altitude: float = 0.0  # meter
    MagLat: float = 0.0  # degree_east
    MagLon: float = 0.0  # degree_east
    MagDip: float = 0.0  # degree
    GyroFreq: float = 0.0  # Station GyroFrequency at 300 km altitude, MHz
    PRI: float = 0.0  # microsecond
    range_gate_offset: float = 0.0  # microsecond
    gate_count: float = 0.0  # counts
    gate_start: float = 0.0  # microsecond
    gate_end: float = 0.0  # microsecond
    gate_step: float = 0.0  # microsecond
    Range0: float = 0.0  # kilometer
    freq_start: float = 0.0  # lower frequency, kilohertz
    freq_end: float = 0.0  # upper frequency, kilohertz
    tune_type: int = 0  # 1=log 2=linear 3=table
    freq_count: int = 0  # count
    linear_step: float = 0.0  # kilohertz
    log_step: float = 0.0  # logarithmic tuning step, percent
    Range: np.array = None  # kilometer
    Frequency: np.array = None  # kilohertz
    Time: np.array = None  # Nominal Observation Time, UT / second
    TxDrive: np.array = None  # decibel
    NumAve: np.array = None  # count
    SCT_version: float = 1.2  #
    SCT: int = 0
    PREFACE: int = 0
    Has_total_power: int = 0  # flag
    total_power: np.array = None  # decibel
    total_noise: np.array = None  # decibel
    Has_O_mode_power: int = 0  # flag
    O_mode_power: np.array = None  # decibel, Shape(Frequency, Range)
    O_mode_noise: np.array = None  # decibel, Shape(Frequency, )
    Has_X_mode_power: int = 0  # flag
    X_mode_power: np.array = None  # decibel, Shape(Frequency, Range)
    X_mode_noise: np.array = None  # decibel, Shape(Frequency, )
    Has_Doppler: int = 0  # flag
    Has_VLoS: int = 0  # flag
    Has_SPGR: int = 0  # flag
    Has_Zenith: int = 0  # flag
    Has_Azimuth: int = 0  # flag
    Has_Coherence: int = 0  # flag

    def __initialize__(self, ds):
        key_map = {
            "Has_O_mode_power": "Has_O-mode_power",
            "O_mode_power": "O-mode_power",
            "O_mode_noise": "O-mode_noise",
            "Has_X_mode_power": "Has_X-mode_power",
            "X_mode_power": "X-mode_power",
            "X_mode_noise": "X-mode_noise",
        }
        for attr in self.__dict__.keys():
            if attr in list(key_map.keys()):
                setattr(self, attr, np.array(ds[key_map[attr]].values))
            elif (type(ds[attr].values) == np.ndarray) and (
                (ds[attr].values.dtype == "|S8") or (ds[attr].values.dtype == "|S64")
            ):
                setattr(self, attr, ds[attr].values.astype(str).tolist())
            else:
                if len(ds[attr].values.shape) == 0:
                    setattr(self, attr, ds[attr].values.tolist())
                else:
                    setattr(self, attr, np.array(ds[attr].values))
        return self


class Ionogram(object):

    def __init__(
        self,
        dates: dt.datetime = None,
        fig_title: str = "",
        num_subplots: int = 1,
        font_size: float = 10,
    ):
        self.dates = dates
        self.num_subplots = num_subplots
        self._num_subplots_created = 0
        self.fig = plt.figure(
            figsize=(4.5, 3 * num_subplots), dpi=300
        )  # Size for website
        self.fig_title = fig_title
        utils.setsize(font_size)
        return

    def add_ionogram(
        self,
        ds: Dataset,
        mode: str = "O",
        xlabel: str = "Frequency, MHz",
        ylabel: str = "Virtual Height, km",
        ylim: List[float] = [50, 800],
        xlim: List[float] = [1, 22],
        add_cbar: bool = True,
        cbar_label: str = "{}-mode Power, dB",
        cmap: str = "Greens",
        prange: List[float] = [5, 70],
        xticks: List[float] = [1.5, 2.0, 3.0, 5.0, 7.0, 10.0, 15.0, 20.0],
        noise_scale: float = 1.5,
    ) -> None:
        ax = self._add_axis()
        ax.set_xlim(np.log10(xlim))
        ax.set_xlabel(xlabel, fontdict={"size": 12})
        ax.set_ylim(ylim)
        ax.set_ylabel(ylabel)
        time = dt.datetime(ds.year, ds.month, ds.day, ds.hour, ds.minute, ds.second)
        ax.text(
            0.05,
            1.05,
            f"{ds.StationName}/{time.strftime('%H:%M:%S UT %d %b %Y')}",
            ha="left",
            va="center",
            transform=ax.transAxes,
        )
        Zval = getattr(ds, f"{mode}_mode_power")
        Zval[Zval < getattr(ds, f"{mode}_mode_noise")[:, np.newaxis] * noise_scale] = (
            np.nan
        )
        im = ax.pcolormesh(
            np.log10(ds.Frequency / 1e3),
            ds.Range,
            Zval.T,
            lw=0.01,
            edgecolors="None",
            cmap=cmap,
            vmax=prange[1],
            vmin=prange[0],
            zorder=3,
        )
        ax.set_xticks(np.log10(xticks))
        ax.set_xticklabels(xticks)
        if add_cbar:
            self._add_colorbar(im, self.fig, ax, label=cbar_label.format(mode))
        return

    def add_interval_plots(
        self,
        df: pd.DataFrame,
        mode: str = "O",
        xlabel: str = "Time, UT",
        ylabel: str = "Virtual Height, km",
        ylim: List[float] = [50, 800],
        xlim: List[dt.datetime] = None,
        add_cbar: bool = True,
        cbar_label: str = "{}-mode Power, dB",
        cmap: str = "Spectral",
        prange: List[float] = [5, 70],
        noise_scale: float = 1.5,
    ) -> None:
        xlim = xlim if xlim is not None else [df.time.min(), df.time.max()]
        ax = self._add_axis()
        ax.set_xlim(xlim)
        ax.set_xlabel(xlabel)
        ax.set_ylim(ylim)
        ax.set_ylabel(ylabel)
        hours = mdates.HourLocator(byhour=range(0, 24, 1))
        ax.xaxis.set_minor_locator(hours)
        ax.xaxis.set_minor_formatter(DateFormatter(r"%H^{%M}"))
        X, Y, Z = utils.get_gridded_parameters(
            df,
            xparam="time",
            yparam="range",
            zparam=f"{mode}_mode_power",
            rounding=False,
        )
        im = ax.pcolormesh(
            X,
            Y,
            Z.T,
            lw=0.01,
            edgecolors="None",
            cmap=cmap,
            vmax=prange[1],
            vmin=prange[0],
            zorder=3,
        )
        if add_cbar:
            self._add_colorbar(im, self.fig, ax, label=cbar_label.format(mode))
        return

    def _add_axis(self):
        self._num_subplots_created += 1
        ax = self.fig.add_subplot(self.num_subplots, 1, self._num_subplots_created)
        if self._num_subplots_created == 1:
            ax.text(
                0.01,
                1.05,
                self.fig_title,
                ha="left",
                va="center",
                transform=ax.transAxes,
            )
        return ax

    def _add_colorbar(self, im, fig, ax, label=""):
        """
        Add a colorbar to the right of an axis.
        """
        pos = ax.get_position()
        cpos = [
            pos.x1 + 0.025,
            pos.y0 + 0.0125,
            0.015,
            pos.height * 0.9,
        ]  # this list defines (left, bottom, width, height
        cax = fig.add_axes(cpos)
        cb = fig.colorbar(im, ax=ax, cax=cax)
        cb.set_label(label)
        return

    def save(self, filepath):
        self.fig.savefig(filepath, bbox_inches="tight")
        return

    def close(self):
        self.fig.clf()
        plt.close()
        return


class DataSource(object):
    """ """

    def __init__(
        self,
        source_folder: str = "./tmp/",
        file_ext: str = "*.ngi",
        file_names: List[str] = [],
    ):
        self.source_folder = source_folder
        self.file_ext = file_ext
        self.file_names = file_names
        # Load full path of the files by file_names or from free space search
        self.file_paths = (
            [os.path.join(source_folder, f) for f in file_names]
            if file_names
            else glob.glob(os.path.join(source_folder, file_ext.lower()))
            + glob.glob(os.path.join(source_folder, file_ext.upper()))
        )
        self.file_paths.sort()
        logger.info(f"Total number of files {len(self.file_paths)}")
        return

    def load_data_sets(self):
        """ """
        self.datasets = []
        for f in self.file_paths:
            logger.info(f"Load file: {f}")
            ds = xr.open_dataset(f)
            self.datasets.append(Dataset().__initialize__(ds))
        return

    def extract_ionograms(
        self,
        folder: str = "tmp/",
        kinds: List[str] = ["O", "X"],
    ) -> None:
        os.makedirs(folder, exist_ok=True)
        for ds in self.datasets:
            time = dt.datetime(ds.year, ds.month, ds.day, ds.hour, ds.minute, ds.second)
            i = Ionogram()
            i.add_ionogram(ds)
            i.save(
                os.path.join(folder, f"{ds.URSI}_{time.strftime('%Y%m%d%H%M%S')}.png")
            )
            i.close()
        return

    def extract_FTI_RTI(
        self,
        folder: str = "tmp/",
        rlim: List[float] = [50, 800],
        flim: List[float] = [3.5, 4.5],
        mode: str = "O",
    ) -> pd.DataFrame:
        logger.info(f"Extract FTI/RTI, based on {flim}MHz {rlim}km")
        rti = pd.DataFrame()
        for ds in self.datasets:
            frequency, range = np.meshgrid(ds.Frequency, ds.Range, indexing="ij")
            noise, _ = np.meshgrid(
                getattr(ds, f"{mode}_mode_noise"), ds.Range, indexing="ij"
            )
            o = pd.DataFrame()
            (
                o["frequency"],
                o["range"],
                o[f"{mode}_mode_power"],
                o[f"{mode}_mode_noise"],
            ) = (
                frequency.ravel() / 1e3,  # to MHz
                range.ravel(),  # in km
                getattr(ds, f"{mode}_mode_power").ravel(),  # in dB
                noise.ravel(),  # in dB
            )
            o["time"] = dt.datetime(
                ds.year, ds.month, ds.day, ds.hour, ds.minute, ds.second
            )
            rti = pd.concat([rti, o])
        if (len(rlim) == 2) and (len(flim) == 2):
            rti = rti[
                (rti.range >= rlim[0])
                & (rti.range <= rlim[1])
                & (rti.frequency >= flim[0])
                & (rti.frequency <= flim[1])
            ]
        fname = f"{ds.URSI}_{rti.time.min().strftime('%Y%m%d.%H%M-')}{rti.time.max().strftime('%H%M')}_{mode}-mode.png"
        i = Ionogram()
        i.add_interval_plots(rti, mode=mode)
        i.save(os.path.join(folder, fname))
        i.close()
        return rti
