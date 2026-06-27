"""Data ingestion helpers for VIPIR NGI ionogram datasets.

Provides structures to represent scaled traces (`Trace`), the full NGI dataset
(`Dataset`), and a `DataSource` orchestrator that loads raw NGI files, extracts
plots, and writes scaled diagnostics.
"""

import bz2
import datetime as dt
import glob
import os
import shutil
from dataclasses import dataclass
from typing import List

import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed
from loguru import logger
from pysolar.solar import get_altitude

from pynasonde.vipir.ngi.plotlib import Ionogram
from pynasonde.vipir.ngi.utils import TimeZoneConversion


@dataclass
class Trace:
    """Container for scaled trace parameters extracted from NGI files.

    Attributes:
        traces: Data frame of individual trace samples.
        trace_params: Data frame describing fitted trace parameters.
    """

    traces: pd.DataFrame = None
    trace_params: pd.DataFrame = None

    @staticmethod
    def load_saved_scaled_parameters(
        folder: str,
        extension="*.nc",
        mode: str = "O",
        local_tz: str = None,
        lat: float = 37.8815,
        long: float = -75.4374,
    ):
        """Load previously scaled trace parameters from NetCDF files.

        Args:
            folder: Directory containing saved scaling products.
            extension: File extension glob (default ``*.nc``).
            mode: Wave mode prefix (e.g., ``'O'`` or ``'X'``).
            local_tz: Optional local timezone name. If omitted the value is
                inferred from latitude/longitude.
            lat: Station latitude in degrees.
            long: Station longitude in degrees.

        Returns:
            Pandas DataFrame with time, local time, solar zenith angle, and
            fitted frequency/height pairs.
        """
        files = glob.glob(os.path.join(folder, mode + extension))
        files.sort()
        traces = {
            "sza": [],
            "time": [],
            "fs": [],
            "hs": [],
            "local_time": [],
        }
        LTC = TimeZoneConversion(local_tz, lat, long)
        for file in files:
            d = xr.open_dataset(file)
            L = len(d.hs.values)
            traces["time"].extend(d.time.values.tolist() * L)
            traces["local_time"].extend(
                LTC.utc_to_local_time(pd.to_datetime(d.time.values).tolist()) * L
            )
            traces["sza"].extend(d.sza.values.tolist() * L)
            traces["hs"].extend(d.hs.values.tolist())
            traces["fs"].extend(d.fs.values.tolist())
        traces = pd.DataFrame.from_dict(traces)
        traces.time = pd.to_datetime(traces.time)
        return traces


@dataclass
class Dataset:
    """Representation of a single NGI dataset and its derived traces."""

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

    def __initialize__(self, ds, unicode: str = "latin-1"):
        """Populate the dataclass fields using an xarray dataset.

        Args:
            ds: `xarray.Dataset` produced from an NGI NetCDF file.
            unicode: Encoding used for legacy byte-string attributes.

        Returns:
            Dataset: The populated instance (also stored as ``self``).
        """
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
        self.StationName = "".join([u.decode("latin-1") for u in self.StationName])
        self.URSI = "".join([u.decode("latin-1") for u in self.URSI])
        self.time = dt.datetime(
            self.year,
            self.month,
            self.day,
            self.hour,
            self.minute,
            self.second,
        )
        self.sza = 90.0 - get_altitude(
            self.latitude,
            self.longitude,
            self.time.replace(
                tzinfo=dt.timezone.utc,
            ),
        )
        logger.info(f"Local zenith angle: {self.sza}")
        return self

    def set_traces(self, traces: dict, trace_params: dict) -> None:
        """Attach fitted trace data produced by the autoscaler.

        Args:
            traces: Mapping of cluster labels to extracted trace samples.
            trace_params: Mapping of cluster labels to metadata dictionaries.
        """
        self.trace = Trace()
        self.trace.traces = pd.concat([traces[k] for k in list(traces.keys())])
        self.trace.trace_params = pd.DataFrame.from_records(
            [trace_params[k] for k in list(trace_params.keys())]
        )
        return

    def get_n_traces(self):
        """Return the number of trace parameter sets currently stored.

        Returns:
            int: Number of available traces (zero when none were set).
        """
        _n = 0
        if hasattr(self, "trace"):
            _n = len(self.trace.trace_params)
        return _n


class DataSource(object):
    """Manage discovery and loading of NGI ionogram files."""

    def __init__(
        self,
        source_folder: str = "./tmp/",
        file_ext: str = "*.ngi.bz2",
        file_names: List[str] = [],
        needs_decompression: bool = False,
    ):
        """Configure the datasource to point at a folder or explicit files.

        Args:
            source_folder: Directory containing NGI files.
            file_ext: Filename glob to match ionogram products.
            file_names: Explicit list of filenames (optional).
            needs_decompression: Force decompression even if extension differs.
        """
        self.source_folder = source_folder
        self.file_ext = file_ext
        self.file_names = file_names
        self.needs_decompression = (
            True if (".bz2" in file_ext) or needs_decompression else False
        )
        # Load full path of the files by file_names or from free space search
        self.file_paths = (
            [os.path.join(source_folder, f) for f in file_names]
            if file_names
            else glob.glob(os.path.join(source_folder, file_ext.lower()))
            + glob.glob(os.path.join(source_folder, file_ext.upper()))
        )
        self.file_paths.sort()
        logger.info(f"Total number of files {len(self.file_paths)}")
        logger.info(f"Needs decompression {self.needs_decompression}")
        return

    def load_data_sets(self, load_start=0, load_end=-1, n_jobs=-1):
        """Populate `self.datasets` with NGI records from disk.

        Args:
            load_start: Start index when slicing the discovered file list.
            load_end: End index (non-inclusive) for the slice; ``-1`` means all.
            n_jobs: Parallelism level passed to `joblib.Parallel`.
        """

        self.datasets = []
        compress = lambda fc, fd: shutil.copyfileobj(
            open(fd, "rb"), bz2.BZ2File(fc, "wb")
        )
        decompress = lambda fc, fd: shutil.copyfileobj(
            bz2.BZ2File(fc, "rb"), open(fd, "wb")
        )
        check_bad_file = lambda f: (
            True if os.path.getsize(f) / (1024 * 1024) >= 5.0 else False
        )

        def process_file(f):
            if not check_bad_file(f):
                return None
            logger.info(f"Load file: {f}")
            needs_decompression = self.needs_decompression
            orig_f = f
            if needs_decompression:
                decompress(f, f.replace(".bz2", ""))
                os.remove(f)
                f = f.replace(".bz2", "")
            ds = Dataset().__initialize__(xr.load_dataset(f, engine="netcdf4"))
            if needs_decompression:
                compress(f + ".bz2", f)
                os.remove(f)
            return ds

        files = self.file_paths[load_start:load_end]
        results = Parallel(n_jobs=n_jobs)(delayed(process_file)(f) for f in files)
        self.datasets = [ds for ds in results if ds is not None]
        return

    def extract_ionograms(self, folder: str = "tmp/", mode: str = "O") -> None:
        """Render quick-look ionograms for each dataset to PNG files.

        Args:
            folder: Destination directory for PNG images.
            mode: Wave mode to visualize (``'O'`` or ``'X'``).
        """
        os.makedirs(folder, exist_ok=True)
        for ds in self.datasets:
            time = dt.datetime(ds.year, ds.month, ds.day, ds.hour, ds.minute, ds.second)
            i = Ionogram()
            i.add_ionogram(
                ds.Frequency, ds.Range, getattr(ds, f"{mode}_mode_power"), mode=mode
            )
            i.save(
                os.path.join(folder, f"{ds.URSI}_{time.strftime('%Y%m%d%H%M%S')}.png")
            )
            i.close()
        return

    def extract_FTI_RTI(
        self,
        rlim: List[float] = [50, 800],
        flim: List[float] = [],
        mode: str = "O",
        noise_scale: float = 1.0,
    ) -> pd.DataFrame:
        """Extract frequency–time ionogram summary points for each dataset.

        Args:
            rlim: Height limits (km) used when scanning for maxima.
            flim: Optional frequency limits (MHz) for filtering results.
            mode: Ionogram mode to process.
            noise_scale: Multiplier applied to the noise threshold.

        Returns:
            DataFrame of RTI points with timestamp, frequency, power, and range.
        """
        logger.info(f"Extract FTI/RTI, based on {rlim}km")
        rti = []
        for ds in self.datasets:
            time = dt.datetime(ds.year, ds.month, ds.day, ds.hour, ds.minute, ds.second)
            logger.info(f"Time: {time}")
            f_max = np.nan
            for i, r in enumerate(ds.Range):
                if (r < rlim[0]) or (r > rlim[1]):
                    continue
                powr = np.array(getattr(ds, f"{mode}_mode_power")[:, i])
                powr[powr < getattr(ds, f"{mode}_mode_noise") * noise_scale] = (
                    np.nan
                )  # remove noise
                f_max = ds.Frequency[np.nanargmax(powr)]
                rti.append(
                    dict(
                        time=time,
                        frequency=f_max / 1e3,  # to MHz
                        power=np.nanmax(powr),  # in dB
                        range=r,
                        noise=getattr(ds, f"{mode}_mode_noise")[
                            np.nanargmax(powr)
                        ],  # in dB
                    )
                )
        rti = pd.DataFrame.from_records(rti)
        if len(flim) == 2:
            rti = rti[(rti.frequency >= flim[0]) & (rti.frequency <= flim[1])]
        return rti

    def extract_Power_RTI(
        self,
        folder: str = "tmp/",
        rlim: List[float] = [50, 800],
        flim: List[float] = [3.95, 4.05],
        mode: str = "O",
        fname: str = None,
        xlabel: str = "Time, UT",
        ylabel: str = "Virtual Height, km",
        xlim: List[dt.datetime] = None,
        add_cbar: bool = True,
        cbar_label: str = "{}-mode Power, dB",
        cmap: str = "Spectral",
        xtick_locator: mdates.HourLocator = mdates.HourLocator(interval=4),
        prange: List[float] = [5, 70],
        noise_scale: float = 1.2,
        date_format: str = r"$%H^{%M}$",
        del_ticks: bool = False,
        xdate_lims: List[dt.datetime] = None,
    ) -> pd.DataFrame:
        """Generate power RTI plots and return the underlying dataframe.

        Args:
            folder: Directory where the PNG plot will be written.
            rlim: Range limits (km) kept in the dataset.
            flim: Frequency limits (MHz) used to narrow the data.
            mode: Ionogram mode (``'O'`` or ``'X'``).
            fname: Optional filename for the saved figure.
            xlabel: X-axis label text.
            ylabel: Y-axis label text.
            xlim: Datetime limits applied to the plot.
            add_cbar: Whether to include a colorbar.
            cbar_label: Format string for the colorbar label.
            cmap: Matplotlib colormap name.
            xtick_locator: Locator controlling x-axis tick frequency.
            prange: Min/max dB levels displayed.
            noise_scale: Multiplier applied to the noise floor when masking.
            date_format: Major tick label format.
            del_ticks: Remove axis ticks before plotting (useful for grids).
            xdate_lims: Override for x-axis limits.

        Returns:
            DataFrame containing the filtered RTI samples.
        """
        logger.info(f"Extract Power/RTI, based on {flim}MHz {rlim}km")
        rti = pd.DataFrame()
        for ds in self.datasets:
            time = dt.datetime(ds.year, ds.month, ds.day, ds.hour, ds.minute, ds.second)
            logger.info(f"Time: {time}")
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
            o["time"] = time
            if (len(rlim) == 2) and (len(flim) == 2):
                o = o[
                    (o.range >= rlim[0])
                    & (o.range <= rlim[1])
                    & (o.frequency >= flim[0])
                    & (o.frequency <= flim[1])
                ]
            rti = pd.concat([rti, o])

        if fname is None:
            fname = f"{ds.URSI}_{rti.time.min().strftime('%Y%m%d.%H%M-')}{rti.time.max().strftime('%H%M')}_{mode}-mode.png"
        if xdate_lims is None:
            fig_title = f"""{ds.URSI}/{rti.time.min().strftime('%H%M-')}{rti.time.max().strftime('%H%M')} UT, {rti.time.max().strftime('%d %b %Y')}"""
        else:
            fig_title = f"""{ds.URSI}/{xdate_lims[0].strftime('%H%M-')}{xdate_lims[1].strftime('%H%M')} UT, {xdate_lims[1].strftime('%d %b %Y')}"""
        fig_title += (
            r"/ $f_0\sim$[" + "%.2f" % flim[0] + "-" + "%.2f" % flim[1] + "] MHz"
        )
        i = Ionogram(fig_title=fig_title, nrows=1, ncols=1)
        i.add_interval_plots(
            rti,
            mode=mode,
            xlabel=xlabel,
            ylabel=ylabel,
            xlim=xlim,
            add_cbar=add_cbar,
            cbar_label=cbar_label.format(mode),
            cmap=cmap,
            prange=prange,
            noise_scale=noise_scale,
            del_ticks=del_ticks,
            date_format=date_format,
            ylim=rlim,
            xtick_locator=xtick_locator,
            xdate_lims=xdate_lims,
        )
        i.save(os.path.join(folder, fname))
        i.close()
        return rti

    def save_scaled_dataset(self, params=["hmF2", "foF2"]):
        """Persist selected scaled parameters to a CSV file.

        Args:
            params: Sequence of attribute names to extract per dataset.
        """
        records = []
        for ds in self.datasets:
            rec = dict()
            for p in params:
                rec[p] = getattr(ds, p)
            records.append(rec)
        records = pd.DataFrame.from_records(records)
        fname = os.path.join(self.source_folder, "scaled.csv")
        records.to_csv(fname, float_format="%g", header=True, index=False)
        return

    def save_scaled_parameters(self, attr: dict = dict(), mode: str = "O"):
        """Write trace parameters into per-epoch NetCDF files.

        Args:
            attr: Extra dataset-level attributes to store in each NetCDF file.
            mode: Mode suffix applied to the output filenames.
        """
        folder = os.path.join(self.source_folder, "scaled")
        os.makedirs(folder, exist_ok=True)
        for ds in self.datasets:
            fname = os.path.join(
                folder, f"{mode}_mode_sp_{ds.time.strftime('%Y%m%d%H%M%S')}.nc"
            )
            logger.info(f"Saved to {fname}")
            max_n_trace = ds.get_n_traces()
            fs, hs = (
                np.zeros(max_n_trace) * np.nan,
                np.zeros(max_n_trace) * np.nan,
            )
            if ds.get_n_traces():
                fs, hs = (
                    ds.trace.trace_params.fs,
                    ds.trace.trace_params.hs,
                )
            data = {
                "fs": (("max_n_trace"), fs),
                "hs": (("max_n_trace"), hs),
                "sza": (("time",), [ds.sza]),
            }
            coords = dict(time=[ds.time], max_n_trace=np.arange(max_n_trace))
            d = xr.Dataset(data, coords=coords, attrs=attr)
            d.fs.attrs["units"], d.hs.attrs["units"] = "MHz", "km"
            d.fs.attrs["description"], d.hs.attrs["description"] = (
                "Plasma Frequencies",
                "Virtual Height/Reflection Height",
            )
            d.to_netcdf(fname)
            del d
        return
