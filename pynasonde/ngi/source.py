import bz2
import datetime as dt
import glob
import os
import shutil
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger


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
        return self


class DataSource(object):
    """ """

    def __init__(
        self,
        source_folder: str = "./tmp/",
        file_ext: str = "*.ngi.bz2",
        file_names: List[str] = [],
        needs_decompression: bool = False,
    ):
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

    def load_data_sets(self, load_start=0, load_end=-1):
        """ """
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
        for f in self.file_paths[load_start:load_end]:
            if check_bad_file(f):
                logger.info(f"Load file: {f}")
                if self.needs_decompression:
                    decompress(f, f.replace(".bz2", ""))
                    os.remove(f)
                    f = f.replace(".bz2", "")
                self.datasets.append(
                    Dataset().__initialize__(xr.load_dataset(f, engine="netcdf4"))
                )
                if self.needs_decompression:
                    compress(f + ".bz2", f)
                    os.remove(f)
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
        flim: List[float] = [3.95, 4.05],
        mode: str = "O",
        index: int = 0,
    ) -> pd.DataFrame:
        logger.info(f"Extract FTI/RTI, based on {flim}MHz {rlim}km")
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

        fname = f"{index}_{ds.URSI}_{rti.time.min().strftime('%Y%m%d.%H%M-')}{rti.time.max().strftime('%H%M')}_{mode}-mode.png"
        fig_title = f"""{ds.URSI}/{rti.time.min().strftime('%H%M-')}{rti.time.max().strftime('%H%M')} UT, {rti.time.max().strftime('%d %b %Y')}"""
        fig_title += (
            r"/ $f_0\sim$[" + "%.2f" % flim[0] + "-" + "%.2f" % flim[1] + "] MHz"
        )
        i = Ionogram(fig_title=fig_title)
        i.add_interval_plots(rti, mode=mode)
        i.save(os.path.join(folder, fname))
        i.close()
        return rti

    def save_scaled_dataset(self, params=["hmF2", "foF2"]):
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
