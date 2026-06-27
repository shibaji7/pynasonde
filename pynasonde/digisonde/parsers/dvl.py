"""DVL (drift/velocity) file parser utilities for Digisonde outputs.

This module provides :class:`DvlExtractor` — a small parser for DVL
records exported by Digisonde software. It focuses on simple text-based
DVL records where each file contains a single line of whitespace-
separated fields. The extractor exposes convenience helpers that return
pandas.DataFrame objects suitable for plotting or downstream analysis.
"""

import datetime as dt
from typing import List

import numpy as np
import pandas as pd
from loguru import logger

from pynasonde.digisonde.digi_utils import (
    apply_filename_metadata,
    load_files_to_dataframe,
    to_namespace,
)


class DvlExtractor(object):
    """Parser for DVL-format records.

    The extractor assumes one DVL record per file containing a fixed set
    of whitespace-separated fields (see :attr:`key_order`). It parses the
    record into a structured dictionary (:attr:`dvl_struct`) and provides
    convenience static methods for batch-loading into pandas.
    """

    def __init__(
        self,
        filename: str,
        extract_time_from_name: bool = False,
        extract_stn_from_name: bool = False,
    ):
        """Create a DvlExtractor instance.

        Args:
            filename (str): Path to the DVL-format file to parse.
            extract_time_from_name (bool, optional): If True, attempt to parse a timestamp from the filename
                (default False). The expected filename format includes a
                YYYYDDDHHMMSS-like timestamp token at the end.
            extract_stn_from_name (bool, optional): If True, attempt to parse a station code from the filename
                (default False).
        """
        # Initialize the data structure to hold extracted data
        self.filename = filename
        apply_filename_metadata(
            self,
            self.filename,
            extract_time_from_name=extract_time_from_name,
            extract_stn_from_name=extract_stn_from_name,
        )
        date = getattr(self, "date", None)
        self.key_order = [
            "type",
            "version",
            "station_id",
            "ursi_tag",
            "lat",
            "lon",
            "date",
            "doy",
            "time",
            "Vx",
            "Vx_err",
            "Vy",
            "Vy_err",
            "Az",
            "Az_err",
            "Vh",
            "Vh_err",
            "Vz",
            "Vz_err",
            "Cord",
            "Hb",
            "Ht",
            "Fl",
            "Fu",
        ]
        self.dvl_struct = dict(
            type="",
            version="",
            station_id=np.nan,
            ursi_tag="",
            lat=np.nan,
            lon=np.nan,
            date=date.date() if date is not None else None,
            doy=date.timetuple().tm_yday if date is not None else np.nan,
            time=date.time() if date is not None else None,
            Vx=np.nan,  # in m/s; magnetic north direction
            Vx_err=np.nan,
            Vy=np.nan,  # in m/s; magnetic east direction
            Vy_err=np.nan,
            Az=np.nan,  # in m/s; counted clockwise from the magnetic north
            Az_err=np.nan,
            Vh=np.nan,
            Vh_err=np.nan,
            Vz=np.nan,  # in m/s; vertical velocity component
            Vz_err=np.nan,  #
            Cord="",  # in COM means Compass, GEO means Geographic, CGm means Corrected Geromagnetic
            Hb=np.nan,  # in km
            Ht=np.nan,  # in km
            Fl=np.nan,  # in MHz
            Fu=np.nan,  # in MHz
        )
        return

    def read_file(self) -> List[str]:
        """
        Reads the file line by line into a list.
        Returns:
            A list of strings, each representing a line from the file.
        """
        with open(self.filename, "r") as f:
            return f.readlines()

    def extract(self) -> dict:
        """Parse the DVL file and populate :attr:`dvl_struct`.

        The parser expects a single-line record of whitespace-separated
        fields in the order given by :attr:`key_order`. Each value is
        cast to the type of the corresponding entry in :attr:`dvl_struct`.

        Returns:
            The populated dictionary of parsed fields (also available as :attr:`dvl_struct`).
        """
        # Read file lines
        dvl_arch = self.read_file()
        dvl_arch = list(filter(None, dvl_arch[0].split()))
        for i, key in enumerate(self.key_order):
            if type(self.dvl_struct[key]) == dt.date:
                logger.info(
                    f"Already holds date information: {self.dvl_struct['date']}"
                )
            elif type(self.dvl_struct[key]) == dt.time:
                logger.info(
                    f"Already holds time information: {self.dvl_struct['time']}"
                )
            else:
                self.dvl_struct[key] = type(self.dvl_struct[key])(dvl_arch[i])
        self.dvl = to_namespace(self.dvl_struct)
        return self.dvl_struct

    @staticmethod
    def extract_DVL_pandas(
        file: str,
        extract_time_from_name: bool = True,
        extract_stn_from_name: bool = True,
    ) -> pd.DataFrame:
        """Convenience wrapper to extract a single file into a pandas row.

        Args:
            file (str): Path to the DVL file.
            extract_time_from_name (bool, optional): See :meth:`__init__`.
            extract_stn_from_name (bool, optional): See :meth:`__init__`.

        Returns:
            A 1-row DataFrame containing the parsed DVL record and two timestamp columns: ``datetime`` (UTC) and ``local_datetime``.
        """
        extractor = DvlExtractor(file, extract_time_from_name, extract_stn_from_name)
        df = pd.DataFrame.from_records([extractor.extract()])
        df["datetime"] = extractor.date
        df["local_datetime"] = extractor.local_time
        return df

    @staticmethod
    def load_DVL_files(
        folders: List[str] = None,
        ext: str = "*.DVL",
        n_procs: int = 4,
        extract_time_from_name: bool = True,
        extract_stn_from_name: bool = True,
    ) -> pd.DataFrame:
        """Recursively load DVL files from folders into a single DataFrame.

        Args:
            folders (list[str], optional): List of folders (globbed) to search for files.
            ext (str, optional): Filename glob pattern to match DVL files.
            n_procs (int, optional): Number of worker processes to use for parallel parsing.
            extract_time_from_name (bool, optional): See :meth:`__init__`.
            extract_stn_from_name (bool, optional): See :meth:`__init__`.

        Returns:
            Concatenated DataFrame containing parsed rows for all files discovered under the provided folders.
        """
        if folders is None:
            folders = []
        return load_files_to_dataframe(
            folders=folders,
            exts=ext,
            extractor=DvlExtractor.extract_DVL_pandas,
            n_procs=n_procs,
            extractor_kwargs=dict(
                extract_time_from_name=extract_time_from_name,
                extract_stn_from_name=extract_stn_from_name,
            ),
        )