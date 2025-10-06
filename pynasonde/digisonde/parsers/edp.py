import datetime as dt
import glob
import os
from functools import partial
from multiprocessing import Pool
from typing import List

import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm

from pynasonde.digisonde.digi_utils import to_namespace
from pynasonde.ngi.utils import TimeZoneConversion


class EdpExtractor(object):

    def __init__(
        self,
        filename: str,
        extract_time_from_name: bool = False,
        extract_stn_from_name: bool = False,
    ) -> None:
        """EDP (electron density profile) text-file parser for Digisonde outputs.

        This module provides :class:`EdpExtractor`, a lightweight parser for
        EDP files that contain F2-layer and profile information exported by
        Digisonde software. It focuses on parsing small text files into plain
        Python structures and pandas DataFrames suitable for documentation
        examples and downstream analysis.

        Parameters:
            filename: str
                Path to the EDP-format file to parse.
            extract_time_from_name: bool, optional
                If True, attempt to parse a timestamp from the filename (default False). The extractor expects a trailing YYYYDDDHHMMSS-like token in the filename when this is used.
            extract_stn_from_name: bool, optional
                If True, attempt to extract a station code from the filename (default False).
        """
        # Initialize the data structure to hold extracted data
        self.filename = filename
        if extract_time_from_name:
            date = self.filename.split("_")[-1].replace(".SAO", "").replace(".sao", "")
            self.date = dt.datetime(int(date[:4]), 1, 1) + dt.timedelta(
                int(date[4:7]) - 1
            )
            self.date = self.date.replace(
                hour=int(date[7:9]), minute=int(date[9:11]), second=int(date[11:13])
            )
            logger.info(f"Date: {self.date}")
        if extract_stn_from_name:
            self.stn_code = self.filename.split("/")[-1].split("_")[0]
        return

    def __update_tz__(self) -> None:
        """Update station timezone information from parsed F2 metadata."""
        self.stn_info = dict(
            LAT=self.edp_struct["f2"]["lat"], LONG=self.edp_struct["f2"]["lon"]
        )
        self.local_timezone_converter = TimeZoneConversion(
            lat=self.stn_info["LAT"], long=self.stn_info["LONG"]
        )
        self.local_time = self.local_timezone_converter.utc_to_local_time([self.date])[
            0
        ]
        logger.info(f"Station code: {self.stn_code}; {self.stn_info}")
        return

    def read_file(self) -> List[str]:
        """Update station timezone information from parsed F2 metadata.

        This helper extracts latitude/longitude from the parsed F2
        header (``edp_struct['f2']``), constructs a
        :class:`TimeZoneConversion` and computes :attr:`local_time`.

        Returns:
            Lines from the file including newline characters.
        """
        with open(self.filename, "r") as f:
            return f.readlines()

    def __check_issues__(self, line1: str, n: int) -> bool:
        """Inspect header lines for problem flags or truncated files.

        Parameters:
            line1: str
                The second header line from the file (used to detect flags).
            n: int
                Number of lines in the file (used to detect truncated output).

        Returns:
            True if the file appears to have an issue (bad flags or too few lines), False otherwise.
        """
        tag = (
            True
            if (("Problem Flags.Check" in line1) and ("failed" in line1) or (n < 5))
            else False
        )
        return tag

    def __parse_F2_datasets__(self, lines: List[str]) -> None:
        """Parse the F2 header and value lines into ``edp_struct['f2']``.

        Parameters:
            lines: list[str]
                Two-line sequence where the first is a whitespace-separated header and the second contains numeric values.
        """
        header, values = (
            list(filter(None, lines[0].replace("\n", "").split(" "))),
            list(filter(None, lines[1].replace("\n", "").split(" "))),
        )
        self.edp_struct["f2"] = dict(zip(header, values))
        for k in self.edp_struct["f2"].keys():
            self.edp_struct["f2"][k] = float(self.edp_struct["f2"][k])
        self.edp_struct["f2"]["lon"] = (self.edp_struct["f2"]["lon"] + 180) % 360 - 180
        return

    def extract(self) -> SimpleNamespace:
        """Parse the EDP file and return a namespace-wrapped structure.

        Behavior
        --------
        - Loads the file lines via :meth:`read_file`.
        - Parses the first two F2 lines with :meth:`__parse_F2_datasets__`.
        - Updates timezone information with :meth:`__update_tz__`.
        - Parses the remaining profile rows into ``edp_struct['profile']``
            unless problem flags are detected by :meth:`__check_issues__`.

        Returns:
            A namespace wrapping the parsed dictionary. The raw dict is available as :attr:`edp_struct`.
        """
        logger.info(f"Loading file: {self.filename}")
        self.edp_struct = dict(f2=dict(), profile=[], other=dict())
        edp_arch_list = self.read_file()
        self.__parse_F2_datasets__(edp_arch_list[:2])
        self.__update_tz__()
        if not self.__check_issues__(edp_arch_list[1], len(edp_arch_list)):
            header = list(filter(None, edp_arch_list[2].replace("\n", "").split(" ")))
            head_base, head_tail = header[:7], header[7:]
            for j, l in enumerate(edp_arch_list[3:]):
                l = l[:56] + l[57:65] + l[66:]
                l = list(filter(None, l.replace("\n", "").split(" ")))
                if j == 0:
                    self.edp_struct["other"] = dict(
                        zip(head_tail, np.array(l[7:]).astype(float))
                    )
                self.edp_struct["profile"].append(
                    dict(
                        zip(
                            head_base,
                            [float(x) if i < 5 else x for i, x in enumerate(l[:7])],
                        )
                    )
                )
        else:
            logger.warning(f"Error in flags of file: {self.filename}")

        self.edp = to_namespace(self.edp_struct)
        return self.edp

    @staticmethod
    def extract_EDP(
        file: str,
        extract_time_from_name: bool = True,
        extract_stn_from_name: bool = True,
        func_name: str = "height_profile",
    ) -> pd.DataFrame:
        """Convenience function to extract a single EDP file into a DataFrame.

        Parameters:
            file: str
                Path to the EDP file.
            extract_time_from_name: bool, optional
                See :meth:`__init__`.
            extract_stn_from_name: bool, optional
                See :meth:`__init__`.
            func_name: str, optional
                Which view to return. ``'height_profile'`` returns the profile
                rows as a DataFrame (default). ``'scaled'`` returns the F2
                header as a single-row DataFrame.

        Returns:
            DataFrame corresponding to the requested view (profile or scaled F2 header). If ``func_name`` is unrecognized an empty DataFrame is returned.
        """

        ex = EdpExtractor(file, extract_time_from_name, extract_stn_from_name)
        ex.extract()
        if func_name == "height_profile":
            df = pd.DataFrame.from_records(ex.edp_struct["profile"])
            df = set_timestamp(df, ex)
        elif func_name == "scaled":
            df = pd.DataFrame.from_records([ex.edp_struct["f2"]])
            df = set_timestamp(df, ex)
        else:
            df = pd.DataFrame()
        return df

    @staticmethod
    def load_EDP_files(
        folders: List[str] = [],
        ext: str = "*.EDP",
        n_procs: int = 8,
        extract_time_from_name: bool = True,
        extract_stn_from_name: bool = True,
        func_name: str = "height_profile",
    ) -> pd.DataFrame:
        """Load EDP files from one or more folders into a single DataFrame.

        Parameters:
            folders: list[str], optional
                Folders to search for files (can be empty). Each folder will be
                globbed using ``ext``.
            ext: str, optional
                File glob pattern to match (default ``'*.EDP'``).
            n_procs: int, optional
                Number of worker processes to use for parallel extraction.
            extract_time_from_name: bool, optional
                See :meth:`__init__`.
            extract_stn_from_name: bool, optional
                See :meth:`__init__`.
            func_name: str, optional
                Passed to :meth:`extract_EDP` to select the returned view.

        Returns:
            Concatenated DataFrame containing extracted rows from all discovered files.
        """


if __name__ == "__main__":
    coll = EdpExtractor.load_EDP_files(
        [
            "/media/chakras4/Crucial X9/NOAA_Archives/profilers-sounders/ionosonde/mids09/BC840/2017/233/scaled/"
        ],  # func_name="scaled"
    )
    print(coll.columns)
