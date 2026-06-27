"""Parser for Digisonde EDP electron-density profile text outputs."""

from types import SimpleNamespace
from typing import List

import numpy as np
import pandas as pd
from loguru import logger

from pynasonde.digisonde.digi_utils import (
    apply_filename_metadata,
    load_files_to_dataframe,
    to_namespace,
)
from pynasonde.vipir.ngi.utils import TimeZoneConversion


def set_timestamp(df: pd.DataFrame, extractor: "EdpExtractor") -> pd.DataFrame:
    """Attach UTC/local timestamp and station metadata when available."""
    if hasattr(extractor, "date"):
        df["datetime"] = extractor.date
    if hasattr(extractor, "local_time"):
        df["local_datetime"] = extractor.local_time
    if hasattr(extractor, "stn_info"):
        df["lat"] = extractor.stn_info["LAT"]
        df["lon"] = extractor.stn_info["LONG"]
    if hasattr(extractor, "stn_code"):
        df["station"] = extractor.stn_code
    df["source_file"] = extractor.filename
    return df


class EdpExtractor(object):
    """Extract F2 metadata and height profiles from one EDP file."""

    def __init__(
        self,
        filename: str,
        extract_time_from_name: bool = False,
        extract_stn_from_name: bool = False,
    ) -> None:
        """Create an EDP extractor.

        Args:
            filename: Path to the EDP-format file to parse.
            extract_time_from_name: If True, parse a timestamp from the
                filename.
            extract_stn_from_name: If True, parse a station code from the
                filename.
        """
        # Initialize the data structure to hold extracted data
        self.filename = filename
        apply_filename_metadata(
            self,
            self.filename,
            extract_time_from_name=extract_time_from_name,
            extract_stn_from_name=extract_stn_from_name,
            load_station_info=False,
        )
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
        """Read the EDP text file.

        Returns:
            Lines from the file including newline characters.
        """
        with open(self.filename, "r") as f:
            return f.readlines()

    def __check_issues__(self, line1: str, n: int) -> bool:
        """Inspect header lines for problem flags or truncated files.

        Args:
            line1 (str): The second header line from the file (used to detect flags).
            n (int): Number of lines in the file (used to detect truncated output).

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

        Args:
            lines (list[str]): Two-line sequence where the first is a whitespace-separated header and the second contains numeric values.
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

        Args:
            file (str): Path to the EDP file.
            extract_time_from_name (bool, optional): See :meth:`__init__`.
            extract_stn_from_name (bool, optional): See :meth:`__init__`.
            func_name (str, optional): Which view to return. ``'height_profile'`` returns the profile
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

        Args:
            folders (list[str], optional): Folders to search for files (can be empty). Each folder will be
                globbed using ``ext``.
            ext (str, optional): File glob pattern to match (default ``'*.EDP'``).
            n_procs (int, optional): Number of worker processes to use for parallel extraction.
            extract_time_from_name (bool, optional): See :meth:`__init__`.
            extract_stn_from_name (bool, optional): See :meth:`__init__`.
            func_name (str, optional): Passed to :meth:`extract_EDP` to select the returned view.

        Returns:
            Concatenated DataFrame containing extracted rows from all discovered files.
        """
        return load_files_to_dataframe(
            folders=folders,
            exts=ext,
            extractor=EdpExtractor.extract_EDP,
            n_procs=n_procs,
            extractor_kwargs=dict(
                extract_time_from_name=extract_time_from_name,
                extract_stn_from_name=extract_stn_from_name,
                func_name=func_name,
            ),
        )
