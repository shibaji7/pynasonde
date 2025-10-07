"""SAO file parsers and helpers for Digisonde Standard Archiving Output.

This module provides: class:`SaoExtractor` which can read legacy fixed-
width SAO files as well as modern XML SAO exports. It offers helpers to
convert parsed content into pandas DataFrames and lightweight
namespaces for examples and downstream analysis. The implementation is
kept intentionally small and avoids heavy optional dependencies at
import-time so it can be used in documentation builds.
"""

import datetime as dt
import glob
import os
import re
from functools import partial
from multiprocessing import Pool
from typing import List

import numpy as np
import pandas as pd
from loguru import logger
from lxml import etree
from tqdm import tqdm

from pynasonde.digisonde.datatypes.saoxmldatatypes import SAORecordList
from pynasonde.digisonde.digi_plots import SaoSummaryPlots
from pynasonde.digisonde.digi_utils import (
    get_digisonde_info,
    is_valid_xml_data_string,
    load_dtd_file,
    to_namespace,
)
from pynasonde.vipir.ngi.utils import TimeZoneConversion


class SaoExtractor(object):
    """Extractor for SAO-format files (text and XML).

    The extractor supports two input styles:
    - legacy fixed-width ``.SAO`` text files parsed into a structured
      dictionary (``sao_struct``), and
    - XML exports validated and loaded via the datatypes loader
      (``SAORecordList``).
    """

    def __init__(
        self,
        filename: str,
        extract_time_from_name: bool = False,
        extract_stn_from_name: bool = False,
        dtd_file: str = None,
    ) -> None:
        """Create an extractor for the provided file.

        Parameters:
            filename:  str
                Path to the SAO or XML file to parse.
            extract_time_from_name:  bool, optional
                If True, attempt to parse a timestamp from the filename.
            extract_stn_from_name:  bool, optional
                If True, fetch station metadata and compute local time.
            dtd_file:  str, optional
                Optional DTD file used to validate XML inputs.
        """
        # Initialize the data structure to hold extracted data
        self.filename = filename
        self.xml_file = True if filename.split(".")[-1].lower() == "xml" else False
        self.dtd_file = dtd_file
        self.sao_struct = {}
        if self.xml_file:
            date = self.filename.split("_")[-2]
            self.stn_code = self.filename.split("/")[-1].split("_")[0]
        else:
            date = self.filename.split("_")[-1].replace(".SAO", "").replace(".sao", "")
            self.stn_code = self.filename.split("/")[-1].split("_")[0]
        if extract_time_from_name:
            self.date = dt.datetime(int(date[:4]), 1, 1) + dt.timedelta(
                int(date[4:7]) - 1
            )
            self.date = self.date.replace(
                hour=int(date[7:9]), minute=int(date[9:11]), second=int(date[11:13])
            )
            logger.info(f"Date: {self.date}")
        if extract_stn_from_name:
            self.stn_info = get_digisonde_info(self.stn_code)
            self.local_timezone_converter = TimeZoneConversion(
                lat=self.stn_info["LAT"], long=self.stn_info["LONG"]
            )
            self.local_time = self.local_timezone_converter.utc_to_local_time(
                [self.date]
            )[0]
            logger.info(f"Station code: {self.stn_code}; {self.stn_info}")
        return

    def read_file(self) -> List[str]:
        """Read the file and return a list of lines without trailing newline.

        Returns:
            List of strings (lines) with trailing newline removed.
        """
        with open(self.filename, "r") as f:
            SAOarch = [line.rstrip("\n") for line in f]
            return SAOarch

    def pad(self, s, length, pad_char=" ") -> str:
        return s.ljust(length, pad_char)

    def parse_line(self, line: str, fmt: str, num_ch: int) -> List:
        """Parse a fixed-width chunked line according to a format token.

        Parameters:
            line:  str
                The line to parse (a concatenated fixed-width string).
            fmt:  str
                Format token (e.g. '%7.3f', '%120c') that controls parsing and
                type coercion.
            num_ch:  int
                Width of each chunk in characters.

        Returns:
            Parsed values; numeric tokens are converted to float when
                possible, otherwise None is used for parsing failures.
        """
        results = []
        for i in range(0, len(line), num_ch):
            chunk = line[i : i + num_ch]
            if fmt in ["%1c", "%120c"]:
                results.append(chunk)
            elif fmt in ["%1d", "%2d", "%3d", "%7.3f", "%8.3f", "%11.6f", "%20.12f"]:
                try:
                    results.append(float(chunk.strip()))
                except ValueError:
                    results.append(None)
            else:
                results.append(chunk)
        return results

    def extract_xml(self) -> None:
        """Load XML SAO data into: attr:`sao` using: class:`SAORecordList`.

        The optional attr:`dtd_file` is used for validation when provided.
        """
        self.sao = SAORecordList.load_from_xml(
            xml_path=self.filename, dtd_path=self.dtd_file
        )
        return

    def get_height_profile_xml(self, plot_ionogram: str = None) -> tuple:
        """Extract height profile and trace DataFrames from loaded XML SAO.

        Parameters:
            plot_ionogram:  str or None, optional
                Optional filename to save an ionogram plot. If provided, a
                plot is generated using: class:`SaoSummaryPlots`.

        Returns:
            (profile_df, trace_df) extracted from the XML records. Both
                DataFrames include ``datetime`` and ``local_datetime`` columns
                when station metadata is available.
        """
        profile, trace = pd.DataFrame(), dict(RangeList=[], FrequencyList=[])
        for sao_record in self.sao.SAORecord:
            for prof in sao_record.ProfileList.Profile:
                for profVal in prof.Tabulated.ProfileValueList:
                    profile[profVal.Name] = profVal.values
                profile["AltitudeList"] = prof.Tabulated.AltitudeList
            for trc in sao_record.TraceList.Trace:
                trace["RangeList"].extend(trc.RangeList)
                trace["FrequencyList"].extend(trc.FrequencyList)
        trace = pd.DataFrame.from_records(trace)
        (
            profile["datetime"],
            profile["local_datetime"],
            profile["lat"],
            profile["lon"],
        ) = (
            self.date,
            self.local_time,
            self.stn_info["LAT"],
            self.stn_info["LONG"],
        )
        (trace["datetime"], trace["local_datetime"], trace["lat"], trace["lon"]) = (
            self.date,
            self.local_time,
            self.stn_info["LAT"],
            self.stn_info["LONG"],
        )
        trace["datetime"] = self.date
        trace["local_datetime"] = self.local_time
        trace["lat"], trace["lon"] = (
            self.stn_info["LAT"],
            self.stn_info["LONG"],
        )
        if plot_ionogram:
            logger.info("Save figures...")
            sao_plot = SaoSummaryPlots()
            ax = sao_plot.plot_ionogram(
                profile,
                xparam="PlasmaFrequency",
                yparam="AltitudeList",
                text=f"{self.stn_code}/{self.date.strftime('%Y-%m-%d %H:%M:%S')}",
            )
            sao_plot.plot_ionogram(trace, ax=ax, lw=2, kind="trace", lcolor="b")
            sao_plot.save(plot_ionogram)
            sao_plot.close()
        return profile, trace

    def get_scaled_datasets_xml(
        self, params: List[str] = ["foEs", "foF1", "foF2", "h`Es", "hmF1", "hmF2"]
    ) -> pd.DataFrame:
        """Return selected characteristic parameters from XML SAO as a DataFrame.

        Parameters:
        params:  list[str], optional
            List of characteristic names to extract. Defaults to common
                scaled parameters like 'foF2', 'hmF2', etc.

        Returns:
            One-row-per-record DataFrame with extracted characteristic
                values; missing parameters are filled with NaN.
        """
        df = []
        for sao_record in self.sao.SAORecord:
            d = dict()
            for cid, ursi in enumerate(sao_record.CharacteristicList.URSI):
                if ursi.Name in params:
                    d.update({ursi.Name: ursi.Val})
            if len(d) == 0:
                d.update(zip(params, [np.nan] * len(params)))
            df.append(d)
        df = pd.DataFrame.from_records(df)
        if len(df):
            df["datetime"] = self.date
            df["local_datetime"] = self.local_time
            df["lat"], df["lon"] = (
                self.stn_info["LAT"],
                self.stn_info["LONG"],
            )
        return df

    def extract(self) -> dict:
        """Parse a legacy fixed-width SAO text file into a structured dict.

        Behavior
        --------
        The routine inspects the first two header lines to determine the
        number and sizes of subsequent arrays, then walks the remainder
        of the file concatenating fixed-width lines into fields. The
        output is stored in: attr:`SAOstruct` and: attr:`sao`.

        Returns:
            The raw dictionary of parsed SAO fields (same as
                : attr:`SAOstruct`).
        """
        # Read file lines
        SAOarch, self.SAOstruct = self.read_file(), dict()

        Dindex1 = [int(x) for x in re.findall(r".{3}", self.pad(SAOarch[0], 120))]
        Dindex2 = [int(x) for x in re.findall(r".{3}", self.pad(SAOarch[1], 120))]
        noe = Dindex1 + Dindex2

        fmt_cell = [
            "%7.3f",
            "%120c",
            "%1c",
            "%8.3f",
            "%2d",
            "%7.3f",
            "%8.3f",
            "%8.3f",
            "%3d",
            "%1d",
            "%8.3f",
            "%8.3f",
            "%8.3f",
            "%3d",
            "%1d",
            "%8.3f",
            "%8.3f",
            "%8.3f",
            "%3d",
            "%1d",
            "%8.3f",
            "%8.3f",
            "%3d",
            "%1d",
            "%8.3f",
            "%8.3f",
            "%3d",
            "%1d",
            "%8.3f",
            "%8.3f",
            "%3d",
            "%1d",
            "%8.3f",
            "%3d",
            "%3d",
            "%3d",
            "%11.6f",
            "%11.6f",
            "%11.6f",
            "%20.12f",
            "%1d",
            "%11.6f",
            "%8.3f",
            "%3d",
            "%1d",
            "%8.3f",
            "%8.3f",
            "%3d",
            "%1d",
            "%8.3f",
            "%8.3f",
            "%8.3f",
            "%8.3f",
            "%1c",
            "%1c",
            "%1c",
            "%11.6f",
            "%8.3f",
            "%8.3f",
            "%8.3f",
        ]
        var_cell = [
            "geophcon",
            "sysdes",
            "timesound",
            "Scaled",
            "aflags",
            "dopptab",
            "OTF2vh",
            "OTF2th",
            "OTF2amp",
            "OTF2dpn",
            "OTF2fq",
            "OTF1vh",
            "OTF1th",
            "OTF1amp",
            "OTF1dpn",
            "OTF1fq",
            "OTEvh",
            "OTEth",
            "OTEamp",
            "OTEdpn",
            "OTEfq",
            "XTF2vh",
            "XTF2amp",
            "XTF2dpn",
            "XTF2fq",
            "XTF1vh",
            "XTF1amp",
            "XTF1dpn",
            "XTF1fq",
            "XTEvh",
            "XTEamp",
            "XTEdpn",
            "XTEfq",
            "mdampF",
            "mdampE",
            "mdampEs",
            "thcf2",
            "thcf1",
            "thce",
            "QPS",
            "edflags",
            "vlydesc",
            "OTEsvh",
            "OTEsamp",
            "OTEsdpn",
            "OTEsfq",
            "OTEavh",
            "OTEaamp",
            "OTEadpn",
            "OTEafq",
            "TH",
            "PF",
            "ED",
            "Qletter",
            "Dletter",
            "edflagstp",
            "thcea",
            "thea",
            "pfea",
            "edea",
        ]
        scal_cell = [
            "foF2",
            "foF1",
            "M3000F",
            "MUF3000",
            "fmin",
            "foEs",
            "fminF",
            "fminE",
            "foE",
            "fxI",
            "hF",
            "hF2",
            "hE",
            "hEs",
            "hmE",
            "ymE",
            "QF",
            "QE",
            "downF",
            "downE",
            "downEs",
            "FF",
            "FE",
            "D",
            "fMUF",
            "hMUF",
            "dfoF",
            "foEp",
            "fhF",
            "fhF2",
            "foF1p",
            "hmF2",
            "hmF1",
            "h05NmF2",
            "foFp",
            "fminEs",
            "ymF2",
            "ymF1",
            "TEC",
            "Ht",
            "B0",
            "B1",
            "D1",
            "foEa",
            "hEa",
            "foP",
            "hP",
            "fbEs",
            "TypeEs",
        ]
        count = 2
        for i0, fmt in enumerate(fmt_cell):
            if noe[i0] == 0:
                continue
            # Determine num_ch from fmt
            if fmt == "%7.3f":
                num_ch = 7
            elif fmt == "%8.3f":
                num_ch = 8
            elif fmt == "%11.6f":
                num_ch = 11
            elif fmt == "%20.12f":
                num_ch = 20
            elif fmt == "%120c":
                num_ch = 120
            elif fmt == "%1c":
                num_ch = 1
            elif fmt == "%1d":
                num_ch = 1
            elif fmt == "%2d":
                num_ch = 2
            elif fmt == "%3d":
                num_ch = 3
            else:
                num_ch = 1

            # Concatenate lines until enough data
            expected_items = noe[i0]
            total_chars_needed = num_ch * expected_items
            line_in = ""
            while len(line_in) < total_chars_needed and count < len(SAOarch):
                line_in += self.pad(
                    SAOarch[count],
                    num_ch * ((len(SAOarch[count]) + num_ch - 1) // num_ch),
                )
                count += 1

            # Special case for Qletter/Dletter
            if var_cell[i0] in ["Qletter", "Dletter"]:
                line_in = self.pad(line_in, expected_items)

            # Parse data
            if var_cell[i0] != "Scaled":
                self.SAOstruct[var_cell[i0]] = []
                for i1 in range(expected_items):
                    chunk = line_in[num_ch * i1 : num_ch * (i1 + 1)]
                    aux_out = self.parse_line(chunk, fmt, num_ch)
                    self.SAOstruct[var_cell[i0]].append(aux_out[0] if aux_out else None)
            else:
                self.SAOstruct[var_cell[i0]] = {}
                for i1 in range(expected_items):
                    chunk = line_in[num_ch * i1 : num_ch * (i1 + 1)]
                    aux_out = self.parse_line(chunk, fmt, num_ch)
                    self.SAOstruct[var_cell[i0]][scal_cell[i1]] = (
                        aux_out[0] if aux_out else None
                    )

        if "sysdes" in self.SAOstruct:
            self.SAOstruct["sysdes"] = np.array(self.SAOstruct["sysdes"])

        for key in ["ED", "TH", "PF"]:
            if key in self.SAOstruct:
                self.SAOstruct[key] = list(
                    filter(
                        None,
                        [
                            x.strip() if type(x) is str else x
                            for x in self.SAOstruct[key]
                        ],
                    )
                )
                if type(self.SAOstruct[key][0]) is str:
                    self.SAOstruct[key] = [float(x) for x in self.SAOstruct[key]]
        self.sao = to_namespace(self.SAOstruct)
        return self.SAOstruct

    def get_scaled_datasets(self, asdf: bool = True) -> pd.DataFrame:
        """Return scaled dataset fields from parsed legacy SAO as a DataFrame.

        Parameters:
            asdf:  Retured as `df` if `True`.

        Returns:
            Single-row DataFrame containing scaled parameters with
                datetime/local_datetime columns when available.
        """
        for key in vars(self.sao.Scaled).keys():
            if type(vars(self.sao.Scaled)[key]) == str:
                setattr(self.sao.Scaled, key, [np.nan])

        o = pd.DataFrame.from_records(vars(self.sao.Scaled), index=[0])
        o.replace(9999.0, np.nan, inplace=True)
        if hasattr(self, "date"):
            o["datetime"] = self.date
        if hasattr(self, "local_time"):
            o["local_datetime"] = self.local_time
        return o

    def get_height_profile(
        self, asdf: bool = True, plot_ionogram: bool = False
    ) -> pd.DataFrame:
        """Return the height profile DataFrame extracted from the parsed SAO.

        Parameters:
            asdf:  Retured as `df` if `True`.
            plot_ionogram:  If True, generate and save an ionogram plot alongside the
                returned DataFrame.

        Returns:
            DataFrame with columns for height (`th`), plasma frequency
                (`pf`), electron density (`ed`) and timestamps when
                available.
        """
        o = pd.DataFrame()
        if (
            hasattr(self.sao, "PF")
            and hasattr(self.sao, "TH")
            and hasattr(self.sao, "ED")
        ):
            hlen, o["th"] = len(self.sao.TH), self.sao.TH
            if len(self.sao.PF) == hlen:
                o["pf"] = self.sao.PF
                o.pf = o.pf.astype(float)
            if len(self.sao.ED) == hlen:
                o["ed"] = self.sao.ED
                o.ed = o.ed.astype(float)
            o.th = o.th.astype(float)
            if hasattr(self, "date"):
                o["datetime"] = self.date
            if hasattr(self, "local_time"):
                o["local_datetime"] = self.local_time
            o["lat"], o["lon"] = (self.stn_info["LAT"], self.stn_info["LONG"])
            if plot_ionogram:
                logger.info("Save figures...")
                sao_plot = SaoSummaryPlots()
                sao_plot.plot_ionogram(
                    o, text=f"{self.stn_code}/{self.date.strftime('%Y-%m-%d %H:%M:%S')}"
                )
                sao_plot.save(self.filename.split(".")[0] + ".png")
                sao_plot.close()
        return o

    def display_struct(self) -> None:
        """Log the raw parsed SAO structure.

        This is a lightweight helper used for debugging and quick
        inspection during development.
        """
        logger.info(self.sao_struct)
        return

    @staticmethod
    def extract_SAO(
        file: str,
        extract_time_from_name: bool = True,
        extract_stn_from_name: bool = True,
        func_name: str = "height_profile",
    ) -> pd.DataFrame:
        """Convenience function to extract a single SAO/XML file into a DataFrame.

        Parameters:
            file:  str
                Path to the SAO or XML file.
            extract_time_from_name:  bool, optional
                If True, infer timestamps from the filename.
            extract_stn_from_name:  bool, optional
                If True, infer station and compute local time.
            func_name:  str, optional
                Which view to return: ``'height_profile'`` or ``'scaled'``.

        Returns:
            DataFrame corresponding to the requested view. For XML inputs
                both the scaled and height_profile views are supported.
        """
        extractor = SaoExtractor(file, extract_time_from_name, extract_stn_from_name)
        if extractor.xml_file:
            extractor.extract_xml()
        else:
            extractor.extract()
        if func_name == "height_profile":
            if extractor.xml_file:
                df, _ = extractor.get_height_profile_xml()
            else:
                df = extractor.get_height_profile()
        elif func_name == "scaled":
            if extractor.xml_file:
                df = extractor.get_scaled_datasets_xml()
            else:
                df = extractor.get_scaled_datasets()
        else:
            df = pd.DataFrame()
        return df

    @staticmethod
    def load_SAO_files(
        folders: List[str] = ["tmp/SKYWAVE_DPS4D_2023_10_13"],
        ext: str = "*.SAO",
        n_procs: int = 4,
        extract_time_from_name: bool = True,
        extract_stn_from_name: bool = True,
        func_name: str = "height_profile",
    ) -> pd.DataFrame:
        """Load fixed-width SAO files from folders into a single DataFrame.

        Parameters:
            folders:  list[str], optional
                List of folders to search for files.
            ext:  str, optional
                Glob pattern to match files (default ``'*.SAO'``).
            n_procs:  int, optional
                Number of worker processes used for parallel extraction.
            extract_time_from_name:  bool, optional
                See: meth:`extract_SAO`.
            extract_stn_from_name:  bool, optional
                See: meth:`extract_SAO`.
            func_name:  str, optional
                View to extract (``'height_profile'`` or ``'scaled'``).

        Returns:
            Concatenated DataFrame of all extracted rows.
        """
        collections = []
        for folder in folders:
            logger.info(f"Searching for files under: {os.path.join(folder, ext)}")
            files = glob.glob(os.path.join(folder, ext))
            files.sort()
            logger.info(f"N files: {len(files)}")
            with Pool(n_procs) as pool:
                df_collection = list(
                    tqdm(
                        pool.imap(
                            partial(
                                SaoExtractor.extract_SAO,
                                extract_time_from_name=extract_time_from_name,
                                extract_stn_from_name=extract_stn_from_name,
                                func_name=func_name,
                            ),
                            files,
                        ),
                        total=len(files),
                    )
                )
            collections.extend(df_collection)
        collections = pd.concat(collections)
        return collections

    @staticmethod
    def load_XML_files(
        folders: List[str] = [],
        ext: str = "*.XML",
        n_procs: int = 4,
        extract_time_from_name: bool = True,
        extract_stn_from_name: bool = True,
        func_name: str = "height_profile",
    ) -> pd.DataFrame:
        """Load XML SAO files from folders into a single DataFrame.

        Same behaviour as: meth:`load_SAO_files` but matches XML file
        extensions by default.
        """
        collections = []
        for folder in folders:
            logger.info(f"Searching for files under: {os.path.join(folder, ext)}")
            files = glob.glob(os.path.join(folder, ext))
            files.sort()
            logger.info(f"N files: {len(files)}")
            with Pool(n_procs) as pool:
                df_collection = list(
                    tqdm(
                        pool.imap(
                            partial(
                                SaoExtractor.extract_SAO,
                                extract_time_from_name=extract_time_from_name,
                                extract_stn_from_name=extract_stn_from_name,
                                func_name=func_name,
                            ),
                            files,
                        ),
                        total=len(files),
                    )
                )
            collections.extend(df_collection)
        collections = pd.concat(collections)
        return collections
