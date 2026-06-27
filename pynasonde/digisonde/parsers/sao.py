"""SAO file parsers and helpers for Digisonde Standard Archiving Output.

This module provides: class:`SaoExtractor` which can read legacy fixed-
width SAO files as well as modern XML SAO exports. It offers helpers to
convert parsed content into pandas DataFrames and lightweight
namespaces for examples and downstream analysis. The implementation is
kept intentionally small and avoids heavy optional dependencies at
import-time so it can be used in documentation builds.
"""

import datetime as dt
import re
from typing import List, Optional

import numpy as np
import pandas as pd
from loguru import logger

from pynasonde.digisonde.datatypes.saoxmldatatypes import SAORecordList
from pynasonde.digisonde.digi_plots import SaoSummaryPlots
from pynasonde.digisonde.digi_utils import (
    apply_filename_metadata,
    extract_datetime_token_from_filename,
    load_files_to_dataframe,
    parse_digisonde_datetime_token,
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

        Args:
            filename (str): Path to the SAO or XML file to parse.
            extract_time_from_name (bool, optional): If True, attempt to parse a timestamp from the filename.
            extract_stn_from_name (bool, optional): If True, fetch station metadata and compute local time.
            dtd_file (str, optional): Optional DTD file used to validate XML inputs.
        """
        # Initialize the data structure to hold extracted data
        self.filename = filename
        self.xml_file = True if filename.split(".")[-1].lower() == "xml" else False
        self.dtd_file = dtd_file
        self.sao_struct = {}
        apply_filename_metadata(
            self,
            self.filename,
            extract_time_from_name=extract_time_from_name,
            extract_stn_from_name=extract_stn_from_name,
            station_code_always=True,
        )
        return

    def _extract_datetime_token(self) -> str:
        """Return the trailing filename token likely containing date/time."""
        return extract_datetime_token_from_filename(self.filename)

    @staticmethod
    def _parse_datetime_token(token: str):
        """Parse known SAO datetime token formats."""
        return parse_digisonde_datetime_token(token)

    def read_file(self) -> List[str]:
        """Read the file and return a list of lines without trailing newline.

        Returns:
            List of strings (lines) with trailing newline removed.
        """
        with open(self.filename, "r") as f:
            SAOarch = [line.rstrip("\n") for line in f]
            return SAOarch

    def pad(self, s, length, pad_char=" ") -> str:
        """Right-pad a string to the requested fixed width."""
        return s.ljust(length, pad_char)

    def parse_line(self, line: str, fmt: str, num_ch: int) -> List:
        """Parse a fixed-width chunked line according to a format token.

        Args:
            line (str): The line to parse (a concatenated fixed-width string).
            fmt (str): Format token (e.g. '%7.3f', '%120c') that controls parsing and
                type coercion.
            num_ch (int): Width of each chunk in characters.

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

    @staticmethod
    def _is_index_line(line: str) -> bool:
        """Return True when line resembles a 120-char SAO index header line."""
        try:
            chunks = re.findall(r".{3}", line.ljust(120)[:120])
            if len(chunks) != 40:
                return False
            for chunk in chunks:
                token = chunk.strip()
                if token == "":
                    token = "0"
                int(token)
            return True
        except ValueError:
            return False

    @staticmethod
    def _parse_index_line(line: str) -> List[int]:
        """Parse a SAO index header line into its 40 integer slots."""
        out = []
        for chunk in re.findall(r".{3}", line.ljust(120)[:120]):
            token = chunk.strip()
            out.append(int(token) if token else 0)
        return out

    @staticmethod
    def _parse_ff_datetime(ff_line: str) -> Optional[dt.datetime]:
        """Parse UTC timestamp from a SAO FF line.

        Expected prefix layout:
        ``FFYYYYDDDMMDDHHMM...``
        """
        m = re.match(
            r"^FF(?P<year>\d{4})(?P<doy>\d{3})(?P<mmdd>\d{4})(?P<hhmm>\d{4}).*$",
            ff_line.strip(),
        )
        if not m:
            return None
        year = int(m.group("year"))
        doy = int(m.group("doy"))
        hhmm = m.group("hhmm")
        date = dt.datetime(year, 1, 1) + dt.timedelta(doy - 1)
        date = date.replace(hour=int(hhmm[:2]), minute=int(hhmm[2:4]), second=0)
        if date.strftime("%m%d") != m.group("mmdd"):
            logger.warning(
                f"FF line month/day mismatch: parsed {date.strftime('%m%d')} != token {m.group('mmdd')}"
            )
        return date

    @classmethod
    def _find_record_starts(cls, lines: List[str]) -> List[int]:
        """Find start line indices of SAO records in a text file."""
        starts = []
        for i in range(max(0, len(lines) - 1)):
            if not (cls._is_index_line(lines[i]) and cls._is_index_line(lines[i + 1])):
                continue
            d1 = cls._parse_index_line(lines[i])
            d2 = cls._parse_index_line(lines[i + 1])
            if not (len(d1) > 0 and len(d2) > 0 and d1[0] == 5 and d2[0] == 49):
                continue
            lookahead = lines[i + 2 : i + 8]
            if any(re.match(r"^FF\d+", ln) for ln in lookahead):
                starts.append(i)
        if (
            not starts
            and len(lines) >= 2
            and cls._is_index_line(lines[0])
            and cls._is_index_line(lines[1])
        ):
            starts = [0]
        return starts

    @classmethod
    def _detect_sao_layout(cls, lines: List[str]) -> str:
        """Classify the SAO text layout as single or multi record."""
        starts = cls._find_record_starts(lines)
        return "multi" if len(starts) > 1 else "single"

    @staticmethod
    def _resolve_record_index(record_index: int, n_records: int) -> int:
        """Resolve positive/negative record index and validate bounds."""
        if n_records <= 0:
            raise ValueError("No SAO records detected in file.")
        idx = record_index if record_index >= 0 else n_records + record_index
        if idx < 0 or idx >= n_records:
            raise IndexError(
                f"record_index {record_index} out of bounds for {n_records} records."
            )
        return idx

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

        Args:
            plot_ionogram (str or None, optional): Optional filename to save an ionogram plot. If provided, a
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
        self,
        params: List[str] = [
            "foEs",
            "foF1",
            "foF2",
            "h`Es",
            "hmF1",
            "hmF2",
            "hmE",
            "foEp",
            "foE",
        ],
    ) -> pd.DataFrame:
        """Return selected characteristic parameters from XML SAO as a DataFrame.

        Args:
            params: List of characteristic names to extract. Defaults to common
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
            for cid, mod in enumerate(sao_record.CharacteristicList.Modeled):
                if mod.Name in params:
                    d.update({mod.Name: mod.Val})
            if len(d) == 0:
                d.update(zip(params, [np.nan] * len(params)))
            d.update(
                dict(
                    datetime=sao_record.StartTimeUTC,
                    lat=sao_record.GeoLatitude,
                    lon=sao_record.GeoLongitude,
                    ursi_code=sao_record.URSICode,
                )
            )
            df.append(d)
        df = pd.DataFrame.from_records(df)
        return df

    def _extract_record_struct(self, lines: List[str]) -> dict:
        """Parse one SAO record chunk into a structured dictionary."""
        if len(lines) < 2:
            return {}
        Dindex1 = self._parse_index_line(lines[0])
        Dindex2 = self._parse_index_line(lines[1])
        noe = Dindex1 + Dindex2
        sao_struct = dict()
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
            elif fmt in ["%1c", "%1d"]:
                num_ch = 1
            elif fmt == "%2d":
                num_ch = 2
            elif fmt == "%3d":
                num_ch = 3
            else:
                num_ch = 1

            expected_items = int(noe[i0])
            total_chars_needed = num_ch * expected_items
            line_in = ""
            while len(line_in) < total_chars_needed and count < len(lines):
                line_in += self.pad(
                    lines[count], num_ch * ((len(lines[count]) + num_ch - 1) // num_ch)
                )
                count += 1
            if var_cell[i0] in ["Qletter", "Dletter"]:
                line_in = self.pad(line_in, expected_items)
            if var_cell[i0] != "Scaled":
                sao_struct[var_cell[i0]] = []
                for i1 in range(expected_items):
                    chunk = line_in[num_ch * i1 : num_ch * (i1 + 1)]
                    aux_out = self.parse_line(chunk, fmt, num_ch)
                    sao_struct[var_cell[i0]].append(aux_out[0] if aux_out else None)
            else:
                sao_struct[var_cell[i0]] = {}
                for i1 in range(expected_items):
                    chunk = line_in[num_ch * i1 : num_ch * (i1 + 1)]
                    aux_out = self.parse_line(chunk, fmt, num_ch)
                    sao_struct[var_cell[i0]][scal_cell[i1]] = (
                        aux_out[0] if aux_out else None
                    )

        if "sysdes" in sao_struct:
            sao_struct["sysdes"] = np.array(sao_struct["sysdes"])
        for key in ["ED", "TH", "PF"]:
            if key not in sao_struct:
                continue
            cleaned = list(
                filter(
                    None,
                    [x.strip() if isinstance(x, str) else x for x in sao_struct[key]],
                )
            )
            if len(cleaned) > 0 and isinstance(cleaned[0], str):
                cleaned = [float(x) for x in cleaned]
            sao_struct[key] = cleaned
        return sao_struct

    def _to_local_datetime(
        self, utc_time: Optional[dt.datetime]
    ) -> Optional[dt.datetime]:
        """Convert one UTC datetime to local station time when possible."""
        if utc_time is None or not hasattr(self, "stn_info"):
            return None
        if not hasattr(self, "local_timezone_converter"):
            self.local_timezone_converter = TimeZoneConversion(
                lat=self.stn_info["LAT"], long=self.stn_info["LONG"]
            )
        return self.local_timezone_converter.utc_to_local_time([utc_time])[0]

    def extract(self, mode: str = "auto", record_index: int = 0):
        """Parse legacy SAO text with single/multi record support.

        Args:
            mode: ``'auto'`` | ``'single'`` | ``'multi'``.
            record_index: Record index used when ``mode='single'`` and file is multi.
        """
        lines = self.read_file()
        starts = self._find_record_starts(lines)
        if len(starts) == 0:
            starts = [0]
        detected_mode = "multi" if len(starts) > 1 else "single"
        if mode == "auto":
            mode = detected_mode
        if mode not in ["single", "multi"]:
            raise ValueError("mode must be one of: auto, single, multi")

        records = []
        if mode == "single":
            idx = self._resolve_record_index(record_index, len(starts))
            selected = [idx]
        else:
            selected = list(range(len(starts)))

        for idx in selected:
            start = starts[idx]
            end = starts[idx + 1] if idx + 1 < len(starts) else len(lines)
            section = lines[start:end]
            record = self._extract_record_struct(section)
            ff_line = next((ln for ln in section if re.match(r"^FF\d+", ln)), None)
            record_dt = (
                self._parse_ff_datetime(ff_line)
                if ff_line
                else getattr(self, "date", None)
            )
            records.append(
                dict(
                    struct=record,
                    record_datetime=record_dt,
                    record_index=idx,
                )
            )

        self.sao_records = records
        if len(records) == 1:
            self.SAOstruct = records[0]["struct"]
            self.sao = to_namespace(self.SAOstruct)
            if records[0]["record_datetime"] is not None:
                self.date = records[0]["record_datetime"]
                local_dt = self._to_local_datetime(self.date)
                if local_dt is not None:
                    self.local_time = local_dt
            return self.SAOstruct

        self.SAOstruct = [rec["struct"] for rec in records]
        self.sao = [to_namespace(rec["struct"]) for rec in records]
        return self.SAOstruct

    def get_scaled_datasets(self, asdf: bool = True) -> pd.DataFrame:
        """Return scaled dataset fields from parsed legacy SAO as a DataFrame.

        Args:
            asdf: Return as a DataFrame when ``True``.

        Returns:
            Single-row DataFrame containing scaled parameters with
                datetime/local_datetime columns when available.
        """
        if hasattr(self, "sao_records") and len(self.sao_records) > 1:
            rows = []
            for rec in self.sao_records:
                scaled = dict(rec["struct"].get("Scaled", {}))
                for key, value in scaled.items():
                    if isinstance(value, str):
                        scaled[key] = np.nan
                o = pd.DataFrame.from_records([scaled])
                o.replace(9999.0, np.nan, inplace=True)
                rec_dt = rec.get("record_datetime")
                if rec_dt is not None:
                    o["datetime"] = rec_dt
                    local_dt = self._to_local_datetime(rec_dt)
                    if local_dt is not None:
                        o["local_datetime"] = local_dt
                elif hasattr(self, "date"):
                    o["datetime"] = self.date
                    if hasattr(self, "local_time"):
                        o["local_datetime"] = self.local_time
                if hasattr(self, "stn_info"):
                    o["lat"], o["lon"] = self.stn_info["LAT"], self.stn_info["LONG"]
                o["record_index"] = rec.get("record_index")
                o["source_file"] = self.filename
                if rec.get("ff_line"):
                    o["ff_line"] = rec["ff_line"]
                rows.append(o)
            return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

        for key in vars(self.sao.Scaled).keys():
            if isinstance(vars(self.sao.Scaled)[key], str):
                setattr(self.sao.Scaled, key, [np.nan])
        o = pd.DataFrame.from_records(vars(self.sao.Scaled), index=[0])
        o.replace(9999.0, np.nan, inplace=True)
        if hasattr(self, "date"):
            o["datetime"] = self.date
        if hasattr(self, "local_time"):
            o["local_datetime"] = self.local_time
        if hasattr(self, "stn_info"):
            o["lat"], o["lon"] = self.stn_info["LAT"], self.stn_info["LONG"]
        o["record_index"] = 0
        o["source_file"] = self.filename
        if (
            hasattr(self, "sao_records")
            and len(self.sao_records) > 0
            and self.sao_records[0].get("ff_line")
        ):
            o["ff_line"] = self.sao_records[0]["ff_line"]
        return o

    def get_height_profile(
        self, asdf: bool = True, plot_ionogram: bool = False
    ) -> pd.DataFrame:
        """Return the height profile DataFrame extracted from the parsed SAO.

        Args:
            asdf: Return as a DataFrame when ``True``.
            plot_ionogram: If True, generate and save an ionogram plot
                alongside the returned DataFrame.

        Returns:
            DataFrame with columns for height (`th`), plasma frequency
                (`pf`), electron density (`ed`) and timestamps when
                available.
        """
        if hasattr(self, "sao_records") and len(self.sao_records) > 1:
            profiles = []
            for rec in self.sao_records:
                struct = rec["struct"]
                if not all(k in struct for k in ["TH", "PF", "ED"]):
                    continue
                o = pd.DataFrame()
                hlen, o["th"] = len(struct["TH"]), struct["TH"]
                if len(struct["PF"]) == hlen:
                    o["pf"] = pd.Series(struct["PF"]).astype(float)
                if len(struct["ED"]) == hlen:
                    o["ed"] = pd.Series(struct["ED"]).astype(float)
                o.th = o.th.astype(float)
                rec_dt = rec.get("record_datetime")
                if rec_dt is not None:
                    o["datetime"] = rec_dt
                    local_dt = self._to_local_datetime(rec_dt)
                    if local_dt is not None:
                        o["local_datetime"] = local_dt
                elif hasattr(self, "date"):
                    o["datetime"] = self.date
                    if hasattr(self, "local_time"):
                        o["local_datetime"] = self.local_time
                if hasattr(self, "stn_info"):
                    o["lat"], o["lon"] = self.stn_info["LAT"], self.stn_info["LONG"]
                o["record_index"] = rec.get("record_index")
                o["source_file"] = self.filename
                if rec.get("ff_line"):
                    o["ff_line"] = rec["ff_line"]
                profiles.append(o)
            out = pd.concat(profiles, ignore_index=True) if profiles else pd.DataFrame()
            if plot_ionogram and len(out) > 0:
                logger.warning("plot_ionogram is ignored for multi-record SAO files.")
            return out

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
            if hasattr(self, "stn_info"):
                o["lat"], o["lon"] = self.stn_info["LAT"], self.stn_info["LONG"]
            o["record_index"] = 0
            o["source_file"] = self.filename
            if (
                hasattr(self, "sao_records")
                and len(self.sao_records) > 0
                and self.sao_records[0].get("ff_line")
            ):
                o["ff_line"] = self.sao_records[0]["ff_line"]
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
        mode: str = "auto",
        record_index: int = 0,
    ) -> pd.DataFrame:
        """Convenience function to extract a single SAO/XML file into a DataFrame.

        Args:
            file (str): Path to the SAO or XML file.
            extract_time_from_name (bool, optional): If True, infer timestamps from the filename.
            extract_stn_from_name (bool, optional): If True, infer station and compute local time.
            func_name (str, optional): Which view to return: ``'height_profile'`` or ``'scaled'``.
            mode (str, optional): SAO parsing mode for text files: ``'auto'``, ``'single'``, ``'multi'``.
            record_index (int, optional): Record index when ``mode='single'`` on multi-record SAO files.

        Returns:
            DataFrame corresponding to the requested view. For XML inputs
                both the scaled and height_profile views are supported.
        """
        extractor = SaoExtractor(file, extract_time_from_name, extract_stn_from_name)
        if extractor.xml_file:
            extractor.extract_xml()
        else:
            extractor.extract(mode=mode, record_index=record_index)
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
        mode: str = "auto",
        record_index: int = 0,
    ) -> pd.DataFrame:
        """Load fixed-width SAO files from folders into a single DataFrame.

        Args:
            folders (list[str], optional): List of folders to search for files.
            ext (str, optional): Glob pattern to match files (default ``'*.SAO'``).
            n_procs (int, optional): Number of worker processes used for parallel extraction.
            extract_time_from_name: See :meth:`extract_SAO`.
            extract_stn_from_name: See :meth:`extract_SAO`.
            func_name (str, optional): View to extract (``'height_profile'`` or ``'scaled'``).
            mode (str, optional): SAO parsing mode for text files.
            record_index (int, optional): Record index used when forcing ``mode='single'``.

        Returns:
            Concatenated DataFrame of all extracted rows.
        """
        return load_files_to_dataframe(
            folders=folders,
            exts=ext,
            extractor=SaoExtractor.extract_SAO,
            n_procs=n_procs,
            extractor_kwargs=dict(
                extract_time_from_name=extract_time_from_name,
                extract_stn_from_name=extract_stn_from_name,
                func_name=func_name,
                mode=mode,
                record_index=record_index,
            ),
        )

    @staticmethod
    def load_XML_files(
        folders: List[str] = [],
        ext: str = "*.XML",
        n_procs: int = 4,
        extract_time_from_name: bool = True,
        extract_stn_from_name: bool = True,
        func_name: str = "height_profile",
        mode: str = "auto",
        record_index: int = 0,
    ) -> pd.DataFrame:
        """Load XML SAO files from folders into a single DataFrame.

        Same behaviour as: meth:`load_SAO_files` but matches XML file
        extensions by default.
        """
        return load_files_to_dataframe(
            folders=folders,
            exts=ext,
            extractor=SaoExtractor.extract_SAO,
            n_procs=n_procs,
            extractor_kwargs=dict(
                extract_time_from_name=extract_time_from_name,
                extract_stn_from_name=extract_stn_from_name,
                func_name=func_name,
                mode=mode,
                record_index=record_index,
            ),
        )
