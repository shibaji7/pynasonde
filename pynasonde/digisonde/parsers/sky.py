"""Skymap/sky-format file parser utilities for Digisonde.

This module contains :class:`SkyExtractor`, a compact parser for
skymap/sky-format outputs (both skymap and velocity-style datasets).
The extractor reads simple text files, parses nested fixed-format
blocks and exposes helpers to convert parsed content into pandas
DataFrames suitable for plotting with :class:`SkySummaryPlots`.
"""

import datetime as dt
from types import SimpleNamespace
from typing import List

import pandas as pd
from loguru import logger

from pynasonde.digisonde.digi_plots import SkySummaryPlots
from pynasonde.digisonde.digi_utils import get_digisonde_info, to_namespace
from pynasonde.vipir.ngi.utils import TimeZoneConversion


def get_indent(line: str) -> int:
    """Return the number of leading spaces in a line.

    Parameters:
        line: Input text line.

    Returns:
        Number of leading space characters. This helper assumes spaces
         are used for indentation (not tabs).
    """
    return len(line) - len(line.lstrip())


class SkyExtractor(object):
    """Parser for SKY-format files.

    The :class:`SkyExtractor` reads text files produced by Digisonde's
    skymap/sky output. It parses nested blocks (data header, frequency
    headers, source sky-data) into a lightweight dictionary
    (:attr:`sky_struct`) and provides :meth:`to_pandas` to convert the
    parsed records into a flattened pandas.DataFrame.
    """

    def __init__(
        self,
        filename: str,
        extract_time_from_name: bool = False,
        extract_stn_from_name: bool = False,
        n_fft: int = 2048,
        delta_freq: float = 50,  # in Hz
    ) -> None:
        """Create a SkyExtractor.

        Parameters:
            filename: str
                Path to the SKY-format file to parse.
            extract_time_from_name: bool, optional
                If True, attempt to parse a timestamp token from the
                filename (default False).
            extract_stn_from_name: bool, optional
                If True, attempt to determine station code and local
                timezone information (default False).
            n_fft: int, optional
                FFT length used to compute Doppler frequency scaling.
            delta_freq: float, optional
                Frequency step in Hz used by :meth:`get_doppler_freq`.
        """
        # Initialize the data structure to hold extracted data
        self.filename = filename
        self.n_fft = n_fft
        self.delta_freq = delta_freq
        self.l0 = n_fft // 2
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
        """Read the input file and return a list of lines.

        Returns:
            List of lines including their trailing newline characters.
        """
        with open(self.filename, "r") as f:
            return f.readlines()

    def parse_line(self, sky_arch_list: List[str], _i: int):
        """Parse a single line from the raw lines list.

        Parameters:
            sky_arch_list:  List of file lines as returned by :meth:`read_file`.
            _i: Index of the line to parse.

        Returns:
            (indent_level, token_list) where ``indent_level`` is the
                number of leading spaces and ``token_list`` is the
                whitespace-split tokens from the line with 'D' markers
                removed when present.
        """
        sky_arch = sky_arch_list[_i]
        if "D" in sky_arch:
            sky_arch = sky_arch.replace("D", "")
        line_indent = get_indent(sky_arch)
        sky_arch = sky_arch.strip()
        sky_arch = list(filter(None, sky_arch.split()))
        return line_indent, sky_arch

    def parse_data_header(self, sky_arch):
        """Parse a data-header line into a dictionary of fields.

        Parameters:
            sky_arch: Tokenized line (as returned by :meth:`parse_line`).

        Returns:
            Parsed header fields including type, version, number of
                spectrums and other control fields.
        """
        parsed_data_header = dict(
            # Data Format Type (1 Skymap) (2 Velocity) (3 Quality control)
            type=int(sky_arch[0]),
            # Program version Number
            version=float(sky_arch[1]),
            # 57Z1 Digisonde Preface Header
            digi_preface_no=sky_arch[2],
            # Number of Frequencies/Height Spectrums
            n_spectrums=int(sky_arch[3]),
            # Number of Rows in data.
            n_rows_data=int(sky_arch[4]),
            # Others / not eligible
            others=int(sky_arch[6]),
        )
        return parsed_data_header

    def parse_freq_header(self, sky_arch: List[str]):
        """Parse a frequency/height header line into a dict of fields.

        Parameters:
            sky_arch: Tokenized line (as returned by :meth:`parse_line`).

        Returns:
            Parsed frequency/height header fields including sampling
                frequency, group range, polarization and number of sources.
        """
        parsed_freq_header = dict(
            # Frequcncy number (DGS) / Height spectrum number (DPS)
            frq_height_num=int(sky_arch[0]),
            # Maximum Main reciever array lobe zenith angle
            zenith_angle=float(sky_arch[1]),
            # Sampling Frequency in MHz
            sampl_freq=float(sky_arch[2]),
            # Group Range in KM.
            group_range=float(sky_arch[3]),
            # Gain Amplitude number
            gain_ampl=float(sky_arch[4]),
            # Height Spectrum Amplitude number
            height_spctrum_ampl=float(sky_arch[5]),
            # Maximum Height Spectrum Amplitude number
            max_height_spctrum_ampl=float(sky_arch[6]),
            # Number of sources, Maximum set for each line is for 26 sources
            # more sources are wrapped around below the first 26 as shown for
            # height spectrum No.2 below.
            n_sources=int(sky_arch[7]),
            # Height Spectrum cleaning threshold
            height_spctrum_cl_th=float(sky_arch[8]),
            # Spectral line cleaning threshold
            spect_line_cl_th=float(sky_arch[9]),
            # Polarization Identifier 0/1=O/X-mode
            polarization=int(sky_arch[10]),
            # Skymap dataset
            sky_data=None,
        )
        return parsed_freq_header

    def extract(self) -> SimpleNamespace:
        """Parse the SKY file into :attr:`sky_struct`.

        The parser walks the file line-by-line, identifies data headers
        (indentation 1 or 2), then parses frequency headers and nested
        sky-data blocks for each detected source. The final structure is
        wrapped into :attr:`sky` (a namespace) and returned.

        Returns:
            Namespace wrapper around the parsed ``sky_struct`` dictionary.
        """
        self.sky_struct = dict(dataset=[])
        # Read file lines and set the datastructure by indent
        sky_arch_list = self.read_file()
        _i = 0
        while _i < len(sky_arch_list):
            line_indent, sky_arch = self.parse_line(sky_arch_list, _i)
            # print(">", _i, line_indent, sky_arch)
            if line_indent in [1, 2]:  # Consider it first 'Data Header'
                ds = dict(
                    data_header=self.parse_data_header(sky_arch),
                    freq_headers=[],
                )
                _i_nest, n_spectrums = 0, ds["data_header"]["n_spectrums"]
                # print("n_spectrums", ds["data_header"]["n_spectrums"])
                while _i_nest < n_spectrums:
                    # add next line to above to this nest (previous while loop)
                    _i += 1  # you need to add this previous to work on sky_arch
                    line_indent, sky_arch = self.parse_line(sky_arch_list, _i)
                    # print(">>", _i, line_indent, sky_arch)
                    if line_indent == 4:  # Frequency header
                        fh = self.parse_freq_header(sky_arch)
                        # Nesting 2nd layer for datasets from sources 'n_sources'
                        n_sources = fh["n_sources"]
                        # print("n_sources", n_sources)
                        if n_sources > 0:  # Check if data esists
                            fh["sky_data"], _i = self.parse_sky_data(
                                sky_arch_list, _i, n_sources
                            )
                        ds["freq_headers"].append(fh)
                    _i_nest += 1
                self.sky_struct["dataset"].append(ds)
            # Loop add next line
            _i += 1
        self.sky = to_namespace(self.sky_struct)
        return self.sky

    def parse_sky_data(self, sky_arch_list, _i, n_sources):
        """Parse a nested sky-data block belonging to a frequency header.

        The sky-data block encodes coordinates and spectral arrays across
        multiple following lines. Depending on ``n_sources`` the block may
        wrap to multiple groups of five lines.

        Parameters:
            sky_arch_list: list[str]
                Full list of file lines.
            _i: int
                Current index in the file (frequency header line index).
            n_sources: int
                Number of sky sources to extract; determines how many lines
                to consume.

        Returns:
            (data_dict, new_index) where ``data_dict`` contains keys
                ``y_coords``, ``x_coords``, ``spect_amp``, ``spect_dop`` and
                ``rms_error`` and ``new_index`` points to the last consumed
                input index.
        """
        # if n_sources <= 26 then they are in 1 line otherwise check on _i+5 lines
        import re

        y_coords = re.findall(r"-?\d+\.\d+", sky_arch_list[_i + 1].strip())
        y_coords = [float(y) for y in y_coords]
        x_coords = re.findall(r"-?\d+\.\d+", sky_arch_list[_i + 2].strip())
        x_coords = [float(x) for x in x_coords]
        spect_amp = re.findall(r"-?\d+(?:\.\d+)?", sky_arch_list[_i + 3].strip())
        spect_amp = [float(a) for a in spect_amp]
        spect_dop = re.findall(r"-?\d+(?:\.\d+)?", sky_arch_list[_i + 4].strip())
        spect_dop = [float(d) for d in spect_dop]
        rms_error = re.findall(r"-?\d+(?:\.\d+)?", sky_arch_list[_i + 5].strip())
        rms_error = [float(r) for r in rms_error]
        data = dict(
            y_coords=y_coords,  # Y coordinate
            x_coords=x_coords,  # X coordinate
            spect_amp=spect_amp,  # Spectral Amplitude number
            spect_dop=spect_dop,  # Spectral Doppler line number
            rms_error=rms_error,  # Least Squares Fit RMS error
        )
        _i = _i + 5 if n_sources <= 26 else _i + (5 * (int(n_sources / 26) + 1))
        return data, _i

    def get_doppler_freq(self, L: float) -> float:
        """Convert a Doppler bin index to Doppler frequency in MHz.

        Parameters:
            L: float
                Doppler bin index (integer or float) returned by the skymap
                processing.

        Returns:
            Doppler frequency in MHz computed as ``L * delta_freq / n_fft``.
        """
        return L * self.delta_freq / self.n_fft

    def to_pandas(self) -> pd.DataFrame:
        """Flatten parsed sky records into a pandas DataFrame.

        Returns:
            One row per sky-source with columns for coordinates,
                amplitudes, Doppler values (and Doppler frequency), RMS error
                and timestamps where available.
        """
        df = pd.DataFrame()
        for ds in self.sky.dataset:
            for fh in ds.freq_headers:
                d = []
                if fh.sky_data:
                    d.extend(
                        [
                            dict(
                                zenith_angle=fh.zenith_angle,
                                sampl_freq=fh.sampl_freq,
                                group_range=fh.group_range,
                                gain_ampl=fh.gain_ampl,
                                height_spctrum_ampl=fh.height_spctrum_ampl,
                                max_height_spctrum_ampl=fh.max_height_spctrum_ampl,
                                n_sources=fh.n_sources,
                                height_spctrum_cl_th=fh.height_spctrum_cl_th,
                                spect_line_cl_th=fh.spect_line_cl_th,
                                polarization=fh.polarization,
                                x_coord=fh.sky_data.x_coords[i],
                                y_coord=fh.sky_data.y_coords[i],
                                spect_amp=fh.sky_data.spect_amp[i],
                                spect_dop=fh.sky_data.spect_dop[i],
                                spect_dop_freq=self.get_doppler_freq(
                                    fh.sky_data.spect_dop[i]
                                ),
                                rms_error=fh.sky_data.rms_error[i],
                                datetime=self.date if hasattr(self, "date") else None,
                                local_datetime=(
                                    self.local_time
                                    if hasattr(self, "local_time")
                                    else None
                                ),
                            )
                            for i in range(len(fh.sky_data.y_coords))
                        ]
                    )
                d = pd.DataFrame.from_records(d)
                df = pd.concat([df, d])
        return df


if __name__ == "__main__":
    extractor = SkyExtractor(
        # "tmp/SKYWAVE_DPS4D_2023_10_13/KR835_2023286000915.SKY",
        "tmp/20250527/KW009_2025147000426.SKY",
        True,
        True,
    )
    extractor.extract().dataset[-1].freq_headers
    df = extractor.to_pandas()
    skyplot = SkySummaryPlots()
    skyplot.plot_skymap(
        df,
        zparam="spect_dop_freq",
        text=f"Skymap:\n {extractor.stn_code} / {extractor.date.strftime('%H:%M:%S UT, %d %b %Y')}",
        # cmap="jet",
        clim=[-1, 1],
        rlim=6,
    )
    skyplot.save("tmp/extract_sky.png")
    skyplot.close()
