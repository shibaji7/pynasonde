import datetime as dt

import pandas as pd
from loguru import logger

from pynasonde.digisonde.digi_plots import SkySummaryPlots
from pynasonde.digisonde.digi_utils import get_digisonde_info, to_namespace
from pynasonde.ngi.utils import TimeZoneConversion


def get_indent(line):
    return len(line) - len(line.lstrip())


class SkyExtractor(object):

    def __init__(
        self,
        filename: str,
        extract_time_from_name: bool = False,
        extract_stn_from_name: bool = False,
    ):
        """
        Initialize the SkyExtractor with the given file.

        Args:
            filename (str): Path to the sky file to be processed.
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
            self.stn_info = get_digisonde_info(self.stn_code)
            self.local_timezone_converter = TimeZoneConversion(
                lat=self.stn_info["LAT"], long=self.stn_info["LONG"]
            )
            self.local_time = self.local_timezone_converter.utc_to_local_time(
                [self.date]
            )[0]
            logger.info(f"Station code: {self.stn_code}; {self.stn_info}")
        return

    def read_file(self):
        """
        Reads the file line by line into a list.

        Returns:
            list: A list of strings, each representing a line from the file.
        """
        with open(self.filename, "r") as f:
            return f.readlines()

    def parse_line(self, sky_arch_list, _i):
        sky_arch = sky_arch_list[_i]
        if "D" in sky_arch:
            sky_arch = sky_arch.replace("D", "")
        line_indent = get_indent(sky_arch)
        sky_arch = sky_arch.strip()
        sky_arch = list(filter(None, sky_arch.split()))
        return line_indent, sky_arch

    def parse_data_header(self, sky_arch):
        """
        Parses a single line of input and extracts its data header components.

        Args:
            sky_arch (str): The input line to parse.

        Returns:
            dict: Parsed values organized in a dictionary.
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

    def parse_freq_header(self, sky_arch):
        """
        Parses a single line of input and extracts its freq header components.

        Args:
            sky_arch (str): The input line to parse.

        Returns:
            dict: Parsed values organized in a dictionary.
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

    def extract(self):
        """
        Main method to extract data from the sky file and populate the sky_struct dictionary.

        Returns:
            dict: The populated sky_struct dictionary containing all extracted data.
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
        # add next line to above to this nest (previous while loop)
        # we need to add 5 more lines to main index

        # if n_sources <= 26 then they are in 1 line otherwise check on _i+5 lines
        _, y_coords = self.parse_line(sky_arch_list, _i + 1)
        y_coords = [float(y) for y in y_coords]
        _, x_coords = self.parse_line(sky_arch_list, _i + 2)
        x_coords = [float(x) for x in x_coords]
        _, spect_amp = self.parse_line(sky_arch_list, _i + 3)
        spect_amp = [float(a) for a in spect_amp]
        _, spect_dop = self.parse_line(sky_arch_list, _i + 4)
        spect_dop = [float(d) for d in spect_dop]
        _, rms_error = self.parse_line(sky_arch_list, _i + 5)
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

    def to_pandas(self) -> pd.DataFrame:
        """Convert dataset to pandas dataframe"""
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
    pass

    extractor = SkyExtractor(
        "tmp/SKYWAVE_DPS4D_2023_10_13/KR835_2023286000915.SKY", True, True
    )
    extractor.extract().dataset[-1].freq_headers
    df = extractor.to_pandas()
    skyplot = SkySummaryPlots()
    skyplot.plot_skymap(
        df,
        text=f"Skymap:\n {extractor.stn_code} / {extractor.local_time.strftime('%H:%M LT, %d %b %Y')}",
    )
    skyplot.save("tmp/extract_sky.png")
    skyplot.close()
