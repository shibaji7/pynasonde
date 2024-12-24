import datetime as dt

from loguru import logger

from pynasonde.digisonde.digi_utils import to_namespace


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
            logger.info(f"Station code: {self.stn_code}")
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
        print(sky_arch.split(" "))
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
            sky_data=[],
        )
        return parsed_freq_header

    def extract(self):
        """
        Main method to extract data from the sky file and populate the sao_struct dictionary.

        Returns:
            dict: The populated sky_struct dictionary containing all extracted data.
        """
        self.sky_struct = dict(dataset=[])
        # Read file lines and set the datastructure by indent
        sky_arch_list = self.read_file()
        _i = 0
        while _i < len(sky_arch_list):
            line_indent, sky_arch = self.parse_line(sky_arch_list, _i)
            print(">", _i, line_indent, sky_arch)
            if line_indent in [1, 2]:  # Consider it first 'Data Header'
                ds = dict(
                    data_header=self.parse_data_header(sky_arch),
                    freq_headers=[],
                )
                _i_nest, n_spectrums = 0, ds["data_header"]["n_spectrums"]
                print("n_spectrums", ds["data_header"]["n_spectrums"])
                while _i_nest < n_spectrums:
                    # add next line to above to this nest (previous while loop)
                    _i += 1  # you need to add this previous to work on sky_arch
                    line_indent, sky_arch = self.parse_line(sky_arch_list, _i)
                    print(">>", _i, line_indent, sky_arch)
                    if line_indent == 4:  # Frequency header
                        fh = self.parse_freq_header(sky_arch)
                        # Nesting 2nd layer for datasets from sources 'n_sources'
                        n_sources = fh["n_sources"]
                        print("n_sources", n_sources)
                        if n_sources > 0:  # Check if data esists
                            # if n_sources <= 26:
                            for _i_nest_nest in range(5):
                                # add next line to above to this nest (previous while loop)
                                _i += 1  # you need to add this previous to work on sky_arch
                                line_indent, sky_arch = self.parse_line(
                                    sky_arch_list, _i
                                )
                                print(">>>", _i, line_indent, sky_arch)
                                data = dict(
                                    y_coords=[],  # Y coordinate
                                    x_coords=[],  # X coordinate
                                    spect_amp=[],  # Spectral Amplitude number
                                    spect_dop=[],  # Spectral Doppler line number
                                    rms_error=[],  # Least Squares Fit RMS error
                                )
                                fh["sky_data"].append(data)
                        ds["freq_headers"].append(fh)
                    _i_nest += 1
                self.sky_struct["dataset"].append(ds)
            if _i >= 20:
                break
            # Loop add next line
            _i += 1
        self.sky = to_namespace(self.sky_struct)
        print(self.sky_struct)
        return


if __name__ == "__main__":
    extractor = SkyExtractor("tmp/KR835_2023286235715.SKY", True, True)
    extractor.extract()
