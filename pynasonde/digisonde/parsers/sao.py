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
from tqdm import tqdm

from pynasonde.digisonde.digi_plots import SaoSummaryPlots
from pynasonde.digisonde.digi_utils import get_digisonde_info, to_namespace
from pynasonde.ngi.utils import TimeZoneConversion


class SaoExtractor(object):
    """
    A class to extract and process data from SAO (Standard Archiving Output) files.

    Attributes:
        filename (str): The path to the SAO file.
        sao_struct (dict): A dictionary to store the parsed data from the SAO file.
    """

    def __init__(
        self,
        filename: str,
        extract_time_from_name: bool = False,
        extract_stn_from_name: bool = False,
    ):
        """
        Initialize the SaoExtractor with the given file.

        Args:
            filename (str): Path to the SAO file to be processed.
        """
        # Initialize the data structure to hold extracted data
        self.filename = filename
        self.sao_struct = {}
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

    @staticmethod
    def parse_index_line(line, num_chunks):
        """
        Parses a line into a list of integers based on fixed-width chunks.

        Args:
            line (str): The line to parse.
            num_chunks (int): Number of 3-character chunks to extract.

        Returns:
            list: A list of integers extracted from the line.
        """
        padded_line = line.ljust(120)  # Pad line to 120 characters
        return [
            int(chunk.strip())
            for chunk in re.findall(".{3}", padded_line[: num_chunks * 3])
        ]

    def process_line(self, line, num_ch, num_elements, format_type):
        """
        Processes a line based on the format type and the number of elements.

        Args:
            line (str): The line to be processed.
            num_ch (int): Width of each element.
            num_elements (int): Number of elements to extract.
            format_type (str): Format string indicating the data type.

        Returns:
            list: A list of parsed values.
        """
        padded_line = line.ljust(num_ch * ((len(line) + num_ch - 1) // num_ch))
        values = []
        for i in range(num_elements):
            start = i * num_ch
            end = start + num_ch
            chunk = padded_line[start:end].strip()
            if format_type in ["%7.3f", "%8.3f", "%11.6f", "%20.12f"]:
                values.append(float(chunk) if chunk else None)
            elif format_type in ["%1d", "%2d", "%3d"]:
                values.append(int(chunk) if chunk else None)
            else:
                values.append(chunk)
        return values

    def extract(self):
        """
        Main method to extract data from the SAO file and populate the sao_struct dictionary.

        Returns:
            dict: The populated sao_struct dictionary containing all extracted data.
        """
        # Read file lines
        sao_arch = self.read_file()

        # Parse first two lines for indices
        dindex1 = self.parse_index_line(sao_arch[0], 40)
        dindex2 = self.parse_index_line(sao_arch[1], 40)
        noe = dindex1 + dindex2  # Concatenate indices

        # Define formats and variable names
        fmt_cell = [
            "%7.3f",
            "%120c",
            "%1c",
            "%8.3f",
            "%2d",
            "%7.3f",  # Data Index, Geof. Const.,Description,Time and Sounder,Scaled,Flags,Doppler Table.
            "%8.3f",
            "%8.3f",
            "%3d",
            "%1d",
            "%8.3f",  # O trace F2
            "%8.3f",
            "%8.3f",
            "%3d",
            "%1d",
            "%8.3f",  # O trace F1
            "%8.3f",
            "%8.3f",
            "%3d",
            "%1d",
            "%8.3f",  # O trace E
            "%8.3f",
            "%3d",
            "%1d",
            "%8.3f",  # X trace F2
            "%8.3f",
            "%3d",
            "%1d",
            "%8.3f",  # X trace F1
            "%8.3f",
            "%3d",
            "%1d",
            "%8.3f",  # X trace E
            "%3d",
            "%3d",
            "%3d",  # Median Amplitude of F, E and Es
            "%11.6f",
            "%11.6f",
            "%11.6f",  # T. Height Coeff. UMLCAR
            "%20.12f",  # Quazi-parabolic segments
            "%1d",  # Edit Flags
            "%11.6f",  # Valley description
            "%8.3f",
            "%3d",
            "%1d",
            "%8.3f",  # O trace Es
            "%8.3f",
            "%3d",
            "%1d",
            "%8.3f",  # O trace E Auroral
            "%8.3f",
            "%8.3f",
            "%8.3f",  # True Height Profile
            "%1c",
            "%1c",
            "%1c",  # URSI Q/D Letters and Edit Flags Traces/Profile
            "%11.6f",
            "%8.3f",
            "%8.3f",
            "%8.3f",  # Auroral E Profile Data
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
            "foF",
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

        count = 2  # Initialize the line counter
        for i, fmt in enumerate(fmt_cell):
            if noe[i] == 0:
                continue  # Skip if no elements

            # Determine field width based on format
            num_ch = {
                "%7.3f": 7,
                "%8.3f": 8,
                "%11.6f": 11,
                "%20.12f": 20,
                "%120c": 120,
                "%1c": 1,
                "%1d": 1,
                "%2d": 2,
                "%3d": 3,
            }[fmt]

            line_in = sao_arch[count].strip()  # Adjust for 0-based indexing
            fixln = len(line_in)  # Length of the line
            # If the line length isn't divisible by num_ch, pad it to the next multiple of num_ch
            if fixln % num_ch != 0:
                line_in = line_in.rjust(
                    num_ch * ((fixln + num_ch - 1) // num_ch)
                )  # Pad to the left
            # Additional padding for specific variable names
            if var_cell[i] in ["Qletter", "Dletter"]:
                line_in = line_in.rjust(noe[i])  # Pad to the left based on noe[i0]

            if var_cell[i] != "Scaled":  # Check if the current variable is not 'Scaled'
                in_off = 0
                for i1 in range(noe[i]):  # Iterate over the number of elements
                    i2 = i1 + in_off
                    if i2 * num_ch > len(line_in):  # Check if index exceeds line length
                        in_off -= (fixln + num_ch - 1) // num_ch  # Adjust offset
                        count += 1  # Move to the next line
                        line_in = sao_arch[count].strip()  # Read next line
                        fixln = len(line_in)
                        if fixln % num_ch != 0:  # Pad line if necessary
                            line_in = line_in.rjust(
                                num_ch * ((fixln + num_ch - 1) // num_ch)
                            )
                        i2 = i1 + in_off

                    # Extract chunk based on current index
                    chunk = line_in[num_ch * (i2 - 1) : num_ch * i2].strip()
                    if not chunk:  # Handle empty or invalid data
                        chunk = " "
                    self.sao_struct.setdefault(var_cell[i], []).append(
                        float(chunk)
                        if chunk.strip().replace(".", "", 1).isdigit()
                        else chunk
                    )
            else:
                in_off = 0
                for i1 in range(noe[i]):  # Iterate over the number of scaled fields
                    i2 = i1 + in_off
                    if i2 * num_ch > len(line_in):  # Check if index exceeds line length
                        in_off -= (fixln + num_ch - 1) // num_ch  # Adjust offset
                        count += 1  # Move to the next line
                        line_in = sao_arch[count].strip()  # Read next line
                        fixln = len(line_in)
                        if fixln % num_ch != 0:  # Pad line if necessary
                            line_in = line_in.rjust(
                                num_ch * ((fixln + num_ch - 1) // num_ch)
                            )
                        i2 = i1 + in_off

                    # Extract chunk based on current index
                    chunk = line_in[num_ch * (i2 - 1) : num_ch * i2].strip()
                    field_name = scal_cell[i1]  # Get the scaled field name
                    if not chunk:  # Handle empty or invalid data
                        chunk = None
                    self.sao_struct.setdefault(var_cell[i], {}).setdefault(
                        field_name, []
                    ).append(
                        float(chunk)
                        if chunk and chunk.strip().replace(".", "", 1).isdigit()
                        else chunk
                    )

            count += 1

        # Correct specific fields if necessary
        if "sysdes" in self.sao_struct:
            self.sao_struct["sysdes"] = "".join(self.sao_struct["sysdes"])

        for key in ["ED", "TH", "PF"]:
            if key in self.sao_struct:
                self.sao_struct[key] = list(
                    filter(
                        None,
                        [
                            x.strip() if type(x) is str else x
                            for x in self.sao_struct[key]
                        ],
                    )
                )
                if type(self.sao_struct[key][0]) is str:
                    self.sao_struct[key] = [float(x) for x in self.sao_struct[key]]
        self.sao = to_namespace(self.sao_struct)
        return self.sao_struct

    def get_scaled_datasets(self, asdf=True):
        for key in vars(self.sao.Scaled).keys():
            if (
                len(vars(self.sao.Scaled)[key]) == 1
                and type(vars(self.sao.Scaled)[key][0]) == str
            ):
                setattr(self.sao.Scaled, key, [np.nan])
        o = pd.DataFrame.from_records(vars(self.sao.Scaled))
        o.replace(9999.0, np.nan, inplace=True)
        if hasattr(self, "date"):
            o["datetime"] = self.date
        if hasattr(self, "local_time"):
            o["local_datetime"] = self.local_time
        return o

    def get_height_profile(self, asdf=True, plot_ionogram=False):
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

    def display_struct(self):
        """
        Prints the extracted SAO structure in a readable format.
        """
        logger.info(self.sao_struct)
        return

    @staticmethod
    def extract_SAO(
        file: str,
        extract_time_from_name: bool = True,
        extract_stn_from_name: bool = True,
        func_name: str = "height_profile",
    ):
        extractor = SaoExtractor(file, extract_time_from_name, extract_stn_from_name)
        extractor.extract()
        if func_name == "height_profile":
            df = extractor.get_height_profile()
        elif func_name == "scaled":
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
    ):
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


# Example Usage
if __name__ == "__main__":
    extractor = SaoExtractor("tmp/SKYWAVE_DPS4D_2023_10_13/KR835_2023286000437.SAO")
    extractor.extract()
    print(extractor.sao)
    # coll1 = SaoExtractor.load_SAO_files(
    #     folders=["tmp/SKYWAVE_DPS4D_2023_10_14"],
    #     func_name="height_profile",
    # )
    # coll1.ed = coll1.ed / 1e6
    # sao_plot = SaoSummaryPlots(
    #     figsize=(6, 3), fig_title="KR835/13-14 Oct, 2023", draw_local_time=True
    # )
    # sao_plot.add_TS(
    #     coll1,
    #     zparam="ed",
    #     prange=[0, 1],
    #     zparam_lim=10,
    #     cbar_label=r"$N_e$,$\times 10^{6}$ /cc",
    # )
    # sao_plot.save("tmp/example_pf.png")
    # sao_plot.close()
    # coll2 = SaoExtractor.load_SAO_files(
    #     folders=["tmp/SKYWAVE_DPS4D_2023_10_14"], func_name="scaled"
    # )
    # sao_plot = SaoSummaryPlots(
    #     figsize=(6, 3), fig_title="KR835/13-14 Oct, 2023", draw_local_time=True
    # )
    # sao_plot.plot_TS(coll2, left_yparams=["foF1"], left_ylim=[1, 15])
    # sao_plot.save("tmp/example_ts.png")
    # sao_plot.close()
    # SaoSummaryPlots.plot_isodensity_contours(
    #     coll1,
    #     xlim=[dt.datetime(2023, 10, 13, 12), dt.datetime(2023, 10, 14)],
    #     fname="tmp/example_id.png",
    # )
