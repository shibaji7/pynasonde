import datetime as dt
import struct
from dataclasses import dataclass

import numpy as np
from loguru import logger

from pynasonde.digisonde.datatypes.rsfdatatypes import RsfHeader
from pynasonde.digisonde.digi_utils import get_digisonde_info
from pynasonde.vipir.ngi.utils import TimeZoneConversion

RSF_IONOGRAM_SETTINGS = {
    "128": dict(
        number_freq_blocks=15,
        number_range_bins=128,
        byte_length=262,
    ),
    "256": dict(
        number_freq_blocks=8,
        number_range_bins=249,
        byte_length=504,
    ),
    "512": dict(
        number_freq_blocks=4,
        number_range_bins=501,
        byte_length=1008,
    ),
}


class RsfExtractor(object):

    def __init__(
        self,
        filename: str,
        extract_time_from_name: bool = False,
        extract_stn_from_name: bool = False,
        DATA_BLOCK_SIZE: int = 4096,
    ):
        """
        Initialize the SkyExtractor with the given file.

        Args:
            filename (str): Path to the sky file to be processed.
        """
        # Initialize the data structure to hold extracted data
        self.filename = filename
        self.DATA_BLOCK_SIZE = DATA_BLOCK_SIZE
        with open(self.filename, "rb") as file:
            self.BLOCKS = int(len(file.read()) / self.DATA_BLOCK_SIZE)
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

    def extract(self):
        """
        Main method to extract data from the rsf file and populate the rsf_struct dictionary.

        Returns:
            dict: The populated rsf_struct dictionary containing all extracted data.
        """
        with open(self.filename, "rb") as file:
            for n in range(self.BLOCKS):
                blk_size = self.DATA_BLOCK_SIZE
                logger.debug(f"Reading block {n+1} of {self.BLOCKS}")
                h = RsfHeader(
                    record_type=struct.unpack("B", file.read(1))[0],
                    header_length=struct.unpack("B", file.read(1))[0],
                    version_maker=hex(struct.unpack("B", file.read(1))[0]),
                    year=self.unpack_bcd(file.read(1)[0]) + 2000,
                    doy=self.unpack_bcd(file.read(1)[0]) * 100
                    + self.unpack_bcd(file.read(1)[0]),
                    month=self.unpack_bcd(file.read(1)[0]),
                    dom=self.unpack_bcd(file.read(1)[0]),
                    hour=self.unpack_bcd(file.read(1)[0]),
                    minute=self.unpack_bcd(file.read(1)[0]),
                    second=self.unpack_bcd(file.read(1)[0]),
                    stn_code_rx=file.read(3).decode("ascii"),
                    stn_code_tx=file.read(3).decode("ascii"),
                    schedule=self.unpack_bcd(file.read(1)[0]),
                    program=self.unpack_bcd(file.read(1)[0]),
                    start_frequency=(
                        self.unpack_bcd(file.read(1)[0]) * 1e5
                        + self.unpack_bcd(file.read(1)[0]) * 1e3
                        + self.unpack_bcd(file.read(1)[0])
                    ),
                    coarse_frequency_step=self.unpack_bcd(file.read(1)[0]) * 1e3
                    + self.unpack_bcd(file.read(1)[0]),
                    stop_frequency=(
                        self.unpack_bcd(file.read(1)[0]) * 1e5
                        + self.unpack_bcd(file.read(1)[0]) * 1e3
                        + self.unpack_bcd(file.read(1)[0])
                    ),
                    fine_frequency_step=(
                        self.unpack_bcd(file.read(1)[0]) * 1e3
                        + self.unpack_bcd(file.read(1)[0])
                    ),
                    num_small_steps_in_scan=struct.unpack("b", file.read(1))[0],
                    phase_code=struct.unpack("b", file.read(1))[0],
                    option_code=struct.unpack("b", file.read(1))[0],
                    number_of_samples=self.unpack_bcd(file.read(1)[0]),
                    pulse_repetition_rate=(
                        self.unpack_bcd(file.read(1)[0]) * 1e2
                        + self.unpack_bcd(file.read(1)[0])
                    ),
                    range_start=(
                        self.unpack_bcd(file.read(1)[0]) * 1e3
                        + self.unpack_bcd(file.read(1)[0])
                    ),
                    range_increment=self.unpack_bcd(file.read(1)[0]),
                    number_of_heights=(
                        self.unpack_bcd(file.read(1)[0]) * 1e2
                        + self.unpack_bcd(file.read(1)[0])
                    ),
                    delay=(
                        self.unpack_bcd(file.read(1)[0]) * 1e3
                        + self.unpack_bcd(file.read(1)[0])
                    ),
                    base_gain=self.unpack_bcd(file.read(1)[0]),
                    frequency_search=self.unpack_bcd(file.read(1)[0]),
                    operating_mode=self.unpack_bcd(file.read(1)[0]),
                    data_format=self.unpack_bcd(file.read(1)[0]),
                    printer_output=self.unpack_bcd(file.read(1)[0]),
                    threshold=self.unpack_bcd(file.read(1)[0]),
                    constant_gain=self.unpack_bcd(file.read(1)[0]),
                    spare=file.read(2),
                    cit_length=struct.unpack("H", file.read(2))[0],
                    journal=struct.unpack("B", file.read(1))[0],
                    bottom_height_window=(
                        self.unpack_bcd(file.read(1)[0]) * 1e3
                        + self.unpack_bcd(file.read(1)[0])
                    ),
                    top_height_window=(
                        self.unpack_bcd(file.read(1)[0]) * 1e3
                        + self.unpack_bcd(file.read(1)[0])
                    ),
                    number_of_heights_stored=(
                        self.unpack_bcd(file.read(1)[0]) * 1e2
                        + self.unpack_bcd(file.read(1)[0])
                    ),
                )
                blk_size -= 60
                print(h)
                # file.read()
                pol, group_size = self.unpack_bcd(file.read(1)[0], "tuple")
                pol = "O" if pol == 3 else "X"
                group_size = RSF_IONOGRAM_SETTINGS[str(int(h.number_of_heights))]
                t_freq = self.unpack_bcd(file.read(1)[0]) * 1e3 + self.unpack_bcd(
                    file.read(1)[0]
                )
                print(
                    pol, group_size, t_freq, self.unpack_bcd(file.read(1)[0], "tuple")
                )
                print(self.unpack_bcd(file.read(1)[0]))
                print(self.unpack_bcd(file.read(1)[0]))
                # datasets
                print(self.unpack_5_3(file.read(1)[0]))
                break
        return

    def unpack_5_3(self, bcd_byte):
        """Unpacks a 1-byte packed BCD into 5 bit MSB and 3 bit LSB."""
        high_nibble = (bcd_byte >> 5) & 0x1F
        low_nibble = bcd_byte & 0x07
        return high_nibble, low_nibble

    def unpack_bcd(self, bcd_byte, format="int"):
        """Unpacks a 1-byte packed BCD into two decimal digits."""
        high_nibble = (bcd_byte >> 4) & 0x0F
        low_nibble = bcd_byte & 0x0F
        if format == "int":
            return 10 * high_nibble + low_nibble
        elif format == "tuple":
            return high_nibble, low_nibble
        else:
            raise ValueError("Invalid format specified. Use 'int' or 'tuple'.")


if __name__ == "__main__":
    extractor = RsfExtractor(
        "tmp/SKYWAVE_DPS4D_2023_10_13/KR835_2023286235456.RSF", True, True
    )
    extractor.extract()
