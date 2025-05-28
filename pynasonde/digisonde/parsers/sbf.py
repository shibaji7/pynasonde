import copy
import datetime as dt
import struct
from typing import List

import numpy as np
import pandas as pd
from loguru import logger

from pynasonde.digisonde.datatypes.sbfdatatypes import (
    SbfDataFile,
    SbfDataUnit,
    SbfFreuencyGroup,
    SbfHeader,
)
from pynasonde.digisonde.digi_utils import get_digisonde_info
from pynasonde.vipir.ngi.utils import TimeZoneConversion

SBF_IONOGRAM_SETTINGS = {
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


class SbfExtractor(object):

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
        Main method to extract data from the sbf file and populate the sbf_struct dictionary.

        Returns:
            dict: The populated sbf_struct dictionary containing all extracted data.
        """
        self.sbf_data = SbfDataFile(sbf_data_units=[])
        with open(self.filename, "rb") as file:
            for n in range(self.BLOCKS):
                sbf_data_unit = SbfDataUnit(frequency_groups=[])
                blk_size = self.DATA_BLOCK_SIZE
                logger.debug(f"Reading block {n+1} of {self.BLOCKS}")
                h = SbfHeader(
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
                        self.unpack_bcd(file.read(1)[0]) * 1e3
                        + self.unpack_bcd(file.read(1)[0]) * 1e2
                        + self.unpack_bcd(file.read(1)[0])
                    ),
                    coarse_frequency_step=self.unpack_bcd(file.read(1)[0]) * 1e2
                    + self.unpack_bcd(file.read(1)[0]),
                    stop_frequency=(
                        self.unpack_bcd(file.read(1)[0]) * 1e3
                        + self.unpack_bcd(file.read(1)[0]) * 1e2
                        + self.unpack_bcd(file.read(1)[0])
                    ),
                    fine_frequency_step=(
                        self.unpack_bcd(file.read(1)[0]) * 1e2
                        + self.unpack_bcd(file.read(1)[0])
                    ),
                    num_small_steps_in_scan=struct.unpack("b", file.read(1))[0],
                    phase_code=self.unpack_bcd(file.read(1)[0]),
                    option_code=struct.unpack("b", file.read(1))[0],
                    number_of_samples=self.unpack_bcd(file.read(1)[0]),
                    pulse_repetition_rate=(
                        self.unpack_bcd(file.read(1)[0]) * 1e2
                        + self.unpack_bcd(file.read(1)[0])
                    ),
                    range_start=(
                        self.unpack_bcd(file.read(1)[0]) * 1e2
                        + self.unpack_bcd(file.read(1)[0])
                    ),
                    range_increment=self.unpack_bcd(file.read(1)[0]),
                    number_of_heights=(
                        self.unpack_bcd(file.read(1)[0]) * 1e2
                        + self.unpack_bcd(file.read(1)[0])
                    ),
                    delay=(
                        self.unpack_bcd(file.read(1)[0]) * 1e2
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
                        self.unpack_bcd(file.read(1)[0]) * 1e2
                        + self.unpack_bcd(file.read(1)[0])
                    ),
                    top_height_window=(
                        self.unpack_bcd(file.read(1)[0]) * 1e2
                        + self.unpack_bcd(file.read(1)[0])
                    ),
                    number_of_heights_stored=(
                        self.unpack_bcd(file.read(1)[0]) * 1e2
                        + self.unpack_bcd(file.read(1)[0])
                    ),
                )
                freq_group_settings = SBF_IONOGRAM_SETTINGS[
                    str(int(h.number_of_heights))
                ]
                h.number_of_frequency_groups = freq_group_settings["number_freq_blocks"]
                blk_size -= 60

                for _ in range(h.number_of_frequency_groups):
                    pol, group_size = self.unpack_bcd(file.read(1)[0], "tuple")
                    pol = "O" if pol == 3 else "X"
                    fg = SbfFreuencyGroup(
                        pol=pol,
                        group_size=group_size,
                        frequency_reading=(
                            self.unpack_bcd(file.read(1)[0]) * 1e2
                            + self.unpack_bcd(file.read(1)[0])
                        ),
                    )
                    fg.offset, fg.additional_gain = self.unpack_bcd(
                        file.read(1)[0], "tuple"
                    )
                    fg.seconds = self.unpack_bcd(file.read(1)[0])
                    fg.mpa = self.unpack_bcd(file.read(1)[0])
                    one_byte = [
                        [self.unpack_5_3(file.read(1)[0])]
                        for _ in range(freq_group_settings["number_range_bins"])
                    ]
                    two_bytes = np.array(two_bytes)
                    fg.amplitude = two_bytes[:, 0, 0]
                    fg.dop_num = two_bytes[:, 0, 1]
                    fg.phase = two_bytes[:, 1, 0]
                    fg.azimuth = two_bytes[:, 1, 1]
                    fg.setup(h.threshold)
                    blk_size -= 2 * freq_group_settings["number_range_bins"] + 6
                    sbf_data_unit.frequency_groups.append(fg)
                sbf_data_unit.header = h
                # Cleaning remaining bytes
                if blk_size > 0:
                    logger.debug(f"Cleaning remaining {blk_size} bytes")
                    file.read(blk_size)
                sbf_data_unit.setup()
                self.sbf_data.sbf_data_units.append(sbf_data_unit)
        return

    def add_dicts_selected_keys(self, d0, du, keys=None) -> dict:
        if keys is None:
            return d0 | du
        else:
            return d0 | {k: du[k] for k in keys}

    def to_pandas(self) -> pd.DataFrame:
        """Converts the extracted SBF data to a pandas DataFrame."""
        self.records = []
        for du in self.sbf_data.sbf_data_units:
            for fg in du.frequency_groups:
                recs = []
                d0 = self.add_dicts_selected_keys(dict(), du.header.__dict__)
                d0 = self.add_dicts_selected_keys(
                    d0,
                    fg.__dict__,
                    keys=[
                        "pol",
                        "group_size",
                        "frequency_reading",
                        "offset",
                        "additional_gain",
                        "seconds",
                        "mpa",
                    ],
                )
                for am, dn, p, az, h in zip(
                    fg.amplitude,
                    fg.dop_num,
                    fg.phase,
                    fg.azimuth,
                    fg.height,
                ):
                    d = copy.copy(d0)
                    d["amplitude"] = am
                    d["dop_num"] = dn
                    d["phase"] = p
                    d["azimuth"] = az
                    d["height"] = h
                    recs.append(d)
                self.records.extend(recs)
        if len(self.records):
            logger.info(f"Extracted {len(self.records)} records from SBF file.")
            self.records = pd.DataFrame.from_dict(self.records)
            self.records["date"] = pd.to_datetime(self.records["date"], utc=True)
        return self.records

    def unpack_5_3(self, bcd_byte: int) -> List[int]:
        """Unpacks a 1-byte packed BCD into 5 bit MSB and 3 bit LSB."""
        high_nibble = (bcd_byte >> 3) & 0b00011111
        low_nibble = bcd_byte & 0b00000111
        return [high_nibble, low_nibble]

    def unpack_bcd(self, bcd_byte: int, format: str = "int") -> int | tuple:
        """Unpacks a 1-byte packed BCD into two decimal digits."""
        high_nibble = (bcd_byte >> 4) & 0x0F
        low_nibble = bcd_byte & 0x0F
        if format == "int":
            return 10 * high_nibble + low_nibble
        elif format == "tuple":
            return high_nibble, low_nibble
        else:
            raise ValueError("Invalid format specified. Use 'int' or 'tuple'.")
