import copy
import datetime as dt
import struct
from typing import List

import numpy as np
import pandas as pd
from loguru import logger

from pynasonde.digisonde.datatypes.mmmdatatypes import ModMaxFreuencyGroup, ModMaxHeader
from pynasonde.digisonde.digi_utils import get_digisonde_info
from pynasonde.vipir.ngi.utils import TimeZoneConversion

MMM_IONOGRAM_SETTINGS = {
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


class ModMaxExtractor(object):

    def __init__(
        self,
        filename: str,
        extract_time_from_name: bool = False,
        extract_stn_from_name: bool = False,
        DATA_BLOCK_SIZE: int = 4096,
    ):
        """
        Initialize the ModMaxExtractor with the given file.

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
        Main method to extract data from the mmm file and populate the mmm_struct dictionary.

        Returns:
            dict: The populated mmm_struct dictionary containing all extracted data.
        """
        # self.mmm_data = RsfDataFile(mmm_data_units=[])
        with open(self.filename, "rb") as file:
            for n in range(self.BLOCKS):
                # mmm_data_unit = RsfDataUnit(frequency_groups=[])
                blk_size = self.DATA_BLOCK_SIZE
                logger.debug(f"Reading block {n+1} of {self.BLOCKS}")
                # Helper to read and unpack a byte
                ub = lambda: struct.unpack("B", file.read(1))[0]

                h = ModMaxHeader(
                    record_type=ub(),
                    header_length=ub(),
                    version_maker=hex(ub()),
                    year=(2000 + (10 * ub()) + ub()),
                    doy=(ub() * 1e2 + ub() * 1e1 + ub()),
                    hour=(ub() * 1e1 + ub()),
                    minute=(ub() * 1e1 + ub()),
                    second=(ub() * 1e1 + ub()),
                    program_set=hex(ub()),
                    program_type=hex(ub()),
                    journal=[ub(), ub(), ub(), ub(), ub(), ub()],
                    nom_frequency=(
                        ub() * 1e5
                        + ub() * 1e4
                        + ub() * 1e3
                        + ub() * 1e2
                        + ub() * 1e1
                        + ub()
                    ),
                    tape_ctrl=hex(ub()),
                    print_ctrl=hex(ub()),
                    mmm_opt=hex(ub()),
                    print_clean_ctrl=hex(ub()),
                    print_gain_lev=hex(ub()),
                    ctrl_intm_tx=hex(ub()),
                    drft_use=hex(ub()),
                    start_frequency=(ub() * 1e1 + ub()),
                    freq_step=ub(),
                    stop_frequency=(ub() * 1e1 + ub()),
                    trg=hex(ub()),
                    ch_a=hex(ub()),
                    ch_b=hex(ub()),
                    sta_id=f"{ub()}{ub()}{ub()}",
                    phase_code=ub(),
                    ant_azm=ub(),
                    ant_scan=ub(),
                    ant_opt=ub(),
                    num_samples=ub(),
                    rep_rate=ub(),
                    pwd_code=ub(),
                    time_ctrl=ub(),
                    freq_cor=ub(),
                    gain_cor=ub(),
                    range_inc=ub(),
                    range_start=ub(),
                    f_search=ub(),
                    nom_gain=ub(),
                )
                print(h, blk_size)
                blk_size -= 60
                fg = ModMaxFreuencyGroup(
                    blk_type=ub(),
                    frequency=self.unpack_bcd(ub(), format="int"),
                    frequency_k=self.unpack_bcd(ub(), format="int") * 10,
                )
                fg.frequency_search, fg.gain_param = self.unpack_bcd(
                    ub(), format="tuple"
                )
                fg.sec = self.unpack_bcd(ub(), format="int")
                fg.mpa = ub()
                # if fg.blk_type == 1:
                #     for blk in range(30):

                print(fg)
                file.read(blk_size)
                if n == 0:
                    break
        return

    def add_dicts_selected_keys(self, d0, du, keys=None) -> dict:
        if keys is None:
            return d0 | du
        else:
            return d0 | {k: du[k] for k in keys}

    def to_pandas(self) -> pd.DataFrame:
        """Converts the extracted RSF data to a pandas DataFrame."""
        self.records = []
        for du in self.mmm_data.mmm_data_units:
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
            logger.info(f"Extracted {len(self.records)} records from RSF file.")
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


if __name__ == "__main__":
    extractor = ModMaxExtractor(
        "/media/chakras4/ERAU/SpeedDemon/WP937/individual/2022/233/ionogram/WP937_2022233235510.MMM",
        True,
        True,
    )
    extractor.extract()
