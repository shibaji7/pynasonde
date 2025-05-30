import copy
import datetime as dt
import struct
from typing import List

import numpy as np
import pandas as pd
from loguru import logger

from pynasonde.digisonde.datatypes.rsfdatatypes import (
    RsfDataFile,
    RsfDataUnit,
    RsfFreuencyGroup,
    RsfHeader,
)
from pynasonde.digisonde.digi_utils import get_digisonde_info
from pynasonde.vipir.ngi.utils import TimeZoneConversion

RSF_IONOGRAM_SETTINGS = {
    "128": dict(number_freq_blocks=15, number_range_bins=128, byte_length=262),
    "256": dict(number_freq_blocks=8, number_range_bins=249, byte_length=504),
    "512": dict(number_freq_blocks=4, number_range_bins=501, byte_length=1008),
}


class RsfExtractor:
    def __init__(
        self,
        filename: str,
        extract_time_from_name: bool = False,
        extract_stn_from_name: bool = False,
        DATA_BLOCK_SIZE: int = 4096,
    ):
        self.filename = filename
        self.DATA_BLOCK_SIZE = DATA_BLOCK_SIZE
        with open(filename, "rb") as f:
            self.BLOCKS = len(f.read()) // DATA_BLOCK_SIZE

        if extract_time_from_name:
            date = filename.split("_")[-1].replace(".SAO", "").replace(".sao", "")
            self.date = (
                dt.datetime(int(date[:4]), 1, 1) + dt.timedelta(int(date[4:7]) - 1)
            ).replace(
                hour=int(date[7:9]), minute=int(date[9:11]), second=int(date[11:13])
            )
            logger.info(f"Date: {self.date}")

        if extract_stn_from_name:
            self.stn_code = filename.split("/")[-1].split("_")[0]
            self.stn_info = get_digisonde_info(self.stn_code)
            self.local_timezone_converter = TimeZoneConversion(
                lat=self.stn_info["LAT"], long=self.stn_info["LONG"]
            )
            self.local_time = self.local_timezone_converter.utc_to_local_time(
                [self.date]
            )[0]
            logger.info(f"Station code: {self.stn_code}; {self.stn_info}")

    def extract(self):
        self.rsf_data = RsfDataFile(rsf_data_units=[])
        with open(self.filename, "rb") as file:
            for n in range(self.BLOCKS):
                rsf_data_unit = RsfDataUnit(frequency_groups=[])
                blk_size = self.DATA_BLOCK_SIZE
                logger.debug(f"Reading block {n+1} of {self.BLOCKS}")

                # Helper to read and unpack a byte
                rb = lambda: file.read(1)[0]
                ub = lambda: struct.unpack("B", file.read(1))[0]
                sb = lambda: struct.unpack("b", file.read(1))[0]
                uh = lambda: struct.unpack("H", file.read(2))[0]

                h = RsfHeader(
                    record_type=ub(),
                    header_length=ub(),
                    version_maker=hex(ub()),
                    year=self.unpack_bcd(rb()) + 2000,
                    doy=self.unpack_bcd(rb()) * 100 + self.unpack_bcd(rb()),
                    month=self.unpack_bcd(rb()),
                    dom=self.unpack_bcd(rb()),
                    hour=self.unpack_bcd(rb()),
                    minute=self.unpack_bcd(rb()),
                    second=self.unpack_bcd(rb()),
                    stn_code_rx=file.read(3).decode("ascii"),
                    stn_code_tx=file.read(3).decode("ascii"),
                    schedule=self.unpack_bcd(rb()),
                    program=self.unpack_bcd(rb()),
                    start_frequency=self.unpack_bcd(rb()) * 1e3
                    + self.unpack_bcd(rb()) * 1e2
                    + self.unpack_bcd(rb()),
                    coarse_frequency_step=self.unpack_bcd(rb()) * 1e2
                    + self.unpack_bcd(rb()),
                    stop_frequency=self.unpack_bcd(rb()) * 1e3
                    + self.unpack_bcd(rb()) * 1e2
                    + self.unpack_bcd(rb()),
                    fine_frequency_step=self.unpack_bcd(rb()) * 1e2
                    + self.unpack_bcd(rb()),
                    num_small_steps_in_scan=sb(),
                    phase_code=self.unpack_bcd(rb()),
                    option_code=sb(),
                    number_of_samples=self.unpack_bcd(rb()),
                    pulse_repetition_rate=self.unpack_bcd(rb()) * 1e2
                    + self.unpack_bcd(rb()),
                    range_start=self.unpack_bcd(rb()) * 1e2 + self.unpack_bcd(rb()),
                    range_increment=self.unpack_bcd(rb()),
                    number_of_heights=self.unpack_bcd(rb()) * 1e2
                    + self.unpack_bcd(rb()),
                    delay=self.unpack_bcd(rb()) * 1e2 + self.unpack_bcd(rb()),
                    base_gain=self.unpack_bcd(rb()),
                    frequency_search=self.unpack_bcd(rb()),
                    operating_mode=self.unpack_bcd(rb()),
                    data_format=self.unpack_bcd(rb()),
                    printer_output=self.unpack_bcd(rb()),
                    threshold=self.unpack_bcd(rb()),
                    constant_gain=self.unpack_bcd(rb()),
                    spare=file.read(2),
                    cit_length=uh(),
                    journal=ub(),
                    bottom_height_window=self.unpack_bcd(rb()) * 1e2
                    + self.unpack_bcd(rb()),
                    top_height_window=self.unpack_bcd(rb()) * 1e2
                    + self.unpack_bcd(rb()),
                    number_of_heights_stored=self.unpack_bcd(rb()) * 1e2
                    + self.unpack_bcd(rb()),
                )
                freq_group_settings = RSF_IONOGRAM_SETTINGS[
                    str(int(h.number_of_heights))
                ]
                h.number_of_frequency_groups = freq_group_settings["number_freq_blocks"]
                blk_size -= 60

                for _ in range(h.number_of_frequency_groups):
                    pol, group_size = self.unpack_bcd(rb(), "tuple")
                    pol = "O" if pol == 3 else "X"
                    fg = RsfFreuencyGroup(
                        pol=pol,
                        group_size=group_size,
                        frequency_reading=self.unpack_bcd(rb()) * 1e2
                        + self.unpack_bcd(rb()),
                    )
                    fg.offset, fg.additional_gain = self.unpack_bcd(rb(), "tuple")
                    fg.seconds = self.unpack_bcd(rb())
                    fg.mpa = self.unpack_bcd(rb())
                    two_bytes = np.array(
                        [
                            [self.unpack_5_3(rb()), self.unpack_5_3(rb())]
                            for _ in range(freq_group_settings["number_range_bins"])
                        ]
                    )
                    fg.amplitude = two_bytes[:, 0, 0]
                    fg.dop_num = two_bytes[:, 0, 1]
                    fg.phase = two_bytes[:, 1, 0]
                    fg.azimuth = two_bytes[:, 1, 1]
                    fg.setup(h.threshold)
                    blk_size -= 2 * freq_group_settings["number_range_bins"] + 6
                    rsf_data_unit.frequency_groups.append(fg)
                rsf_data_unit.header = h
                if blk_size > 0:
                    logger.debug(f"Cleaning remaining {blk_size} bytes")
                    file.read(blk_size)
                rsf_data_unit.setup()
                self.rsf_data.rsf_data_units.append(rsf_data_unit)
        return

    def add_dicts_selected_keys(self, d0, du, keys=None) -> dict:
        return d0 | (du if keys is None else {k: du[k] for k in keys})

    def to_pandas(self) -> pd.DataFrame:
        """Converts the extracted RSF data to a pandas DataFrame."""
        records = []
        for du in self.rsf_data.rsf_data_units:
            for fg in du.frequency_groups:
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
                    fg.amplitude, fg.dop_num, fg.phase, fg.azimuth, fg.height
                ):
                    d = copy.copy(d0)
                    d.update(
                        {
                            "amplitude": am,
                            "dop_num": dn,
                            "phase": p,
                            "azimuth": az,
                            "height": h,
                        }
                    )
                    records.append(d)
        if records:
            logger.info(f"Extracted {len(records)} records from RSF file.")
            df = pd.DataFrame.from_dict(records)
            df["date"] = pd.to_datetime(df["date"], utc=True)
            self.records = df
            return df
        return pd.DataFrame()

    @staticmethod
    def unpack_5_3(bcd_byte: int) -> List[int]:
        return [(bcd_byte >> 3) & 0b00011111, bcd_byte & 0b00000111]

    @staticmethod
    def unpack_bcd(bcd_byte: int, format: str = "int") -> int | tuple:
        high, low = (bcd_byte >> 4) & 0x0F, bcd_byte & 0x0F
        if format == "int":
            return 10 * high + low
        elif format == "tuple":
            return high, low
        raise ValueError("Invalid format specified. Use 'int' or 'tuple'.")


# if __name__ == "__main__":
#     extractor = RsfExtractor(
#         "tmp/SKYWAVE_DPS4D_2023_10_13/KR835_2023286235456.RSF", True, True
#     )
#     extractor.extract()
#     # df = extractor.to_pandas()


if __name__ == "__main__":
    extractor = RsfExtractor(
        "tmp/SKYWAVE_DPS4D_2023_10_13/KR835_2023286235456.RSF", True, True
    )
    extractor.extract()
    df = extractor.to_pandas()
    from pynasonde.digisonde.digi_plots import RsfIonogram

    print(df.head())
    print(df[df.pol == "X"].amplitude.min(), df[df.pol == "X"].amplitude.max())
    r = RsfIonogram()
    r.add_ionogram(
        df[df.pol == "O"],
        xparam="frequency_reading",
        yparam="height",
        zparam="amplitude",
    )
    r.save("tmp/extract_rsf.png")
    r.close()
