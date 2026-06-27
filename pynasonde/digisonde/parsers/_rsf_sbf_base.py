"""Shared binary block parser for RSF/SBF ionogram formats.

RSF and SBF use the same 4096-byte block structure, packed-BCD header fields,
and two-packed-byte range-bin payload layout. The concrete parser classes keep
their own dataclasses and public attribute names; this base class only
centralizes the duplicated byte-reading and DataFrame flattening logic.
"""

from typing import List

import numpy as np
import pandas as pd
from loguru import logger

from pynasonde.digisonde.digi_utils import (
    apply_filename_metadata,
    merge_dicts_selected_keys,
    read_i8,
    read_u8,
    read_u16,
    unpack_5_3_byte,
    unpack_bcd_byte,
)


class RsfSbfBinaryBlockExtractor:
    """Base implementation for RSF/SBF fixed-block binary parsers."""

    data_file_class = None
    data_unit_class = None
    header_class = None
    frequency_group_class = None
    ionogram_settings: dict = {}
    data_attr: str = ""
    units_attr: str = ""
    format_label: str = ""
    include_azm_directions: bool = False
    log_block_reads: bool = False
    empty_returns_dataframe: bool = True

    def __init__(
        self,
        filename: str,
        extract_time_from_name: bool = False,
        extract_stn_from_name: bool = False,
        DATA_BLOCK_SIZE: int = 4096,
    ):
        """Create a fixed-block binary extractor.

        Args:
            filename: Path to the RSF/SBF file.
            extract_time_from_name: If True, parse the timestamp from the
                filename.
            extract_stn_from_name: If True, parse station metadata from the
                filename.
            DATA_BLOCK_SIZE: Fixed binary block size in bytes.
        """
        self.filename = filename
        self.DATA_BLOCK_SIZE = DATA_BLOCK_SIZE
        with open(self.filename, "rb") as file:
            self.BLOCKS = int(len(file.read()) / self.DATA_BLOCK_SIZE)

        apply_filename_metadata(
            self,
            self.filename,
            extract_time_from_name=extract_time_from_name,
            extract_stn_from_name=extract_stn_from_name,
        )

    def extract(self):
        """Read fixed-size blocks into the concrete RSF/SBF data container."""
        data_file = self.data_file_class(**{self.units_attr: []})
        data_units = getattr(data_file, self.units_attr)
        setattr(self, self.data_attr, data_file)

        with open(self.filename, "rb") as file:
            for block_index in range(self.BLOCKS):
                data_unit = self.data_unit_class(frequency_groups=[])
                blk_size = self.DATA_BLOCK_SIZE
                if self.log_block_reads:
                    logger.debug(f"Reading block {block_index + 1} of {self.BLOCKS}")

                header = self._read_header(file)
                settings = self.ionogram_settings[str(int(header.number_of_heights))]
                header.number_of_frequency_groups = settings["number_freq_blocks"]
                blk_size -= 60

                for _ in range(header.number_of_frequency_groups):
                    group = self._read_frequency_group(file, settings)
                    group.setup(header.threshold)
                    blk_size -= 2 * settings["number_range_bins"] + 6
                    data_unit.frequency_groups.append(group)

                data_unit.header = header
                if blk_size > 0:
                    if self.log_block_reads:
                        logger.debug(f"Cleaning remaining {blk_size} bytes")
                    file.read(blk_size)
                data_unit.setup()
                data_units.append(data_unit)
        return

    def _read_header(self, file):
        """Read one concrete RSF/SBF header dataclass from ``file``."""
        rb = lambda: read_u8(file)
        sb = lambda: read_i8(file)
        uh = lambda: read_u16(file)

        return self.header_class(
            record_type=rb(),
            header_length=rb(),
            version_maker=hex(rb()),
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
            start_frequency=(
                self.unpack_bcd(rb()) * 1e3
                + self.unpack_bcd(rb()) * 1e2
                + self.unpack_bcd(rb())
            ),
            coarse_frequency_step=self.unpack_bcd(rb()) * 1e2
            + self.unpack_bcd(rb()),
            stop_frequency=(
                self.unpack_bcd(rb()) * 1e3
                + self.unpack_bcd(rb()) * 1e2
                + self.unpack_bcd(rb())
            ),
            fine_frequency_step=self.unpack_bcd(rb()) * 1e2 + self.unpack_bcd(rb()),
            num_small_steps_in_scan=sb(),
            phase_code=self.unpack_bcd(rb()),
            option_code=sb(),
            number_of_samples=self.unpack_bcd(rb()),
            pulse_repetition_rate=self.unpack_bcd(rb()) * 1e2 + self.unpack_bcd(rb()),
            range_start=self.unpack_bcd(rb()) * 1e2 + self.unpack_bcd(rb()),
            range_increment=self.unpack_bcd(rb()),
            number_of_heights=self.unpack_bcd(rb()) * 1e2 + self.unpack_bcd(rb()),
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
            journal=rb(),
            bottom_height_window=self.unpack_bcd(rb()) * 1e2 + self.unpack_bcd(rb()),
            top_height_window=self.unpack_bcd(rb()) * 1e2 + self.unpack_bcd(rb()),
            number_of_heights_stored=self.unpack_bcd(rb()) * 1e2
            + self.unpack_bcd(rb()),
        )

    def _read_frequency_group(self, file, settings: dict):
        """Read one frequency group using the selected ionogram settings."""
        rb = lambda: read_u8(file)

        pol, group_size = self.unpack_bcd(rb(), "tuple")
        pol = "O" if pol == 3 else "X"
        group = self.frequency_group_class(
            pol=pol,
            group_size=group_size,
            frequency_reading=self.unpack_bcd(rb()) * 1e2 + self.unpack_bcd(rb()),
        )
        group.offset, group.additional_gain = self.unpack_bcd(rb(), "tuple")
        group.seconds = self.unpack_bcd(rb())
        group.mpa = self.unpack_bcd(rb())
        packed = np.array(
            [
                [self.unpack_5_3(rb()), self.unpack_5_3(rb())]
                for _ in range(settings["number_range_bins"])
            ]
        )
        group.amplitude = packed[:, 0, 0]
        group.dop_num = packed[:, 0, 1]
        group.phase = packed[:, 1, 0]
        group.azimuth = packed[:, 1, 1]
        return group

    def add_dicts_selected_keys(
        self, d0: dict, du: dict, keys: List[str] = None
    ) -> dict:
        """Merge two dictionaries, optionally selecting keys from the second."""
        return merge_dicts_selected_keys(d0, du, keys=keys)

    def to_pandas(self) -> pd.DataFrame:
        """Convert parsed RSF/SBF records into a pandas DataFrame."""
        records = []
        data_file = getattr(self, self.data_attr)
        for data_unit in getattr(data_file, self.units_attr):
            for group in data_unit.frequency_groups:
                base = merge_dicts_selected_keys(dict(), data_unit.header.__dict__)
                base = merge_dicts_selected_keys(
                    base,
                    group.__dict__,
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
                for row_values in self._iter_group_rows(group):
                    records.append(merge_dicts_selected_keys(base, row_values))

        if records:
            logger.info(
                f"Extracted {len(records)} records from {self.format_label} file."
            )
            self.records = pd.DataFrame.from_dict(records)
            self.records["date"] = pd.to_datetime(self.records["date"], utc=True)
            return self.records

        if self.empty_returns_dataframe:
            return pd.DataFrame()
        self.records = []
        return self.records

    def _iter_group_rows(self, group):
        """Yield row dictionaries for each range bin in a frequency group."""
        arrays = [
            group.amplitude,
            group.dop_num,
            group.phase,
            group.azimuth,
            group.height,
        ]
        if self.include_azm_directions:
            arrays.append(group.azm_directions)

        for values in zip(*arrays):
            row = {
                "amplitude": values[0],
                "dop_num": values[1],
                "phase": values[2],
                "azimuth": values[3],
                "height": values[4],
            }
            if self.include_azm_directions:
                row["azm_directions"] = values[5]
            yield row

    @staticmethod
    def unpack_5_3(bcd_byte: int) -> List[int]:
        """Unpack a byte into 5-bit and 3-bit fields."""
        return unpack_5_3_byte(bcd_byte)

    @staticmethod
    def unpack_bcd(bcd_byte: int, format: str = "int") -> int | tuple:
        """Unpack a BCD-encoded byte."""
        return unpack_bcd_byte(bcd_byte, format=format)
