import copy
import datetime as dt
import struct

import numpy as np
from loguru import logger

from pynasonde.digisonde.datatypes.dftdatatypes import (
    DftHeader,
    DopplerSpectra,
    DopplerSpectralBlock,
)


class DftExtractor(object):

    def __init__(
        self,
        filename: str,
        extract_time_from_name: bool = False,
        extract_stn_from_name: bool = False,
        DATA_BLOCK_SIZE: int = 4096,
        SUB_CASE_NUMBER: int = 16,
    ):
        """
        Initialize the DftExtractor with the given file.

        Args:
            filename (str): Path to the dft file to be processed.
        """
        # Initialize the data structure to hold extracted data
        self.filename = filename
        self.DATA_BLOCK_SIZE = DATA_BLOCK_SIZE
        self.SUB_CASE_NUMBER = SUB_CASE_NUMBER
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
            logger.info(f"Station code: {self.stn_code}")
        return

    def extract(self):
        """
        Main method to extract data from the dft file and populate the sao_struct dictionary.

        Returns:
            dict: The populated dft_struct dictionary containing all extracted data.
        """
        with open(self.filename, "rb") as file:
            for block_index in range(self.BLOCKS):
                logger.debug(f"Reading block {block_index+1} of {self.BLOCKS}")
                dsb = DopplerSpectralBlock(header=DftHeader(), spectra_line=[])
                header_bits_ampl_bytes = []
                for sub_case_index in range(self.SUB_CASE_NUMBER):
                    amplitude_bytes = [file.read(1) for _ in range(128)]
                    phase_bytes = [file.read(1) for _ in range(128)]
                    header_bits_ampl_bytes.extend(copy.copy(amplitude_bytes))

                    if sub_case_index == 0:
                        dsb.header.record_type = hex(
                            struct.unpack("B", amplitude_bytes[0])[0]
                        )
                        logger.debug(f"Record type:{dsb.header.record_type}")
                    ds = DopplerSpectra(
                        amplitude=np.array(
                            [self.unpack_7_1(a[0], False) for a in amplitude_bytes]
                        ).astype(np.float64),
                        phase=np.array([struct.unpack("B", p)[0] for p in phase_bytes]),
                    )
                    dsb.spectra_line.append(ds)

                    # header_bits.extend(lsb_hb)
                self.extract_header_from_amplitudes(header_bits_ampl_bytes)
                # header_bits = "".join(map(str, header_bits))
                # print(header_bits)
                # print(
                #     self.to_int(header_bits[:8]),
                #     dsb.header,
                #     # self.to_int(header_bits[8:24]),
                #     # hex(self.to_int(header_bits[24:32])),
                # )
                # print(header_bits)
                if block_index == 1:
                    break

        return

    def extract_header_from_amplitudes(self, amplitude_bytes: list):
        header_bits = [(b[0] & 0x01) for b in amplitude_bytes]
        # Convert bit list to string of bits
        header_bitstring = "".join(str(b) for b in header_bits)
        # Convert to integers
        record_type = hex(int(header_bitstring[0:4], 2))
        # header_length = int(header_bitstring[8:24], 2)
        # version = hex(int(header_bitstring[24:32], 2))
        print(
            self.to_int(header_bitstring[0:4]),
            self.to_int(header_bitstring[4:8]),
            self.to_int(header_bitstring[8:12]),
        )
        # print(self.to_int(header_bitstring[32:36][::-1]), self.to_int(header_bitstring[36:40][::-1]))
        return

    def to_int(self, bin_strs: str, base: int = 2):
        """
        Converts a binary string to an integer, padding with zeros to ensure it is at least 8 bits.

        Args:
            bin_strs (str): The binary string to convert.

        Returns:
            int: The integer representation of the binary string.
        """
        return int(f"0b{bin_strs.zfill(8)}", base=base)

    def unpack_7_1(self, bcd_byte: int, return_lsb=True):
        """Unpacks a 1-byte packed BCD into 7 bit MSB and 1 bit LSB."""
        msb = (bcd_byte >> 1) & 0x7F
        lsb = bcd_byte & 0x01
        if return_lsb:
            return lsb
        else:
            return msb


if __name__ == "__main__":
    extractor = DftExtractor(
        "tmp/SKYWAVE_DPS4D_2023_10_13/KR835_2023286235715.DFT", True, True
    )
    extractor.extract()
