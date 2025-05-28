import copy
import datetime as dt
import struct

import numpy as np
from loguru import logger

from pynasonde.digisonde.datatypes.dftdatatypes import (
    DftHeader,
    DopplerSpectra,
    DopplerSpectralBlock,
    SubCaseHeader,
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
                dsb = DopplerSpectralBlock(spectra_line=[])
                header_bits_ampl_bytes = []
                for _ in range(self.SUB_CASE_NUMBER):
                    amplitude_bytes = [file.read(1) for _ in range(128)]
                    phase_bytes = [file.read(1) for _ in range(128)]
                    header_bits_ampl_bytes.extend(copy.copy(amplitude_bytes))
                    ds = DopplerSpectra(
                        amplitude=np.array(
                            [self.unpack_7_1(a[0], False) for a in amplitude_bytes]
                        ).astype(np.float64),
                        phase=np.array([struct.unpack("B", p)[0] for p in phase_bytes]),
                    )
                    dsb.spectra_line.append(ds)
                dsb.header = self.extract_header_from_amplitudes(header_bits_ampl_bytes)
                if block_index == 1:
                    break
        return

    def extract_header_from_amplitudes(self, amplitude_bytes: list) -> DftHeader:
        """
        Extracts the header information from the amplitude bytes.
        Args:
            amplitude_bytes (list): List of bytes containing amplitude data.
        Returns:
            DftHeader: An instance of DftHeader populated with the extracted information.
        """

        header_bits = [(b[0] & 0x01) for b in amplitude_bytes]
        # Convert bit list to string of bits
        header_bitstring = "".join(str(b) for b in header_bits)
        # Extracting header information from the bitstring
        header = DftHeader(
            record_type=hex(int(header_bitstring[:4][::-1], 2)),
            year=(
                int(header_bitstring[4:8][::-1], 2) * 10
                + int(header_bitstring[8:12][::-1], 2)
            ),
            doy=(
                int(header_bitstring[12:16][::-1], 2) * 1e2
                + int(header_bitstring[16:20][::-1], 2) * 1e1
                + int(header_bitstring[20:24][::-1], 2)
            ),
            hour=(
                int(header_bitstring[24:28][::-1], 2) * 10
                + int(header_bitstring[28:32][::-1], 2)
            ),
            minute=(
                int(header_bitstring[32:36][::-1], 2) * 10
                + int(header_bitstring[36:40][::-1], 2)
            ),
            second=(
                int(header_bitstring[40:44][::-1], 2) * 10
                + int(header_bitstring[44:48][::-1], 2)
            ),
            schdule=int(header_bitstring[48:52][::-1], 2),
            program=int(header_bitstring[52:56][::-1], 2),
            drift_data_flag=(hex(int(header_bitstring[56:64][::-1], 2))),
            journal=hex(int(header_bitstring[64:68][::-1], 2)),
            first_height_sampling_winodw=int(header_bitstring[68:72][::-1], 2),
            height_resolution=int(header_bitstring[72:76][::-1], 2),
            number_of_heights=int(header_bitstring[76:80][::-1], 2),
            start_frequency=(
                int(header_bitstring[80:84][::-1], 2) * 1e5
                + int(header_bitstring[84:88][::-1], 2) * 1e4
                + int(header_bitstring[88:92][::-1], 2) * 1e3
                + int(header_bitstring[92:96][::-1], 2) * 1e2
                + int(header_bitstring[96:100][::-1], 2) * 1e1
                + int(header_bitstring[100:104][::-1], 2)
            ),
            disk_io=hex(int(header_bitstring[104:108][::-1], 2)),
            freq_search_enabled=bool(int(header_bitstring[108:112][::-1], 2)),
            fine_frequency_step=(
                int(
                    header_bitstring[116:120][::-1] + header_bitstring[112:116][::-1], 2
                )
            ),
            number_small_steps_scan_abs=int(header_bitstring[120:124][::-1], 2),
            number_small_steps_scan=(
                np.mod(
                    int(
                        header_bitstring[128:132][::-1]
                        + header_bitstring[124:128][::-1],
                        2,
                    )
                    + 16,
                    32,
                )
                - 16
            ),  # Adjusting for signed byte
            start_frequency_case=(
                int(header_bitstring[132:136][::-1], 2) * 10
                + int(header_bitstring[136:140][::-1], 2)
            ),
            coarse_frequency_step=int(header_bitstring[140:144][::-1], 2),
            end_frequency=(
                int(header_bitstring[148:152][::-1], 2) * 10
                + int(header_bitstring[144:148][::-1], 2)
            ),
            bottom_height=(int(header_bitstring[152:156][::-1], 2)),
            top_height=(int(header_bitstring[156:160][::-1], 2)),
            unused=int(header_bitstring[160:164][::-1], 2),
            stn_id=(
                int(header_bitstring[164:168][::-1], 2) * 1e2
                + int(header_bitstring[168:172][::-1], 2) * 10
                + int(header_bitstring[172:176][::-1], 2)
            ),
            phase_code=int(header_bitstring[176:180][::-1], 2),
            multi_antenna_sequence=int(header_bitstring[180:184][::-1], 2),
            cit_length=int(header_bitstring[184:192][::-1], 2),
            num_doppler_lines=int(header_bitstring[192:196][::-1], 2),
            pulse_repeat_rate=int(header_bitstring[196:200][::-1], 2),
            waveform_type=hex(int(header_bitstring[200:204][::-1], 2)),
            delay=int(header_bitstring[204:208][::-1], 2),
            frequency_search_offset=int(header_bitstring[208:212][::-1], 2),
            auto_gain=int(header_bitstring[212:216][::-1], 2),
            heights_to_output=int(
                header_bitstring[220:224][::-1] + header_bitstring[216:220][::-1], 2
            ),
            num_of_polarizations=int(header_bitstring[224:228][::-1], 2),
            start_gain=int(header_bitstring[228:232][::-1], 2),
            subcases=[],
        )
        print(header)
        logger.debug(f"Record type:{header.record_type}")
        sub = SubCaseHeader(
            frequency=(
                int(header_bitstring[232:240][::-1], 2) * 1e4
                + int(header_bitstring[240:248][::-1], 2) * 1e3
                + int(header_bitstring[248:256][::-1], 2) * 1e2
                + int(header_bitstring[256:264][::-1], 2) * 1e1
                + int(header_bitstring[264:272][::-1], 2)
            ),
            height_mpa=(
                int(header_bitstring[272:280][::-1], 2) * 1e3
                + int(header_bitstring[280:288][::-1], 2) * 1e2
                + int(header_bitstring[288:296][::-1], 2) * 1e1
                + int(header_bitstring[296:304][::-1], 2)
            ),
            height_bin=(
                int(header_bitstring[288:296][::-1], 2) * 1e1
                + int(header_bitstring[296:304][::-1], 2)
            ),
            agc_offset=int(header_bitstring[304:312][::-1], 2),
            polarization=int(header_bitstring[312:316][::-1], 2),
        )
        print(sub, header_bitstring[312:316][::-1])
        return header

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
