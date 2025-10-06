"""DFT parser utilities for Digisonde DFT-format files.

This module provides :class:`DftExtractor`, a compact reader that
unpacks Digisonde DFT binary blocks into the lightweight dataclasses
defined in :mod:`pynasonde.digisonde.datatypes.dftdatatypes`.

The implementation focuses on bit-level unpacking and construction of
``DftHeader``, ``DopplerSpectra`` and ``DopplerSpectralBlock`` objects
so the parsed records can be consumed by higher-level tooling or used
directly in documentation examples.
"""

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
    """Low-level reader for DFT-format files.

    This class provides a minimal API to read a DFT-format file and
    produce block-level containers. It intentionally avoids complex
    dependence on external libraries beyond numpy so it can be used in
    lightweight docs and tests.
    """

    def __init__(
        self,
        filename: str,
        extract_time_from_name: bool = False,
        extract_stn_from_name: bool = False,
        DATA_BLOCK_SIZE: int = 4096,
        SUB_CASE_NUMBER: int = 16,
    ) -> None:
        """Create a DftExtractor instance.

        Parameters:
            filename: str
                Path to the DFT-format file to read.
            extract_time_from_name: bool, optional
                If True, attempt to parse a timestamp from the filename.
            extract_stn_from_name: bool, optional
                If True, attempt to parse a station code from the filename.
            DATA_BLOCK_SIZE: int, optional
                Block size in bytes used by the DFT format.
            SUB_CASE_NUMBER: int, optional
                Number of sub-cases per block.
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

    def extract(self) -> None:
        """Read the DFT file and construct DopplerSpectralBlock objects.

        The method iterates over blocks in the file and assembles
        ``DopplerSpectralBlock`` containers holding a header and a list
        of :class:`DopplerSpectra` objects. The current implementation
        is minimal and intended for examples and testing; callers may
        adapt it to return or yield parsed records instead of mutating
        internal state.

        Returns:
            The method populates local variables and currently returns
                None. Future revisions may return an iterable of parsed
                blocks.
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
        """Decode header bits embedded in the amplitude LSBs.

        The DFT format stores header bits spread across the least
        significant bits of amplitude bytes. This routine reconstructs
        the bitstring and converts the bit fields into a
        :class:`DftHeader` dataclass instance.

        Parameters:
            amplitude_bytes: list
                Sequence of 1-byte objects (as returned by ``file.read(1)``)
                that contain the embedded header bits in their LSB.

        Returns:
            Populated header dataclass with parsed integer and raw
            (hex) fields where applicable.
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

    def to_int(self, bin_strs: str, base: int = 2) -> int:
        """Convert a binary string fragment to an integer.

        The helper pads the provided bit string to at least 8 bits and
        converts it to an integer using the provided base.

        Parameters:
            bin_strs: str
                Bitstring fragment (e.g. '1010').
            base: int, optional
                Numeric base to use for conversion (default 2).

        Returns:
            Integer representation.
        """
        return int(f"0b{bin_strs.zfill(8)}", base=base)

    def unpack_7_1(self, bcd_byte: int, return_lsb: bool = True) -> int:
        """Unpack a 1-byte packed BCD into 7-bit MSB and 1-bit LSB.

        Parameters:
            bcd_byte: int
                The raw byte value (0-255).
            return_lsb: bool, optional
                If True return the least-significant-bit, otherwise return
                the seven most-significant bits.

        Returns:
            Either the LSB (0/1) or the 7-bit MSB integer.
        """
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
