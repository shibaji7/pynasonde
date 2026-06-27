"""Binary MMM/ModMax format parser for Digisonde ionogram data.

This module provides :class:`ModMaxExtractor`, a parser for the MMM
(ModMax) binary files produced by Digisonde instruments.  MMM files store
sounder output in fixed-size 4096-byte blocks similar to the SBF format but
with a distinct header and frequency-group layout.

Key constants:
    ``MMM_IONOGRAM_SETTINGS`` — maps the number of height bins (128, 256,
    or 512) to the frequency-block/range-bin/byte-length metadata used
    during parsing.

Typical usage::

    extractor = ModMaxExtractor("WP937_2022233235510.MMM",
                                 extract_time_from_name=True,
                                 extract_stn_from_name=True)
    extractor.extract()
"""

import datetime as dt
from typing import List

import numpy as np
import pandas as pd
from loguru import logger

from pynasonde.digisonde.datatypes.mmmdatatypes import ModMaxHeader
from pynasonde.digisonde.digi_utils import (
    apply_filename_metadata,
    low_nibble,
    merge_dicts_selected_keys,
    read_u8,
    unpack_4_4_byte,
    unpack_5_3_byte,
    unpack_bcd_byte,
)

MMM_IONOGRAM_SETTINGS = {
    # IH < 8: 128 range bins, 30 frequency groups per block
    # Fortran (unpack_dpsmmm.f): DBSCALE=6/16, MASKH=0xF0, MASKL=0x0F
    # → amplitude = (byte >> 4) × 6 dB;  channel/status = byte & 0x0F
    "block_type_1": dict(
        blk_type=1,
        n_groups=30,
        n_bins=128,
        byte_length=134,  # 6 prelude + 128 data bytes
        amp_shift=4,
        status_mask=0x0F,
        db_per_level=6,
    ),
    # IH >= 8: 256 range bins, 15 frequency groups per block
    # Fortran uses the same 4+4 split (MASKH/MASKL unchanged).
    # Note: Fortran authors flagged "need 256 range data for testing" —
    # treat as same encoding until confirmed otherwise.
    "block_type_2": dict(
        blk_type=2,
        n_groups=15,
        n_bins=256,
        byte_length=262,  # 6 prelude + 256 data bytes
        amp_shift=4,
        status_mask=0x0F,
        db_per_level=6,
    ),
}


class ModMaxExtractor(object):
    """Extract ionogram rows from MMM/ModMax binary files."""

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
            filename (str): Path to the MMM file to be processed.
        """
        # Initialize the data structure to hold extracted data
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
        return

    def extract(self):
        """Extract all blocks and frequency groups from the MMM file."""
        self.records = []
        with open(self.filename, "rb") as file:
            for n in range(self.BLOCKS):
                # Seek to exact block boundary to guard against any padding bytes
                file.seek(n * self.DATA_BLOCK_SIZE)
                logger.debug(f"Reading block {n+1} of {self.BLOCKS}")
                ub = lambda: read_u8(file)
                # Preface bytes store one BCD digit per byte in the lower
                # nibble only (pref_mmm.f: IAND(IBUF(IE), 15)).
                nb = lambda: low_nibble(ub())

                # ── 60-byte header ─────────────────────────────────────────
                # Read fixed fields first so year can be decoded correctly
                record_type = ub()
                header_length = ub()
                version_maker = hex(ub())
                yr_tens, yr_ones = nb(), nb()
                year_2d = 10 * yr_tens + yr_ones
                # BCD two-digit year: ≥90 → 1900s, else → 2000s
                year = 1900 + year_2d if year_2d >= 90 else 2000 + year_2d

                h = ModMaxHeader(
                    record_type=record_type,
                    header_length=header_length,
                    version_maker=version_maker,
                    year=year,
                    doy=(nb() * 1e2 + nb() * 1e1 + nb()),
                    hour=(nb() * 1e1 + nb()),
                    minute=(nb() * 1e1 + nb()),
                    second=(nb() * 1e1 + nb()),
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
                    # Walrus operators capture raw ints for CIT computation below
                    # while still storing hex strings in the header.
                    # PREFACE(28-32) → block bytes 30-34 (PREFACE(N) = block[N+2])
                    mmm_opt=hex(_b30 := ub()),          # PREFACE(28): phase code → NCOMP
                    print_clean_ctrl=hex(_b31 := ub()), # PREFACE(29): pol. count → NPOL
                    print_gain_lev=hex(_b32 := ub()),   # PREFACE(30): FFT size exp → NFFT
                    ctrl_intm_tx=hex(_b33 := ub()),     # PREFACE(31): PRF high byte
                    drft_use=hex(_b34 := ub()),         # PREFACE(32): PRF low byte
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
                logger.debug(f"  Header: {h}")

                # ── Select block type from range_inc (preface field H index) ──
                # range_inc is an index 0-15 (field H in D256 preface), NOT km.
                # IH >= 8 → 256 bins; IH < 8 → 128 bins  (from rg_d256.f)
                ih = low_nibble(h.range_inc)   # lower nibble = index H
                ie = low_nibble(h.range_start)  # lower nibble = index E
                cfg = MMM_IONOGRAM_SETTINGS[
                    "block_type_1" if ih < 8 else "block_type_2"
                ]
                n_bins = cfg["n_bins"]
                n_groups = cfg["n_groups"]
                amp_shift = cfg["amp_shift"]
                status_mask = cfg["status_mask"]
                db_per_level = cfg["db_per_level"]

                datetime_ = dt.datetime(
                    int(h.year), 1, 1, int(h.hour), int(h.minute), int(h.second)
                ) + dt.timedelta(days=int(h.doy) - 1)

                # ── Height range gate table (rg_d256.f) ───────────────────────
                # HTBL: km spacing per range-inc index H (0-15)
                # ETBL: start height km per range-start index E (1-5)
                _HTBL = [2.5, 5.0, 10.0, 2.5, 0.0, 0.0, 0.0, 0.0,
                         2.5, 5.0, 10.0,  2.5,  5.0, 0.0, 0.0, 0.0]
                _ETBL = [0.0, 10.0, 60.0, 160.0, 380.0, 760.0]  # 1-indexed; [0] unused

                dh = _HTBL[ih] if ih < len(_HTBL) else 5.0
                # ie is 1-5; guard against 0 (malformed header)
                ie_safe = max(1, min(ie, 5))
                h0 = _ETBL[ie_safe]

                # Bilinear modes (ih=3,11,12): lower IBL gates at dh, upper at 2×dh
                _IBL = {3: 40, 11: 128, 12: 128}
                ibl = _IBL.get(ih, 0)
                if ibl == 0:
                    heights = h0 + np.arange(n_bins) * dh
                else:
                    lower = h0 + np.arange(ibl) * dh
                    upper = h0 + np.arange(ibl, n_bins) * (2.0 * dh)
                    heights = np.concatenate([lower, upper])

                logger.debug(
                    f"  Block {n}: IH={ih} IE={ie_safe}  "
                    f"h0={h0:.1f} km  dh={dh:.1f} km  "
                    f"height range {heights[0]:.1f}–{heights[-1]:.1f} km"
                )

                # ── CIT → Doppler Hz conversion (dop_dps.f) ──────────────────
                # Computed once per block; used for all range bins in this block.
                # PREFACE(28) → IX  → NCOMP: complementary phase code multiplier
                # PREFACE(29) → IA  → NPOL:  number of polarizations sounded
                # PREFACE(30) → IN  → NFFT:  FFT size = 2^IN
                # PREFACE(31-32)   → PRF:   pulse repetition frequency (Hz, BCD)
                # CIT = NFFT * NPOL * NCOMP / PRF  (coherent integration time, s)
                _bcd2 = lambda b: unpack_bcd_byte(b)
                _ncomp = 2 if _bcd2(_b30) == 1 else 1
                _npol  = 1 if _bcd2(_b31) > 7  else 2
                _nfft  = 1 << _bcd2(_b32)
                _prf_raw = 100 * _bcd2(_b33) + _bcd2(_b34)
                if _prf_raw == 0:
                    # Newer DPS-4D firmware leaves PRF bytes zero.
                    # dop_dps.f warns CIT is unreliable for ionogram mode;
                    # doppler_hz will use the PRF=1 fallback.
                    logger.debug(
                        "  PRF bytes (PREFACE 31-32) = 0; DPS-4D firmware "
                        "may not populate these. doppler_hz uses PRF=1 Hz fallback."
                    )
                _prf   = max(1.0, float(_prf_raw))
                _cit   = (_nfft * _npol * _ncomp) / _prf
                logger.debug(
                    f"  CIT: NFFT={_nfft} NPOL={_npol} NCOMP={_ncomp} "
                    f"PRF={_prf:.1f} Hz → CIT={_cit:.4f} s"
                )

                # ── Frequency groups ──────────────────────────────────────────
                for _ in range(n_groups):
                    # First prelude byte: upper nibble = IPOL, lower nibble = blk_type
                    # (prelude_mmm.f lines 99-100)
                    first_byte = ub()
                    unpack_4_4_byte(first_byte)
                    freq_mhz = self.unpack_bcd(ub(), format="int")
                    freq_k = self.unpack_bcd(ub(), format="int") * 10
                    freq_search, gain_param = self.unpack_bcd(
                        ub(), format="tuple"
                    )  # 4+4 bits
                    self.unpack_bcd(ub(), format="int")
                    mpa = ub()
                    frequency_mhz = freq_mhz + freq_k / 1000.0
                    mpa_db = float(mpa * db_per_level)

                    # 1 byte per range bin: packed amplitude (MSB) + channel (LSB)
                    # Channel lower-nibble layout — T=1 mode (2 beams × 8 Dopplers)
                    # channel_map.f IC2S(3,:) + IBM(3)=1:
                    #   ICH = channel + 1;  IB = IC2S(3,ICH) & 1
                    #   channel EVEN → ICH ODD → IB=0 → O-mode
                    #   channel ODD  → ICH EVEN → IB=1 → X-mode
                    # Doppler IDD from IS2RD(3, status): pairs (0-1)=+1, (2-3)=+2,
                    #   (4-5)=+3, (6-7)=+4, (8-9)=-1, (10-11)=-2, (12-13)=-3, (14-15)=-4
                    raw = np.frombuffer(file.read(n_bins), dtype=np.uint8)
                    # DPS: remove first 6 and last 12 bins (unpack_dpsmmm.f: MAXRBIN-=18)
                    raw = raw[6: n_bins - 12]
                    heights_trim = heights[6: n_bins - 12]
                    amps    = (raw >> amp_shift).astype(float) * db_per_level  # → dB
                    channel = (raw & status_mask).astype(int)                  # 0-15
                    pol_arr = np.where((channel % 2) == 0, "X", "O")
                    _IDD = {0:+1,1:+1, 2:+2,3:+2, 4:+3,5:+3, 6:+4,7:+4,
                            8:-1,9:-1, 10:-2,11:-2, 12:-3,13:-3, 14:-4,15:-4}
                    dop_arr = np.vectorize(lambda c: _IDD[c])(channel)

                    # ── Doppler Hz (dop_dps.f) ────────────────────────────────
                    # DOPTAB[NDOP] = [-7,-5,-3,-1,+1,+3,+5,+7]
                    # IDD→NDOP: IDD<0 → NDOP=IDD+4;  IDD>0 → NDOP=IDD+3
                    # doppler_hz = DOPTAB[NDOP] / (2 * CIT)
                    ndop_arr = np.where(dop_arr < 0, dop_arr + 4, dop_arr + 3)
                    doptab_arr = (2 * ndop_arr - 7).astype(float)  # = DOPTAB[NDOP]
                    dop_hz_arr = doptab_arr / (2.0 * _cit)

                    for amp_db, ch, pol, dop, dop_hz, hgt in zip(
                        amps, channel, pol_arr, dop_arr, dop_hz_arr, heights_trim
                    ):
                        self.records.append(
                            {
                                "datetime": datetime_,
                                "frequency_mhz": frequency_mhz,
                                "range_km": float(hgt),
                                "amplitude_dB": float(amp_db),
                                "channel": int(ch),
                                "polarization": pol,
                                "doppler_channel": int(dop),
                                "doppler_hz": float(dop_hz),
                                "mpa_dB": mpa_db,
                                "block": n,
                            }
                        )

        logger.info(f"Extracted {len(self.records)} records from {self.filename}")
        return

    def add_dicts_selected_keys(self, d0, du, keys=None) -> dict:
        """Merge parser dictionaries, optionally selecting keys from ``du``."""
        return merge_dicts_selected_keys(d0, du, keys=keys)

    def to_pandas(self) -> pd.DataFrame:
        """Return extracted records as a DataFrame. Call extract() first."""
        if not hasattr(self, "records") or not self.records:
            logger.warning("No records found — call extract() first.")
            return pd.DataFrame()
        df = pd.DataFrame(self.records)
        logger.info(f"DataFrame shape: {df.shape}  columns: {list(df.columns)}")
        return df

    def unpack_5_3(self, bcd_byte: int) -> List[int]:
        """Unpacks a 1-byte packed BCD into 5 bit MSB and 3 bit LSB."""
        return unpack_5_3_byte(bcd_byte)

    def unpack_bcd(self, bcd_byte: int, format: str = "int") -> int | tuple:
        """Unpacks a 1-byte packed BCD into two decimal digits."""
        return unpack_bcd_byte(bcd_byte, format=format)
