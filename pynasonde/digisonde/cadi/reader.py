"""Low-level binary reader for CADI MD2/MD4 files.

This module keeps parsing logic close to the documented on-disk layout so
higher-level extractors can reuse a stable decoded representation.
"""

from __future__ import annotations

import datetime as dt
import struct
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class CadiHeader:
    """Decoded MD2/MD4 header fields."""

    site: str
    ascii_datetime: str
    filetype: str
    nfreqs: int
    ndops: int
    minheight: int
    maxheight: int
    pps: int
    npulses_avgd: int
    base_thr100: int
    noise_thr100: int
    min_dop_forsave: int
    dtime: int
    gain_control: str
    sig_process: str
    noofreceivers: int
    spares: str
    header_datetime: Optional[dt.datetime]


@dataclass
class CadiDetection:
    """One detected Doppler sample tied to time/frequency/height bins."""

    time_index: int
    time_min: int
    time_sec: int
    frequency_index: int
    noise_flag: int
    noise_power10: int
    gain_flag: int
    height_flag: int
    doppler_flag: int
    record_datetime: Optional[dt.datetime]
    iq_samples: Tuple[Tuple[int, int], ...]


@dataclass
class CadiDataset:
    """Container holding decoded CADI header, frequencies, and detections."""

    header: CadiHeader
    frequencies_hz: Tuple[float, ...]
    dheight_km: float
    detections: List[CadiDetection]


class CadiReader:
    """Binary MD2/MD4 reader.

    Notes:
        Parsing follows the same control-flow as the reference converter scripts
        bundled under ``tmp/CADI`` for compatibility with existing datasets.
    """

    def __init__(self, filename: str, dheight_km: float = 3.0):
        self.filename = filename
        self.dheight_km = float(dheight_km)

    @staticmethod
    def _parse_header_datetime(ascii_datetime: str) -> Optional[dt.datetime]:
        """Parse CADI header datetime like ``May 13 12:00:00 2026``."""
        clean = " ".join(ascii_datetime.strip().split())
        try:
            return dt.datetime.strptime(clean, "%b %d %H:%M:%S %Y")
        except ValueError:
            return None

    @staticmethod
    def _u8(fobj) -> int:
        raw = fobj.read(1)
        if len(raw) != 1:
            raise EOFError("Unexpected EOF while reading uint8")
        return struct.unpack("<B", raw)[0]

    @staticmethod
    def _u16(fobj) -> int:
        raw = fobj.read(2)
        if len(raw) != 2:
            raise EOFError("Unexpected EOF while reading uint16")
        return struct.unpack("<H", raw)[0]

    def parse(self) -> CadiDataset:
        """Decode the MD2/MD4 file into a :class:`CadiDataset`."""
        with open(self.filename, "rb") as f:
            f.seek(-1, 2)
            eof = f.tell()
            f.seek(0, 0)

            site = f.read(3).decode("utf-8", errors="replace")
            ascii_datetime = f.read(22).decode("utf-8", errors="replace")
            filetype = f.read(1).decode("utf-8", errors="replace")

            nfreqs = self._u16(f)
            ndops = self._u8(f)
            minheight = self._u16(f)
            maxheight = self._u16(f)
            pps = self._u8(f)
            npulses_avgd = self._u8(f)
            base_thr100 = self._u16(f)
            noise_thr100 = self._u16(f)
            min_dop_forsave = self._u8(f)
            dtime = self._u16(f)
            gain_control = f.read(1).decode("utf-8", errors="replace")
            sig_process = f.read(1).decode("utf-8", errors="replace")
            noofreceivers = self._u8(f)
            spares = f.read(11).decode("utf-8", errors="replace")

            freqs = struct.unpack("<" + "f" * nfreqs, f.read(4 * nfreqs))

            header_dt = self._parse_header_datetime(ascii_datetime)
            header = CadiHeader(
                site=site,
                ascii_datetime=ascii_datetime,
                filetype=filetype,
                nfreqs=nfreqs,
                ndops=ndops,
                minheight=minheight,
                maxheight=maxheight,
                pps=pps,
                npulses_avgd=npulses_avgd,
                base_thr100=base_thr100,
                noise_thr100=noise_thr100,
                min_dop_forsave=min_dop_forsave,
                dtime=dtime,
                gain_control=gain_control,
                sig_process=sig_process,
                noofreceivers=noofreceivers,
                spares=spares,
                header_datetime=header_dt,
            )

            detections: List[CadiDetection] = []
            timex = -1
            time_min = self._u8(f)

            while time_min != 255:
                time_sec = self._u8(f)
                flag = self._u8(f)
                timex += 1

                rec_dt = None
                if header_dt is not None:
                    rec_dt = header_dt + dt.timedelta(
                        minutes=int(time_min), seconds=int(time_sec)
                    )

                for freqx in range(nfreqs):
                    noise_flag = self._u8(f)
                    noise_power10 = self._u16(f)
                    gain_flag = flag
                    flag = self._u8(f)

                    while flag < 224:
                        hflag = flag
                        ndops_oneh = self._u8(f)
                        if ndops_oneh >= 128:
                            ndops_oneh -= 128
                            hflag += 200

                        for _ in range(ndops_oneh):
                            doppler_flag = self._u8(f)
                            iq_samples: List[Tuple[int, int]] = []
                            for _rec in range(noofreceivers):
                                i_raw = self._u8(f)
                                q_raw = self._u8(f)
                                iq_samples.append((i_raw, q_raw))

                            detections.append(
                                CadiDetection(
                                    time_index=timex,
                                    time_min=time_min,
                                    time_sec=time_sec,
                                    frequency_index=freqx,
                                    noise_flag=noise_flag,
                                    noise_power10=noise_power10,
                                    gain_flag=gain_flag,
                                    height_flag=hflag,
                                    doppler_flag=doppler_flag,
                                    record_datetime=rec_dt,
                                    iq_samples=tuple(iq_samples),
                                )
                            )

                        flag = self._u8(f)

                time_min = flag
                if (f.tell() - 1) != eof:
                    time_min = self._u8(f)

        return CadiDataset(
            header=header,
            frequencies_hz=tuple(freqs),
            dheight_km=self.dheight_km,
            detections=detections,
        )
