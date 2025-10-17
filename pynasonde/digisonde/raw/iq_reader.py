"""
Python port of the Julia `IQReader` module.

This module provides utilities for working with IQ sample recordings that are
organized into one-second binary files whose names encode the RF center
frequency and sample rate.  Files are arranged in a time-based directory tree:

    root/YYYY-mm-dd/HH/MM/<timestamp>_<channel>_fcXXXXXkHz_bwYYYYYkHz.bin

Each binary file stores interleaved 16-bit (little endian) I and Q samples.
"""

from __future__ import annotations

import datetime as dt
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import numpy as np
from loguru import logger

_SECONDS = dt.timedelta(seconds=1)
_MINUTES = dt.timedelta(minutes=1)
_UTC = dt.timezone.utc


def _ensure_utc(epoch: dt.datetime) -> dt.datetime:
    """Return the epoch as a timezone-aware UTC datetime."""
    if epoch.tzinfo is None:
        return epoch.replace(tzinfo=_UTC)
    return epoch.astimezone(_UTC)


def _subdir_for_epoch(
    dir_iq: Path, epoch: dt.datetime, fmt: Optional[str] = "%H/%M"
) -> Path:
    """Build the sub-directory path that holds the IQ file for ``epoch``."""
    return dir_iq / epoch.strftime(fmt)


def _filename_stub(epoch: dt.datetime, rx_tag: str) -> str:
    """Base filename (without frequency suffixes) used for IQ recordings."""
    return f"{epoch.strftime('%Y-%m-%d_%H%M%S')}_{rx_tag}"


def _parse_frequency(token: str, prefix: str) -> float:
    """
    Extract the frequency (Hz) from a filename token such as ``fc05000kHz``.
    """
    if not token.startswith(prefix):
        raise ValueError(f"Token '{token}' does not start with '{prefix}'")
    value_khz = token[len(prefix) :].split("kHz")[0]
    return float(value_khz) * 1e3


@dataclass
class IQStream:
    """
    Direct stream reader mirroring the Julia ``IQStream`` object.

    Parameters
    ----------
    dir_iq:
        Root directory that contains the time-partitioned IQ recordings.
    epoch:
        Approximate epoch from which to start the stream.  This can be naive
        (assumed UTC) or timezone-aware.
    rx_tag:
        Channel tag that identifies which stream to use (defaults to ``"ch0"``).
    """

    dir_iq: Path
    rx_tag: str
    file_size: int
    f_center: float
    f_sample: float
    epoch: dt.datetime
    _stream: Optional[object] = field(default=None, init=False, repr=False)
    _raw_samples: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.complex64), init=False, repr=False
    )

    def __init__(
        self, dir_iq: Path | str, epoch: dt.datetime, rx_tag: str = "ch0"
    ) -> None:
        dir_path = Path(dir_iq)
        epoch_utc = _ensure_utc(epoch)
        f_center, f_sample = get_frequencies(dir_path, epoch_utc)

        self.dir_iq = dir_path
        self.rx_tag = rx_tag
        self.file_size = int(round(f_sample))  # hard-coded for one-second files
        self.f_center = f_center
        self.f_sample = f_sample
        self.epoch = dt.datetime(1970, 1, 1, tzinfo=_UTC)
        self._stream = None
        self._raw_samples = np.empty(0, dtype=np.complex64)

    # ------------------------------------------------------------------
    def close(self) -> None:
        """Close the underlying file handle if it is open."""
        if self._stream is not None:
            self._stream.close()
            self._stream = None

    # ------------------------------------------------------------------
    def read_samples(
        self, epoch0: dt.datetime, n_samples: Optional[int] = None
    ) -> np.ndarray:
        """
        Read ``n_samples`` complex IQ samples starting near ``epoch0``.

        Results are returned as a NumPy array of ``np.complex64`` values.  The
        method reuses internal buffers across calls to minimize allocations.  When
        ``n_samples`` is ``None`` the entire one-second file that contains
        ``epoch0`` is returned.
        """
        epoch0_utc = _ensure_utc(epoch0)
        read_full_file = n_samples is None

        if read_full_file:
            epoch0_utc = epoch0_utc.replace(microsecond=0)
            n_samples = self.file_size

        if n_samples <= 0:
            return np.empty(0, dtype=np.complex64)

        if self._raw_samples.shape[0] != n_samples:
            self._raw_samples = np.empty(n_samples, dtype=np.complex64)

        # Determine the byte offset within the currently open file.  If the
        # request sits outside the active file window, move to the appropriate
        # file that covers the requested epoch.
        delta_seconds = (epoch0_utc - self.epoch).total_seconds()
        file_position = int(round(delta_seconds * self.f_sample))
        if file_position < 0 or file_position >= self.file_size:
            self.close()
            self._seek_file_from_epoch(epoch0_utc)
            delta_seconds = (epoch0_utc - self.epoch).total_seconds()
            file_position = int(round(delta_seconds * self.f_sample))

        if read_full_file:
            file_position = 0

        samples_remaining = n_samples
        write_idx = 0
        file_index = 0

        while samples_remaining > 0:
            if file_position >= self.file_size:
                self.close()
                file_index += 1
                next_epoch = epoch0_utc + file_index * _SECONDS
                self._seek_file_from_epoch(next_epoch)
                file_position = 0

            n_collect = min(self.file_size - file_position, samples_remaining)
            self._stream.seek(file_position * 4)  # 4 bytes per I/Q pair
            raw_bytes = self._stream.read(n_collect * 4)

            if len(raw_bytes) != n_collect * 4:
                raise EOFError("Unexpected end of IQ file while reading samples")

            ints = np.frombuffer(raw_bytes, dtype="<i2").astype(np.float32)
            iq_pairs = ints.reshape(-1, 2)
            complex_chunk = iq_pairs[:, 0] + 1j * iq_pairs[:, 1]
            self._raw_samples[write_idx : write_idx + n_collect] = complex_chunk.astype(
                np.complex64
            )

            write_idx += n_collect
            samples_remaining -= n_collect
            file_position += n_collect

        return self._raw_samples.copy()

    # ------------------------------------------------------------------
    def _seek_file_from_epoch(
        self, epoch: dt.datetime, fmt: Optional[str] = "%H/%M"
    ) -> None:
        """
        Locate and open the IQ file that contains ``epoch``.

        The method waits for future files (with a two-second cushion) and
        supports a fallback glob search if the canonical filename is not yet
        known.
        """

        epoch = _ensure_utc(epoch)
        subdir = _subdir_for_epoch(self.dir_iq, epoch, fmt)
        stub = _filename_stub(epoch, self.rx_tag)

        # Wait for files scheduled in the future.
        time_to_wait = (epoch - dt.datetime.now(_UTC)).total_seconds() + 2.0
        if time_to_wait > 0:
            time.sleep(time_to_wait)

        canonical_name = (
            f"{stub}"
            f"_fc{self.f_center * 1e-3:05.0f}kHz"
            f"_bw{self.f_sample * 1e-3:05.0f}kHz.bin"
        )
        candidate = subdir / canonical_name

        try:
            stream = candidate.open("rb")
            path_bin = candidate
        except FileNotFoundError:
            matches = sorted(subdir.glob(f"{stub}*.bin"))
            if not matches:
                raise FileNotFoundError(
                    f"No IQ file matching '{stub}*.bin' in {subdir}"
                )
            path_bin = matches[0]
            stream = path_bin.open("rb")

        self.epoch = epoch.replace(microsecond=0)
        self._stream = stream


def get_frequencies(
    dir_iq: Path | str, epoch: dt.datetime, fmt: Optional[str] = "%H/%M"
) -> tuple[float, float]:
    """
    Derive center and sampling frequencies based on the closest available file.

    The search scans forward in one-minute increments for up to 24 hours, then
    backwards for another 24 hours, mirroring the Julia implementation.
    """
    dir_path = Path(dir_iq)
    epoch_utc = _ensure_utc(epoch)

    offsets: Sequence[int] = list(range(0, 24 * 60 + 1)) + list(
        range(-1, -(24 * 60) - 1, -1)
    )
    last_bins: List[Path] = []

    for minutes_offset in offsets:
        epoch_scan = epoch_utc + minutes_offset * _MINUTES
        subdir = _subdir_for_epoch(dir_path, epoch_scan, fmt)
        if not subdir.exists():
            continue
        bins = sorted(subdir.glob("*.bin"))
        if bins:
            last_bins = bins
            break

    if not last_bins:
        logger.info(
            "IQReader: closest file not found. System time=%s, requested epoch=%s",
            dt.datetime.now(_UTC).strftime("%Y/%m/%d %H:%M:%S"),
            epoch_utc.isoformat(),
        )
        subdir = _subdir_for_epoch(dir_path, epoch_utc, fmt)
        raise FileNotFoundError(f"No IQ files under {subdir}")

    filename = last_bins[0].name
    tokens = filename.split("_")

    try:
        fc_token = next(token for token in tokens if token.startswith("fc"))
        bw_token = next(token for token in tokens if token.startswith("bw"))
    except StopIteration as exc:
        raise ValueError(
            f"Filename '{filename}' does not encode frequency metadata"
        ) from exc

    f_center = _parse_frequency(fc_token, "fc")
    f_sample = _parse_frequency(bw_token, "bw")

    return f_center, f_sample


def get_channels(
    dir_iq: Path | str,
    epoch: Optional[dt.datetime] = None,
    fmt: Optional[str] = "%H/%M",
) -> List[str]:
    """
    Return the unique channel tags available in the current minute directory.

    Parameters
    ----------
    dir_iq:
        Root IQ directory.
    epoch:
        Optional epoch to inspect.  Defaults to ``datetime.now(timezone.utc)``.
    """
    dir_path = Path(dir_iq)
    epoch = _ensure_utc(epoch or dt.datetime.now(_UTC))
    subdir = _subdir_for_epoch(dir_path, epoch, fmt)
    tags: List[str] = []

    if not subdir.exists():
        return tags

    for bin_file in sorted(subdir.glob("*.bin")):
        parts = bin_file.name.split("_")
        if len(parts) < 3:
            continue
        tag = parts[2]
        if tag not in tags:
            tags.append(tag)

    return tags


__all__ = ["IQStream", "get_frequencies", "get_channels"]
