"""GRM archive splitter for Digisonde batch files from GIRO DIDBase.

GRM files are raw concatenations of 4096-byte ionogram blocks downloaded
from GIRO DIDBase.  Each ionogram type is identified by ``block[0]``
(the record-type byte):

    RSF → 7   SBF → 3   MMM → 9

This module exposes :class:`GrmSplitter`, which can:

- **auto-detect** the ionogram format from the file itself
- **split** the archive to individual files on disk (parallel-safe)
- **load** each ionogram as a lazily-constructed parser object
- **load_dataframes** – extract all ionograms and return a flat DataFrame
  (parallel-safe)

Parallelism strategy
--------------------
All three methods use a two-phase design:

1. ``_scan_offsets()`` — **sequential** linear scan that records only
   ``(datetime, byte_offset, byte_length)`` per ionogram.  No ionogram
   data is copied.
2. Worker functions — **parallel**, each opens the file independently
   and ``seek()``-s to its own slice, so no large bytes objects are
   pickled across process/thread boundaries.

- ``split()``           → I/O-bound  → :class:`ThreadPoolExecutor`
- ``load_dataframes()`` → CPU-bound  → :class:`ProcessPoolExecutor`
- ``load()``            → creates lazy wrappers, stays sequential

Usage::

    spl = GrmSplitter("AU930_20171470000.GRM")
    print(spl.fmt)                               # "MMM"

    paths = spl.split("/tmp/out/")               # sequential
    paths = spl.split("/tmp/out/", n_workers=4)  # parallel (threads)

    df = spl.load_dataframes()                   # sequential
    df = spl.load_dataframes(n_workers=4)        # parallel (processes)

    objs = spl.load()                            # list[_LazyExtractor]
    objs[0].extract(); objs[0].to_pandas()
"""

from __future__ import annotations

import datetime as dt
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import List, Literal, Optional, Tuple, Union

import pandas as pd
from loguru import logger

from pynasonde.digisonde.parsers.mmm import ModMaxExtractor
from pynasonde.digisonde.parsers.rsf import RsfExtractor
from pynasonde.digisonde.parsers.sbf import SbfExtractor

# ── Constants ─────────────────────────────────────────────────────────────────

FmtStr = Literal["RSF", "SBF", "MMM"]

_RECORD_TYPE: dict[str, int] = {"RSF": 7, "SBF": 3, "MMM": 9}
_RECORD_TYPE_INV: dict[int, str] = {v: k for k, v in _RECORD_TYPE.items()}
_EXTRACTOR: dict[str, type] = {
    "RSF": RsfExtractor,
    "SBF": SbfExtractor,
    "MMM": ModMaxExtractor,
}

BLOCK_SIZE = 4096

# Type alias for offset metadata produced by _scan_offsets
_Slice = Tuple[dt.datetime, int, int]  # (date, start_byte, byte_length)


# ── Date helpers ──────────────────────────────────────────────────────────────

def _parse_block_date(block: bytes, fmt: str) -> dt.datetime:
    """Extract the timestamp from the first block of an ionogram."""
    if fmt == "MMM":
        yr_tens = block[3] & 0x0F
        yr_ones = block[4] & 0x0F
        year_2d = 10 * yr_tens + yr_ones
        year   = 1900 + year_2d if year_2d >= 90 else 2000 + year_2d
        doy    = 100 * (block[5] & 0x0F) + 10 * (block[6] & 0x0F) + (block[7] & 0x0F)
        hour   = 10  * (block[8] & 0x0F) + (block[9] & 0x0F)
        minute = 10  * (block[10] & 0x0F) + (block[11] & 0x0F)
        second = 10  * (block[12] & 0x0F) + (block[13] & 0x0F)
        return dt.datetime(year, 1, 1, hour, minute, second) + dt.timedelta(days=doy - 1)

    # RSF / SBF: packed BCD
    raw = (
        block[3:4].hex() + block[6:7].hex() + block[7:8].hex()
        + block[8:9].hex() + block[9:10].hex() + block[10:11].hex()
    )
    return dt.datetime.strptime(raw, "%y%m%d%H%M%S")


# ── Module-level worker functions (must be top-level for pickle) ──────────────

def _read_slice(path: str, start: int, length: int) -> bytes:
    """Read a byte slice from a GRM archive."""
    with open(path, "rb") as f:
        f.seek(start)
        return f.read(length)


def _worker_split(args: tuple) -> Path:
    """Write one ionogram slice to disk.  Called by ThreadPoolExecutor."""
    path, station, fmt, date, start, length, out_dir = args
    data = _read_slice(path, start, length)
    fname = Path(out_dir) / f"{station}_{date.strftime('%Y%j%H%M%S')}.{fmt}"
    fname.write_bytes(data)
    return fname


def _worker_extract(args: tuple) -> Optional[pd.DataFrame]:
    """Extract one ionogram slice to a DataFrame.  Called by ProcessPoolExecutor."""
    path, cls_name, date, start, length = args
    cls = _EXTRACTOR[cls_name]
    data = _read_slice(path, start, length)
    suffix = f".{cls_name}"
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        ext = cls(tmp_path)
        ext.extract()
        df = ext.to_pandas()
        return df
    except Exception as exc:
        logger.warning(f"Failed to extract ionogram @ {date}: {exc}")
        return None
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


# ── GrmSplitter ───────────────────────────────────────────────────────────────

class GrmSplitter:
    """Split or load a GRM batch archive from GIRO DIDBase.

    Args:
        path:      Path to the ``.GRM`` file.
        fmt:       Force a specific format (``"RSF"``, ``"SBF"``, ``"MMM"``).
                   If ``None`` (default) the format is auto-detected.
        station:   Five-character station code used in output filenames.
                   Defaults to the first five characters of the filename.
    """

    def __init__(
        self,
        path: Union[str, Path],
        fmt: Optional[str] = None,
        station: Optional[str] = None,
    ):
        """Create a GRM archive splitter.

        Args:
            path: Path to the ``.GRM`` file.
            fmt: Optional forced format, one of ``"RSF"``, ``"SBF"``, or
                ``"MMM"``. If omitted, the format is auto-detected.
            station: Station code used when writing split files.
        """
        self.path = Path(path)
        self._fmt: Optional[str] = fmt.upper() if fmt else None
        self.station: str = station or self.path.stem[:5]
        self._offsets: Optional[List[_Slice]] = None  # cached after first scan

    # ── Format detection ──────────────────────────────────────────────────────

    @property
    def fmt(self) -> str:
        """Detected or user-specified ionogram format."""
        if self._fmt is None:
            self._fmt = self.detect_format(self.path)
        return self._fmt

    @staticmethod
    def detect_format(path: Union[str, Path]) -> str:
        """Scan the first matching block and return the format string.

        Raises:
            ValueError: if no known record-type byte is found.
        """
        with open(path, "rb") as f:
            while True:
                block = f.read(BLOCK_SIZE)
                if not block:
                    break
                rt = block[0]
                if rt in _RECORD_TYPE_INV:
                    fmt = _RECORD_TYPE_INV[rt]
                    logger.info(f"Auto-detected format: {fmt} (record_type={rt})")
                    return fmt
        raise ValueError(
            f"Could not detect ionogram format in {path}. "
            f"Expected record_type in {list(_RECORD_TYPE_INV)}"
        )

    # ── Phase 1: sequential offset scan ──────────────────────────────────────

    def _scan_offsets(self) -> List[_Slice]:
        """Scan the GRM file once and return (date, start, length) per ionogram.

        Result is cached — subsequent calls are free.
        """
        if self._offsets is not None:
            return self._offsets

        fmt = self.fmt
        target_rt = _RECORD_TYPE[fmt]
        starts: List[Tuple[dt.datetime, int]] = []  # (date, byte_offset)

        with open(self.path, "rb") as f:
            file_size = f.seek(0, 2)
            f.seek(0)
            offset = 0
            while True:
                block = f.read(BLOCK_SIZE)
                if not block:
                    break
                if block[0] == target_rt:
                    starts.append((_parse_block_date(block, fmt), offset))
                offset += BLOCK_SIZE

        slices: List[_Slice] = []
        for i, (date, start) in enumerate(starts):
            end = starts[i + 1][1] if i + 1 < len(starts) else file_size
            slices.append((date, start, end - start))

        logger.info(
            f"Scanned {self.path.name}: {len(slices)} {fmt} ionograms "
            f"({file_size / 1024 / 1024:.1f} MB)"
        )
        self._offsets = slices
        return slices

    # ── Public methods ────────────────────────────────────────────────────────

    def split(
        self,
        out_dir: Union[str, Path],
        n_workers: int = 1,
    ) -> List[Path]:
        """Write each ionogram to a separate file in *out_dir*.

        Files are named ``<station>_<YYYYDDDHHMMSS>.<fmt>``.

        Args:
            out_dir:    Output directory (created if it does not exist).
            n_workers:  Number of threads for parallel file writes.
                        ``1`` (default) → sequential.
                        ``-1`` → use all CPU cores.

        Returns:
            List of :class:`~pathlib.Path` for every file written,
            in ionogram order.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        fmt = self.fmt
        slices = self._scan_offsets()
        n_workers = _resolve_workers(n_workers)

        args_list = [
            (str(self.path), self.station, fmt, date, start, length, str(out_dir))
            for date, start, length in slices
        ]

        if n_workers == 1:
            written = [_worker_split(a) for a in args_list]
        else:
            written = [None] * len(args_list)
            with ThreadPoolExecutor(max_workers=n_workers) as pool:
                futures = {pool.submit(_worker_split, a): i for i, a in enumerate(args_list)}
                for fut in as_completed(futures):
                    written[futures[fut]] = fut.result()

        logger.info(f"split → {len(written)} files in {out_dir}")
        return written

    def load(self) -> List["_LazyExtractor"]:
        """Return lazy extractor wrappers — one per ionogram.

        Each object is cheap to construct.  Call ``.extract()`` and
        ``.to_pandas()`` to materialise the data.  Loading is always
        sequential since the wrappers themselves hold no data.

        Returns:
            Ordered list of :class:`_LazyExtractor` objects.
        """
        fmt = self.fmt
        cls = _EXTRACTOR[fmt]
        slices = self._scan_offsets()
        objs = [
            _LazyExtractor(cls, str(self.path), start, length, date)
            for date, start, length in slices
        ]
        logger.info(f"load → {len(objs)} lazy {fmt} extractors")
        return objs

    def load_dataframes(self, n_workers: int = 1) -> pd.DataFrame:
        """Extract all ionograms and return a single concatenated DataFrame.

        Args:
            n_workers:  Number of processes for parallel extraction.
                        ``1`` (default) → sequential.
                        ``-1`` → use all CPU cores.

        Returns:
            Combined :class:`~pandas.DataFrame`.  Empty if nothing extracted.
        """
        fmt = self.fmt
        slices = self._scan_offsets()
        n_workers = _resolve_workers(n_workers)

        # Pass cls as a string key so the tuple is always picklable
        args_list = [
            (str(self.path), fmt, date, start, length)
            for date, start, length in slices
        ]

        if n_workers == 1:
            frames = [_worker_extract(a) for a in args_list]
        else:
            frames = [None] * len(args_list)
            with ProcessPoolExecutor(max_workers=n_workers) as pool:
                futures = {pool.submit(_worker_extract, a): i for i, a in enumerate(args_list)}
                for fut in as_completed(futures):
                    frames[futures[fut]] = fut.result()

        good = [f for f in frames if f is not None and not f.empty]
        if not good:
            logger.warning("No ionograms successfully extracted.")
            return pd.DataFrame()

        result = pd.concat(good, ignore_index=True)
        logger.info(
            f"load_dataframes → {len(good)}/{len(slices)} ionograms, "
            f"{len(result)} total rows"
        )
        return result


# ── Lazy extractor wrapper ────────────────────────────────────────────────────

class _LazyExtractor:
    """Returned by :meth:`GrmSplitter.load`.

    Defers temp-file creation until ``.extract()`` is first called.
    """

    def __init__(self, cls: type, path: str, start: int, length: int, date: dt.datetime):
        """Create a lazy wrapper for one GRM ionogram slice."""
        self._cls = cls
        self._path = path
        self._start = start
        self._length = length
        self.date = date
        self._inner = None
        self._tmp: Optional[str] = None

    def extract(self) -> "_LazyExtractor":
        """Materialize the temporary file and run the wrapped extractor."""
        if self._inner is None:
            data = _read_slice(self._path, self._start, self._length)
            suffix = "." + self._cls.__name__.replace("Extractor", "").upper()
            suffix = suffix if suffix in (".RSF", ".SBF", ".MMM") else ".tmp"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(data)
                self._tmp = tmp.name
            self._inner = self._cls(self._tmp)
        self._inner.extract()
        return self

    def to_pandas(self) -> pd.DataFrame:
        """Return the wrapped extractor output as a DataFrame."""
        if self._inner is None:
            self.extract()
        return self._inner.to_pandas()

    def __del__(self):
        """Remove the temporary extracted ionogram file when possible."""
        if self._tmp:
            try:
                os.unlink(self._tmp)
            except Exception:
                pass

    def __repr__(self) -> str:
        """Return a compact representation with timestamp and block count."""
        blocks = self._length // BLOCK_SIZE
        return f"<{self._cls.__name__} @ {self.date}  {blocks} blocks>"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _resolve_workers(n: int) -> int:
    """Map -1 → os.cpu_count(), clamp minimum to 1."""
    if n == -1:
        return os.cpu_count() or 1
    return max(1, n)
