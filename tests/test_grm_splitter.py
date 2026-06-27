"""Tests for GrmSplitter (pynasonde.digisonde.parsers.grm).

Test strategy
-------------
- **Synthetic GRM bytes** are used for all unit tests — no real files needed.
  A helper ``_make_grm()`` builds minimal valid GRM archives in memory and
  writes them to a ``tmp_path`` fixture directory.
- A single **integration test** against the real AU930 MMM file is skipped
  automatically if the file is not present.
- Parallel paths (``n_workers > 1``) are tested with ``n_workers=2`` on a
  small synthetic file so CI stays fast.
"""

import datetime as dt
import struct
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from pynasonde.digisonde.parsers.grm import (
    BLOCK_SIZE,
    GrmSplitter,
    _parse_block_date,
    _resolve_workers,
    _worker_extract,
    _worker_split,
)

# ── Synthetic GRM helpers ─────────────────────────────────────────────────────

_AU930_MMM = "/home/chakras4/Research/ERAUCodeBase/apep_eclipse/AU930_2017147000005.MMM"

# Record-type bytes for each format
_RT = {"RSF": 7, "SBF": 3, "MMM": 9}


def _mmm_header_bytes(year: int, doy: int, hour: int, minute: int, second: int) -> bytes:
    """Build a minimal 60-byte MMM block header with correct BCD preface fields."""
    hdr = bytearray(BLOCK_SIZE)
    hdr[0] = _RT["MMM"]   # record_type = 9
    hdr[1] = 60           # header_length
    hdr[2] = 0x01         # version_maker
    year_2d = year % 100
    hdr[3] = year_2d // 10
    hdr[4] = year_2d % 10
    hdr[5] = doy // 100
    hdr[6] = (doy % 100) // 10
    hdr[7] = doy % 10
    hdr[8]  = hour // 10
    hdr[9]  = hour % 10
    hdr[10] = minute // 10
    hdr[11] = minute % 10
    hdr[12] = second // 10
    hdr[13] = second % 10
    # range_inc index H=1 (5 km), range_start index E=2 (60 km)
    hdr[56] = 1
    hdr[57] = 2
    return bytes(hdr)


def _rsf_header_bytes(year: int, month: int, day: int, hour: int, minute: int, second: int) -> bytes:
    """Build a minimal 4096-byte RSF block with packed BCD datetime."""
    hdr = bytearray(BLOCK_SIZE)
    hdr[0] = _RT["RSF"]
    # packed BCD: year in byte 3, month in 6, day in 7, hour in 8, min in 9, sec in 10
    yr_2d = year % 100
    hdr[3] = ((yr_2d // 10) << 4) | (yr_2d % 10)
    hdr[6] = ((month // 10) << 4) | (month % 10)
    hdr[7] = ((day // 10) << 4) | (day % 10)
    hdr[8] = ((hour // 10) << 4) | (hour % 10)
    hdr[9] = ((minute // 10) << 4) | (minute % 10)
    hdr[10] = ((second // 10) << 4) | (second % 10)
    return bytes(hdr)


def _make_grm(tmp_path: Path, fmt: str, n_ionograms: int = 2, blocks_per: int = 3) -> Path:
    """Write a synthetic GRM file with ``n_ionograms`` ionograms of ``blocks_per`` blocks each."""
    grm = tmp_path / f"TEST_{fmt}.GRM"
    with grm.open("wb") as f:
        for i in range(n_ionograms):
            # First block of each ionogram has the ionogram record_type
            if fmt == "MMM":
                first = _mmm_header_bytes(2017, 147 + i, i, 0, 5)
            else:
                first = _rsf_header_bytes(2017, 5, 27 + i, i, 0, 0)
            f.write(first)
            # Subsequent blocks have a different (continuation) record_type
            for _ in range(blocks_per - 1):
                cont = bytearray(BLOCK_SIZE)
                cont[0] = 0x00  # not a header block
                f.write(bytes(cont))
    return grm


# ── _parse_block_date ─────────────────────────────────────────────────────────

class TestParseBlockDate:
    def test_mmm_roundtrip(self):
        block = _mmm_header_bytes(2017, 147, 0, 0, 5)
        parsed = _parse_block_date(block, "MMM")
        assert parsed == dt.datetime(2017, 5, 27, 0, 0, 5)

    def test_mmm_nibble_masking(self):
        """Upper nibble flags in BCD bytes must be masked away."""
        block = bytearray(_mmm_header_bytes(2017, 147, 0, 0, 5))
        # Corrupt upper nibbles of yr_tens and doy fields
        block[3] |= 0xF0
        block[5] |= 0x10
        parsed = _parse_block_date(bytes(block), "MMM")
        assert parsed == dt.datetime(2017, 5, 27, 0, 0, 5)

    def test_mmm_year_pre_2000(self):
        block = _mmm_header_bytes(1999, 1, 12, 30, 0)
        parsed = _parse_block_date(block, "MMM")
        assert parsed.year == 1999

    def test_mmm_year_post_2000(self):
        block = _mmm_header_bytes(2025, 365, 23, 59, 59)
        parsed = _parse_block_date(block, "MMM")
        assert parsed.year == 2025
        assert parsed.hour == 23
        assert parsed.second == 59

    def test_rsf_roundtrip(self):
        block = _rsf_header_bytes(2017, 5, 27, 12, 30, 45)
        parsed = _parse_block_date(block, "RSF")
        assert parsed == dt.datetime(2017, 5, 27, 12, 30, 45)


# ── detect_format ─────────────────────────────────────────────────────────────

class TestDetectFormat:
    def test_detects_mmm(self, tmp_path):
        grm = _make_grm(tmp_path, "MMM")
        assert GrmSplitter.detect_format(grm) == "MMM"

    def test_detects_rsf(self, tmp_path):
        grm = _make_grm(tmp_path, "RSF")
        assert GrmSplitter.detect_format(grm) == "RSF"

    def test_unknown_raises(self, tmp_path):
        bad = tmp_path / "bad.GRM"
        bad.write_bytes(bytes(BLOCK_SIZE))  # all zeros → record_type=0
        with pytest.raises(ValueError, match="Could not detect"):
            GrmSplitter.detect_format(bad)

    def test_fmt_property_caches(self, tmp_path):
        grm = _make_grm(tmp_path, "MMM")
        spl = GrmSplitter(grm)
        _ = spl.fmt
        _ = spl.fmt  # second access must not re-scan
        assert spl._fmt == "MMM"

    def test_fmt_override(self, tmp_path):
        """User-supplied fmt overrides auto-detection."""
        grm = _make_grm(tmp_path, "MMM")
        spl = GrmSplitter(grm, fmt="MMM")
        assert spl.fmt == "MMM"


# ── station name ──────────────────────────────────────────────────────────────

class TestStation:
    def test_default_from_filename(self, tmp_path):
        grm = tmp_path / "AU930_20171470000.GRM"
        grm.write_bytes(_make_grm(tmp_path, "MMM").read_bytes())
        assert GrmSplitter(grm).station == "AU930"

    def test_explicit_override(self, tmp_path):
        grm = _make_grm(tmp_path, "MMM")
        assert GrmSplitter(grm, station="WP937").station == "WP937"


# ── _scan_offsets ─────────────────────────────────────────────────────────────

class TestScanOffsets:
    def test_count(self, tmp_path):
        grm = _make_grm(tmp_path, "MMM", n_ionograms=3, blocks_per=2)
        slices = GrmSplitter(grm)._scan_offsets()
        assert len(slices) == 3

    def test_lengths(self, tmp_path):
        blocks_per = 4
        grm = _make_grm(tmp_path, "MMM", n_ionograms=2, blocks_per=blocks_per)
        slices = GrmSplitter(grm)._scan_offsets()
        for _, start, length in slices:
            assert length == blocks_per * BLOCK_SIZE

    def test_dates_ordered(self, tmp_path):
        grm = _make_grm(tmp_path, "MMM", n_ionograms=3)
        slices = GrmSplitter(grm)._scan_offsets()
        dates = [s[0] for s in slices]
        assert dates == sorted(dates)

    def test_cached(self, tmp_path):
        grm = _make_grm(tmp_path, "MMM", n_ionograms=2)
        spl = GrmSplitter(grm)
        s1 = spl._scan_offsets()
        s2 = spl._scan_offsets()
        assert s1 is s2  # same list object — not re-scanned


# ── split ─────────────────────────────────────────────────────────────────────

class TestSplit:
    def test_file_count(self, tmp_path):
        grm = _make_grm(tmp_path, "MMM", n_ionograms=3)
        out = tmp_path / "out"
        files = GrmSplitter(grm).split(out)
        assert len(files) == 3

    def test_filenames(self, tmp_path):
        grm = _make_grm(tmp_path, "MMM", n_ionograms=1)
        out = tmp_path / "out"
        files = GrmSplitter(grm, station="AU930").split(out)
        assert files[0].suffix == ".MMM"
        assert files[0].name.startswith("AU930_")

    def test_file_sizes(self, tmp_path):
        blocks_per = 5
        grm = _make_grm(tmp_path, "MMM", n_ionograms=2, blocks_per=blocks_per)
        out = tmp_path / "out"
        files = GrmSplitter(grm).split(out)
        for f in files:
            assert f.stat().st_size == blocks_per * BLOCK_SIZE

    def test_creates_out_dir(self, tmp_path):
        grm = _make_grm(tmp_path, "MMM", n_ionograms=1)
        out = tmp_path / "deep" / "nested"
        GrmSplitter(grm).split(out)
        assert out.is_dir()

    def test_returns_paths(self, tmp_path):
        grm = _make_grm(tmp_path, "MMM", n_ionograms=2)
        out = tmp_path / "out"
        files = GrmSplitter(grm).split(out)
        assert all(isinstance(p, Path) for p in files)
        assert all(p.exists() for p in files)

    def test_parallel_same_as_sequential(self, tmp_path):
        grm = _make_grm(tmp_path, "MMM", n_ionograms=3)
        out_seq = tmp_path / "seq"
        out_par = tmp_path / "par"
        spl = GrmSplitter(grm, station="TEST1")
        seq_files = spl.split(out_seq, n_workers=1)
        spl2 = GrmSplitter(grm, station="TEST1")
        par_files = spl2.split(out_par, n_workers=2)
        assert sorted(f.name for f in seq_files) == sorted(f.name for f in par_files)
        for sf, pf in zip(
            sorted(seq_files, key=lambda p: p.name),
            sorted(par_files, key=lambda p: p.name),
        ):
            assert sf.read_bytes() == pf.read_bytes()


# ── load (lazy) ───────────────────────────────────────────────────────────────

class TestLoad:
    def test_count(self, tmp_path):
        grm = _make_grm(tmp_path, "MMM", n_ionograms=2)
        objs = GrmSplitter(grm).load()
        assert len(objs) == 2

    def test_repr(self, tmp_path):
        grm = _make_grm(tmp_path, "MMM", n_ionograms=1, blocks_per=3)
        obj = GrmSplitter(grm).load()[0]
        r = repr(obj)
        assert "ModMaxExtractor" in r
        assert "3 blocks" in r

    def test_date_attribute(self, tmp_path):
        grm = _make_grm(tmp_path, "MMM", n_ionograms=1)
        obj = GrmSplitter(grm).load()[0]
        assert isinstance(obj.date, dt.datetime)
        assert obj.date.year == 2017

    def test_lazy_no_inner_until_extract(self, tmp_path):
        grm = _make_grm(tmp_path, "MMM", n_ionograms=1)
        obj = GrmSplitter(grm).load()[0]
        assert obj._inner is None

    def test_to_pandas_without_explicit_extract(self, tmp_path):
        """to_pandas() must trigger extract() automatically."""
        grm = _make_grm(tmp_path, "MMM", n_ionograms=1)
        obj = GrmSplitter(grm).load()[0]
        # Synthetic MMM block has no valid frequency groups → empty but no crash
        result = obj.to_pandas()
        assert isinstance(result, pd.DataFrame)


# ── _resolve_workers ──────────────────────────────────────────────────────────

class TestResolveWorkers:
    def test_positive(self):
        assert _resolve_workers(3) == 3

    def test_one(self):
        assert _resolve_workers(1) == 1

    def test_minus_one(self):
        import os
        assert _resolve_workers(-1) == (os.cpu_count() or 1)

    def test_clamp_zero(self):
        assert _resolve_workers(0) == 1


# ── _worker_split and _worker_extract (module-level, must be picklable) ───────

class TestWorkerFunctions:
    def test_worker_split_writes_file(self, tmp_path):
        grm = _make_grm(tmp_path, "MMM", n_ionograms=1, blocks_per=2)
        slices = GrmSplitter(grm)._scan_offsets()
        date, start, length = slices[0]
        out = tmp_path / "workers"
        out.mkdir()
        result = _worker_split((str(grm), "TEST1", "MMM", date, start, length, str(out)))
        assert result.exists()
        assert result.stat().st_size == length

    def test_worker_extract_returns_dataframe_or_none(self, tmp_path):
        grm = _make_grm(tmp_path, "MMM", n_ionograms=1)
        slices = GrmSplitter(grm)._scan_offsets()
        date, start, length = slices[0]
        result = _worker_extract((str(grm), "MMM", date, start, length))
        # Synthetic block has no frequency groups → may return empty df or None
        assert result is None or isinstance(result, pd.DataFrame)


# ── Integration test (real AU930 file) ───────────────────────────────────────

@pytest.mark.skipif(
    not Path(_AU930_MMM).exists(),
    reason="AU930 MMM file not available",
)
class TestIntegrationAU930:
    def test_detect(self):
        assert GrmSplitter.detect_format(_AU930_MMM) == "MMM"

    def test_station(self):
        assert GrmSplitter(_AU930_MMM).station == "AU930"

    def test_scan_offsets(self):
        slices = GrmSplitter(_AU930_MMM)._scan_offsets()
        assert len(slices) >= 1
        date, start, length = slices[0]
        assert date == dt.datetime(2017, 5, 27, 0, 0, 5)
        assert start == 0
        assert length == 7 * BLOCK_SIZE

    def test_split(self, tmp_path):
        files = GrmSplitter(_AU930_MMM).split(tmp_path)
        assert len(files) == 1
        assert files[0].name == "AU930_2017147000005.MMM"
        assert files[0].stat().st_size == 7 * BLOCK_SIZE

    def test_load_lazy(self):
        objs = GrmSplitter(_AU930_MMM).load()
        assert len(objs) == 1
        assert "ModMaxExtractor" in repr(objs[0])
        assert objs[0].date == dt.datetime(2017, 5, 27, 0, 0, 5)

    def test_load_dataframes_shape(self):
        df = GrmSplitter(_AU930_MMM).load_dataframes()
        assert not df.empty
        assert set(df.columns) >= {"datetime", "frequency_mhz", "range_km", "amplitude_dB"}

    def test_load_dataframes_height_range(self):
        df = GrmSplitter(_AU930_MMM).load_dataframes()
        assert df["range_km"].min() >= 90.0
        assert df["range_km"].max() <= 635.0

    def test_split_parallel_matches_sequential(self, tmp_path):
        out_seq = tmp_path / "seq"
        out_par = tmp_path / "par"
        seq = GrmSplitter(_AU930_MMM, station="AU930").split(out_seq, n_workers=1)
        par = GrmSplitter(_AU930_MMM, station="AU930").split(out_par, n_workers=2)
        assert len(seq) == len(par)
        for sf, pf in zip(sorted(seq), sorted(par)):
            assert sf.name == pf.name
            assert sf.read_bytes() == pf.read_bytes()

    def test_load_dataframes_parallel_matches_sequential(self):
        spl = GrmSplitter(_AU930_MMM)
        df_seq = spl.load_dataframes(n_workers=1)
        df_par = spl.load_dataframes(n_workers=2)
        assert df_seq.shape == df_par.shape
