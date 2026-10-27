"""Tests for the raw-IQ reader and the DPS4D pipeline on synthetic recordings.

A tiny time-partitioned IQ tree (one-second files at 64 kHz) is written to a temp
directory, so ``IQStream``, ``read_pulse`` and ``process`` run end to end without
any field data. The IQ values are either a known ramp (exact read-back checks) or
seeded noise (pipeline checks).
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("scipy")
pytest.importorskip("netCDF4")

from pynasonde.digisonde.raw import iq_reader, raw_parse
from pynasonde.digisonde.raw.iq_reader import IQStream, get_channels, get_frequencies
from pynasonde.digisonde.raw.raw_parse import IonogramResult, process, read_pulse

UTC = dt.timezone.utc
EPOCH = dt.datetime(2024, 1, 1, 12, 0, 0, tzinfo=UTC)
FC, FS = 5.0e6, 64.0e3
N_PER_FILE = int(FS)


def iq_path(root: Path, epoch: dt.datetime, tag="ch0", name=None) -> Path:
    sub = root / epoch.strftime("%H/%M")
    sub.mkdir(parents=True, exist_ok=True)
    stub = f"{epoch.strftime('%Y-%m-%d_%H%M%S')}_{tag}"
    return sub / (name or f"{stub}_fc05000kHz_bw00064kHz.bin")


def write_iq(path: Path, i: np.ndarray, q: np.ndarray) -> None:
    inter = np.empty(2 * len(i), dtype="<i2")
    inter[0::2], inter[1::2] = i, q
    path.write_bytes(inter.tobytes())


def ramp_file(root, epoch, tag="ch0", name=None, offset=0):
    """File whose sample n (within the second) is (offset + n) + 1j * -(offset + n)."""
    n = np.arange(N_PER_FILE) + offset
    write_iq(iq_path(root, epoch, tag, name), n % 30000, -(n % 30000))


def noise_tree(root, seconds=3, seed=0):
    rng = np.random.default_rng(seed)
    for s in range(seconds):
        i = rng.integers(-800, 800, N_PER_FILE)
        q = rng.integers(-800, 800, N_PER_FILE)
        write_iq(iq_path(root, EPOCH + dt.timedelta(seconds=s)), i, q)


# ── filename helpers ──────────────────────────────────────────────────────────


def test_helpers():
    naive = dt.datetime(2024, 1, 1, 12)
    assert iq_reader._ensure_utc(naive).tzinfo == UTC
    est = dt.timezone(dt.timedelta(hours=-5))
    assert iq_reader._ensure_utc(naive.replace(tzinfo=est)).hour == 17
    assert iq_reader._filename_stub(EPOCH, "ch1") == "2024-01-01_120000_ch1"
    assert iq_reader._subdir_for_epoch(Path("r"), EPOCH) == Path("r/2024-01-01/12/00")
    assert iq_reader._parse_frequency("fc05000kHz", "fc") == 5.0e6
    with pytest.raises(ValueError, match="does not start"):
        iq_reader._parse_frequency("bw00064kHz", "fc")


# ── get_frequencies / get_channels ────────────────────────────────────────────


def test_get_frequencies_reads_the_file_name(tmp_path):
    ramp_file(tmp_path, EPOCH)
    assert get_frequencies(tmp_path, EPOCH) == (FC, FS)


def test_get_frequencies_scans_neighbouring_minutes(tmp_path):
    ramp_file(tmp_path, EPOCH)
    # forward and backward searches both find the minute that holds data
    assert get_frequencies(tmp_path, EPOCH - dt.timedelta(minutes=7)) == (FC, FS)
    assert get_frequencies(tmp_path, EPOCH + dt.timedelta(minutes=7)) == (FC, FS)


def test_get_frequencies_errors(tmp_path):
    with pytest.raises(FileNotFoundError, match="No IQ files"):
        get_frequencies(tmp_path, EPOCH)
    bad = tmp_path / "12" / "00"
    bad.mkdir(parents=True)
    (bad / "mystery.bin").write_bytes(b"")
    with pytest.raises(ValueError, match="does not encode"):
        get_frequencies(tmp_path, EPOCH)


def test_get_channels_lists_unique_tags(tmp_path):
    ramp_file(tmp_path, EPOCH, tag="ch0")
    ramp_file(tmp_path, EPOCH + dt.timedelta(seconds=1), tag="ch0")
    ramp_file(tmp_path, EPOCH + dt.timedelta(seconds=2), tag="ch1")
    iq_path(tmp_path, EPOCH, name="short_name.bin").write_bytes(b"")  # ignored
    assert get_channels(tmp_path, EPOCH) == ["ch0", "ch1"]
    assert get_channels(tmp_path, EPOCH + dt.timedelta(hours=3)) == []


def test_get_channels_defaults_to_now(tmp_path):
    assert get_channels(tmp_path) == []


# ── IQStream ──────────────────────────────────────────────────────────────────


def test_stream_metadata(tmp_path):
    ramp_file(tmp_path, EPOCH)
    s = IQStream(tmp_path, EPOCH.replace(tzinfo=None))
    assert (s.f_center, s.f_sample, s.file_size) == (FC, FS, N_PER_FILE)
    s.close()  # nothing open yet: harmless


def test_read_full_file_returns_the_whole_second(tmp_path):
    ramp_file(tmp_path, EPOCH)
    s = IQStream(tmp_path, EPOCH)
    x = s.read_samples(EPOCH + dt.timedelta(milliseconds=300))  # sub-second ignored
    s.close()
    assert x.dtype == np.complex64 and len(x) == N_PER_FILE
    n = np.arange(N_PER_FILE)
    np.testing.assert_array_equal(x, (n % 30000) - 1j * (n % 30000))


def test_read_partial_block_at_an_offset(tmp_path):
    ramp_file(tmp_path, EPOCH)
    s = IQStream(tmp_path, EPOCH)
    start = EPOCH + dt.timedelta(seconds=0.25)
    x = s.read_samples(start, 100)
    s.close()
    first = int(0.25 * FS)
    np.testing.assert_array_equal(x.real, first + np.arange(100))
    np.testing.assert_array_equal(x.imag, -(first + np.arange(100)))


def test_read_reuses_open_file_and_moves_on_when_out_of_window(tmp_path):
    ramp_file(tmp_path, EPOCH)
    ramp_file(tmp_path, EPOCH + dt.timedelta(seconds=1), offset=5000)
    s = IQStream(tmp_path, EPOCH)
    a = s.read_samples(EPOCH, 10)
    b = s.read_samples(EPOCH + dt.timedelta(seconds=0.5), 10)  # same open file
    c = s.read_samples(EPOCH + dt.timedelta(seconds=1), 10)  # next file
    s.close()
    assert (
        a.real[0] == 0 and b.real[0] == (N_PER_FILE // 2) % 30000 and c.real[0] == 5000
    )


def test_read_spans_two_files(tmp_path):
    ramp_file(tmp_path, EPOCH)
    ramp_file(tmp_path, EPOCH + dt.timedelta(seconds=1), offset=40000)
    s = IQStream(tmp_path, EPOCH)
    start = EPOCH + dt.timedelta(seconds=1) - dt.timedelta(seconds=50 / FS)
    x = s.read_samples(start, 100)
    s.close()
    np.testing.assert_array_equal(
        x.real[:50], (N_PER_FILE - 50 + np.arange(50)) % 30000
    )
    np.testing.assert_array_equal(x.real[50:], (40000 + np.arange(50)) % 30000)


def test_read_zero_samples_and_buffer_reuse(tmp_path):
    ramp_file(tmp_path, EPOCH)
    s = IQStream(tmp_path, EPOCH)
    assert s.read_samples(EPOCH, 0).size == 0
    x = s.read_samples(EPOCH, 8)
    y = s.read_samples(EPOCH, 8)  # buffer of the right size is reused
    s.close()
    np.testing.assert_array_equal(x, y)


def test_truncated_file_raises_eof(tmp_path):
    n = np.arange(100)
    write_iq(iq_path(tmp_path, EPOCH), n, n)  # far fewer samples than a second
    s = IQStream(tmp_path, EPOCH)
    with pytest.raises(EOFError, match="Unexpected end"):
        s.read_samples(EPOCH, 1000)
    s.close()


def test_missing_second_file_raises(tmp_path):
    ramp_file(tmp_path, EPOCH)
    s = IQStream(tmp_path, EPOCH)
    with pytest.raises(FileNotFoundError, match="No IQ file matching"):
        s.read_samples(EPOCH + dt.timedelta(seconds=1), 10)
    s.close()


def test_glob_fallback_when_the_canonical_name_is_missing(tmp_path):
    # get_frequencies parses the name fine, but the canonical 5-digit form is absent
    ramp_file(tmp_path, EPOCH, name="2024-01-01_120000_ch0_fc5000kHz_bw64kHz.bin")
    s = IQStream(tmp_path, EPOCH)
    assert s.read_samples(EPOCH, 4).real.tolist() == [0, 1, 2, 3]
    s.close()


def test_future_epoch_waits_for_the_file(tmp_path, monkeypatch):
    future = dt.datetime.now(UTC).replace(microsecond=0) + dt.timedelta(seconds=30)
    ramp_file(tmp_path, future)
    slept = []
    monkeypatch.setattr(iq_reader.time, "sleep", lambda s: slept.append(s))
    s = IQStream(tmp_path, future)
    s.read_samples(future, 4)
    s.close()
    assert slept and 30 < slept[0] < 33


# ── IonogramResult ────────────────────────────────────────────────────────────


def make_result(phase=True):
    rng = np.random.default_rng(1)
    return IonogramResult(
        frequency_hz=np.array([2e6, 3e6, 4e6]),
        range_km=np.array([100.0, 200.0, 300.0, 400.0]),
        time_unix=np.array([1.0, 2.0, 3.0]),
        power_o=rng.uniform(1, 100, (3, 4)),
        power_x=rng.uniform(1, 100, (3, 4)),
        phase_o=rng.uniform(-3, 3, (3, 4)) if phase else None,
        phase_x=rng.uniform(-3, 3, (3, 4)) if phase else None,
        program_id="TEST",
        epoch=EPOCH,
    )


def test_result_properties_and_power_db():
    r = make_result()
    np.testing.assert_allclose(r.frequency_mhz, [2, 3, 4])
    np.testing.assert_allclose(r.power_total, r.power_o + r.power_x)
    for mode in ("total", "O", "X"):
        db = r.power_db(mode)
        assert db.shape == (3, 4) and db.min() >= 0 and db.max() <= 255
    # median normalisation: about half the cells sit at/above 0 dB
    assert (r.power_db("O") > 0).sum() <= 6
    with pytest.raises(KeyError):
        r.power_db("Z")


@pytest.mark.filterwarnings("ignore:All-NaN slice")
def test_power_db_handles_zero_power():
    r = make_result()
    r.power_o[:] = 0
    r.power_x[:] = 0
    assert np.all(r.power_db() == 0)


@pytest.mark.parametrize("phase", [True, False])
def test_to_netcdf_round_trip(tmp_path, phase):
    import netCDF4

    r = make_result(phase)
    out = tmp_path / "x.nc"
    r.to_netcdf(out)
    with netCDF4.Dataset(out) as ds:
        assert ds["power"].shape == (3, 4)
        assert ("phaseOMode" in ds.variables) is phase
        assert ("phaseXMode" in ds.variables) is phase
        np.testing.assert_allclose(ds["frequency"][:], [2, 3, 4])
        np.testing.assert_allclose(ds["range"][:], r.range_km)
        np.testing.assert_allclose(ds["time"][:], r.time_unix)
        assert ds["ID"][:].tobytes().decode() == "TEST"


@pytest.mark.parametrize("phase", [True, False])
def test_to_xarray(phase):
    pytest.importorskip("xarray")
    ds = make_result(phase).to_xarray()
    assert ds["power_db"].dims == ("frequency", "range")
    assert ("phase_o" in ds) is phase and ("phase_x" in ds) is phase
    assert ds.attrs["program_id"] == "TEST"
    assert ds.attrs["epoch"] == EPOCH.isoformat()


# ── small numeric helpers ─────────────────────────────────────────────────────


def test_datetime_helper_rejects_other_types():
    with pytest.raises(TypeError, match="Unsupported epoch"):
        raw_parse._ensure_datetime("2024-01-01")


# ── read_pulse / process ──────────────────────────────────────────────────────


def program(**over):
    p = {
        "Epoch": EPOCH,
        "ID": "SYN",
        "Freq Stepping Law": "linear",
        "Lower Freq Limit": 4.93e6,  # first two steps lie below the band edge
        "Upper Freq Limit": 5.02e6,
        "Coarse Freq Step": 30e3,
        "Number of Fine Steps": 1,
        "Fine Freq step": 5e3,
        "Fine Multiplexing": False,
        "Inter-Pulse Period": 2e-3,
        "Number of Integrated Repeats": 2,
        "Interpulse Phase Switching": False,
        "Wave Form": "16-chip complementary",
        "Polarization": "O and X",
    }
    p.update(over)
    return p


def test_read_pulse_picks_the_pulse_by_index(tmp_path):
    ramp_file(tmp_path, EPOCH)
    # IPP 2 ms = 128 samples, so pulse k starts at sample 128 k
    p0 = read_pulse(program(), tmp_path, coarse_index=0)
    assert p0.shape == (128,) and p0.real[0] == 0
    # coarse 1, pulse 0 -> pulse_index = pulses_per_coarse = 8 -> sample 1024
    assert read_pulse(program(), tmp_path, coarse_index=1).real[0] == 1024
    # O/X polarisation and complementary index shift by 2 and 1 pulses
    assert read_pulse(program(), tmp_path, 0, pol_index=1).real[0] == 2 * 128
    assert read_pulse(program(), tmp_path, 0, comp_index=1).real[0] == 128
    # non-multiplexed: rep index steps by 2 * n_pol = 4 pulses
    assert read_pulse(program(), tmp_path, 0, rep_index=1).real[0] == 4 * 128


def test_read_pulse_multiplexed_and_single_polarisation(tmp_path):
    ramp_file(tmp_path, EPOCH)
    mux = program(**{"Fine Multiplexing": True, "Number of Fine Steps": 2})
    # multiplexed: rep index steps by 2 * n_pol * fine_steps = 8 pulses
    assert read_pulse(mux, tmp_path, 0, rep_index=1).real[0] == 8 * 128
    single = program(Polarization="O")
    # single polarisation: rep index steps by 2 pulses
    assert read_pulse(single, tmp_path, 0, rep_index=1).real[0] == 2 * 128


@pytest.fixture
def iq_root(tmp_path):
    root = tmp_path / "iq"
    noise_tree(root)
    return root


def test_process_shapes_axes_and_netcdf(iq_root, tmp_path):
    out = tmp_path / "out"
    r = process(program(), iq_root, out, verbose=False)
    # tune frequencies below f_center - fs/2 = 4.968 MHz are skipped
    np.testing.assert_allclose(r.frequency_hz, [4.99e6, 5.02e6])
    assert r.power_o.shape == r.power_x.shape == (2, 60)
    assert r.phase_o is None and r.phase_x is None
    assert (r.power_o > 0).all() and (r.power_x > 0).all()
    c = raw_parse.SPEED_OF_LIGHT
    assert r.range_km[0] == pytest.approx(-220e-6 * c)
    assert np.diff(r.range_km) == pytest.approx(c / 30000)
    assert len(r.time_unix) == 2
    ncs = list(out.rglob("*.nc"))
    assert len(ncs) == 1 and ncs[0].name == "SYN_2024-01-01_120000.nc"
    # an existing product is skipped
    assert process(program(), iq_root, out, verbose=False) is None


def test_process_is_deterministic_and_can_skip_netcdf(iq_root, tmp_path):
    a = process(program(), iq_root, tmp_path, nc_flag=False, verbose=False)
    b = process(program(), iq_root, tmp_path, nc_flag=False, verbose=False)
    np.testing.assert_array_equal(a.power_o, b.power_o)
    assert not list(tmp_path.rglob("*.nc"))


def test_process_saves_phase_for_both_polarisations(iq_root, tmp_path):
    r = process(
        program(**{"Save Phase": True}), iq_root, tmp_path, nc_flag=False, verbose=False
    )
    assert r.phase_o.shape == r.phase_x.shape == (2, 60)
    assert np.abs(r.phase_o).max() <= 2 * np.pi * 1.01


def test_process_single_polarisation_leaves_x_empty(iq_root, tmp_path):
    r = process(
        program(Polarization="O", **{"Save Phase": True}),
        iq_root,
        tmp_path,
        nc_flag=False,
        verbose=False,
    )
    assert (r.power_x == 0).all() and r.phase_x is None
    assert (r.power_o > 0).all() and r.phase_o is not None


@pytest.mark.parametrize(
    "over",
    [
        {"Interpulse Phase Switching": True},
        {"Fine Multiplexing": True, "Number of Fine Steps": 2},
        {"Number of Fine Steps": 2, "Fine Freq step": 7e3},
    ],
)
def test_process_other_program_variants(iq_root, tmp_path, over):
    r = process(program(**over), iq_root, tmp_path, nc_flag=False, verbose=True)
    assert r.power_o.shape[1] == 60 and np.isfinite(r.power_o).all()
    assert (r.power_o > 0).any()


def test_process_with_silent_input_gives_zero_power(tmp_path):
    root = tmp_path / "iq"
    zeros = np.zeros(N_PER_FILE, dtype=int)
    write_iq(iq_path(root, EPOCH), zeros, zeros)
    r = process(program(), root, tmp_path, nc_flag=False, verbose=False)
    assert r.power_o.sum() == 0 and r.power_x.sum() == 0


def test_process_survives_a_missing_file(iq_root, tmp_path):
    # the program needs data beyond the one-second files that were written
    long = program(**{"Number of Integrated Repeats": 64, "Upper Freq Limit": 5.2e6})
    r = process(long, iq_root, tmp_path, nc_flag=False, verbose=True)
    assert r is not None and r.power_o.shape[0] > 2


def test_process_rejects_unsupported_programs(iq_root, tmp_path):
    with pytest.raises(NotImplementedError):
        process(program(FFTMode=True), iq_root, tmp_path)
    with pytest.raises(ValueError, match="linear"):
        process(program(**{"Freq Stepping Law": "log"}), iq_root, tmp_path / "a")
    with pytest.raises(ValueError, match="16-chip"):
        process(program(**{"Wave Form": "barker"}), iq_root, tmp_path / "b")


# ── main() harness ────────────────────────────────────────────────────────────


def test_main_reports_a_result_for_every_epoch(monkeypatch):
    seen = []

    def fake_process(prog, **_kw):
        seen.append(prog["Epoch"])
        return make_result()

    monkeypatch.setattr(raw_parse, "process", fake_process)
    raw_parse.main()
    assert len(seen) == 10
    assert seen[1] - seen[0] == dt.timedelta(minutes=12)


def test_main_stops_at_the_first_failure(monkeypatch):
    calls = []

    def boom(prog, **_kw):
        calls.append(1)
        raise RuntimeError("no data")

    monkeypatch.setattr(raw_parse, "process", boom)
    raw_parse.main()
    assert len(calls) == 1
