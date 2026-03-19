"""
Python translation of the Julia ``Digisonde`` module.

The ``process`` function reproduces the DPS4D Digisonde ionogram pipeline:
reading complex baseband IQ recordings, performing complementary-code
correlation, assembling range–frequency power grids, and writing results to a
NetCDF product.  The logic mirrors the original Julia implementation as closely
as possible while adopting Pythonic structure and naming.

----------------------------------------------------------------------
Integration depth and the "how many .bin files?" question
----------------------------------------------------------------------
Each sounding epoch requires reading a block of one-second IQ recordings.
The total duration (and therefore the number of files) is determined by
the program parameters::

    n_coarse = (upper_freq - lower_freq) / coarse_step   # e.g. 435
    pulses_per_freq = n_rep × 2 (comp. pair) × n_pol     # e.g. 32
    total_pulses = n_coarse × pulses_per_freq              # e.g. 13 920
    sounding_duration_s = total_pulses × IPP               # e.g. 139 s

For the default Kirtland schedule (IPP = 10 ms, nRep = 8, O+X, 2–15 MHz):
each sounding reads **~139 one-second .bin files**.  The 12-minute cadence
gives enough margin so that the *next* sounding starts only after all data
for the current one has been collected.

Reducing ``Number of Integrated Repeats`` (nRep) trades SNR for cadence:

+------+-------------------+-------------------+------------+
| nRep | Sounding duration | Min. cadence      | SNR loss   |
+======+===================+===================+============+
|  8   |     ~139 s        |  ~3 min           | baseline   |
|  4   |      ~70 s        |  ~2 min           |  −1.5 dB   |
|  2   |      ~35 s        |  ~1 min           |  −3.0 dB   |
|  1   |      ~17 s        |  ~30 s            |  −4.5 dB   |
+------+-------------------+-------------------+------------+

SNR scales as √nRep because the correlation voltage averages coherently
across repeats before the Doppler FFT.  For strong daytime F-layer echoes
nRep = 4 is usually sufficient; for weak or disturbed conditions keep nRep = 8.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import math
import socket
from pathlib import Path
from typing import Dict, List, Optional

import netCDF4
import numpy as np
from loguru import logger
from scipy import fft as sp_fft

from pynasonde.digisonde.raw.iq_reader import IQStream

# ---------------------------------------------------------------------------
# Constants and phase codes

#: 16-chip complementary code A (Barker-like, confirmed against Julia source)
p_code_a = np.array(
    [+1, +1, -1, +1, +1, +1, +1, -1, -1, +1, +1, +1, -1, +1, -1, -1],
    dtype=np.complex128,
)
#: 16-chip complementary code B
p_code_b = np.array(
    [-1, -1, +1, -1, -1, -1, -1, +1, -1, +1, +1, +1, -1, +1, -1, -1],
    dtype=np.complex128,
)

#: Chip bandwidth of the DPS4D waveform (Hz).  Sets range resolution
#: (≈ 5 km) and the number of range gates per IPP.
CHIP_BW: float = 30_000.0

#: Speed of light used for group-path → range conversion (km s⁻¹).
SPEED_OF_LIGHT: float = 2.99792458e5

#: Artificial group-path offset applied by the Digisonde hardware (s).
#: Confirmed by E. Dao 2017-07-26.
DIGISONDE_DELAY: float = 220e-6

HOSTNAME = socket.gethostname()
_UTC = dt.timezone.utc


# ---------------------------------------------------------------------------
# Result container


@dataclasses.dataclass
class IonogramResult:
    """Container for a single processed Digisonde sounding.

    All arrays are in physical units.  Use :meth:`to_netcdf` to write the
    standard compressed NetCDF product, or :meth:`to_xarray` to get an
    in-memory ``xarray.Dataset`` for interactive analysis.

    Attributes
    ----------
    frequency_hz:
        RF frequencies at which the sounding was taken, shape ``(n_freqs,)``,
        in Hz.
    range_km:
        Virtual heights (group range) corresponding to each range gate, shape
        ``(n_ranges,)``, in km.  The 220 µs Digisonde hardware delay has
        already been subtracted.
    time_unix:
        Unix timestamp of the first pulse for each sounding frequency, shape
        ``(n_freqs,)``.
    power_o:
        Ordinary-mode receive power (linear, arbitrary units), shape
        ``(n_freqs, n_ranges)``.
    power_x:
        Extraordinary-mode receive power (linear, arbitrary units), same shape.
        All zeros when ``Polarization != "O and X"``.
    phase_o:
        Ordinary-mode phase at the Doppler peak (radians), or ``None`` when
        ``Save Phase`` was not requested.
    phase_x:
        Extraordinary-mode phase at the Doppler peak (radians), or ``None``.
    program_id:
        Identifier string from the program dictionary (e.g. ``"DPS4D_Kirtland0"``).
    epoch:
        UTC epoch of the sounding.
    """

    frequency_hz: np.ndarray
    range_km: np.ndarray
    time_unix: np.ndarray
    power_o: np.ndarray
    power_x: np.ndarray
    phase_o: Optional[np.ndarray]
    phase_x: Optional[np.ndarray]
    program_id: str
    epoch: dt.datetime

    # ------------------------------------------------------------------
    # Derived helpers

    @property
    def frequency_mhz(self) -> np.ndarray:
        """Frequency axis in MHz."""
        return self.frequency_hz * 1e-6

    @property
    def power_total(self) -> np.ndarray:
        """Sum of O- and X-mode power (linear)."""
        return self.power_o + self.power_x

    def power_db(self, mode: str = "total") -> np.ndarray:
        """Return SNR in dB, normalised so the median equals 0 dB.

        Parameters
        ----------
        mode:
            ``"total"`` (default), ``"O"``, or ``"X"``.

        Returns
        -------
        np.ndarray
            Array clipped to ``[0, 255]`` dB, shape ``(n_freqs, n_ranges)``.
        """
        src = {"total": self.power_total, "O": self.power_o, "X": self.power_x}[mode]
        with np.errstate(divide="ignore", invalid="ignore"):
            db = 10.0 * np.log10(src)
        db[~np.isfinite(db)] = np.nan
        med = np.nanmedian(db)
        db -= med if np.isfinite(med) else 0.0
        return np.clip(np.nan_to_num(db, nan=0.0), 0, 255)

    # ------------------------------------------------------------------
    # Export methods

    def to_netcdf(self, path: Path | str) -> None:
        """Write the ionogram to a compressed NetCDF4 file.

        Parameters
        ----------
        path:
            Destination file path (parent directories must exist).
        """
        pow_uint8 = self.power_db("total").astype(np.uint8)
        nc_path = Path(path)

        dataset = netCDF4.Dataset(nc_path, "w", format="NETCDF4")
        try:
            n_freqs = len(self.frequency_hz)
            n_ranges = len(self.range_km)
            dataset.createDimension("frequency", n_freqs)
            dataset.createDimension("range", n_ranges)

            for name, value in (
                ("system", "SORcer"),
                ("ID", self.program_id),
            ):
                dataset.createDimension(f"{name}_strlen", len(value))
                v = dataset.createVariable(name, "S1", (f"{name}_strlen",))
                v[:] = np.array(list(value), dtype="S1")

            dataset.createDimension("channel_dim", 1)
            ch_var = dataset.createVariable("channel", "u1", ("channel_dim",))
            ch_var[:] = np.array([0], dtype="u1")

            pw = dataset.createVariable(
                "power", "u1", ("frequency", "range"), zlib=True, complevel=9
            )
            pw.setncatts(
                {
                    "units": "dB",
                    "long_name": "receive power",
                    "notes": "SNR, noise estimate via median power",
                }
            )
            pw[:, :] = pow_uint8

            if self.phase_o is not None:
                po = dataset.createVariable(
                    "phaseOMode", "f4", ("frequency", "range"), zlib=True, complevel=9
                )
                po.setncatts({"units": "radians", "long_name": "ordinary mode phase"})
                po[:, :] = self.phase_o.astype(np.float32)

            if self.phase_x is not None:
                px = dataset.createVariable(
                    "phaseXMode", "f4", ("frequency", "range"), zlib=True, complevel=9
                )
                px.setncatts(
                    {"units": "radians", "long_name": "extraordinary mode phase"}
                )
                px[:, :] = self.phase_x.astype(np.float32)

            fv = dataset.createVariable(
                "frequency", "f4", ("frequency",), zlib=True, complevel=9
            )
            fv.setncatts({"units": "MHz", "long_name": "radio frequency"})
            fv[:] = self.frequency_mhz.astype(np.float32)

            rv = dataset.createVariable(
                "range", "f4", ("range",), zlib=True, complevel=9
            )
            rv.setncatts(
                {
                    "units": "km",
                    "long_name": "group path",
                    "notes": "group delay * speed of light",
                }
            )
            rv[:] = self.range_km.astype(np.float32)

            tv = dataset.createVariable(
                "time", "f8", ("frequency",), zlib=True, complevel=9
            )
            tv.setncatts(
                {
                    "units": "seconds",
                    "long_name": "Unix time",
                    "notes": "since 1970-01-01T00:00:00Z, ignoring leap seconds",
                }
            )
            tv[:] = self.time_unix
        finally:
            dataset.close()

    def to_xarray(self):
        """Return the ionogram as an ``xarray.Dataset``.

        The dataset has dimensions ``(frequency, range)`` with frequency in MHz
        and range in km.  Both linear power arrays (O- and X-mode) and the
        normalised dB power are included as data variables.

        Returns
        -------
        xarray.Dataset
        """
        import xarray as xr

        coords = {
            "frequency": ("frequency", self.frequency_mhz, {"units": "MHz"}),
            "range": ("range", self.range_km, {"units": "km"}),
        }
        data_vars = {
            "power_o": (
                ("frequency", "range"),
                self.power_o,
                {"long_name": "O-mode power (linear)"},
            ),
            "power_x": (
                ("frequency", "range"),
                self.power_x,
                {"long_name": "X-mode power (linear)"},
            ),
            "power_db": (
                ("frequency", "range"),
                self.power_db("total"),
                {"units": "dB", "long_name": "Total SNR (median-normalised)"},
            ),
        }
        if self.phase_o is not None:
            data_vars["phase_o"] = (
                ("frequency", "range"),
                self.phase_o,
                {"units": "radians", "long_name": "O-mode Doppler phase"},
            )
        if self.phase_x is not None:
            data_vars["phase_x"] = (
                ("frequency", "range"),
                self.phase_x,
                {"units": "radians", "long_name": "X-mode Doppler phase"},
            )
        return xr.Dataset(
            data_vars=data_vars,
            coords=coords,
            attrs={
                "program_id": self.program_id,
                "epoch": self.epoch.isoformat(),
                "created_by": "pynasonde.digisonde.raw.raw_parse",
            },
        )


# ---------------------------------------------------------------------------
# Utility helpers


def _ensure_datetime(value: dt.datetime | float | int) -> dt.datetime:
    """Normalize program epochs to timezone-aware UTC datetimes."""
    if isinstance(value, dt.datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=_UTC)
        return value.astimezone(_UTC)
    if isinstance(value, (int, float)):
        return dt.datetime.fromtimestamp(float(value), tz=_UTC)
    raise TypeError(f"Unsupported epoch type: {type(value)!r}")


def _next_power_of_two(value: float) -> int:
    """Return the next power of two greater than or equal to ``value``."""
    if value <= 1:
        return 1
    return 1 << (math.ceil(math.log2(value)))


def _prev_power_of_two(value: float) -> int:
    """Return the previous power of two less than or equal to ``value``."""
    if value <= 1:
        return 1
    return 1 << (math.floor(math.log2(value)))


def _next_smooth_235(target: int) -> int:
    """Return the smallest 5-smooth (2^a 3^b 5^c) number >= target."""
    best = None
    limit = target * 4
    for a in range(0, 20):
        for b in range(0, 20):
            value = (2**a) * (3**b)
            if value > limit:
                break
            c = 0
            while True:
                candidate = value * (5**c)
                if candidate >= target:
                    if best is None or candidate < best:
                        best = candidate
                    break
                if candidate > limit:
                    break
                c += 1
    return best if best is not None else target


def _interp_complex(
    x_old: np.ndarray, y_old: np.ndarray, x_new: np.ndarray
) -> np.ndarray:
    """Linear interpolation for complex arrays via real/imag components."""
    real = np.interp(x_new, x_old, y_old.real)
    imag = np.interp(x_new, x_old, y_old.imag)
    return real + 1j * imag


# ---------------------------------------------------------------------------
# Single-pulse IQ accessor


def read_pulse(
    program: Dict[str, object],
    dir_iq: Path | str,
    coarse_index: int,
    fine_index: int = 0,
    pol_index: int = 0,
    rep_index: int = 0,
    comp_index: int = 0,
) -> np.ndarray:
    """Read the raw complex IQ samples for one specific pulse.

    This is the entry point for inspecting individual pulses without running
    the full sounding pipeline.  The returned samples are in the ADC's native
    baseband (full receiver bandwidth); use the FFT-domain mixing logic in
    :func:`process` to tune to a narrower band around a specific frequency.

    Parameters
    ----------
    program:
        Same program dictionary passed to :func:`process`.
    dir_iq:
        Root directory of the IQ recording tree.
    coarse_index:
        Which coarse-frequency step to read (0-based).
    fine_index:
        Fine-frequency sub-step within the coarse step (0 for most programs).
    pol_index:
        Polarisation index: 0 = O-mode, 1 = X-mode.
    rep_index:
        Repeat (integration) index within the CIT, 0 … nRep-1.
    comp_index:
        Complementary-code index: 0 = code A, 1 = code B.

    Returns
    -------
    np.ndarray
        Complex64 array of length ``n_samples`` (next power-of-two of
        ``IPP × f_sample``).

    Examples
    --------
    >>> samples = read_pulse(program, "/media/data", coarse_index=0)
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(samples.real[:500])
    """
    epoch = _ensure_datetime(program["Epoch"])
    ipp_seconds = float(program["Inter-Pulse Period"])
    fine_steps = int(program["Number of Fine Steps"])
    repeats = int(program["Number of Integrated Repeats"])
    multiplexing = bool(program["Fine Multiplexing"])
    waveform = program["Wave Form"]
    polarization = program.get("Polarization", "O and X")
    n_pol = 2 if polarization == "O and X" else 1
    rx_tag = program.get("rxTag", "ch0")

    pulses_per_coarse = fine_steps * repeats
    if waveform == "16-chip complementary":
        pulses_per_coarse *= 2
    if polarization == "O and X":
        pulses_per_coarse *= 2

    if multiplexing:
        pulse_in_coarse = (
            2 * n_pol * fine_steps * rep_index
            + 2 * n_pol * fine_index
            + 2 * pol_index
            + comp_index
        )
    else:
        pulse_in_coarse = (
            2 * n_pol * repeats * fine_index
            + 2 * n_pol * rep_index
            + 2 * pol_index
            + comp_index
        )

    pulse_index = coarse_index * pulses_per_coarse + pulse_in_coarse
    sub_time = epoch + dt.timedelta(seconds=pulse_index * ipp_seconds)

    stream = IQStream(dir_iq, epoch, rx_tag=rx_tag)
    n_samples = _next_power_of_two(ipp_seconds * stream.f_sample)
    try:
        samples = stream.read_samples(sub_time, n_samples)
    finally:
        stream.close()

    return samples


# ---------------------------------------------------------------------------
# Processing pipeline


def process(
    program: Dict[str, object],
    dir_iq: Path | str = "/mnt/Data/",
    out_dir: Path | str = "out/",
    min_range: float = -math.inf,
    max_range: float = math.inf,
    nc_flag: bool = True,
    verbose: bool = True,
) -> Optional[IonogramResult]:
    """Run a single Digisonde sounding program and return an :class:`IonogramResult`.

    The pipeline mirrors the Julia ``Digisonde.process`` function:

    1. Open the IQ stream via :class:`~pynasonde.digisonde.raw.iq_reader.IQStream`.
    2. For every (frequency, polarisation, repeat, complementary-code) tuple:

       a. Read ``n_samples`` raw IQ samples for the pulse's sub-time.
       b. FFT → frequency-shift to baseband + decimate to chip bandwidth.
       c. IFFT → optional fine-frequency residual mix.
       d. Interpolate onto the 30 kHz chip grid.
       e. Correlate with the reversed phase code via FFT convolution.
       f. Accumulate into the CIT voltage array.

    3. Vectorised Doppler FFT across all range gates → peak power and phase.
    4. Optionally write a compressed NetCDF product.

    Parameters
    ----------
    program:
        Dictionary of sounding-program parameters.  Required keys:

        * ``"Epoch"`` – :class:`datetime.datetime` or Unix timestamp
        * ``"ID"`` – string identifier written into the NetCDF
        * ``"Freq Stepping Law"`` – must be ``"linear"``
        * ``"Wave Form"`` – must be ``"16-chip complementary"``
        * ``"Lower Freq Limit"`` / ``"Upper Freq Limit"`` – Hz
        * ``"Coarse Freq Step"`` / ``"Fine Freq step"`` – Hz
        * ``"Number of Fine Steps"`` – int
        * ``"Fine Multiplexing"`` – bool
        * ``"Inter-Pulse Period"`` – seconds (IPP)
        * ``"Number of Integrated Repeats"`` – int (nRep)
        * ``"Interpulse Phase Switching"`` – bool
        * ``"Polarization"`` – ``"O and X"`` or ``"O"``

        Optional keys: ``"rxTag"`` (default ``"ch0"``),
        ``"Save Phase"`` (default ``False``), ``"FFTMode"`` (default ``False``).
    dir_iq:
        Root directory containing the time-partitioned IQ recordings.
    out_dir:
        Root directory for NetCDF output.  A sub-path
        ``<hostname>/<ID>/<YYYY-mm-dd>/`` is created automatically.
    min_range, max_range:
        Reserved for future range-gate masking (not yet applied).
    nc_flag:
        Write the NetCDF product when ``True`` (default).
    verbose:
        Emit progress messages via ``loguru``.

    Returns
    -------
    IonogramResult or None
        ``None`` only if the output file already existed and was skipped.

    Notes
    -----
    **Integration depth and file count** — see module docstring for the full
    table.  For the default Kirtland schedule (IPP = 10 ms, nRep = 8, O+X,
    2–15 MHz at 30 kHz steps) each call reads ≈139 one-second ``.bin`` files
    and takes about 139 s of wall-clock IQ data.  The 12-minute cadence in
    :func:`main` leaves margin before the next sounding.  Reducing ``nRep``
    proportionally shortens the sounding and trades SNR (scales as √nRep).
    """

    # ------------------------------------------------------------------
    # Program metadata and output bookkeeping
    program_id = str(program["ID"])
    epoch = _ensure_datetime(program["Epoch"])
    rx_tag = program.get("rxTag", "ch0")
    fft_mode = bool(program.get("FFTMode", False))

    if fft_mode:
        raise NotImplementedError("FFTMode is not supported in the Python port.")

    save_phase = bool(program.get("Save Phase", False))
    out_dir_path = Path(out_dir) / HOSTNAME / program_id / epoch.strftime("%Y-%m-%d")
    out_dir_path.mkdir(parents=True, exist_ok=True)
    timestamp = epoch.strftime("%Y-%m-%d_%H%M%S")
    nc_path = out_dir_path / f"{program_id}_{timestamp}.nc"
    if nc_path.exists():
        if verbose:
            logger.info(f"Skipping {nc_path} (already exists)")
        return None
    if verbose:
        logger.info(f"Processing {program_id} at {epoch.isoformat()}")

    # ------------------------------------------------------------------
    # Validate program assumptions
    if program.get("Freq Stepping Law") != "linear":
        raise ValueError("Only linear frequency stepping is implemented.")
    if program.get("Wave Form") != "16-chip complementary":
        raise ValueError("Only the 16-chip complementary waveform is supported.")

    lower_freq = float(program["Lower Freq Limit"])
    upper_freq = float(program["Upper Freq Limit"])
    coarse_step = float(program["Coarse Freq Step"])
    fine_steps = int(program["Number of Fine Steps"])
    fine_step = float(program["Fine Freq step"])
    multiplexing = bool(program["Fine Multiplexing"])
    ipp_seconds = float(program["Inter-Pulse Period"])
    repeats = int(program["Number of Integrated Repeats"])
    phase_switching = bool(program["Interpulse Phase Switching"])
    waveform = program["Wave Form"]
    polarization = program.get("Polarization", "O and X")
    n_pol = 2 if polarization == "O and X" else 1
    if multiplexing:
        phase_switching = False  # Matches Julia's "undocumented quirk"

    coarse_frequencies = np.arange(
        lower_freq, upper_freq + coarse_step / 2, coarse_step
    )
    n_coarse = len(coarse_frequencies)
    pulses_per_coarse = fine_steps * repeats
    if waveform == "16-chip complementary":
        pulses_per_coarse *= 2
    if polarization == "O and X":
        pulses_per_coarse *= 2

    n_pulses_total = n_coarse * pulses_per_coarse
    sounding_duration_s = n_pulses_total * ipp_seconds
    if verbose:
        logger.info(
            f"Sounding: {n_coarse} coarse freqs × {pulses_per_coarse} pulses "
            f"= {n_pulses_total} pulses, {sounding_duration_s:.0f} s of IQ data "
            f"(≈{math.ceil(sounding_duration_s)} .bin files)"
        )

    # ------------------------------------------------------------------
    # Open IQ stream and derive tuning parameters
    iq_stream = IQStream(dir_iq, epoch, rx_tag=rx_tag)
    f_low = iq_stream.f_center - iq_stream.f_sample / 2

    # Range bins and axes
    n_ranges = int(math.floor(ipp_seconds * CHIP_BW))
    range_axis = (
        np.arange(n_ranges) / CHIP_BW
    ) * SPEED_OF_LIGHT - DIGISONDE_DELAY * SPEED_OF_LIGHT

    # Frequency axis assembly
    freq_axis: List[float] = []
    for coarse in coarse_frequencies:
        for step_index in range(fine_steps):
            tune_freq = coarse + fine_step * step_index
            if tune_freq < f_low:
                continue
            freq_axis.append(tune_freq)
    n_freqs = len(freq_axis)

    time_axis: List[float] = []
    cit_voltage = np.zeros((n_ranges, repeats), dtype=np.complex64)
    iono_power_o = np.zeros((n_freqs, n_ranges), dtype=np.float64)
    iono_power_x = np.zeros_like(iono_power_o)
    iono_phase_o = np.zeros_like(iono_power_o) if save_phase else None
    iono_phase_x = np.zeros_like(iono_power_o) if save_phase and n_pol == 2 else None

    # Sample buffers and FFT helpers
    n_samples = _next_power_of_two(ipp_seconds * iq_stream.f_sample)
    logger.debug(f"n_samples={n_samples}, f_sample={iq_stream.f_sample:.0f} Hz")
    samples = np.zeros(n_samples, dtype=np.complex64)
    hz_per_bin = iq_stream.f_sample / n_samples

    decimation_factor = max(1, _prev_power_of_two(iq_stream.f_sample / CHIP_BW))
    samples_ds = np.zeros(n_samples // decimation_factor, dtype=np.complex64)
    n_ds = samples_ds.size
    idx_front = np.arange(0, n_ds // 2 + (n_ds % 2))
    idx_back = np.arange(0, n_ds // 2)
    f_sample_ds = iq_stream.f_sample * n_ds / n_samples
    window_ds = np.hamming(n_ds).astype(np.float32)
    mix_signal = np.zeros(n_ds, dtype=np.complex64)

    t_chips = np.arange(0, ipp_seconds, 1.0 / CHIP_BW)
    n_chips = t_chips.size
    n_pcode = p_code_a.size
    n_conv = n_chips + n_pcode - 1
    fft_length = (
        _next_smooth_235(n_conv) if n_conv > 1024 else _next_power_of_two(n_conv)
    )

    p_code_a_pad = np.concatenate(
        (p_code_a[::-1], np.zeros(fft_length - n_pcode, dtype=np.complex128))
    )
    p_code_b_pad = np.concatenate(
        (p_code_b[::-1], np.zeros(fft_length - n_pcode, dtype=np.complex128))
    )
    p_code_a_fft = sp_fft.fft(p_code_a_pad)
    p_code_b_fft = sp_fft.fft(p_code_b_pad)
    s_cg_pad = np.zeros(fft_length, dtype=np.complex128)

    # t_ds is fixed for the whole sounding — compute once.
    t_ds = np.arange(n_ds, dtype=np.float64) / f_sample_ds

    # ------------------------------------------------------------------
    # Main processing loops
    freq_counter = 0
    for coarse_index, coarse_freq in enumerate(coarse_frequencies):
        for fine_index in range(fine_steps):
            tune_freq = coarse_freq + fine_step * fine_index
            if tune_freq < f_low:
                continue
            freq_counter += 1
            new_mix = True
            mix_offset = 0.0
            new_freq = True

            for pol_index in range(n_pol):
                cit_voltage.fill(0.0)

                for rep_index in range(repeats):
                    for comp_index in range(2):  # complementary pair A/B
                        if multiplexing:
                            pulse_in_coarse = (
                                2 * n_pol * fine_steps * rep_index
                                + 2 * n_pol * fine_index
                                + 2 * pol_index
                                + comp_index
                            )
                        else:
                            pulse_in_coarse = (
                                2 * n_pol * repeats * fine_index
                                + 2 * n_pol * rep_index
                                + 2 * pol_index
                                + comp_index
                            )
                        pulse_index = coarse_index * pulses_per_coarse + pulse_in_coarse
                        sub_time = epoch + dt.timedelta(
                            seconds=pulse_index * ipp_seconds
                        )

                        if new_freq:
                            time_axis.append(sub_time.timestamp())
                            new_freq = False

                        try:
                            samples[:] = iq_stream.read_samples(sub_time, n_samples)
                        except Exception as exc:
                            if verbose:
                                logger.warning(f"IQ sample read failed: {exc}")
                            continue

                        # Coarse mixing + decimation (FFT domain)
                        samples_fft = sp_fft.fft(samples, workers=-1)
                        i_tune_precise = (iq_stream.f_center - tune_freq) / hz_per_bin
                        i_tune = int(round(i_tune_precise))
                        samples_ds[idx_front] = samples_fft[
                            (-i_tune + idx_front) % n_samples
                        ]
                        samples_ds[-idx_back.size :] = samples_fft[
                            (-1 - i_tune - idx_back) % n_samples
                        ]
                        f_intermediate = iq_stream.f_center - i_tune * hz_per_bin
                        samples_ds *= window_ds
                        samples_time = sp_fft.ifft(samples_ds, workers=-1)

                        # Fine-frequency residual mix (cached per frequency step)
                        if mix_offset != 0.0 or new_mix:
                            mix_offset = tune_freq - f_intermediate
                            if mix_offset != 0.0:
                                mix_signal[:] = np.exp(-2j * np.pi * mix_offset * t_ds)
                                samples_time *= mix_signal
                            new_mix = False

                        # Interpolate onto 30 kHz chip grid
                        samples_chip = _interp_complex(t_ds, samples_time, t_chips)

                        # Phase-code selection (code A or B; optional phase switching)
                        code_fft = (
                            p_code_a_fft.copy()
                            if comp_index == 0
                            else p_code_b_fft.copy()
                        )
                        if phase_switching and (rep_index % 2 == 1):
                            code_fft *= -1

                        # Matched-filter correlation via FFT convolution
                        s_cg_pad[:n_chips] = samples_chip
                        s_cg_pad[n_chips:] = 0.0
                        s_fft = sp_fft.fft(s_cg_pad, workers=-1)
                        s_fft *= code_fft
                        s_ifft = sp_fft.ifft(s_fft, workers=-1)
                        correlation = s_ifft[n_pcode - 1 : n_conv]

                        cit_voltage[:, rep_index] += correlation[:n_ranges].astype(
                            np.complex64
                        )

                # Doppler FFT — vectorised across all range gates at once
                doppler_spectra = sp_fft.fft(cit_voltage, axis=1, workers=-1)
                doppler_abs = np.abs(doppler_spectra)
                range_powers = np.max(doppler_abs, axis=1) ** 2  # (n_ranges,)
                if pol_index == 0:
                    iono_power_o[freq_counter - 1, :] += range_powers
                    if save_phase and iono_phase_o is not None:
                        max_bins = np.argmax(doppler_abs, axis=1)
                        iono_phase_o[freq_counter - 1, :] += np.angle(
                            doppler_spectra[np.arange(n_ranges), max_bins]
                        )
                else:
                    iono_power_x[freq_counter - 1, :] += range_powers
                    if save_phase and iono_phase_x is not None:
                        max_bins = np.argmax(doppler_abs, axis=1)
                        iono_phase_x[freq_counter - 1, :] += np.angle(
                            doppler_spectra[np.arange(n_ranges), max_bins]
                        )

    iq_stream.close()
    if verbose:
        logger.info(f"Completed {program_id}")

    # ------------------------------------------------------------------
    # Build result object
    result = IonogramResult(
        frequency_hz=np.asarray(freq_axis, dtype=np.float64),
        range_km=range_axis.astype(np.float64),
        time_unix=np.asarray(time_axis, dtype=np.float64),
        power_o=iono_power_o,
        power_x=iono_power_x,
        phase_o=iono_phase_o,
        phase_x=iono_phase_x,
        program_id=program_id,
        epoch=epoch,
    )

    if nc_flag:
        result.to_netcdf(nc_path)
        if verbose:
            logger.info(f"Wrote {nc_path}")

    return result


# ---------------------------------------------------------------------------
# Convenience harness for manual testing (mirrors Julia main())


def main() -> None:
    """Process 10 consecutive soundings from the 2023-10-14 Kirtland dataset.

    Integration tradeoff note
    -------------------------
    Each call to :func:`process` with the parameters below reads approximately
    139 one-second ``.bin`` files (≈ 139 s of IQ data).  The 12-minute spacing
    between epochs ensures the sounding finishes before the next one begins.

    To shorten sounding time at the cost of SNR, reduce
    ``"Number of Integrated Repeats"``.  Halving nRep from 8 → 4 cuts
    acquisition time to ≈70 s (enabling a 2-minute cadence) but reduces the
    coherent-integration gain by ≈1.5 dB.  See the module docstring for the
    full tradeoff table.
    """
    epochs = [
        dt.datetime(2023, 10, 14, 16, 0, tzinfo=_UTC) + dt.timedelta(minutes=12 * idx)
        for idx in range(0, 120 // 12)
    ]
    for epoch in epochs:
        program = {
            "Epoch": epoch,
            "FFTMode": False,
            "rxTag": "ch0",
            "Save Phase": True,
            "Signal Type": "DPS4D",
            "ID": "DPS4D_Kirtland0",
            "Freq Stepping Law": "linear",
            "Lower Freq Limit": 2e6,
            "Upper Freq Limit": 15e6,
            "Coarse Freq Step": 30e3,
            "Number of Fine Steps": 1,
            "Fine Freq step": 5e3,
            "Fine Multiplexing": False,
            "Inter-Pulse Period": 2 * 5e-3,
            "Number of Integrated Repeats": 8,  # reduce to 4 for 2-min cadence (−1.5 dB)
            "Interpulse Phase Switching": False,
            "Wave Form": "16-chip complementary",
            "Polarization": "O and X",
        }
        try:
            result = process(
                program, dir_iq="/media/chakras4/69F9D939661D263B", verbose=True
            )
            if result is not None:
                # Access arrays directly, e.g. for a quick plot:
                #   import matplotlib.pyplot as plt
                #   plt.pcolormesh(result.range_km, result.frequency_mhz,
                #                  result.power_db())
                #   plt.show()
                #
                # Or convert to xarray for label-aware analysis:
                #   ds = result.to_xarray()
                #   ds["power_db"].plot()
                logger.info(
                    f"Result: freq {result.frequency_mhz[0]:.2f}–"
                    f"{result.frequency_mhz[-1]:.2f} MHz, "
                    f"{len(result.range_km)} range gates"
                )
        except Exception as exc:
            import traceback

            traceback.print_exc()
            logger.error(f"Processing failed for {epoch.isoformat()}: {exc}")
            break


if __name__ == "__main__":
    main()
