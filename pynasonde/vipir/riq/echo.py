"""Dynasonde-style seven-parameter echo extraction for VIPIR RIQ data.

For each pulse set (one frequency step) this module derives the seven
physical parameters of every ionospheric echo following the principles of
the Dynasonde 21 system (Zabotin et al., 2005).  The seven parameters are:

    φ₀  — Gross / mean phase across the pulse set (degrees)
    V*  — Doppler / phase-path velocity (m/s)
    R'  — Stationary-phase group range / virtual height (km)
    XL  — Eastward echo echolocation (km)
    YL  — Northward echo echolocation (km)
    PP  — Polarization chirality / rotation (degrees)
    EP  — Least-squares planar-wavefront residual (degrees)
    A   — Echo peak amplitude (dB)

Each pulse set delivers a complex IQ cube of shape
``(pulse_count, gate_count, rx_count)``.  For every range gate whose
coherent SNR exceeds a configurable threshold the extractor computes all
eight quantities and packages them in an :class:`Echo` dataclass.

A collection of echoes from all pulse sets is managed by
:class:`EchoExtractor`, which exposes :meth:`~EchoExtractor.to_dataframe`
and :meth:`~EchoExtractor.to_xarray` for downstream analysis.

References
----------
Zabotin, N. A., Wright, J. W., Bullett, T. W., & Zabotina, L. Ye. (2005).
Dynasonde 21 principles of data processing, transmission, storage and web
service. *Proc. Ionospheric Effects Symposium 2005*, p. 7B3-1.

Wright, J. W. & Pitteway, M. L. V. (1999). A new data acquisition concept
for digital ionosondes: Phase-based echo recognition and real-time parameter
estimation. *Radio Science*, 34, 871–882.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger

from pynasonde.vipir.riq.datatypes.sct import SctType

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------
_C: float = 299_792_458.0  # speed of light (m/s)
_RAD2DEG: float = 180.0 / np.pi  # radians → degrees conversion factor


# ---------------------------------------------------------------------------
# Echo dataclass
# ---------------------------------------------------------------------------


@dataclass
class Echo:
    """All seven Dynasonde-style physical parameters for one ionospheric echo.

    An echo is a single significant return at a specific
    ``(frequency, virtual height)`` point within one VIPIR pulse set.
    The parameter names follow the Dynasonde 21 convention
    (Zabotin et al., 2005).

    Primary seven parameters
    ------------------------
    gross_phase_deg : float
        **φ₀** — Mean complex phase of the pulse set, averaged over all
        pulses and all receivers at this gate (degrees).  This is the
        "stationary-phase" carrier phase of the echo.
    velocity_mps : float
        **V*** — Phase-path velocity derived from the linear temporal phase
        rate across the pulse set (m/s).  Positive values indicate motion
        away from the sounder (receding layer).
    height_km : float
        **R'** — Stationary-phase group range / virtual height (km),
        computed from the range-gate index and the gate-step timing.
    xl_km : float
        **XL** — Eastward echo echolocation (km).  Derived from the spatial
        least-squares fit of inter-antenna phase differences, projected along
        the East axis: ``XL = R' · l`` where *l* is the East direction cosine.
    yl_km : float
        **YL** — Northward echo echolocation (km): ``YL = R' · m``.
    polarization_deg : float
        **PP** — Chirality / polarization rotation (degrees).  Estimated
        from the differential phase between quasi-orthogonally oriented
        antenna pairs.  Returns NaN when the antenna array does not contain
        antennas with sufficiently different orientations.
    residual_deg : float
        **EP** — RMS residual of the planar-wavefront least-squares fit to
        all inter-antenna phase differences (degrees).  Small values indicate
        a coherent, well-localised echo; large values suggest either multipath
        or a non-planar wavefront.
    amplitude_db : float
        **A** — Coherent peak echo amplitude (dB), averaged over receivers.

    Diagnostic / bookkeeping fields
    --------------------------------
    doppler_hz : float
        Doppler frequency (Hz) from the linear phase-rate regression.
        Related to V* by ``V* = f_d · c / (2 · f₀)``.
    snr_db : float
        Coherent SNR at the echo gate relative to the median noise floor (dB).
    gate_index : int
        Range-gate index within the pulse set (0-based).
    pulse_ut : float
        Universal-time stamp of the first pulse in the set (seconds from the
        file epoch, taken from ``PCT.pri_ut``).
    rx_count : int
        Number of receivers used in the spatial LS fit for XL/YL/EP.
    frequency_khz : float
        Sounding frequency (kHz), taken from ``PCT.frequency``.
    """

    # ── Seven Dynasonde parameters ─────────────────────────────────────────
    frequency_khz: float = np.nan
    height_km: float = np.nan  # R'
    amplitude_db: float = np.nan  # A
    gross_phase_deg: float = np.nan  # φ₀
    doppler_hz: float = np.nan  # temporal phase rate (Hz)
    velocity_mps: float = np.nan  # V*
    xl_km: float = np.nan  # XL
    yl_km: float = np.nan  # YL
    polarization_deg: float = np.nan  # PP
    residual_deg: float = np.nan  # EP

    # ── Diagnostics ────────────────────────────────────────────────────────
    snr_db: float = np.nan
    gate_index: int = -1
    pulse_ut: float = np.nan
    rx_count: int = 0

    def to_dict(self) -> dict:
        """Return a plain ``dict`` representation of this echo."""
        return asdict(self)


# ---------------------------------------------------------------------------
# EchoExtractor
# ---------------------------------------------------------------------------


class EchoExtractor:
    """Extract Dynasonde-style echo parameters from VIPIR RIQ pulsets.

    For each pulse set (one frequency step) the extractor:

    1. Assembles the complex IQ cube ``C[pulse, gate, rx]``.
    2. Computes the coherent mean phasor over pulses: ``C_mean[gate, rx]``.
    3. Estimates the noise floor from the per-gate amplitude distribution.
    4. Selects candidate echo gates whose coherent SNR exceeds
       ``snr_threshold_db``.
    5. Computes all eight parameters at each qualifying gate.

    Parameters
    ----------
    sct : SctType
        Sounder configuration table — must have been fully populated from a
        binary RIQ file.  Key sub-structures used:

        * ``sct.station.rx_count``       — number of active receivers
        * ``sct.station.rx_position``    — ``(rx_count, 3)`` array of
          ``[East_m, North_m, Up_m]`` receiver positions
        * ``sct.station.rx_direction``   — ``(rx_count, 3)`` unit direction
          vectors (used for PP estimation)
        * ``sct.timing.{gate_start, gate_end, gate_step}`` — gate timing (µs)
        * ``sct.timing.pri``             — pulse repetition interval (µs)

    pulsets : list of Pulset
        Grouped pulse data as produced by
        :class:`~pynasonde.vipir.riq.parsers.read_riq.RiqDataset`.
        Each element must expose a ``.pcts`` attribute that is a list of
        :class:`~pynasonde.vipir.riq.datatypes.pct.PctType` objects, each
        carrying ``pulse_i`` and ``pulse_q`` arrays of shape
        ``(gate_count, rx_count)``.
    snr_threshold_db : float, default 3.0
        Minimum coherent SNR (dB) a gate must exceed to qualify as an echo
        candidate.  Gates below this value are silently discarded.
    min_height_km : float, default 50.0
        Minimum virtual height (km) considered for echo detection.  Gates
        below this threshold are skipped unconditionally.  Set to 0 to
        disable the filter.  The default (50 km) excludes direct-wave and
        near-field clutter, which can easily dominate the first few gates
        and crowd out ionospheric returns when ``max_echoes_per_pulset`` is
        small.
    max_height_km : float, default 1000.0
        Maximum virtual height (km) considered for echo detection.  Gates
        above this threshold are skipped.  Ionospheric echoes above ~1000 km
        are extremely rare; setting this avoids aliased end-of-range samples
        that are common in the last few percent of range gates.
    min_rx_for_direction : int, default 3
        Minimum number of receivers required to attempt the spatial
        least-squares fit that yields XL, YL, and EP.  When fewer receivers
        are available those three parameters remain NaN.  Set to 0 or 1 to
        attempt the fit even with only two antennas (only one baseline —
        the system will be exactly determined rather than overdetermined).
    max_echoes_per_pulset : int or None, default 5
        Maximum number of echo candidates retained per frequency step.
        Candidates are ranked by descending amplitude before truncation.
        Pass ``None`` to keep all candidates above the SNR threshold.

    Examples
    --------
    >>> riq = RiqDataset.create_from_file("WI937_2022233235902.RIQ")
    >>> extractor = EchoExtractor(riq.sct, riq.pulsets)
    >>> extractor.extract()
    >>> df = extractor.to_dataframe()
    >>> ds = extractor.to_xarray()
    """

    def __init__(
        self,
        sct: SctType,
        pulsets: List,
        snr_threshold_db: float = 3.0,
        min_height_km: float = 50.0,
        max_height_km: float = 1000.0,
        min_rx_for_direction: int = 3,
        max_echoes_per_pulset: Optional[int] = 5,
    ) -> None:
        self.sct = sct
        self.pulsets = pulsets
        self.snr_threshold_db = snr_threshold_db
        self.min_height_km = min_height_km
        self.max_height_km = max_height_km
        self.min_rx_for_direction = min_rx_for_direction
        self.max_echoes_per_pulset = max_echoes_per_pulset

        n_rx = int(sct.station.rx_count)

        # rx_position: (rx_count, 3) — [East_m, North_m, Up_m] per receiver
        # Populated from Station_default_factory shape (32, 3) then trimmed
        # by StationType.clean() to (rx_count, 3).
        self._rx_pos = np.asarray(
            sct.station.rx_position[:n_rx], dtype=float
        )  # (n_rx, 3)

        # rx_direction: (rx_count, 3) — unit direction vector per receiver.
        # Used for polarization (PP) estimation.
        self._rx_dir = np.asarray(
            sct.station.rx_direction[:n_rx], dtype=float
        )  # (n_rx, 3)

        # Virtual-height axis (km): gate delay (µs) × c/2 = 0.15 km/µs
        # Use integer-index arange so the array has exactly gate_count elements.
        # np.arange(float_start, float_end, float_step) can produce gate_count-1
        # elements due to floating-point precision, making every echo at the top
        # of the range return h_km=NaN → XL=YL=NaN.
        _gate_count = int(sct.timing.gate_count)
        self._heights: np.ndarray = (
            float(sct.timing.gate_start)
            + np.arange(_gate_count, dtype=np.float64) * float(sct.timing.gate_step)
        ) * 0.15

        # Pulse repetition interval in seconds
        self._pri_s: float = float(sct.timing.pri) * 1e-6

        self._echoes: Optional[List[Echo]] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract(self) -> "EchoExtractor":
        """Run the full extraction pipeline over all pulse sets.

        Iterates every pulset, assembles the IQ cube, and accumulates
        :class:`Echo` objects in ``self._echoes``.

        Returns
        -------
        EchoExtractor
            ``self``, enabling method chaining (e.g.,
            ``EchoExtractor(...).extract().to_dataframe()``).
        """
        echoes: List[Echo] = []
        n_total = len(self.pulsets)
        for i, pulset in enumerate(self.pulsets):
            if not pulset.pcts:
                # Can happen for the last unflushed group in tune_type >= 4
                # when pri_count is not an exact multiple of pulse_count * 2.
                continue
            if np.mod(i, 200) == 0:
                logger.info(
                    f"EchoExtractor: pulset {i}/{n_total} " f"({100 * i // n_total}%)"
                )
            freq_khz = float(pulset.pcts[0].frequency)
            pulse_ut = float(pulset.pcts[0].pri_ut)
            C = self._build_iq_cube(pulset)
            echoes.extend(self._extract_from_pulset(C, freq_khz, pulse_ut))

        self._echoes = echoes
        logger.info(
            f"EchoExtractor: extraction complete — {len(echoes)} echoes "
            f"from {n_total} pulsets."
        )
        return self

    def to_dataframe(self) -> pd.DataFrame:
        """Return all extracted echoes as a :class:`pandas.DataFrame`.

        Each row is one :class:`Echo`; columns correspond to its fields.
        The DataFrame index is the sequential echo number.

        Returns
        -------
        pandas.DataFrame
            Empty DataFrame when no echoes were found.

        Raises
        ------
        RuntimeError
            If :meth:`extract` has not been called.
        """
        self._require_extracted()
        if not self._echoes:
            return pd.DataFrame()
        return pd.DataFrame([e.to_dict() for e in self._echoes])

    def to_xarray(self) -> xr.Dataset:
        """Return all extracted echoes as an :class:`xarray.Dataset`.

        Each :class:`Echo` field becomes a 1-D data variable indexed by
        ``echo_index``.  CF-convention ``units`` and ``long_name`` attributes
        are attached to every variable.

        Returns
        -------
        xarray.Dataset
            Empty Dataset when no echoes were found.

        Raises
        ------
        RuntimeError
            If :meth:`extract` has not been called.
        """
        self._require_extracted()
        df = self.to_dataframe()
        if df.empty:
            return xr.Dataset()

        ds = xr.Dataset.from_dataframe(df)
        ds = ds.rename({"index": "echo_index"})

        _meta = {
            "frequency_khz": ("kHz", "Sounding frequency"),
            "height_km": ("km", "Virtual height R-prime"),
            "amplitude_db": ("dB", "Echo amplitude A"),
            "gross_phase_deg": ("deg", "Mean pulse-set phase phi_0"),
            "doppler_hz": ("Hz", "Doppler frequency"),
            "velocity_mps": ("m/s", "Phase-path velocity V-star"),
            "xl_km": ("km", "Eastward echolocation XL"),
            "yl_km": ("km", "Northward echolocation YL"),
            "polarization_deg": ("deg", "Chirality PP"),
            "residual_deg": ("deg", "LS wavefront residual EP"),
            "snr_db": ("dB", "Coherent SNR"),
            "gate_index": ("1", "Range gate index"),
            "pulse_ut": ("s", "Pulse universal time"),
            "rx_count": ("1", "Receivers used in LS fit"),
        }
        for var, (units, long_name) in _meta.items():
            if var in ds:
                ds[var].attrs["units"] = units
                ds[var].attrs["long_name"] = long_name

        ds.attrs["description"] = (
            "Dynasonde-style seven-parameter echo extraction from VIPIR RIQ data. "
            "Parameters follow Zabotin et al. (2005)."
        )
        return ds

    @property
    def echoes(self) -> List[Echo]:
        """Read-only list of extracted :class:`Echo` objects.

        Raises
        ------
        RuntimeError
            If :meth:`extract` has not been called.
        """
        self._require_extracted()
        return self._echoes

    def fit_drift_velocity(
        self,
        height_bin_km: Optional[float] = None,
        min_echoes: int = 6,
        snr_weight: bool = True,
        n_sigma: float = 2.5,
        max_iter: int = 5,
        max_ep_deg: Optional[float] = None,
    ) -> pd.DataFrame:
        """Estimate the 3-D ionospheric drift velocity from extracted echoes.

        Each echo contributes one line-of-sight (LOS) equation:

        .. math::

            V^*_i = l_i V_x + m_i V_y + n_i V_z

        where the direction cosines follow from the echo's own XL, YL, R':

        .. math::

            l = X_L / R', \\quad m = Y_L / R', \\quad n = \\sqrt{1 - l^2 - m^2}

        Solving the overdetermined system via weighted least-squares over many
        echoes with geometrically diverse arrival directions yields the 3-D
        drift vector **[Vx, Vy, Vz]**.

        Parameters
        ----------
        height_bin_km : float or None, default None
            If given, echoes are grouped into non-overlapping height bins of
            this width (km) and a separate **[Vx, Vy, Vz]** is fit per bin.
            Pass ``None`` (default) to use all echoes from the sounding in a
            single whole-sounding fit.
        min_echoes : int, default 6
            Minimum number of valid echoes (finite XL, YL, V*) required to
            attempt a fit.  Bins with fewer echoes return NaN for all velocity
            components.  This count is checked *after* EP filtering and *after*
            sigma-clipping, so the actual echoes used may be fewer than the
            raw count.
        snr_weight : bool, default True
            Weight each echo by its linear SNR (``10 ** (snr_db / 20)``) so
            high-confidence echoes drive the fit more than weak ones.
        n_sigma : float, default 2.5
            Sigma-clipping threshold.  After each LS fit the per-echo LOS
            residual ``r_i = V*_i − (l·Vx + m·Vy + n·Vz)`` is computed.
            Echoes with ``|r_i| > n_sigma * std(r)`` are rejected and the fit
            is repeated.  Iteration continues until no more echoes are rejected
            or *max_iter* is reached.  Set to ``np.inf`` to disable clipping.
        max_iter : int, default 5
            Maximum number of sigma-clipping iterations per bin.
        max_ep_deg : float or None, default None
            If given, echoes whose planar-wavefront residual EP
            (``residual_deg``) exceeds this value are removed *before* the
            velocity fit.  EP is large for multipath / non-planar echoes that
            would corrupt the direction-cosine matrix.  A value of 20–40 °
            is typical.  ``None`` disables this pre-filter.

        Returns
        -------
        pandas.DataFrame
            One row per height bin when *height_bin_km* is given, or a single
            row for the whole-sounding fit.  Columns:

            * ``height_bin_km``  — bin centre (km); absent in whole-sounding mode
            * ``vx_mps``         — eastward drift velocity (m/s)
            * ``vy_mps``         — northward drift velocity (m/s)
            * ``vz_mps``         — vertical drift velocity (m/s)
            * ``residual_mps``   — RMS LOS misfit after sigma-clipping (m/s)
            * ``condition_number`` — condition number of the direction-cosine
              matrix A; values > 100 indicate poor geometric diversity
              (nearly all echoes near-vertical → Vx/Vy unreliable)
            * ``n_echoes``       — echoes used in the final fit
            * ``n_rejected``     — echoes removed by sigma-clipping

        Raises
        ------
        RuntimeError
            If :meth:`extract` has not been called.
        """
        self._require_extracted()
        df = self.to_dataframe()

        _COLS = [
            "height_bin_km",
            "vx_mps",
            "vy_mps",
            "vz_mps",
            "residual_mps",
            "condition_number",
            "n_echoes",
            "n_rejected",
        ]

        if df.empty:
            return pd.DataFrame(columns=_COLS)

        # Keep only echoes with all required fields finite
        required = ["height_km", "xl_km", "yl_km", "velocity_mps", "snr_db"]
        valid = df.dropna(subset=required).copy()
        if valid.empty:
            return pd.DataFrame(columns=_COLS)

        # Optional EP pre-filter: remove non-planar wavefront echoes
        if max_ep_deg is not None and "residual_deg" in valid.columns:
            valid = valid[
                valid["residual_deg"].isna() | (valid["residual_deg"] <= max_ep_deg)
            ].copy()
        if valid.empty:
            return pd.DataFrame(columns=_COLS)

        # Direction cosines from echo geometry
        valid["_l"] = valid["xl_km"] / valid["height_km"]  # East
        valid["_m"] = valid["yl_km"] / valid["height_km"]  # North
        n_sq = (1.0 - valid["_l"] ** 2 - valid["_m"] ** 2).clip(lower=0.0)
        valid["_n"] = np.sqrt(n_sq)  # vertical (Up)

        # SNR weights (linear amplitude)
        valid["_w"] = (
            np.power(10.0, valid["snr_db"] / 20.0)
            if snr_weight
            else np.ones(len(valid))
        )

        def _fit_group(grp: pd.DataFrame) -> dict:
            """Weighted LS fit with iterative sigma-clipping."""
            n_input = len(grp)
            mask = np.ones(n_input, dtype=bool)
            vel = np.zeros(3)

            for _ in range(max_iter):
                sub = grp.iloc[mask]
                if len(sub) < min_echoes:
                    break
                A = sub[["_l", "_m", "_n"]].to_numpy(dtype=float)
                b = sub["velocity_mps"].to_numpy(dtype=float)
                sqrt_w = np.sqrt(sub["_w"].to_numpy(dtype=float))
                vel, *_ = np.linalg.lstsq(A * sqrt_w[:, None], b * sqrt_w, rcond=None)
                # Per-echo LOS residuals (unweighted, for clipping)
                r = b - A @ vel
                std_r = float(np.std(r))
                if std_r == 0.0 or not np.isfinite(n_sigma):
                    break
                new_mask = np.zeros(n_input, dtype=bool)
                new_mask[np.where(mask)[0]] = np.abs(r) <= n_sigma * std_r
                if np.array_equal(new_mask, mask):
                    break
                mask = new_mask

            sub = grp.iloc[mask]
            if len(sub) < min_echoes:
                return dict(
                    vx_mps=np.nan,
                    vy_mps=np.nan,
                    vz_mps=np.nan,
                    residual_mps=np.nan,
                    condition_number=np.nan,
                    n_echoes=len(sub),
                    n_rejected=n_input - len(sub),
                )
            A = sub[["_l", "_m", "_n"]].to_numpy(dtype=float)
            b = sub["velocity_mps"].to_numpy(dtype=float)
            rms = float(np.sqrt(np.mean((b - A @ vel) ** 2)))
            cond = float(np.linalg.cond(A))
            return dict(
                vx_mps=float(vel[0]),
                vy_mps=float(vel[1]),
                vz_mps=float(vel[2]),
                residual_mps=rms,
                condition_number=cond,
                n_echoes=len(sub),
                n_rejected=n_input - len(sub),
            )

        if height_bin_km is None:
            # Whole-sounding single fit
            row = _fit_group(valid)
            return pd.DataFrame([row])

        # Height-binned fit
        h_min = np.floor(valid["height_km"].min() / height_bin_km) * height_bin_km
        h_max = valid["height_km"].max() + height_bin_km
        bin_edges = np.arange(h_min, h_max, height_bin_km)
        bin_centres = bin_edges[:-1] + height_bin_km / 2.0
        valid["_bin"] = pd.cut(
            valid["height_km"],
            bins=bin_edges,
            labels=bin_centres,
            include_lowest=True,
        )

        rows = []
        for centre, grp in valid.groupby("_bin", observed=True):
            row = _fit_group(grp)
            row["height_bin_km"] = float(centre)
            rows.append(row)

        out = pd.DataFrame(rows)
        # Reorder columns
        cols = ["height_bin_km"] + [c for c in _COLS if c != "height_bin_km"]
        return out[[c for c in cols if c in out.columns]]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _require_extracted(self) -> None:
        """Raise RuntimeError if extraction has not been performed yet."""
        if self._echoes is None:
            raise RuntimeError("Call EchoExtractor.extract() before accessing results.")

    def _build_iq_cube(self, pulset) -> np.ndarray:
        """Assemble the complex IQ cube for one pulse set.

        Parameters
        ----------
        pulset : Pulset
            A pulse set whose ``.pcts`` list contains one PCT per pulse.
            Each PCT carries ``pulse_i`` and ``pulse_q`` arrays of shape
            ``(gate_count, rx_count)``.

        Returns
        -------
        np.ndarray, complex128, shape (pulse_count, gate_count, rx_count)
        """
        parts = [
            pct.pulse_i.astype(np.float64) + 1j * pct.pulse_q.astype(np.float64)
            for pct in pulset.pcts
        ]
        return np.stack(parts, axis=0)  # (pulse_count, gate_count, rx_count)

    def _extract_from_pulset(
        self,
        C: np.ndarray,
        freq_khz: float,
        pulse_ut: float,
    ) -> List[Echo]:
        """Compute all parameters at every qualifying gate in one pulset.

        Parameters
        ----------
        C : np.ndarray, complex, shape (pulse_count, gate_count, rx_count)
            Complex IQ cube for one pulse set.
        freq_khz : float
            Sounding frequency (kHz).
        pulse_ut : float
            UT time-stamp of the first pulse (seconds from file epoch).

        Returns
        -------
        list of Echo
            May be empty when no gate exceeds the SNR threshold.
        """
        n_pulse, n_gate, n_rx = C.shape
        freq_hz = freq_khz * 1e3
        wavelength_m = _C / freq_hz

        # ── Step 1: coherent mean phasor over pulses ──────────────────────
        # C_mean[gate, rx] captures the phase and amplitude stable across pulses.
        C_mean = np.mean(C, axis=0)  # (gate, rx)

        # ── Step 2: incoherent amplitude average over receivers ──────────
        # IMPORTANT: use mean of per-receiver magnitudes, NOT magnitude of
        # the complex mean.  For a multi-antenna VIPIR (8–16 Rx) the signals
        # arrive at each receiver with a different spatial phase.  Taking
        # |mean(C_mean)| (vector sum) causes severe phase cancellation that
        # makes real echoes appear below the noise floor.  Averaging the
        # magnitudes first gives a direction-independent amplitude estimate.
        amp_lin = np.mean(np.abs(C_mean), axis=-1)  # (gate,)

        # ── Step 3: noise floor from the per-gate amplitude distribution ──
        # Median is robust to the few high-amplitude echo gates.
        noise_floor = np.nanmedian(amp_lin)
        if noise_floor <= 0.0 or not np.isfinite(noise_floor):
            noise_floor = 1.0

        # ── Step 4: coherent SNR in dB ────────────────────────────────────
        snr_lin = amp_lin / noise_floor
        with np.errstate(divide="ignore", invalid="ignore"):
            snr_db = np.where(snr_lin > 0.0, 20.0 * np.log10(snr_lin), -np.inf)
            amp_db = np.where(amp_lin > 0.0, 20.0 * np.log10(amp_lin), -np.inf)

        # ── Step 5: select gates above threshold and within height window ─
        snr_mask = snr_db >= self.snr_threshold_db
        # Apply height bounds to exclude direct-wave clutter and wrap-around
        height_mask = (self._heights >= self.min_height_km) & (
            self._heights <= self.max_height_km
        )
        candidate_gates = np.where(snr_mask & height_mask)[0]
        if len(candidate_gates) == 0:
            return []

        # Rank by amplitude (strongest first), then truncate
        order = np.argsort(amp_db[candidate_gates])[::-1]
        sorted_gates = candidate_gates[order]
        if self.max_echoes_per_pulset is not None:
            sorted_gates = sorted_gates[: self.max_echoes_per_pulset]

        # ── Step 6: compute all parameters per qualifying gate ─────────────
        echoes: List[Echo] = []
        for g in sorted_gates:
            h_km = float(self._heights[g]) if g < len(self._heights) else np.nan
            echo = Echo(
                frequency_khz=freq_khz,
                height_km=h_km,
                amplitude_db=float(amp_db[g]),
                snr_db=float(snr_db[g]),
                gate_index=int(g),
                pulse_ut=pulse_ut,
                rx_count=n_rx,
            )

            # φ₀ — gross phase: coherent mean phasor at the reference receiver
            # (index 0).  Averaging over all receivers would bias the phase by
            # the array factor, which depends on arrival direction.
            echo.gross_phase_deg = float(np.angle(np.mean(C[:, g, 0])) * _RAD2DEG)

            # V* — Doppler velocity from temporal phase rate
            echo.doppler_hz, echo.velocity_mps = self._compute_doppler(
                C[:, g, :], freq_hz
            )

            # XL, YL, EP — direction from spatial phase LS fit
            if n_rx >= max(self.min_rx_for_direction, 2):
                echo.xl_km, echo.yl_km, echo.residual_deg = self._compute_direction(
                    C_mean[g, :], h_km, wavelength_m
                )

            # PP — polarization chirality from antenna direction vectors
            echo.polarization_deg = self._compute_polarization(C_mean[g, :])

            echoes.append(echo)

        return echoes

    def _compute_doppler(
        self,
        C_gate: np.ndarray,
        freq_hz: float,
    ) -> Tuple[float, float]:
        """Estimate Doppler frequency and phase-path velocity at one gate.

        Averages the complex phasor across receivers to improve SNR, then
        fits a linear trend to the unwrapped phase over the pulse sequence
        using least-squares regression.

        .. math::

            \\dot{\\phi} \\approx \\frac{\\Delta\\phi}{\\Delta t}
            = 2\\pi f_{\\text{d}}

            V^* = \\frac{f_{\\text{d}} \\cdot c}{2 f_0}

        Parameters
        ----------
        C_gate : np.ndarray, complex, shape (pulse_count, rx_count)
            IQ samples at one range gate for every pulse and receiver.
        freq_hz : float
            Sounding frequency (Hz) used to convert Doppler to velocity.

        Returns
        -------
        tuple of (doppler_hz, velocity_mps)
            Both NaN when the pulse count is less than two.
        """
        n_pulse = C_gate.shape[0]
        if n_pulse < 2:
            return np.nan, np.nan

        # Coherent average across receivers → reduce noise
        C_rx_avg = np.mean(C_gate, axis=-1)  # (pulse_count,)
        phase = np.angle(C_rx_avg)
        phase_uw = np.unwrap(phase)  # remove 2π jumps

        t = np.arange(n_pulse, dtype=np.float64) * self._pri_s
        t_c = t - t.mean()
        denom = float(np.dot(t_c, t_c))
        if denom == 0.0:
            return np.nan, np.nan

        slope = float(np.dot(phase_uw - phase_uw.mean(), t_c)) / denom
        doppler_hz = slope / (2.0 * np.pi)
        velocity_mps = doppler_hz * _C / (2.0 * freq_hz)
        return float(doppler_hz), float(velocity_mps)

    def _compute_direction(
        self,
        C_gate_mean: np.ndarray,
        height_km: float,
        wavelength_m: float,
    ) -> Tuple[float, float, float]:
        """Derive eastward/northward echolocation from inter-antenna phases.

        Constructs all ``n_rx*(n_rx-1)/2`` unique antenna-pair phase
        differences and solves the overdetermined planar-wavefront system

        .. math::

            \\Delta\\phi_{mn}
            = \\frac{2\\pi}{\\lambda}
              \\bigl[(x_m - x_n)\\,l + (y_m - y_n)\\,m\\bigr]

        in a least-squares sense for the East/North direction cosines
        ``(l, m)``.  The echolocations follow from

        .. math::

            X_L = R'\\,l, \\quad Y_L = R'\\,m

        and the residual EP is the RMS misfit in degrees.

        Parameters
        ----------
        C_gate_mean : np.ndarray, complex, shape (rx_count,)
            Coherent mean phasors at one gate (one complex number per Rx).
        height_km : float
            Virtual height R' (km) used to project direction cosines to km.
        wavelength_m : float
            Free-space wavelength at the sounding frequency (m).

        Returns
        -------
        tuple of (xl_km, yl_km, residual_deg)
            XL (km), YL (km), and EP (degrees).  All NaN when fewer than
            two antenna pairs can be formed.
        """
        n_rx = len(C_gate_mean)
        rows_A: List[np.ndarray] = []
        rows_b: List[float] = []

        for m in range(n_rx):
            for n in range(m + 1, n_rx):
                # Cross-correlation gives the inter-antenna phase difference
                delta_phi = float(np.angle(C_gate_mean[m] * np.conj(C_gate_mean[n])))
                # East/North baseline (m); Up component neglected (planar approx)
                baseline = self._rx_pos[m, :2] - self._rx_pos[n, :2]
                rows_A.append(baseline * (2.0 * np.pi / wavelength_m))
                rows_b.append(delta_phi)

        if len(rows_b) < 2:
            # Cannot solve with fewer than two unique baselines
            return np.nan, np.nan, np.nan

        A = np.array(rows_A, dtype=np.float64)  # (n_pairs, 2)
        b = np.array(rows_b, dtype=np.float64)  # (n_pairs,)

        # Least-squares solution: A @ [l, m] ≈ b
        result = np.linalg.lstsq(A, b, rcond=None)
        lm = result[0]  # [l_east, m_north]

        # Clip to valid direction-cosine range
        l_cos = float(np.clip(lm[0], -1.0, 1.0))
        m_cos = float(np.clip(lm[1], -1.0, 1.0))

        xl_km = float(height_km * l_cos) if np.isfinite(height_km) else np.nan
        yl_km = float(height_km * m_cos) if np.isfinite(height_km) else np.nan

        # RMS residual in degrees
        b_hat = A @ lm
        ep_deg = float(np.sqrt(np.mean((b - b_hat) ** 2)) * _RAD2DEG)

        return xl_km, yl_km, ep_deg

    def _compute_polarization(
        self,
        C_gate_mean: np.ndarray,
    ) -> float:
        """Estimate echo chirality PP from antenna direction vectors.

        The Dynasonde PP parameter describes the rotation of the polarisation
        ellipse of the echo, which distinguishes O-mode (left-hand) from
        X-mode (right-hand) echoes.  A full derivation requires antennas
        that explicitly sample two orthogonal polarisation components (e.g.,
        crossed dipoles) so that Stokes parameters can be formed.

        When the ``rx_direction`` vectors stored in the SCT reveal quasi-
        orthogonal antenna pairs (dot-product magnitude < 0.5, i.e., angle
        > 60°), the mean differential phase of those pairs is used as a
        proxy for PP.  This is a first-order estimate; substitute a Stokes-
        parameter derivation here once the polarisation basis of each VIPIR
        antenna is documented.

        Parameters
        ----------
        C_gate_mean : np.ndarray, complex, shape (rx_count,)
            Coherent mean phasors at one gate, one per receiver.

        Returns
        -------
        float
            Polarisation rotation PP (degrees), or NaN when indeterminate.
        """
        n_rx = len(C_gate_mean)
        if n_rx < 2 or self._rx_dir.shape[0] < 2:
            return np.nan

        diffs: List[float] = []
        for m in range(n_rx):
            for n in range(m + 1, n_rx):
                dm = self._rx_dir[m]
                dn = self._rx_dir[n]
                norm_m = np.linalg.norm(dm)
                norm_n = np.linalg.norm(dn)
                if norm_m < 1e-9 or norm_n < 1e-9:
                    # Zero-length direction vector — antenna metadata absent
                    continue
                cos_angle = float(np.dot(dm, dn) / (norm_m * norm_n))
                # Quasi-orthogonal: |cos| < 0.5 → opening angle > 60°
                if abs(cos_angle) < 0.5:
                    delta = float(np.angle(C_gate_mean[m] * np.conj(C_gate_mean[n])))
                    diffs.append(delta)

        if not diffs:
            # No quasi-orthogonal pairs found; PP indeterminate
            return np.nan

        # Circular mean of differential phases → robust to ±π wrapping
        pp_rad = float(np.angle(np.mean(np.exp(1j * np.array(diffs)))))
        return pp_rad * _RAD2DEG
