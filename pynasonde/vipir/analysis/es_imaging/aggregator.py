"""aggregator.py — Multi-file Es layer imager with per-file and moving-average modes.

Each RIQ file contains 4 pulses × 8 Rx channels at each sounding frequency.
This module reduces that 3-D IQ cube to a single high-resolution Capon
pseudospectrum per file, then optionally stacks or window-averages across files.

Processing strategy — multi-snapshot Capon
------------------------------------------
Instead of beamforming and then running Capon independently on each pulse,
every ``(pulse, Rx)`` pair is treated as an independent range-profile snapshot
and **all L snapshots contribute to a single averaged covariance matrix** R_f
before inversion:

    R_f = (1 / L·cols) Σ_{l=1}^{L} G_l · G_l^H

where G_l is the Hankel subband matrix of profile l and cols = V − Z + 1.

For ``"per_file"`` with n_pulse=4 and n_rx=8: L = 32 snapshots per column.
For ``"moving_avg"`` with window=8: L = 8 × 4 × 8 = 256 snapshots per column.

A better-conditioned R_f produces a dramatically cleaner Capon pseudospectrum,
making weak Es echoes (typically 40–60 dB below the direct-wave clutter)
visible in the normalised spectrum.

Output modes
------------
``"per_file"``
    One spectrum column per file.  For 60 files → 60 columns in the RTI.
    No cross-file averaging.  Each column reflects only the 4 pulses and
    8 Rx channels from that single 60-second sounding.

``"moving_avg"``
    A sliding window of ``window`` consecutive file spectra is averaged
    incoherently at each step of ``step`` files.  For 60 files, window=8,
    step=1 → 53 output columns, each averaging 8 × 4 = 32 effective
    pulse-equivalents.  This is the preferred mode for cleaner Es imaging
    when a time resolution of ``step × 60 s`` is acceptable.

References
----------
Liu, T., Yang, G., & Jiang, C. (2023). High-resolution sporadic E layer
observation based on ionosonde using a cross-spectrum analysis imaging
technique. *Space Weather*, 21, e2022SW003195.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
from loguru import logger

from pynasonde.vipir.analysis.es_imaging.capon import EsCaponImager, EsImagingResult

_C_KM_US = 299_792.458 / 1e6  # km per μs


class RiqAggregator:
    """Multi-file Es layer imager.

    Parameters
    ----------
    n_subbands:
        Capon Z parameter.  Default ``100``.
    resolution_factor:
        Capon K parameter (output grid = K × V bins).  Default ``10``.
    rx_weights:
        Reserved for future use.  Currently all Rx channels are stacked as
        independent snapshots for the multi-snapshot covariance estimator
        rather than combined by beamforming.
    gate_start_km:
        Height of the first range gate (km).  Overridden from the RIQ
        header when :meth:`load` is called.  Default ``0.0``.
    gate_spacing_km:
        Gate spacing r₀ (km).  Overridden from the RIQ header when
        :meth:`load` is called.  Default ``1.499`` (VIPIR standard).
    diagonal_loading:
        Capon covariance diagonal loading fraction ε.  Default ``1e-3``.
    output_mode:
        ``"per_file"``   — one spectrum column per file (slow RTI).
        ``"moving_avg"`` — sliding-window average of ``window`` files.
    window:
        Number of consecutive files to average per output column.
        Only used when ``output_mode="moving_avg"``.  Default ``8``.
    step:
        Sliding-window step in files.  ``step=1`` → maximum overlap
        (one new column per new file).  Only used when
        ``output_mode="moving_avg"``.  Default ``1``.
    blank_min_km:
        Heights below this value (km) are zeroed in each range profile
        before the Capon covariance is computed.  This suppresses the
        direct-wave / ground-clutter spike (typically at the first 1–3
        gates) so the ionospheric signal dominates the covariance matrix.
        Set to ``0.0`` to disable.  Default ``60.0`` km.

    Examples
    --------
    Per-file RTI from synthetic cubes (L = 4×8 = 32 profiles/column):

    >>> agg = RiqAggregator(n_subbands=50, resolution_factor=4,
    ...                     output_mode="per_file")
    >>> cubes = [np.random.randn(4, 200, 8) + 1j*np.random.randn(4, 200, 8)
    ...          for _ in range(10)]
    >>> result = agg.combine(cubes)
    >>> print(result.summary())   # n_snapshots=10

    Moving-average RTI (window=8, step=1):

    >>> agg = RiqAggregator(n_subbands=50, resolution_factor=4,
    ...                     output_mode="moving_avg", window=8, step=1)
    >>> result = agg.combine(cubes)
    >>> print(result.summary())   # n_snapshots=3  (10-8+1=3)
    """

    def __init__(
        self,
        n_subbands: int = 100,
        resolution_factor: int = 10,
        rx_weights: Optional[np.ndarray] = None,
        gate_start_km: float = 0.0,
        gate_spacing_km: float = 1.499,
        diagonal_loading: float = 1e-3,
        output_mode: str = "per_file",
        window: int = 8,
        step: int = 1,
        blank_min_km: float = 60.0,
    ) -> None:
        if output_mode not in ("per_file", "moving_avg"):
            raise ValueError(
                f"output_mode must be 'per_file' or 'moving_avg', got '{output_mode}'."
            )
        if window < 1:
            raise ValueError(f"window must be >= 1, got {window}.")
        if step < 1:
            raise ValueError(f"step must be >= 1, got {step}.")
        self.n_subbands = n_subbands
        self.resolution_factor = resolution_factor
        self.rx_weights = rx_weights
        self.gate_start_km = gate_start_km
        self.gate_spacing_km = gate_spacing_km
        self.diagonal_loading = diagonal_loading
        self.output_mode = output_mode
        self.window = window
        self.step = step
        self.blank_min_km = blank_min_km

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _to_profiles(self, cube: np.ndarray) -> np.ndarray:
        """Flatten (n_pulse, n_gate[, n_rx]) → (n_pulse × n_rx, n_gate).

        Every (pulse, Rx) pair is treated as an independent range-profile
        snapshot for the multi-snapshot Capon covariance estimator.
        2-D input (already single-channel) is returned unchanged.
        """
        if cube.ndim == 2:
            return cube  # already (n_pulse, n_gate)
        n_pulse, n_gate, n_rx = cube.shape
        # transpose → (n_pulse, n_rx, n_gate) then flatten first two dims
        return cube.transpose(0, 2, 1).reshape(n_pulse * n_rx, n_gate)

    def _make_imager(self) -> EsCaponImager:
        return EsCaponImager(
            n_subbands=self.n_subbands,
            resolution_factor=self.resolution_factor,
            coherent_integrations=1,
            diagonal_loading=self.diagonal_loading,
            gate_start_km=self.gate_start_km,
            gate_spacing_km=self.gate_spacing_km,
        )

    def _gate_blank(self) -> int:
        """Number of leading gates to zero for direct-wave suppression."""
        if self.blank_min_km <= self.gate_start_km:
            return 0
        return max(
            0, int((self.blank_min_km - self.gate_start_km) / self.gate_spacing_km)
        )

    def _image_profiles(
        self,
        profiles: np.ndarray,
        imager: EsCaponImager,
        A: np.ndarray,
    ) -> np.ndarray:
        """Run multi-snapshot Capon on L range profiles → one spectrum (n_hr,).

        Parameters
        ----------
        profiles : complex ndarray, shape (L, n_gate)
            L independent range profiles (all pulses × Rx for one file or window).
        """
        gb = self._gate_blank()
        R_inv = imager._covariance_multi(profiles, gate_blank=gb)
        return imager._capon(R_inv, A)  # (n_hr,)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def combine(self, cubes: List[np.ndarray]) -> EsImagingResult:
        """Produce an Es imaging result from a list of pre-loaded IQ cubes.

        Parameters
        ----------
        cubes : list of complex ndarray
            Each element has shape ``(n_pulse, n_gate)`` or
            ``(n_pulse, n_gate, n_rx)``.  All cubes must share the same
            ``n_gate``.

        Returns
        -------
        EsImagingResult
            ``output_mode="per_file"``   → ``n_snapshots = len(cubes)``
                Each column uses L = n_pulse × n_rx profiles for covariance.
            ``output_mode="moving_avg"`` → ``n_snapshots = (N - window)//step + 1``
                Each column uses L = window × n_pulse × n_rx profiles (e.g. 256)
                for a single averaged covariance before Capon inversion.
        """
        if not cubes:
            raise ValueError("cubes list is empty.")

        first = np.asarray(cubes[0], dtype=complex)
        V = first.shape[1]
        n_files = len(cubes)

        # Validate rx_weights length against the cube's n_rx (future-use check)
        if first.ndim == 3 and self.rx_weights is not None:
            n_rx = first.shape[2]
            w = np.asarray(self.rx_weights, dtype=complex)
            if len(w) != n_rx:
                raise ValueError(f"rx_weights length {len(w)} != cube n_rx={n_rx}.")

        imager = self._make_imager()
        imager._validate(V)
        A = imager._steering_matrix(V)

        gate_heights = self.gate_start_km + np.arange(V) * self.gate_spacing_km
        hr_heights = self.gate_start_km + np.arange(self.resolution_factor * V) * (
            self.gate_spacing_km / self.resolution_factor
        )

        # --- per_file: one multi-snapshot Capon run per file ---
        # L = n_pulse × n_rx profiles per file (e.g. 4×8 = 32)
        if self.output_mode == "per_file":
            file_spectra: List[np.ndarray] = []
            for i, raw in enumerate(cubes):
                cube = np.asarray(raw, dtype=complex)
                if cube.shape[1] != V:
                    raise ValueError(
                        f"Cube {i} has n_gate={cube.shape[1]}, expected {V}."
                    )
                profiles = self._to_profiles(cube)  # (n_pulse*n_rx, n_gate)
                print(
                    f"Processing file {i+1}/{n_files} with shape {profiles.shape} ..."
                )
                file_spectra.append(self._image_profiles(profiles, imager, A))
                logger.debug(f"RiqAggregator: processed file {i+1}/{n_files}")
            spectra_lin = np.stack(file_spectra, axis=0)  # (n_files, n_hr)

        else:  # moving_avg
            # Each window stacks all files' (pulse, Rx) pairs into one big batch.
            # L = window × n_pulse × n_rx (e.g. 8×4×8 = 256) profiles per column.
            # A single _covariance_multi call averages all 256 R_f matrices before
            # inversion — far better conditioned than per-file estimates.
            if self.window > n_files:
                raise ValueError(f"window={self.window} > number of files={n_files}.")
            starts = range(0, n_files - self.window + 1, self.step)
            window_spectra: List[np.ndarray] = []
            for i in starts:
                # Stack window files along pulse axis: (window*n_pulse, n_gate, n_rx)
                window_cube = np.concatenate(
                    [
                        np.asarray(cubes[j], dtype=complex)
                        for j in range(i, i + self.window)
                    ],
                    axis=0,
                )
                profiles = self._to_profiles(
                    window_cube
                )  # (window*n_pulse*n_rx, n_gate)
                window_spectra.append(self._image_profiles(profiles, imager, A))
                logger.debug(
                    f"RiqAggregator moving_avg: window {i//self.step + 1}/{len(starts)}  "
                    f"L={profiles.shape[0]} profiles"
                )
            spectra_lin = np.stack(window_spectra, axis=0)  # (n_windows, n_hr)
            logger.info(
                f"RiqAggregator moving_avg: {n_files} files  "
                f"window={self.window}  step={self.step}  "
                f"→ {len(starts)} output columns  "
                f"L={self.window * first.shape[0] * (first.shape[2] if first.ndim==3 else 1)} profiles/column"
            )

        # normalise to dB
        P_max = spectra_lin.max()
        if P_max <= 0:
            P_max = 1.0
        spectra_db = 10.0 * np.log10(spectra_lin / P_max + 1e-15)

        n_pulse = first.shape[0]
        n_rx = first.shape[2] if first.ndim == 3 else 1
        logger.info(
            f"RiqAggregator: {n_files} files × {n_pulse} pulses × {n_rx} Rx  "
            f"Z={self.n_subbands}  K={self.resolution_factor}  "
            f"mode={self.output_mode}  Δr={self.gate_spacing_km/self.resolution_factor:.3f} km"
        )

        return EsImagingResult(
            pseudospectrum_db=spectra_db,
            heights_km=hr_heights,
            gate_heights_km=gate_heights,
            n_subbands=self.n_subbands,
            resolution_factor=self.resolution_factor,
            coherent_integrations=n_pulse,
            gate_spacing_km=self.gate_spacing_km,
        )

    def load(
        self,
        file_list: List[str],
        freq_target_khz: float,
        freq_tol_khz: float = 50.0,
        vipir_version_idx: int = 1,
    ) -> List[np.ndarray]:
        """Load IQ cubes from RIQ files at a target sounding frequency.

        Iterates over ``file_list``, finds the pulset closest to
        ``freq_target_khz`` in each file, and assembles the full
        ``(pulse_count, gate_count, rx_count)`` complex IQ cube.

        Also updates ``self.gate_start_km`` and ``self.gate_spacing_km`` from
        the first file's RIQ header.

        Parameters
        ----------
        file_list:
            Paths to ``.RIQ`` files.
        freq_target_khz:
            Target sounding frequency in kHz.
        freq_tol_khz:
            Maximum allowed frequency offset.  Files whose closest pulset
            differs by more than this are skipped with a warning.
        vipir_version_idx:
            Index into ``VIPIR_VERSION_MAP.configs``.  Default ``1``.

        Returns
        -------
        list of complex ndarray, shape ``(n_pulse, n_gate, n_rx)`` each.
        """
        from pynasonde.vipir.riq.parsers.read_riq import VIPIR_VERSION_MAP, RiqDataset

        cubes: List[np.ndarray] = []
        header_set = False

        for path in file_list:
            try:
                riq = RiqDataset.create_from_file(
                    path,
                    unicode="latin-1",
                    vipir_config=VIPIR_VERSION_MAP.configs[vipir_version_idx],
                )
            except Exception as exc:
                logger.warning(f"RiqAggregator.load: cannot read '{path}': {exc}")
                continue

            if not header_set:
                self.gate_spacing_km = riq.sct.timing.gate_step * _C_KM_US / 2
                self.gate_start_km = riq.sct.timing.gate_start * _C_KM_US / 2
                header_set = True
                logger.info(
                    f"Gate geometry from '{path}': "
                    f"r₀={self.gate_spacing_km:.3f} km  "
                    f"start={self.gate_start_km:.2f} km"
                )

            best_idx, best_diff = 0, np.inf
            for idx, ps in enumerate(riq.pulsets):
                diff = abs(float(ps.pcts[0].frequency) - freq_target_khz)
                if diff < best_diff:
                    best_idx, best_diff = idx, diff

            if best_diff > freq_tol_khz:
                logger.warning(
                    f"RiqAggregator.load: '{path}' closest pulset "
                    f"Δf={best_diff:.0f} kHz > tol={freq_tol_khz:.0f} kHz — skipped."
                )
                continue

            pulset = riq.pulsets[best_idx]
            parts = [
                pct.pulse_i.astype(np.float64) + 1j * pct.pulse_q.astype(np.float64)
                for pct in pulset.pcts
            ]
            cube = np.stack(parts, axis=0)  # (n_pulse, n_gate, n_rx)
            cubes.append(cube)
            logger.debug(
                f"Loaded '{path}'  pulset #{best_idx}  "
                f"f={float(pulset.pcts[0].frequency)/1e3:.3f} MHz  "
                f"shape={cube.shape}"
            )

        if not cubes:
            raise RuntimeError(
                f"No valid cubes loaded from {len(file_list)} files "
                f"at {freq_target_khz/1e3:.3f} MHz ± {freq_tol_khz:.0f} kHz."
            )
        return cubes

    def fit(
        self,
        file_list: List[str],
        freq_target_khz: float,
        freq_tol_khz: float = 50.0,
        vipir_version_idx: int = 1,
    ) -> EsImagingResult:
        """Load RIQ files and produce a combined Es imaging result.

        Convenience wrapper around :meth:`load` + :meth:`combine`.
        """
        cubes = self.load(
            file_list,
            freq_target_khz=freq_target_khz,
            freq_tol_khz=freq_tol_khz,
            vipir_version_idx=vipir_version_idx,
        )
        return self.combine(cubes)
