"""Viterbi trace extraction for canonical Digisonde echo DataFrames."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from pynasonde.digisonde.scaler.core import (
    _E_H_HI,
    _E_H_LO,
    _F2_H_HI,
    _F2_H_LO,
    IonoscaleBase,
    IonoscaleResult,
)


class ViterbiTracer(IonoscaleBase):
    """Extract the most-probable continuous ionospheric trace using Viterbi.

    Inherits cleaning pipeline from ``IonoscaleBase``.  Accepts either
    raw data (calls ``clean()`` internally) or a pre-cleaned DataFrame
    from ``IonogramAutoScaler.cleaned_``.

    Args:
    dbscan_eps, dbscan_min_samples, min_echoes, col_map
        Passed to ``IonoscaleBase`` (used only when ``fit()`` receives raw data).
    sigma_trans_km : float
        Gaussian transition width [km].  Default 30.
    min_amp_dB : float
        Amplitude gate — grid cells below this are -inf.  Default 0.
    consecutive_gap : int
        Consecutive below-threshold steps before trace termination.  Default 3.
    max_jump_km : float
        Max |Δh| allowed between adjacent trace steps.  Default 80.
    f_region_height_filter : bool
        If True, remove likely second-hop F-region echoes before tracing.
        Default rule keeps F-region echoes at <=400 km for f<=6 MHz and
        <=500 km for f>6 MHz.  Override the cutoff/frequency values with the
        ``f_region_*`` keyword arguments.
    """

    def __init__(
        self,
        dbscan_eps: float = 0.04,
        dbscan_min_samples: int = 4,
        min_echoes: int = 5,
        min_amp_dB: float = 0.0,
        mode: str | None = "O",
        col_map: Optional[dict] = None,
        sigma_trans_km: float = 30.0,
        consecutive_gap: int = 3,
        max_jump_km: float = 80.0,
        f_region_height_filter: bool = True,
        f_region_low_freq_max_mhz: float = 6.0,
        f_region_low_freq_max_height_km: float = 400.0,
        f_region_high_freq_max_height_km: float = 500.0,
        f_region_layer_names: tuple[str, ...] = ("F1", "F2"),
        rfi_filter: object | None = None,
    ) -> None:
        super().__init__(
            dbscan_eps=dbscan_eps,
            dbscan_min_samples=dbscan_min_samples,
            min_echoes=min_echoes,
            min_amp_dB=min_amp_dB,
            mode=mode,
            col_map=col_map,
            rfi_filter=rfi_filter,
        )
        self.sigma_trans = sigma_trans_km
        self.gap_limit = consecutive_gap
        self.max_jump_km = max_jump_km
        self.f_region_height_filter = f_region_height_filter
        self.f_region_low_freq_max_mhz = f_region_low_freq_max_mhz
        self.f_region_low_freq_max_height_km = f_region_low_freq_max_height_km
        self.f_region_high_freq_max_height_km = f_region_high_freq_max_height_km
        self.f_region_layer_names = tuple(f_region_layer_names)
        self.trace_metrics_: dict[str, float] = {}

    # ------------------------------------------------------------------
    # Grid builder
    # ------------------------------------------------------------------

    @staticmethod
    def build_grid(
        df: pd.DataFrame,
        h_lo: float,
        h_hi: float,
        freq_col: str = "frequency_mhz",
        height_col: str = "height_km",
        amp_col: str = "amplitude_db",
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Pivot echoes into a (n_freq × n_height) amplitude grid."""
        band = df[df[height_col].between(h_lo, h_hi)].copy()
        if band.empty:
            return np.empty((0, 0)), np.array([]), np.array([])
        piv = band.pivot_table(
            index=freq_col, columns=height_col, values=amp_col, aggfunc="max"
        )
        freq_axis = piv.index.to_numpy(float)
        height_axis = piv.columns.to_numpy(float)
        return piv.values.astype(float), freq_axis, height_axis

    # ------------------------------------------------------------------
    # Core Viterbi (forward)
    # ------------------------------------------------------------------

    def _viterbi(self, grid: np.ndarray, height_axis: np.ndarray) -> np.ndarray:
        n_freq, n_height = grid.shape
        log_emit = np.where(
            np.isfinite(grid) & (grid >= self.min_amp_dB),
            grid / 10.0,
            -np.inf,
        )
        dh = height_axis[:, None] - height_axis[None, :]
        log_trans = -0.5 * (dh / self.sigma_trans) ** 2

        V = np.full((n_freq, n_height), -np.inf)
        ptr = np.zeros((n_freq, n_height), dtype=np.int32)

        h_init_sigma = max((height_axis[-1] - height_axis[0]) / 4.0, 1.0)
        log_prior = -0.5 * ((height_axis - height_axis[0]) / h_init_sigma) ** 2
        V[0] = log_emit[0] + log_prior

        for t in range(1, n_freq):
            scores = V[t - 1, :, None] + log_trans
            ptr[t] = scores.argmax(axis=0)
            V[t] = scores.max(axis=0) + log_emit[t]

        path = np.empty(n_freq, dtype=np.int32)
        path[-1] = int(V[-1].argmax())
        for t in range(n_freq - 2, -1, -1):
            path[t] = ptr[t + 1, path[t + 1]]
        return path

    # ------------------------------------------------------------------
    # Termination
    # ------------------------------------------------------------------

    def _terminate(
        self,
        path: np.ndarray,
        grid: np.ndarray,
        freq_axis: np.ndarray,
        height_axis: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        path_amps = grid[np.arange(len(path)), path]
        valid = np.isfinite(path_amps) & (path_amps >= self.min_amp_dB)
        last_valid = len(valid) - 1
        gap = 0
        for i, v in enumerate(valid):
            if not v:
                gap += 1
                if gap >= self.gap_limit:
                    last_valid = i - self.gap_limit
                    break
            else:
                gap = 0
        mask = np.zeros(len(path), dtype=bool)
        mask[: last_valid + 1] = True
        mask &= valid
        return freq_axis[mask], height_axis[path[mask]]

    # ------------------------------------------------------------------
    # Post-Viterbi filters
    # ------------------------------------------------------------------

    @staticmethod
    def _jump_filter(
        valid_freq: np.ndarray,
        valid_h: np.ndarray,
        max_jump_km: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Remove spike kinks where |Δh| exceeds threshold."""
        if len(valid_h) < 3:
            return valid_freq, valid_h
        dh = np.diff(valid_h)
        bad = np.zeros(len(valid_h), dtype=bool)
        for k in range(len(dh) - 1):
            if abs(dh[k]) > max_jump_km or abs(dh[k + 1]) > max_jump_km:
                bad[k] = bad[k + 1] = bad[k + 2] = True
        keep = ~bad
        if bad.sum():
            logger.debug(f"jump-filter: removed {bad.sum()} spike points")
        valid_freq, valid_h = valid_freq[keep], valid_h[keep]

        # Trailing-descent trim (last 6 heights)
        N_TAIL = 6
        if len(valid_h) > N_TAIL and (np.diff(valid_h)[-N_TAIL:] < 0).any():
            valid_freq, valid_h = valid_freq[:-N_TAIL], valid_h[:-N_TAIL]
            logger.debug(f"descent-trim: removed last {N_TAIL} points")
        return valid_freq, valid_h

    # ------------------------------------------------------------------
    # Single-layer trace
    # ------------------------------------------------------------------

    def trace(
        self,
        df: pd.DataFrame,
        h_lo: float,
        h_hi: float,
        sigma_trans_km: float | None = None,
        f_lo: float = 0.0,
        f_hi: float = 99.0,
        max_jump_km: float | None = None,
        freq_col: str = "frequency_mhz",
        height_col: str = "height_km",
        amp_col: str = "amplitude_db",
    ) -> pd.DataFrame:
        """Extract Viterbi trace for one layer band.

        Returns DataFrame with columns [frequency_mhz, height_km, amplitude_db].
        """
        sigma = sigma_trans_km if sigma_trans_km is not None else self.sigma_trans
        max_jump = max_jump_km if max_jump_km is not None else self.max_jump_km

        df_band = (
            df[df[freq_col].between(f_lo, f_hi)] if (f_hi < 99.0 or f_lo > 0.0) else df
        )
        grid, freq_axis, height_axis = self.build_grid(
            df_band, h_lo, h_hi, freq_col, height_col, amp_col
        )
        if grid.size == 0:
            return pd.DataFrame(columns=["frequency_mhz", "height_km", "amplitude_db"])

        orig, self.sigma_trans = self.sigma_trans, sigma
        path = self._viterbi(grid, height_axis)
        self.sigma_trans = orig

        valid_freq, valid_h = self._terminate(path, grid, freq_axis, height_axis)
        valid_freq, valid_h = self._jump_filter(valid_freq, valid_h, max_jump)

        fo = float(valid_freq.max()) if valid_freq.size else np.nan
        h_prime = float(valid_h.min()) if valid_h.size else np.nan
        logger.debug(
            f"trace [{h_lo}–{h_hi} km]: fo={fo:.2f} MHz  "
            f"h'={h_prime:.0f} km  pts={len(valid_freq)}"
        )

        path_amps = grid[np.arange(len(path)), path]
        valid_mask = np.isin(freq_axis, valid_freq)
        return pd.DataFrame(
            {
                "frequency_mhz": freq_axis[valid_mask],
                "height_km": height_axis[path[valid_mask]],
                "amplitude_db": path_amps[valid_mask],
            }
        )

    # ------------------------------------------------------------------
    # All-layers entry point
    # ------------------------------------------------------------------

    def _apply_f_region_height_filter(
        self,
        df: pd.DataFrame,
        layer_name: str,
    ) -> tuple[pd.DataFrame, dict[str, float]]:
        """Remove likely second-hop F-region echoes before Viterbi tracing.

        The default rule is:
            frequency <= 6 MHz  -> height <= 400 km
            frequency  > 6 MHz  -> height <= 500 km

        It is applied only to configured F-region layer names.
        """
        prefix = f"{layer_name}_trace"
        metrics = {
            f"{prefix}_prefilter_enabled": False,
            f"{prefix}_prefilter_input": int(len(df)),
            f"{prefix}_prefilter_removed": 0,
            f"{prefix}_prefilter_remaining": int(len(df)),
        }
        if (
            not self.f_region_height_filter
            or layer_name not in self.f_region_layer_names
            or df.empty
        ):
            return df, metrics

        required = {"frequency_mhz", "height_km"}
        if not required.issubset(df.columns):
            return df, metrics

        low_freq = df["frequency_mhz"] <= self.f_region_low_freq_max_mhz
        keep = (
            low_freq & (df["height_km"] <= self.f_region_low_freq_max_height_km)
        ) | (~low_freq & (df["height_km"] <= self.f_region_high_freq_max_height_km))
        filtered = df[keep].copy().reset_index(drop=True)
        metrics.update(
            {
                f"{prefix}_prefilter_enabled": True,
                f"{prefix}_prefilter_removed": int(len(df) - len(filtered)),
                f"{prefix}_prefilter_remaining": int(len(filtered)),
            }
        )
        if metrics[f"{prefix}_prefilter_removed"]:
            logger.debug(
                f"{layer_name} first-hop filter removed "
                f"{metrics[f'{prefix}_prefilter_removed']} echoes "
                f"({len(df)} -> {len(filtered)})"
            )
        return filtered, metrics

    def trace_metrics(self, trace: pd.DataFrame, layer_name: str) -> dict[str, float]:
        """Return compact reliability metrics for one Viterbi trace.

        Metrics are intentionally numeric and flat so callers can write them
        directly to a scaled-parameter CSV.
        """
        prefix = f"{layer_name}_trace"
        if trace.empty or not {"frequency_mhz", "height_km"}.issubset(trace.columns):
            return {
                f"{prefix}_n_points": 0,
                f"{prefix}_freq_min_mhz": np.nan,
                f"{prefix}_freq_max_mhz": np.nan,
                f"{prefix}_freq_span_mhz": np.nan,
                f"{prefix}_height_min_km": np.nan,
                f"{prefix}_height_max_km": np.nan,
                f"{prefix}_height_span_km": np.nan,
                f"{prefix}_max_height_jump_km": np.nan,
                f"{prefix}_median_height_jump_km": np.nan,
                f"{prefix}_mean_height_jump_km": np.nan,
                f"{prefix}_max_abs_slope_km_per_mhz": np.nan,
                f"{prefix}_median_amp_db": np.nan,
                f"{prefix}_reliability_score": 0.0,
            }

        tr = trace.sort_values("frequency_mhz").reset_index(drop=True)
        freq = tr["frequency_mhz"].to_numpy(float)
        height = tr["height_km"].to_numpy(float)
        amps = (
            tr["amplitude_db"].to_numpy(float)
            if "amplitude_db" in tr.columns
            else np.full(len(tr), np.nan)
        )

        n_points = int(len(tr))
        freq_min = float(np.nanmin(freq))
        freq_max = float(np.nanmax(freq))
        freq_span = float(freq_max - freq_min)
        height_min = float(np.nanmin(height))
        height_max = float(np.nanmax(height))
        height_span = float(height_max - height_min)

        if n_points >= 2:
            dh = np.abs(np.diff(height))
            df = np.abs(np.diff(freq))
            with np.errstate(divide="ignore", invalid="ignore"):
                slopes = np.divide(dh, df, out=np.full_like(dh, np.nan), where=df > 0)
            max_jump = float(np.nanmax(dh))
            median_jump = float(np.nanmedian(dh))
            mean_jump = float(np.nanmean(dh))
            max_slope = (
                float(np.nanmax(slopes)) if np.isfinite(slopes).any() else np.nan
            )
        else:
            max_jump = np.nan
            median_jump = np.nan
            mean_jump = np.nan
            max_slope = np.nan

        median_amp = float(np.nanmedian(amps)) if np.isfinite(amps).any() else np.nan

        point_score = min(n_points / 12.0, 1.0)
        span_score = min(max(freq_span, 0.0) / 0.75, 1.0)
        continuity_score = (
            0.0 if not np.isfinite(max_jump) else 1.0 / (1.0 + max_jump / 50.0)
        )
        amp_score = (
            0.5
            if not np.isfinite(median_amp)
            else float(np.clip((median_amp - self.min_amp_dB) / 20.0, 0.0, 1.0))
        )
        reliability = (
            0.35 * point_score
            + 0.25 * span_score
            + 0.25 * continuity_score
            + 0.15 * amp_score
        )

        return {
            f"{prefix}_n_points": n_points,
            f"{prefix}_freq_min_mhz": freq_min,
            f"{prefix}_freq_max_mhz": freq_max,
            f"{prefix}_freq_span_mhz": freq_span,
            f"{prefix}_height_min_km": height_min,
            f"{prefix}_height_max_km": height_max,
            f"{prefix}_height_span_km": height_span,
            f"{prefix}_max_height_jump_km": max_jump,
            f"{prefix}_median_height_jump_km": median_jump,
            f"{prefix}_mean_height_jump_km": mean_jump,
            f"{prefix}_max_abs_slope_km_per_mhz": max_slope,
            f"{prefix}_median_amp_db": median_amp,
            f"{prefix}_reliability_score": float(np.clip(reliability, 0.0, 1.0)),
        }

    def fit(
        self,
        df: pd.DataFrame,
        already_cleaned: bool = False,
        layers: list[dict] = [
            dict(
                name="E",
                h_lo=_E_H_LO,
                h_hi=_E_H_HI,
                sigma_trans_km=8.0,
                max_jump_km=20.0,
                f_lo=1.0,
                f_hi=5.0,
            ),
            dict(
                name="F2",
                h_lo=_F2_H_LO,
                h_hi=_F2_H_HI,
                sigma_trans_km=30.0,
                max_jump_km=80.0,
                f_lo=2.0,
                f_hi=16.0,
            ),
        ],
    ) -> tuple[IonoscaleResult, dict[str, pd.DataFrame]]:
        """Trace all layers and return (IonoscaleResult, traces dict).

        Args:
            df (pd.DataFrame): Either raw data (``already_cleaned=False``) or the pre-cleaned
                ``IonogramAutoScaler.cleaned_`` DataFrame (``already_cleaned=True``).
            already_cleaned (bool): Skip the cleaning pipeline when True — use df directly as-is.
            layers (list of dicts): Per-layer tracing parameters (name, h_lo, h_hi, sigma_trans_km,
                max_jump_km, f_lo, f_hi).

        Returns:
        result : IonoscaleResult
        traces : dict[str, pd.DataFrame]  — raw trace points per layer
        """
        self.trace_metrics_ = {}
        cleaned = df if already_cleaned else self.clean(df)
        if cleaned.empty:
            return IonoscaleResult(quality_flags={"insufficient_data": True}), {}
        cleaned = self.run_rfi_filter(cleaned, pol_col="mode")
        self.cleaned_ = cleaned
        if cleaned.empty:
            return IonoscaleResult(quality_flags={"insufficient_data": True}), {}

        traces: dict[str, pd.DataFrame] = {}
        for layer in layers:
            logger.debug(f"Tracing layer: {layer['name']}")
            layer_cleaned, prefilter_metrics = self._apply_f_region_height_filter(
                cleaned, layer["name"]
            )
            traces[layer["name"]] = self.trace(
                layer_cleaned,
                layer["h_lo"],
                layer["h_hi"],
                sigma_trans_km=layer.get("sigma_trans_km"),
                max_jump_km=layer.get("max_jump_km"),
                f_lo=layer.get("f_lo", 0.0),
                f_hi=layer.get("f_hi", 99.0),
            )
            metrics = self.trace_metrics(traces[layer["name"]], layer["name"])
            metrics.update(prefilter_metrics)
            self.trace_metrics_.update(metrics)

        result = self._build_result(traces, trace_metrics=self.trace_metrics_)
        logger.info(f"Viterbi result — {result.summary()}")
        return result, traces
