"""Coherent multi-stage filter for VIPIR ionospheric echo clouds.

:class:`IonogramFilter` applies a configurable cascade of five stages to a
set of :class:`~pynasonde.vipir.riq.echo.Echo` DataFrames produced by
:class:`~pynasonde.vipir.riq.echo.EchoExtractor`.  It accepts a **single
sounding** or a **list of soundings** (multiple RIQ files / time steps) and
returns a clean, filtered ``pd.DataFrame``.

Filter stages
-------------
1. **RFI blanking** — reject entire frequency steps where the height IQR of
   extracted echoes exceeds a threshold.  RFI illuminates random gates across
   all heights (IQR > 300 km); ionospheric echoes cluster near E/F layers
   (IQR < 150 km).  Count-based detection is unreliable when
   ``max_echoes_per_pulset`` caps the per-frequency echo count.

2. **EP / planar-wavefront filter** — reject echoes whose planar-wavefront
   residual (EP / ``residual_deg``) exceeds a threshold, indicating
   multipath or non-planar clutter.

3. **Multi-hop removal** — at each frequency, identify the lowest-height
   (1F) echo cluster and flag echoes at heights close to integer multiples
   (2F, 3F …) that are also weaker by a configurable SNR margin.

4. **DBSCAN clustering** — in a normalised multi-dimensional feature space
   (frequency, height, V*, amplitude, EP), reject points labelled as noise
   (DBSCAN label = -1) while keeping all cluster members.

5. **RANSAC trace fitting** — fit a smooth polynomial h*(f) curve to the
   (frequency, height) echo cloud via Random Sample Consensus.  Echoes
   further than ``ransac_residual_km`` from the best-fit curve are rejected
   as outliers.  Run independently per sounding so that each sounding's
   ionospheric trace is fitted separately.

6. **Temporal coherence** *(multi-sounding only)* — discretise the
   (frequency, height) plane into bins and retain only echoes that appear
   in at least *min_soundings* out of the provided soundings.  Random
   interference is temporally incoherent; real ionospheric echoes persist.

All stages are independently toggleable and fully parameterised.

Usage
-----
Single sounding::

    from pynasonde.vipir.riq.parsers.filter import IonogramFilter

    filt = IonogramFilter(ep_max_deg=40.0, dbscan_eps=1.2)
    clean = filt.filter(extractor)          # EchoExtractor or DataFrame

Multiple soundings::

    filt = IonogramFilter(temporal_min_soundings=3)
    clean = filt.filter([ext1, ext2, ext3, ext4, ext5])

"""

from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from loguru import logger

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _to_dataframe(source) -> pd.DataFrame:
    """Convert an EchoExtractor or DataFrame to a DataFrame of echoes."""
    if isinstance(source, pd.DataFrame):
        return source.copy()
    # EchoExtractor exposes .to_dataframe()
    if hasattr(source, "to_dataframe"):
        return source.to_dataframe()
    # Plain list of Echo dataclass objects
    if isinstance(source, list):
        return pd.DataFrame([e.to_dict() for e in source])
    raise TypeError(
        f"Cannot convert {type(source)} to DataFrame. "
        "Pass an EchoExtractor, a pd.DataFrame, or a list of Echo objects."
    )


# ---------------------------------------------------------------------------
# IonogramFilter
# ---------------------------------------------------------------------------


class IonogramFilter:
    """Multi-stage coherent filter for VIPIR ionospheric echo clouds.

    Parameters
    ----------
    rfi_enabled : bool
        Toggle Stage 1 (RFI blanking). Default ``True``.
    rfi_height_iqr_km : float
        A frequency step is declared RFI/noise when the inter-quartile range
        of echo heights at that frequency exceeds this value.  Ionospheric
        echoes cluster near E/F-layer heights (height IQR < 150 km); RFI
        scatters echoes across all gates (height IQR > 300–800 km).
        Default ``300.0``.
    rfi_min_echoes : int
        Minimum number of echoes at a frequency before any per-frequency
        Stage 1 test is applied.  Default ``3``.
    ep_filter_enabled : bool
        Toggle Stage 2 (EP / planar-wavefront filter). Default ``True``.
    ep_max_deg : float
        Echoes with ``residual_deg > ep_max_deg`` are rejected (non-planar
        wavefront / multipath).  Ignored when ``ep_filter_enabled=False`` or
        when ``residual_deg`` is NaN (single-receiver echoes).  Set
        conservatively high (e.g. 90°) — oblique real echoes can have
        EP 50–80°; let DBSCAN (Stage 4) handle subtler cases.
        Default ``90.0``.
    multihop_enabled : bool
        Toggle Stage 3 (multi-hop removal). Default ``True``.
    multihop_orders : tuple of int
        Harmonic orders to check.  ``(2, 3)`` checks for 2F and 3F echoes.
        Default ``(2, 3)``.
    multihop_height_tol_km : float
        An echo at height *h* is considered an n-th order multi-hop if
        ``|h - n × h_1F| < multihop_height_tol_km``. Default ``50.0``.
    multihop_snr_margin_db : float
        Additionally, the candidate multi-hop echo must be weaker than the
        1F echo by at least this many dB. Default ``6.0``.
    dbscan_enabled : bool
        Toggle Stage 4 (DBSCAN clustering). Default ``True``.
    dbscan_eps : float
        DBSCAN neighbourhood radius in normalised feature space. Default ``1.0``.
    dbscan_min_samples : int
        Minimum cluster size for DBSCAN. Default ``5``.
    dbscan_features : tuple of str
        Feature columns used for DBSCAN.  Columns absent from the DataFrame
        or entirely NaN are silently skipped. Default
        ``("frequency_khz", "height_km", "velocity_mps", "amplitude_db", "residual_deg")``.
    dbscan_feature_scales : dict, optional
        Per-feature normalisation scale (σ).  Keys are column names, values
        are positive floats.  If a key is missing or ``None``, the standard
        deviation of that feature in the current batch is used.
    ransac_enabled : bool
        Toggle Stage 5 (RANSAC trace fitting). Default ``True``.
    ransac_residual_km : float
        Maximum height residual |h - h*(f)| for an echo to be considered an
        inlier of the fitted trace.  Generous values (100 km) tolerate
        spread-F; tighter values (50 km) enforce a clean single-layer trace.
        Default ``100.0``.
    ransac_min_samples : int
        Number of randomly sampled echoes used to estimate the trace model in
        each RANSAC iteration.  Must be ≥ ``ransac_poly_degree + 1``.
        Default ``10``.
    ransac_n_iter : int
        Number of RANSAC iterations per sounding.  More iterations improve
        robustness at the cost of compute time.  Default ``200``.
    ransac_poly_degree : int
        Degree of the polynomial h*(f) used as the trace model.  Degree 3
        captures the curvature of the F-layer trace; higher degrees risk
        overfitting on sparse soundings.  Default ``3``.
    ransac_min_inlier_fraction : float
        Minimum fraction of active echoes that must be inliers for the
        fitted model to be accepted.  If no iteration reaches this threshold
        the stage is skipped.  Default ``0.3``.
    temporal_enabled : bool
        Toggle Stage 6 (temporal coherence). Automatically disabled when
        only one sounding is provided. Default ``True``.
    temporal_min_soundings : int
        An echo is retained only if a matching echo (within the tolerance
        windows below) exists in at least this many of the provided soundings.
        Default ``3``.
    temporal_freq_bin_khz : float
        Frequency bin width for the temporal coherence grid (kHz). Default ``50``.
    temporal_height_bin_km : float
        Height bin width for the temporal coherence grid (km). Default ``50``.
    """

    def __init__(
        self,
        # ── Stage 1: RFI ──────────────────────────────────────────────────
        rfi_enabled: bool = True,
        rfi_height_iqr_km: float = 300.0,
        rfi_min_echoes: int = 3,
        # ── Stage 2: EP filter ────────────────────────────────────────────
        ep_filter_enabled: bool = True,
        ep_max_deg: float = 90.0,
        # ── Stage 3: Multi-hop ────────────────────────────────────────────
        multihop_enabled: bool = True,
        multihop_orders: Tuple[int, ...] = (2, 3),
        multihop_height_tol_km: float = 50.0,
        multihop_snr_margin_db: float = 6.0,
        # ── Stage 4: DBSCAN ───────────────────────────────────────────────
        dbscan_enabled: bool = True,
        dbscan_eps: float = 1.0,
        dbscan_min_samples: int = 5,
        dbscan_features: Tuple[str, ...] = (
            "frequency_khz",
            "height_km",
            "velocity_mps",
            "amplitude_db",
            "residual_deg",
        ),
        dbscan_feature_scales: Optional[Dict[str, float]] = None,
        # ── Stage 5: RANSAC trace fitting ─────────────────────────────────
        ransac_enabled: bool = True,
        ransac_residual_km: float = 100.0,
        ransac_min_samples: int = 10,
        ransac_n_iter: int = 200,
        ransac_poly_degree: int = 3,
        ransac_min_inlier_fraction: float = 0.3,
        # ── Stage 6: Temporal coherence ───────────────────────────────────
        temporal_enabled: bool = True,
        temporal_min_soundings: int = 3,
        temporal_freq_bin_khz: float = 50.0,
        temporal_height_bin_km: float = 50.0,
    ) -> None:
        # Stage 1
        self.rfi_enabled = rfi_enabled
        self.rfi_height_iqr_km = rfi_height_iqr_km
        self.rfi_min_echoes = rfi_min_echoes
        # Stage 2
        self.ep_filter_enabled = ep_filter_enabled
        self.ep_max_deg = ep_max_deg
        # Stage 3
        self.multihop_enabled = multihop_enabled
        self.multihop_orders = tuple(multihop_orders)
        self.multihop_height_tol_km = multihop_height_tol_km
        self.multihop_snr_margin_db = multihop_snr_margin_db
        # Stage 4
        self.dbscan_enabled = dbscan_enabled
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.dbscan_features = tuple(dbscan_features)
        self.dbscan_feature_scales: Dict[str, float] = dbscan_feature_scales or {}
        # Stage 5
        self.ransac_enabled = ransac_enabled
        self.ransac_residual_km = ransac_residual_km
        self.ransac_min_samples = ransac_min_samples
        self.ransac_n_iter = ransac_n_iter
        self.ransac_poly_degree = ransac_poly_degree
        self.ransac_min_inlier_fraction = ransac_min_inlier_fraction
        # Stage 6
        self.temporal_enabled = temporal_enabled
        self.temporal_min_soundings = temporal_min_soundings
        self.temporal_freq_bin_khz = temporal_freq_bin_khz
        self.temporal_height_bin_km = temporal_height_bin_km

        # Populated after each call to filter()
        self._stats: Dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def filter(
        self,
        sources: Union[object, pd.DataFrame, List],
    ) -> pd.DataFrame:
        """Run all enabled filter stages and return the surviving echoes.

        Parameters
        ----------
        sources : EchoExtractor | pd.DataFrame | list thereof
            One or more sounding sources.  Each element may be an
            :class:`~pynasonde.vipir.riq.echo.EchoExtractor`, a
            ``pd.DataFrame`` of echoes, or a list of
            :class:`~pynasonde.vipir.riq.echo.Echo` objects.

        Returns
        -------
        pd.DataFrame
            Filtered echo DataFrame.  A ``sounding_index`` column (int) is
            always present; for a single sounding it is all zeros.  A
            ``filter_mask`` boolean column marks the echoes that survived
            *all* enabled stages (always ``True`` in the returned frame,
            kept for traceability when the caller retains the original).
        """
        # ── Normalise input to list of DataFrames ────────────────────────
        if not isinstance(sources, (list, tuple)):
            sources = [sources]
        dfs: List[pd.DataFrame] = []
        for idx, src in enumerate(sources):
            df = _to_dataframe(src)
            df["sounding_index"] = idx
            dfs.append(df)

        n_soundings = len(dfs)
        combined = pd.concat(dfs, ignore_index=True)
        n_total = len(combined)

        if n_total == 0:
            logger.warning(
                "IonogramFilter.filter: no echoes in input — returning empty DataFrame."
            )
            return combined

        self._stats = {}

        # ── Stage 1: RFI blanking ────────────────────────────────────────
        mask = pd.Series(True, index=combined.index)

        if self.rfi_enabled:
            rfi_mask = self._stage_rfi(combined)
            n_rfi = (~rfi_mask).sum()
            self._stats["rfi"] = {"rejected": int(n_rfi)}
            logger.info(
                f"[IonogramFilter] Stage 1 RFI: rejected {n_rfi}/{n_total} echoes"
            )
            mask &= rfi_mask

        # ── Stage 2: EP / planar-wavefront ───────────────────────────────
        if self.ep_filter_enabled:
            ep_mask = self._stage_ep(combined)
            n_ep = (~ep_mask & mask).sum()
            self._stats["ep"] = {"rejected": int(n_ep)}
            logger.info(
                f"[IonogramFilter] Stage 2 EP : rejected {n_ep}/{mask.sum()} echoes"
            )
            mask &= ep_mask

        # ── Stage 3: Multi-hop removal ───────────────────────────────────
        # Operates only on echoes surviving stages 1–2 (prior mask applied
        # inside); rejected echoes from earlier stages must not influence the
        # 1F reference height used here.
        if self.multihop_enabled:
            mh_mask = self._stage_multihop(combined, mask)
            n_mh = (~mh_mask & mask).sum()
            self._stats["multihop"] = {"rejected": int(n_mh)}
            logger.info(
                f"[IonogramFilter] Stage 3 MH : rejected {n_mh}/{mask.sum()} echoes"
            )
            mask &= mh_mask

        # ── Stage 4: DBSCAN clustering ───────────────────────────────────
        # Operates only on echoes surviving stages 1–3; noise echoes from
        # earlier stages must not pollute cluster membership.
        if self.dbscan_enabled:
            db_mask = self._stage_dbscan(combined, mask)
            n_db = (~db_mask & mask).sum()
            self._stats["dbscan"] = {"rejected": int(n_db)}
            logger.info(
                f"[IonogramFilter] Stage 4 DBSCAN: rejected {n_db}/{mask.sum()} echoes"
            )
            mask &= db_mask

        # ── Stage 5: RANSAC trace fitting ────────────────────────────────
        # Run per sounding so each sounding's ionospheric trace is fitted
        # independently (the trace moves between soundings).
        if self.ransac_enabled:
            rs_mask = self._stage_ransac(combined, mask)
            n_rs = (~rs_mask & mask).sum()
            self._stats["ransac"] = {"rejected": int(n_rs)}
            logger.info(
                f"[IonogramFilter] Stage 5 RANSAC: rejected {n_rs}/{mask.sum()} echoes"
            )
            mask &= rs_mask

        # ── Stage 6: Temporal coherence ──────────────────────────────────
        if self.temporal_enabled and n_soundings > 1:
            tc_mask = self._stage_temporal(combined, mask)
            n_tc = (~tc_mask & mask).sum()
            self._stats["temporal"] = {"rejected": int(n_tc)}
            logger.info(
                f"[IonogramFilter] Stage 6 TC  : rejected {n_tc}/{mask.sum()} echoes "
                f"(min_soundings={self.temporal_min_soundings}/{n_soundings})"
            )
            mask &= tc_mask
        elif self.temporal_enabled and n_soundings == 1:
            logger.debug("[IonogramFilter] Stage 6 TC: skipped (single sounding).")

        n_kept = mask.sum()
        self._stats["summary"] = {
            "total_input": n_total,
            "total_kept": int(n_kept),
            "total_rejected": int(n_total - n_kept),
            "n_soundings": n_soundings,
        }
        logger.info(
            f"[IonogramFilter] Done: kept {n_kept}/{n_total} echoes "
            f"({100*n_kept/max(n_total,1):.1f}%)"
        )

        result = combined[mask].copy()
        result["filter_mask"] = True
        return result.reset_index(drop=True)

    @property
    def stats(self) -> Dict[str, dict]:
        """Rejection statistics from the most recent :meth:`filter` call."""
        return self._stats

    def summary(self) -> str:
        """Return a human-readable rejection summary string."""
        if not self._stats:
            return "No filter run yet."
        lines = ["IonogramFilter summary"]
        lines.append(f"  Soundings   : {self._stats['summary']['n_soundings']}")
        lines.append(f"  Input echoes: {self._stats['summary']['total_input']}")
        for stage, label in [
            ("rfi", "Stage 1 RFI      "),
            ("ep", "Stage 2 EP       "),
            ("multihop", "Stage 3 Multi-hop"),
            ("dbscan", "Stage 4 DBSCAN   "),
            ("ransac", "Stage 5 RANSAC   "),
            ("temporal", "Stage 6 Temporal "),
        ]:
            if stage in self._stats:
                n = self._stats[stage]["rejected"]
                lines.append(f"  {label}: {n:>6d} rejected")
        s = self._stats["summary"]
        lines.append(
            f"  Output echoes: {s['total_kept']}  "
            f"({100*s['total_kept']/max(s['total_input'],1):.1f}% retained)"
        )
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Stage implementations
    # ------------------------------------------------------------------

    # ── Stage 1: RFI ─────────────────────────────────────────────────────

    def _stage_rfi(self, df: pd.DataFrame) -> pd.Series:
        """Return True for echoes NOT at RFI-contaminated frequencies.

        Detection is based on the **height spread** (IQR of ``height_km``)
        at each frequency step, not echo count.  Count-based detection is
        unreliable when :class:`~pynasonde.vipir.riq.echo.EchoExtractor`
        caps the number of echoes per pulset (e.g. ``max_echoes_per_pulset=5``),
        because both ionospheric and RFI frequencies then return ≤5 echoes.

        Physics: RFI illuminates random gates across all heights (height IQR
        ≈ 300–800 km).  Ionospheric echoes cluster near E- or F-layer heights
        (height IQR < 100–200 km at any given frequency).
        """
        keep = pd.Series(True, index=df.index)
        rfi_freqs = []

        for freq_khz, grp in df.groupby("frequency_khz"):
            if len(grp) < self.rfi_min_echoes:
                continue

            flagged = False
            reason = ""

            # ── Height-spread check ───────────────────────────────────────
            h_iqr = grp["height_km"].quantile(0.75) - grp["height_km"].quantile(0.25)
            if h_iqr > self.rfi_height_iqr_km:
                flagged = True
                reason = f"height IQR={h_iqr:.0f} km > {self.rfi_height_iqr_km:.0f}"

            if flagged:
                rfi_freqs.append(freq_khz)
                logger.debug(f"[RFI] {freq_khz:.0f} kHz blanked ({reason})")
                keep[grp.index] = False

        if rfi_freqs:
            logger.info(
                f"[IonogramFilter] Stage 1 RFI: {len(rfi_freqs)} frequency steps blanked (height IQR threshold exceeded)"
            )

        return keep

    # ── Stage 2: EP filter ───────────────────────────────────────────────

    def _stage_ep(self, df: pd.DataFrame) -> pd.Series:
        """Return True for echoes with EP ≤ ep_max_deg (or NaN EP)."""
        keep = pd.Series(True, index=df.index)
        if "residual_deg" not in df.columns:
            return keep

        ep = df["residual_deg"]
        # NaN EP (single-receiver echoes) are not penalised
        exceeds = ep.notna() & (ep > self.ep_max_deg)
        keep[exceeds] = False
        return keep

    # ── Stage 3: Multi-hop ───────────────────────────────────────────────

    def _stage_multihop(self, df: pd.DataFrame, prior_mask: pd.Series) -> pd.Series:
        """Return True for echoes that are NOT identified as multi-hop.

        Operates only on echoes that survived prior stages (``prior_mask``).
        The 1F reference is the **strongest** echo below
        ``multihop_height_tol_km × min_order`` at each frequency, not simply
        the minimum-height echo — a stray low-height noise echo would
        otherwise corrupt every frequency's 1F reference.
        """
        # Start with all True; we only reject within the surviving subset
        keep = pd.Series(True, index=df.index)

        active = df[prior_mask]
        if active.empty:
            return keep

        for freq_khz, grp in active.groupby("frequency_khz"):
            if len(grp) < 2:
                continue

            # 1F reference: strongest echo in the lower half of the height
            # distribution (below median height), so the reference is
            # anchored to the first ionospheric return, not a high-altitude
            # noise spike.
            h_median = grp["height_km"].median()
            lower_half = grp[grp["height_km"] <= h_median]
            if lower_half.empty:
                lower_half = grp

            ref_idx = lower_half["amplitude_db"].idxmax()
            h_1f = grp.loc[ref_idx, "height_km"]
            amp_1f = grp.loc[ref_idx, "amplitude_db"]

            for order in self.multihop_orders:
                h_expected = order * h_1f
                height_match = (
                    grp["height_km"] - h_expected
                ).abs() < self.multihop_height_tol_km
                # Multi-hop must be weaker than 1F by the configured margin
                weaker = grp["amplitude_db"] < (amp_1f - self.multihop_snr_margin_db)
                multihop_idx = grp.index[height_match & weaker]
                if len(multihop_idx) > 0:
                    logger.debug(
                        f"[MultiHop] freq={freq_khz:.0f} kHz: "
                        f"{len(multihop_idx)} echoes flagged as {order}F "
                        f"(h_1F={h_1f:.0f} km, h_exp={h_expected:.0f} km)"
                    )
                    keep[multihop_idx] = False

        return keep

    # ── Stage 4: DBSCAN ──────────────────────────────────────────────────

    def _stage_dbscan(self, df: pd.DataFrame, prior_mask: pd.Series) -> pd.Series:
        """Return True for echoes belonging to a DBSCAN cluster (label ≥ 0).

        Runs DBSCAN only on the surviving subset (``prior_mask``).  Echoes
        already rejected by earlier stages must not form spurious clusters
        or attract real-echo neighbours into noise territory.

        Normalisation uses **IQR** (inter-quartile range) rather than std so
        that the scale is robust to the remaining outliers.  NaN features
        (e.g. V* or EP absent for single-receiver echoes) are imputed with
        the column **median** of the active subset, keeping those echoes
        neutral in those dimensions rather than forcing them to the origin.
        """
        # Default: keep everything (will be overridden for active subset)
        keep = pd.Series(True, index=df.index)

        try:
            from sklearn.cluster import DBSCAN
        except ImportError:
            warnings.warn(
                "scikit-learn is not installed; DBSCAN stage skipped. "
                "Install it with: pip install scikit-learn",
                RuntimeWarning,
                stacklevel=3,
            )
            return keep

        active = df[prior_mask]
        if len(active) < self.dbscan_min_samples:
            logger.warning(
                f"[DBSCAN] Only {len(active)} echoes survive prior stages "
                f"(< min_samples={self.dbscan_min_samples}) — stage skipped."
            )
            return keep

        available = [
            c
            for c in self.dbscan_features
            if c in active.columns and active[c].notna().any()
        ]
        if not available:
            logger.warning("[DBSCAN] No configured features found — stage skipped.")
            return keep

        feat = active[available].copy()

        # ── IQR-based normalisation (robust to remaining outliers) ────────
        X = np.zeros((len(feat), len(available)), dtype=float)
        for j, col in enumerate(available):
            vals = feat[col].to_numpy(dtype=float)

            # Impute NaN with column median before scaling
            col_median = np.nanmedian(vals)
            vals = np.where(np.isfinite(vals), vals, col_median)

            scale = self.dbscan_feature_scales.get(col)
            if scale is None:
                q75, q25 = np.percentile(vals, [75, 25])
                scale = q75 - q25  # IQR
            if scale == 0 or not np.isfinite(scale):
                scale = np.nanstd(vals) or 1.0  # fall back to std if IQR=0
            X[:, j] = (vals - np.nanmedian(vals)) / scale

        labels = DBSCAN(
            eps=self.dbscan_eps,
            min_samples=self.dbscan_min_samples,
        ).fit_predict(X)

        # Map noise labels back to the full df index
        noise_positions = active.index[labels == -1]
        keep[noise_positions] = False
        return keep

    # ── Stage 5: RANSAC trace fitting ────────────────────────────────────

    def _stage_ransac(self, df: pd.DataFrame, prior_mask: pd.Series) -> pd.Series:
        """Return True for echoes consistent with a smooth ionospheric trace.

        Fits a degree-``ransac_poly_degree`` polynomial h*(f) to the
        (frequency, height) echo cloud using RANSAC.  Echoes further than
        ``ransac_residual_km`` from the best-fit curve are labelled as
        outliers and removed.

        The fit is performed **per sounding_index** so that each sounding's
        independently moving ionospheric trace is captured separately.

        Algorithm
        ---------
        For each iteration:
          1. Randomly sample ``ransac_min_samples`` echoes from the active pool.
          2. Fit polynomial  h*(f_norm) = polyval(coeffs, f_norm).
          3. Count inliers: echoes where |h - h*(f)| < ``ransac_residual_km``.
        Keep the model with the most inliers; refit on all its inliers.
        Reject echoes outside ``ransac_residual_km`` of the final model.

        If no iteration achieves ``ransac_min_inlier_fraction`` of the active
        echoes the stage is skipped for that sounding.
        """
        keep = pd.Series(True, index=df.index)
        rng = np.random.default_rng(seed=0)

        for sounding_idx, s_mask in df.groupby("sounding_index").groups.items():
            # Intersect with echoes surviving previous stages
            active_idx = df.index[prior_mask & (df["sounding_index"] == sounding_idx)]
            n_active = len(active_idx)

            min_needed = max(self.ransac_min_samples, self.ransac_poly_degree + 1)
            if n_active < min_needed:
                logger.debug(
                    f"[RANSAC] sounding {sounding_idx}: only {n_active} echoes "
                    f"(< {min_needed}) — skipped."
                )
                continue

            freqs = df.loc[active_idx, "frequency_khz"].to_numpy(dtype=float)
            heights = df.loc[active_idx, "height_km"].to_numpy(dtype=float)

            # Normalise frequency to [0, 1] for numerical stability
            f_min, f_max = freqs.min(), freqs.max()
            f_range = f_max - f_min if f_max > f_min else 1.0
            f_norm = (freqs - f_min) / f_range

            best_inlier_mask = None
            best_n_inliers = 0
            min_inliers_required = max(
                min_needed,
                int(self.ransac_min_inlier_fraction * n_active),
            )

            for _ in range(self.ransac_n_iter):
                sample = rng.choice(n_active, size=min_needed, replace=False)
                try:
                    coeffs = np.polyfit(
                        f_norm[sample], heights[sample], deg=self.ransac_poly_degree
                    )
                except (np.linalg.LinAlgError, ValueError):
                    continue

                h_pred = np.polyval(coeffs, f_norm)
                residuals = np.abs(heights - h_pred)
                inlier_mask = residuals < self.ransac_residual_km
                n_inliers = inlier_mask.sum()

                if n_inliers > best_n_inliers:
                    best_n_inliers = n_inliers
                    best_inlier_mask = inlier_mask

            if best_inlier_mask is None or best_n_inliers < min_inliers_required:
                logger.warning(
                    f"[RANSAC] sounding {sounding_idx}: best model has only "
                    f"{best_n_inliers} inliers "
                    f"(need {min_inliers_required}) — stage skipped for this sounding."
                )
                continue

            # Refit on all inliers of the best model for a cleaner final curve
            try:
                coeffs_final = np.polyfit(
                    f_norm[best_inlier_mask],
                    heights[best_inlier_mask],
                    deg=self.ransac_poly_degree,
                )
                h_pred_final = np.polyval(coeffs_final, f_norm)
                final_inlier_mask = (
                    np.abs(heights - h_pred_final) < self.ransac_residual_km
                )
            except (np.linalg.LinAlgError, ValueError):
                final_inlier_mask = best_inlier_mask

            n_outliers = (~final_inlier_mask).sum()
            if n_outliers > 0:
                outlier_positions = active_idx[~final_inlier_mask]
                keep[outlier_positions] = False
                logger.debug(
                    f"[RANSAC] sounding {sounding_idx}: {n_outliers} outliers removed "
                    f"({best_n_inliers} inliers kept, residual < {self.ransac_residual_km} km)"
                )

        return keep

    # ── Stage 6: Temporal coherence ──────────────────────────────────────

    def _stage_temporal(self, df: pd.DataFrame, prior_mask: pd.Series) -> pd.Series:
        """Return True for echoes that appear in ≥ temporal_min_soundings.

        The test operates on the (frequency bin, height bin) grid.  Any
        (f_bin, h_bin) cell occupied by at least one echo in ≥ min_soundings
        is considered "coherent" and all its echoes are retained.

        Only echoes surviving ``prior_mask`` are used to populate the grid,
        but the output mask covers all indices in ``df``.
        """
        keep = pd.Series(False, index=df.index)

        if self.temporal_min_soundings < 2:
            keep[:] = True
            return keep

        # Work only on echoes that survived previous stages
        active = df[prior_mask].copy()

        if active.empty:
            return keep

        # Assign grid bins
        f_bins = (active["frequency_khz"] / self.temporal_freq_bin_khz).astype(int)
        h_bins = (active["height_km"] / self.temporal_height_bin_km).astype(int)
        active = active.assign(_fb=f_bins, _hb=h_bins)

        # Count distinct soundings per (f_bin, h_bin) cell
        cell_sounding_counts = active.groupby(["_fb", "_hb"])[
            "sounding_index"
        ].nunique()
        coherent_cells = cell_sounding_counts[
            cell_sounding_counts >= self.temporal_min_soundings
        ].index  # MultiIndex of (f_bin, h_bin)

        if len(coherent_cells) == 0:
            logger.warning(
                "[Temporal] No coherent (f_bin, h_bin) cells found — "
                "consider reducing temporal_min_soundings or bin widths."
            )
            return keep

        # Build a set for fast lookup
        coherent_set = set(map(tuple, coherent_cells.tolist()))

        # Apply to ALL echoes in df (not just active), so the mask aligns
        all_fb = (df["frequency_khz"] / self.temporal_freq_bin_khz).astype(int)
        all_hb = (df["height_km"] / self.temporal_height_bin_km).astype(int)

        coherent_flag = pd.Series(
            [(int(fb), int(hb)) in coherent_set for fb, hb in zip(all_fb, all_hb)],
            index=df.index,
        )
        keep[coherent_flag] = True
        return keep
