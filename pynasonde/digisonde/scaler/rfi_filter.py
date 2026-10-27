"""Ionogram RFI filters — no training data required.

Two complementary approaches to remove RFI vertical lines while
preserving ionospheric trace echoes:

Option 2 — NeighborPredictedFilter
    For each frequency, interpolate the expected virtual height h'(f)
    from the surrounding clean frequency steps.  Keep only echoes
    consistent with that prediction.  Even if the frequency is heavily
    contaminated, any real echo sitting near the predicted h' is saved.

Option 3 — ColumnDensityFilter
    A real ionospheric reflection at one frequency occupies a narrow
    height window.  RFI spans all heights simultaneously → the height
    span at that frequency column is huge.  Flag and clip columns whose
    h_max − h_min exceeds a threshold.

They can be used independently or chained (Option 3 first for gross
removal, then Option 2 for residual contamination):

    from pynasonde.digisonde.scaler import NeighborPredictedFilter, ColumnDensityFilter

    df = ColumnDensityFilter().clean(df)
    df = NeighborPredictedFilter().clean(df)
"""

from __future__ import annotations

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from loguru import logger

from pynasonde.digisonde.scaler.schema import normalize_echo_dataframe

# ===========================================================================
# Option 2 — Neighbor-predicted height filter
# ===========================================================================


class NeighborPredictedFilter:
    """Remove RFI by checking trace continuity against neighboring frequencies.

    **Algorithm**

    1. For each unique frequency bin fₖ, compute the minimum virtual
       height hₖ = min(h at fₖ) as the trace candidate.
    2. Using the ±n_neighbors adjacent frequency steps, build a local
       linear interpolant h_pred(fₖ).
    3. Echoes at fₖ that deviate more than *tolerance_km* from h_pred
       are flagged as RFI and discarded.
    4. At sweep boundaries (< min_neighbors clean neighbours), the step
       is left unfiltered so boundary echoes are never accidentally lost.

    Key property: if a real ionospheric echo sits at an RFI-contaminated
    frequency, it is *kept* because it is consistent with the local trace
    continuity predicted by its clean neighbours.

    Args:
        tolerance_km (float): Maximum allowed deviation from the neighbor-predicted h' [km].
            Default 40 km — tight enough to reject RFI, wide enough to
            accommodate genuine layer curvature near foF2.
        n_neighbors (int): Frequency steps on each side used for the h' prediction.
            Default 4.
        freq_bin (float): Frequency bin width for grouping echoes before neighbour
            prediction.  Units must match the frequency column passed to
            ``clean()``; default 0.05 MHz (50 kHz) for canonical frequency values.
        min_neighbors (int): Minimum clean neighbours required to make a prediction; if
            fewer are available the step is left unfiltered.  Default 2.
    """

    def __init__(
        self,
        tolerance_km: float = 40.0,
        n_neighbors: int = 4,
        freq_bin: float = 0.05,
        min_neighbors: int = 2,
    ) -> None:
        self.tolerance_km = tolerance_km
        self.n_neighbors = n_neighbors
        self.freq_bin = freq_bin
        self.min_neighbors = min_neighbors

    def clean(
        self,
        df: pd.DataFrame,
        freq_col: str = "frequency_mhz",
        height_col: str = "height_km",
        amp_col: str = "amplitude_db",
    ) -> pd.DataFrame:
        """Return DataFrame with RFI echoes removed.

        Adds column ``npf_h_pred_km`` (predicted h' at each echo's
        frequency) for inspection/debugging.
        """
        if df.empty:
            return df.copy()

        # ── Bin frequencies; compute per-bin minimum h' (trace candidate) ─────
        f_binned = (df[freq_col] / self.freq_bin).round() * self.freq_bin
        grp = (
            df.assign(_fbin=f_binned)
            .groupby("_fbin")[height_col]
            .min()
            .reset_index()
            .rename(columns={"_fbin": "freq_bin", height_col: "h_min"})
            .sort_values("freq_bin")
        )
        freq_bins = grp["freq_bin"].values
        h_mins = grp["h_min"].values
        n_bins = len(freq_bins)

        # ── Build neighbor-predicted h' for every bin ─────────────────────────
        h_pred_map: dict[float, float] = {}
        for i in range(n_bins):
            lo = max(0, i - self.n_neighbors)
            hi = min(n_bins, i + self.n_neighbors + 1)
            nb_idx = [j for j in range(lo, hi) if j != i]

            if len(nb_idx) < self.min_neighbors:
                # Not enough neighbours — leave this step unfiltered
                h_pred_map[freq_bins[i]] = h_mins[i]
                continue

            nb_f = freq_bins[nb_idx]
            nb_h = h_mins[nb_idx]
            h_pred = float(
                np.interp(freq_bins[i], nb_f, nb_h, left=nb_h[0], right=nb_h[-1])
            )
            h_pred_map[freq_bins[i]] = h_pred

        # ── Map prediction to every echo row and apply tolerance gate ─────────
        pred_series = f_binned.map(h_pred_map)
        deviation = np.abs(df[height_col].values - pred_series.values)
        keep = deviation <= self.tolerance_km

        n_removed = int((~keep).sum())
        if n_removed:
            logger.info(
                f"NeighborPredictedFilter: removed {n_removed} / {len(df)} "
                f"echoes  (tolerance={self.tolerance_km:.0f} km, "
                f"n_neighbors={self.n_neighbors})"
            )
        else:
            logger.debug("NeighborPredictedFilter: no echoes removed")

        out = df[keep].copy()
        out["npf_h_pred_km"] = pred_series[keep].values
        return out.reset_index(drop=True)


# ===========================================================================
# Option 3 — Column density / height-span filter
# ===========================================================================


class ColumnDensityFilter:
    """Remove RFI vertical stripes by detecting anomalously wide height spans.

    **Algorithm**

    For each frequency bin:
      1. Compute height span  Δh = h_max − h_min.
      2. If Δh > *rfi_span_km* → flag as RFI column.
      3. For flagged columns apply an IQR clip (rather than hard delete)
         so any real echo near the column's median height is preserved,
         while the RFI tails above and below are discarded.

    The IQR-clip strategy is the key improvement over naive deletion:
    a real trace echo at a contaminated frequency sits near the median
    height, so it survives clipping while the RFI tails do not.

    Args:
        rfi_span_km (float): Height-span threshold [km].  Columns with Δh > this are flagged.
            Default 150 km.  Real single-layer reflections rarely exceed
            50–80 km span; RFI typically spans 200–400 km.
        iqr_factor (float): Clip window half-width = iqr_factor × IQR around the median.
            Default 1.5  (matches Tukey fence).
        freq_bin (float): Frequency bin width.  Units must match the frequency column
            passed to ``clean()``; default 50000 (50 kHz) for raw RSF Hz
            values.
        hard_delete (bool): If True, completely remove all echoes in flagged columns (no IQR
            clip). Simpler, but loses any real echo at that frequency.
            Default False.
    """

    def __init__(
        self,
        rfi_span_km: float = 150.0,
        iqr_factor: float = 1.5,
        freq_bin: float = 0.05,
        hard_delete: bool = False,
    ) -> None:
        self.rfi_span_km = rfi_span_km
        self.iqr_factor = iqr_factor
        self.freq_bin = freq_bin
        self.hard_delete = hard_delete

    def clean(
        self,
        df: pd.DataFrame,
        freq_col: str = "frequency_mhz",
        height_col: str = "height_km",
        amp_col: str = "amplitude_db",
    ) -> pd.DataFrame:
        """Return DataFrame with vertical RFI stripes removed or clipped.

        Adds boolean column ``cdf_rfi_col`` marking echoes that were in a
        flagged column (True = came from an RFI column, survived IQR clip).
        """
        if df.empty:
            return df.copy()

        f_binned = (df[freq_col] / self.freq_bin).round() * self.freq_bin

        # Per-column height span
        span_series = (
            df.assign(_fbin=f_binned)
            .groupby("_fbin")[height_col]
            .agg(lambda x: float(x.max() - x.min()))
        )
        rfi_bins = set(span_series[span_series > self.rfi_span_km].index.tolist())

        if not rfi_bins:
            logger.debug("ColumnDensityFilter: no RFI columns detected")
            out = df.copy()
            out["cdf_rfi_col"] = False
            return out

        logger.info(
            f"ColumnDensityFilter: {len(rfi_bins)} RFI columns flagged "
            f"(span > {self.rfi_span_km:.0f} km)"
        )

        in_rfi = f_binned.isin(rfi_bins).values

        if self.hard_delete:
            keep = ~in_rfi
            logger.info(f"  hard_delete: removed {in_rfi.sum()} echoes")
            out = df[keep].copy()
            out["cdf_rfi_col"] = False
            return out.reset_index(drop=True)

        # ── IQR clip per flagged column ───────────────────────────────────────
        keep = np.ones(len(df), dtype=bool)
        for fbin in rfi_bins:
            col_idx = np.where(f_binned.values == fbin)[0]
            h_vals = df[height_col].values[col_idx]
            if len(h_vals) < 4:
                keep[col_idx] = False  # too few points — hard delete
                continue
            q25, q75 = np.percentile(h_vals, [25, 75])
            iqr = q75 - q25
            lo_clip = q25 - self.iqr_factor * iqr
            hi_clip = q75 + self.iqr_factor * iqr
            within = (h_vals >= lo_clip) & (h_vals <= hi_clip)
            keep[col_idx] = within

        n_removed = int((~keep).sum())
        logger.info(
            f"  IQR-clip: removed {n_removed} / {len(df)} echoes "
            f"(iqr_factor={self.iqr_factor})"
        )

        out = df[keep].copy()
        out["cdf_rfi_col"] = in_rfi[keep]
        return out.reset_index(drop=True)


# ===========================================================================
# Vertical RFI stripe filter — count-anomaly + neighbor-trace recovery
# ===========================================================================


class VerticalRfiFilter:
    """Remove vertical RFI stripes using O-mode echo count per frequency bin.

    **Algorithm**

    1. Select O-mode echoes (``pol == "O"``).
    2. Count echoes per frequency bin.
    3. Flag bins whose count exceeds ``max_echoes_per_bin``.
    4. Hard-delete ALL echoes (both polarizations) at flagged frequencies.

    Args:
        max_echoes_per_bin (int): Frequency bins with more O-mode echoes than this are treated as RFI
            and hard-deleted.  Real ionospheric traces have ~5–30 O-mode echoes
            per 50 kHz bin; RFI stripes typically have 50–128.  Default 40.
        freq_bin (float): Bin width in MHz. Default 0.05 MHz (50 kHz) for canonical data.
    """

    def __init__(
        self,
        max_echoes_per_bin: int = 40,
        freq_bin: float = 0.05,
    ) -> None:
        self.max_echoes_per_bin = max_echoes_per_bin
        self.freq_bin = freq_bin

    def clean(
        self,
        df: pd.DataFrame,
        freq_col: str = "frequency_mhz",
        height_col: str = "height_km",
        amp_col: str = "amplitude_db",
        pol_col: str = "mode",
    ) -> pd.DataFrame:
        """Return DataFrame with RFI frequency bins hard-deleted.

        After calling, ``self.o_counts_`` (Series, index = freq MHz) and
        ``self.rfi_freqs_mhz_`` (list of flagged MHz values) are available
        for inspection and plotting via ``plot_diagnostic()``.
        """
        if df.empty:
            self.o_counts_ = pd.Series(dtype=float)
            self.rfi_freqs_mhz_ = []
            return df.copy()

        # Canonical scaler frequency is already in MHz.
        f_binned_mhz = (df[freq_col] / self.freq_bin).round() * self.freq_bin

        # ── 1. Count O-mode echoes per frequency bin (in MHz) ─────────────────
        o_mask = (
            (df[pol_col] == "O")
            if pol_col in df.columns
            else pd.Series(True, index=df.index)
        )
        o_counts = f_binned_mhz[o_mask].value_counts().sort_index()
        self.o_counts_ = o_counts  # store for plot_diagnostic

        # ── 2. Flag bins above threshold ──────────────────────────────────────
        rfi_mhz = set(o_counts[o_counts > self.max_echoes_per_bin].index.tolist())
        self.rfi_freqs_mhz_ = sorted(rfi_mhz)

        if not rfi_mhz:
            logger.debug("VerticalRfiFilter: no RFI bins detected")
            return df.copy()

        logger.info(
            f"VerticalRfiFilter: {len(rfi_mhz)} RFI bins flagged "
            f"(O-mode count > {self.max_echoes_per_bin}): "
            f"{[round(f,3) for f in sorted(rfi_mhz)]} MHz"
        )

        # ── 3. Hard-delete all echoes (all polarizations) at flagged bins ─────
        # Compare in MHz to avoid float precision issues with large Hz values
        keep = ~f_binned_mhz.isin(rfi_mhz)
        n_removed = int((~keep).sum())
        logger.info(f"  removed {n_removed} / {len(df)} echoes")

        return df[keep].reset_index(drop=True)

    def plot_diagnostic(
        self,
        out_path: str | Path | None = None,
        time_label: str = "",
        freq_col: str = "frequency_mhz",
    ) -> "plt.Figure":
        """Bar chart of O-mode echo count vs frequency with threshold line.

        Call after ``clean()`` to visualise which bins were flagged.

        Args:
            out_path (Save PNG here when given.):
            time_label (Added to the figure title (e.g. sounding time string).):
        """
        import matplotlib.pyplot as plt

        if not hasattr(self, "o_counts_") or self.o_counts_.empty:
            logger.warning("plot_diagnostic: call clean() first")
            fig, ax = plt.subplots()
            ax.text(
                0.5,
                0.5,
                "No data — call clean() first",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            return fig

        counts = self.o_counts_.sort_index()
        freqs = counts.index.values  # MHz
        vals = counts.values

        fig, ax = plt.subplots(figsize=(10, 3.5), constrained_layout=True)

        # Bar colour: red if flagged, steelblue if clean
        colors = ["tomato" if f in self.rfi_freqs_mhz_ else "steelblue" for f in freqs]
        # ``freq_bin`` is already expressed in MHz for canonical data.
        ax.bar(
            freqs,
            vals,
            width=self.freq_bin * 0.8,
            color=colors,
            align="center",
            alpha=0.85,
        )

        # Threshold line
        ax.axhline(
            self.max_echoes_per_bin,
            color="red",
            lw=1.2,
            ls="--",
            label=f"threshold = {self.max_echoes_per_bin}",
        )

        ax.set_xlabel("Frequency (MHz)", fontsize=9)
        ax.set_ylabel("O-mode echo count\nper 50 kHz bin", fontsize=9)
        title = "VerticalRfiFilter — O-mode counts per frequency bin"
        if time_label:
            title += f"  [{time_label}]"
        ax.set_title(title, fontsize=9, loc="left")
        ax.legend(fontsize=8)
        ax.set_xlim(freqs.min() - 0.1, freqs.max() + 0.1)

        if out_path is not None:
            out_path = Path(out_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved: {out_path}")

        return fig


# ===========================================================================
# Adaptive amplitude gate — threshold detection from a time window of files
# ===========================================================================


class AdaptiveAmpGate:
    """Estimate the noise/signal amplitude threshold from a window of soundings.

    The amplitude distribution of a raw ionogram is bimodal: a noise/MPA
    bump at low dB and a real-echo bump at higher dB.  This class pools
    amplitudes from a user-supplied list of RSF / SBF / MMM files (typically
    2–3 hours of data), fits the chosen method to the pooled distribution,
    and returns both the threshold and the contiguous datetime window over
    which it is stable.

    **Methods**

    ``"otsu"``   — Otsu's inter-class variance maximisation (numpy only).
    ``"gmm"``    — 2-component Gaussian mixture; threshold = analytic
                   crossover of the two fitted Gaussians (requires sklearn).
    ``"kernel"`` — KDE valley search between the two dominant peaks
                   (requires scipy).

    **Usage**

    ::

        from pynasonde.digisonde.scaler import AdaptiveAmpGate

        gate = AdaptiveAmpGate(method="otsu", n_jobs=4)
        gate.fit(rsf_files)                  # list of file paths

        print(gate.threshold)                # e.g. 24.3 dB
        print(gate.time_window)              # (datetime(...), datetime(...))

        # Apply to a DataFrame:
        df_gated = df[df["amplitude_db"] >= gate.threshold]

    Args:
        method (str): ``"otsu"`` | ``"gmm"`` | ``"kernel"``.  Default ``"otsu"``.
        amp_col (str): Amplitude column name in the parser's ``to_pandas()`` output.
            Default ``"amplitude_db"`` (canonical scaler name).
        n_jobs (int): Parallel file-loading workers (joblib loky backend).  Default 4.
        fallback (float): Returned when detection fails or result is out of bounds.
            Default 27.0.
        n_bins (int): Histogram bin count for Otsu and the diagnostic plot.  Default 128.
        validity_tol_dB (float): A per-file threshold is considered "consistent" with the global
            threshold when |T_file − T_global| ≤ this value.  Used to shrink
            the reported time window when conditions drift.  Default 5.0 dB.
        amp_lo (float): Hard lower bound on the returned threshold.  Any detection
            outside ``[amp_lo, amp_hi]`` triggers the fallback.  RSF ``amplitude`` is
            an integer 0–127 SNR scale, so defaults are 1 / 60.
        amp_hi (float): Hard upper bound on the returned threshold (see ``amp_lo``).
    """

    def __init__(
        self,
        method: str = "otsu",
        amp_col: str = "amplitude_db",
        n_jobs: int = 4,
        fallback: float = 27.0,
        n_bins: int = 64,
        validity_tol_dB: float = 5.0,
        amp_lo: float = 1.0,
        amp_hi: float = 60.0,
    ) -> None:
        self.method = method
        self.amp_col = amp_col
        self.n_jobs = n_jobs
        self.fallback = fallback
        self.n_bins = n_bins
        self.validity_tol_dB = validity_tol_dB
        self.amp_lo = amp_lo
        self.amp_hi = amp_hi

        # Set after fit()
        self.threshold: float = np.nan
        self.time_window: tuple[dt.datetime, dt.datetime] | None = None
        self.per_file_df: pd.DataFrame = pd.DataFrame()
        self.pooled_amps_: np.ndarray = np.array([])

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def fit(self, files: list[str]) -> "AdaptiveAmpGate":
        """Load *files* in parallel, pool amplitudes, detect threshold.

        Args:
            files (list of paths to RSF / SBF / MMM files.):

        Returns:
            self — allows chaining: ``gate.fit(files).threshold``.:
        """
        if not files:
            logger.warning("AdaptiveAmpGate.fit: empty file list")
            self.threshold = self.fallback
            return self

        # ── 1. Load files in parallel ─────────────────────────────────────
        records = Parallel(n_jobs=self.n_jobs, backend="loky")(
            delayed(self._load_one)(f, self.amp_col) for f in files
        )
        records = [r for r in records if r is not None]

        if not records:
            logger.warning("AdaptiveAmpGate.fit: no data loaded — using fallback")
            self.threshold = self.fallback
            return self

        # Sort by time so per-file DataFrame is chronological
        records.sort(key=lambda r: r["time"])
        self.per_file_df = pd.DataFrame(
            [{"time": r["time"], "n_echoes": len(r["amps"])} for r in records]
        )

        # ── 2. Pool non-zero amplitudes → global threshold ────────────────
        # Canonical `amplitude_db` = SNR above instrument noise (0 = empty grid cell).
        # We only analyse real echoes (amplitude > 0) to find the threshold
        # separating weak/residual echoes from strong ionospheric returns.
        pooled = np.concatenate([r["amps"] for r in records])
        pooled = pooled[pooled > 0]  # drop empty grid cells
        self.pooled_amps_ = pooled
        logger.info(
            f"AdaptiveAmpGate: {len(records)} files, "
            f"{len(pooled):,} non-zero amplitude values pooled "
            f"(range {pooled.min():.0f}–{pooled.max():.0f})"
        )
        global_thresh = self._detect(pooled)

        if not (self.amp_lo <= global_thresh <= self.amp_hi):
            logger.warning(
                f"AdaptiveAmpGate ({self.method}): detected {global_thresh:.1f} dB "
                f"outside bounds [{self.amp_lo}, {self.amp_hi}] — "
                f"using fallback {self.fallback}"
            )
            self.threshold = self.fallback
        else:
            self.threshold = global_thresh
            logger.info(
                f"AdaptiveAmpGate ({self.method}): threshold = {self.threshold:.2f} dB"
            )

        # ── 3. Per-file thresholds → valid time window ────────────────────
        per_thresh = []
        for r in records:
            nz = r["amps"][r["amps"] > 0]
            t = self._detect(nz) if len(nz) >= 50 else np.nan
            per_thresh.append(t)

        self.per_file_df["threshold"] = per_thresh
        self.time_window = self._valid_window(
            [r["time"] for r in records],
            np.array(per_thresh),
            self.threshold,
        )
        logger.info(
            f"AdaptiveAmpGate: valid window "
            f"{self.time_window[0].strftime('%Y-%m-%d %H:%M')} – "
            f"{self.time_window[1].strftime('%H:%M')} UT"
        )
        return self

    # ------------------------------------------------------------------
    # Threshold detection methods
    # ------------------------------------------------------------------

    def _detect(self, vals: np.ndarray) -> float:
        """Route to the chosen method."""
        vals = vals[np.isfinite(vals)]
        if len(vals) < 50:
            return self.fallback
        if self.method == "otsu":
            return self._otsu(vals)
        if self.method == "gmm":
            return self._gmm(vals)
        if self.method == "kernel":
            return self._kernel(vals)
        raise ValueError(f"Unknown method '{self.method}'. Choose otsu | gmm | kernel.")

    def _otsu(self, vals: np.ndarray) -> float:
        """Otsu inter-class variance maximisation (numpy only)."""
        hist, edges = np.histogram(vals, bins=self.n_bins)
        centers = (edges[:-1] + edges[1:]) / 2.0
        total = hist.sum()
        if total == 0:
            return self.fallback

        best_thresh = centers[0]
        best_var = -np.inf
        w0 = mu0 = 0.0
        mu_total = float(np.dot(hist, centers)) / total

        for i in range(len(centers) - 1):
            w0 += hist[i]
            mu0 += hist[i] * centers[i]
            w1 = total - w0
            if w0 == 0 or w1 == 0:
                continue
            mu1 = (mu_total * total - mu0) / w1
            var = (w0 * w1 / total**2) * (mu0 / w0 - mu1) ** 2
            if var > best_var:
                best_var = var
                best_thresh = centers[i]

        return float(best_thresh)

    def _gmm(self, vals: np.ndarray) -> float:
        """2-component GMM — analytic crossover of the two Gaussians."""
        try:
            from sklearn.mixture import GaussianMixture
        except ImportError:
            logger.warning("sklearn not available — falling back to Otsu")
            return self._otsu(vals)
        try:
            gm = GaussianMixture(n_components=2, random_state=42, max_iter=300)
            gm.fit(vals.reshape(-1, 1))
        except Exception as exc:
            logger.warning(f"GMM failed ({exc}) — falling back to Otsu")
            return self._otsu(vals)

        means = gm.means_.flatten()
        stds = np.sqrt(gm.covariances_.flatten())
        lo, hi = int(np.argmin(means)), int(np.argmax(means))
        mu0, s0 = float(means[lo]), float(stds[lo])
        mu1, s1 = float(means[hi]), float(stds[hi])

        if np.isclose(s0, s1):
            return float((mu0 + mu1) / 2.0)

        # Weighted midpoint by inverse std — approximation of Gaussian crossover
        return float((mu0 * s1 + mu1 * s0) / (s0 + s1))

    def _kernel(self, vals: np.ndarray, n_eval: int = 512) -> float:
        """KDE valley detection — local minimum between the two main peaks."""
        try:
            from scipy.signal import find_peaks
            from scipy.stats import gaussian_kde
        except ImportError:
            logger.warning("scipy not available — falling back to Otsu")
            return self._otsu(vals)

        kde = gaussian_kde(vals, bw_method="scott")
        x = np.linspace(vals.min(), vals.max(), n_eval)
        density = kde(x)

        peaks, props = find_peaks(density, prominence=density.max() * 0.05)
        if len(peaks) < 2:
            logger.debug("kernel: < 2 peaks found — falling back to Otsu")
            return self._otsu(vals)

        # Two most prominent peaks
        top2 = peaks[np.argsort(props["prominences"])[-2:]]
        lo_peak, hi_peak = int(top2.min()), int(top2.max())
        valley_idx = lo_peak + int(np.argmin(density[lo_peak : hi_peak + 1]))
        return float(x[valley_idx])

    # ------------------------------------------------------------------
    # Valid time window
    # ------------------------------------------------------------------

    @staticmethod
    def _valid_window(
        times: list[dt.datetime],
        per_thresh: np.ndarray,
        global_thresh: float,
        tol: float = 5.0,
    ) -> tuple[dt.datetime, dt.datetime]:
        """Return the largest contiguous block of files where the per-file
        threshold is within *tol* dB of *global_thresh*.

        Falls back to the full time range when no per-file threshold is
        available (too few echoes) or when no stable block is found.
        """
        valid = np.abs(per_thresh - global_thresh) <= tol
        # Treat NaN per-file thresholds (sparse files) as valid — don't
        # shrink the window just because a single file had too few echoes
        valid[~np.isfinite(per_thresh)] = True

        if valid.all():
            return times[0], times[-1]

        # Find the longest contiguous run of True
        best_start = best_end = 0
        best_len = 0
        run_start = None
        for i, v in enumerate(valid):
            if v and run_start is None:
                run_start = i
            elif not v and run_start is not None:
                if i - run_start > best_len:
                    best_len = i - run_start
                    best_start = run_start
                    best_end = i - 1
                run_start = None
        if run_start is not None and len(valid) - run_start > best_len:
            best_len = len(valid) - run_start
            best_start = run_start
            best_end = len(valid) - 1

        if best_len == 0:
            # No stable block found — return full range
            return times[0], times[-1]

        return times[best_start], times[best_end]

    # ------------------------------------------------------------------
    # File loader (static — runs in worker processes)
    # ------------------------------------------------------------------

    @staticmethod
    def _load_one(fpath: str, amp_col: str) -> dict | None:
        """Load one RSF / SBF / MMM file and return its amplitudes + time."""
        ext = Path(fpath).suffix.upper()
        try:
            if ext == ".RSF":
                from pynasonde.digisonde.parsers.rsf import RsfExtractor

                reader = RsfExtractor(fpath, extract_time_from_name=True)
                reader.extract()
                df = normalize_echo_dataframe(reader.to_pandas(), source_format="rsf")
                time = getattr(reader, "date", None)
            elif ext == ".SBF":
                from pynasonde.digisonde.parsers.sbf import SbfExtractor

                reader = SbfExtractor(fpath, extract_time_from_name=True)
                reader.extract()
                df = normalize_echo_dataframe(reader.to_pandas(), source_format="sbf")
                time = getattr(reader, "date", None)
            elif ext == ".MMM":
                from pynasonde.digisonde.parsers.mmm import ModMaxExtractor

                reader = ModMaxExtractor(fpath, extract_time_from_name=True)
                reader.extract()
                df = normalize_echo_dataframe(reader.to_pandas(), source_format="mmm")
                time = getattr(reader, "date", None)
            else:
                logger.debug(f"AdaptiveAmpGate: unsupported format {ext} — {fpath}")
                return None

            if df.empty or amp_col not in df.columns:
                return None

            amps = df[amp_col].dropna().values.astype(float)
            if time is None:
                # Fall back: parse from filename
                time = AdaptiveAmpGate._time_from_name(fpath)
            return {"time": time, "amps": amps}

        except Exception as exc:
            logger.debug(f"AdaptiveAmpGate: failed to load {Path(fpath).name} — {exc}")
            return None

    @staticmethod
    def _time_from_name(fpath: str) -> dt.datetime:
        """Best-effort timestamp extraction from Digisonde filename."""
        try:
            stem = Path(fpath).stem
            date_part = stem.split("_")[-1]  # 2023287151418
            base = dt.datetime(int(date_part[:4]), 1, 1) + dt.timedelta(
                days=int(date_part[4:7]) - 1
            )
            return base.replace(
                hour=int(date_part[7:9]),
                minute=int(date_part[9:11]),
                second=int(date_part[11:13]),
            )
        except Exception:
            return dt.datetime(1970, 1, 1)

    # ------------------------------------------------------------------
    # Diagnostic plot
    # ------------------------------------------------------------------

    def plot_diagnostic(
        self,
        out_path: str | Path | None = None,
        n_bins: int | None = None,
    ) -> None:
        """Plot the pooled amplitude histogram with detected threshold overlaid.

        Also shows the per-file threshold time-series and the valid window
        (shaded) in a second panel.

        Args:
            out_path (save PNG here; calls ``plt.show()`` when None.):
            n_bins (histogram bins (defaults to ``self.n_bins``).):
        """
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt

        if np.isnan(self.threshold):
            logger.warning("plot_diagnostic: call fit() first")
            return

        bins = n_bins or self.n_bins
        fig, (ax_hist, ax_ts) = plt.subplots(
            1, 2, figsize=(10, 3.5), constrained_layout=True
        )

        # ── Panel left: pooled amplitude histogram ────────────────────────
        if len(self.pooled_amps_):
            ax_hist.hist(
                self.pooled_amps_,
                bins=bins,
                color="steelblue",
                alpha=0.75,
                edgecolor="none",
                label=f"Non-zero echoes  (n={len(self.pooled_amps_):,})",
            )
        ax_hist.axvline(
            self.threshold,
            color="red",
            lw=1.5,
            label=f"Adaptive ({self.method}): {self.threshold:.1f}",
        )
        if not np.isclose(self.threshold, self.fallback):
            ax_hist.axvline(
                self.fallback,
                color="grey",
                lw=1.0,
                ls="--",
                label=f"Fallback: {self.fallback:.1f}",
            )
        ax_hist.set_xlabel("Amplitude (SNR units)", fontsize=9)
        ax_hist.set_ylabel("Count", fontsize=9)
        ax_hist.set_title("Pooled amplitude distribution", fontsize=9, loc="left")
        ax_hist.legend(fontsize=8)

        # ── Panel right: per-file threshold time-series ───────────────────
        if not self.per_file_df.empty and "threshold" in self.per_file_df.columns:
            times = self.per_file_df["time"].tolist()
            threshs = self.per_file_df["threshold"].values

            ax_ts.plot(
                times,
                threshs,
                "o-",
                ms=3,
                lw=0.8,
                color="steelblue",
                label="Per-file threshold",
            )
            ax_ts.axhline(
                self.threshold,
                color="red",
                lw=1.2,
                label=f"Global: {self.threshold:.1f} dB",
            )
            ax_ts.axhline(
                self.threshold + self.validity_tol_dB, color="red", lw=0.6, ls=":"
            )
            ax_ts.axhline(
                self.threshold - self.validity_tol_dB, color="red", lw=0.6, ls=":"
            )

            # Shade valid window
            if self.time_window is not None:
                ax_ts.axvspan(
                    self.time_window[0],
                    self.time_window[1],
                    color="green",
                    alpha=0.12,
                    label="Valid window",
                )

            ax_ts.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
            ax_ts.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax_ts.set_xlabel("UT", fontsize=9)
            ax_ts.set_ylabel("Threshold (dB)", fontsize=9)
            ax_ts.set_title("Per-file stability", fontsize=9, loc="left")
            ax_ts.legend(fontsize=8)
            fig.autofmt_xdate(rotation=30)

        if out_path:
            fig.savefig(out_path, dpi=150, bbox_inches="tight")
            logger.info(f"Saved diagnostic: {out_path}")
        else:
            plt.show()
        plt.close(fig)
