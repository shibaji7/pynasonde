from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.cluster import DBSCAN
from sklearn.linear_model import RANSACRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures

from pynasonde.digisonde.scaler.schema import normalize_echo_dataframe

# ---------------------------------------------------------------------------
# Layer height windows (km)
# ---------------------------------------------------------------------------
_ES_H_LO, _ES_H_HI = 90.0, 130.0
_E_H_LO, _E_H_HI = 90.0, 160.0
_F1_H_LO, _F1_H_HI = 160.0, 250.0
_F2_H_LO, _F2_H_HI = 160.0, 800.0


# ===========================================================================
# Output dataclass
# ===========================================================================


@dataclass
class IonoscaleResult:
    """Scaled ionospheric parameters for one ionogram sounding.

    All frequency parameters are in MHz; all height parameters in km.
    NaN signals the parameter was not detected or could not be computed.
    """

    foE_mhz: float = np.nan
    h_prime_E_km: float = np.nan
    foEs_mhz: float = np.nan
    h_prime_Es_km: float = np.nan
    foF1_mhz: float = np.nan
    h_prime_F1_km: float = np.nan
    foF2_mhz: float = np.nan
    h_prime_F2_km: float = np.nan
    quality_flags: Dict[str, bool] = field(default_factory=dict)
    trace_metrics: Dict[str, float] = field(default_factory=dict)

    def to_series(self, prefix: str | None = None) -> pd.Series:
        d = {
            "foE_mhz": self.foE_mhz,
            "h_prime_E_km": self.h_prime_E_km,
            "foEs_mhz": self.foEs_mhz,
            "h_prime_Es_km": self.h_prime_Es_km,
            "foF1_mhz": self.foF1_mhz,
            "h_prime_F1_km": self.h_prime_F1_km,
            "foF2_mhz": self.foF2_mhz,
            "h_prime_F2_km": self.h_prime_F2_km,
            "hmF2_km": self.hmF2_km if hasattr(self, "hmF2_km") else np.nan,
            "hmF2_shimazaki": (
                self.hmF2_shimazaki if hasattr(self, "hmF2_shimazaki") else np.nan
            ),
        }
        d.update(self.trace_metrics)
        d.update({f"flag_{k}": v for k, v in self.quality_flags.items()})
        if prefix is not None:
            d = {f"{prefix}_{k}": v for k, v in d.items()}
        return pd.Series(d)

    def summary(self) -> str:
        return (
            f"foE={self.foE_mhz:.2f}  foEs={self.foEs_mhz:.2f}  "
            f"foF1={self.foF1_mhz:.2f}  foF2={self.foF2_mhz:.2f} MHz  "
            f"h'E={self.h_prime_E_km:.0f}  h'F2={self.h_prime_F2_km:.0f} km"
        )


# ===========================================================================
# Base class — shared cleaning + layer utilities
# ===========================================================================


class IonoscaleBase:
    """Shared data-preparation and parameter-extraction utilities.

    Both ``IonogramAutoScaler`` and ``ViterbiTracer`` inherit from this
    class.  It provides:

    * Column normalisation  (``_normalise``)
    * O-mode / amplitude filtering  (``_select_o_mode``)
    * DBSCAN noise removal  (``_dbscan_clean``)
    * Amplitude gate        (``_amplitude_gate``)
    * Layer band selection  (``_band`` — static)
    * fo / h' extraction    (``_fo``, ``_h_prime`` — static)
    * IonoscaleResult builder (``_build_result`` — static)
    * ``clean(raw_df)``     — runs the full pipeline (normalise → O-mode →
                              DBSCAN → amplitude gate) and stores the result
                              in ``self.cleaned_``
    """

    # Optional parser-column overrides passed through to normalize_echo_dataframe().
    _DEFAULT_COLS = {}

    def __init__(
        self,
        dbscan_eps: float = 0.04,
        dbscan_min_samples: int = 4,
        min_echoes: int = 5,
        min_amp_dB: float = 0.0,
        mode: str | None = "O",
        col_map: Optional[dict] = None,
        rfi_filter: object | None = None,
    ) -> None:
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.min_echoes = min_echoes
        self.min_amp_dB = min_amp_dB
        self.mode = mode.upper() if isinstance(mode, str) else mode
        self.rfi_filter = rfi_filter
        self._cols = dict(self._DEFAULT_COLS)
        if col_map:
            self._cols.update(col_map)
        self.cleaned_: pd.DataFrame = pd.DataFrame()

    # ------------------------------------------------------------------
    # Column normalisation
    # ------------------------------------------------------------------

    def _normalise(self, df: pd.DataFrame) -> pd.DataFrame:
        """Return a copy with canonical scaler column names."""
        return normalize_echo_dataframe(df, col_map=self._cols or None)

    def run_rfi_filter(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """Apply the configured RFI filter if one was provided.

        The filter object is user-created and only needs to provide a
        ``clean(df, **kwargs)`` method.  Additional keyword arguments are
        forwarded directly to that method.
        """
        if self.rfi_filter is None:
            return df
        if not hasattr(self.rfi_filter, "clean"):
            raise TypeError("rfi_filter must provide a clean(df, ...) method")
        return self.rfi_filter.clean(df, **kwargs)

    # ------------------------------------------------------------------
    # O-mode filter + amplitude threshold
    # ------------------------------------------------------------------

    def _select_o_mode(self, df: pd.DataFrame) -> pd.DataFrame:
        o = (
            df[df["mode"] == self.mode].copy()
            if self.mode and "mode" in df.columns
            else df.copy()
        )
        if "amplitude_db" in o.columns and "mpa_db" in o.columns:
            o = o[o["amplitude_db"] >= o["mpa_db"]]
        return o.reset_index(drop=True)

    # ------------------------------------------------------------------
    # DBSCAN noise removal
    # ------------------------------------------------------------------

    def _dbscan_clean(self, o: pd.DataFrame) -> pd.DataFrame:
        if len(o) < self.dbscan_min_samples:
            return o
        f_norm = (o["frequency_mhz"] - o["frequency_mhz"].min()) / (
            o["frequency_mhz"].max() - o["frequency_mhz"].min() + 1e-9
        )
        h_norm = (o["height_km"] - o["height_km"].min()) / (
            o["height_km"].max() - o["height_km"].min() + 1e-9
        )
        X = np.column_stack([f_norm, h_norm])
        labels = DBSCAN(
            eps=self.dbscan_eps, min_samples=self.dbscan_min_samples
        ).fit_predict(X)
        cleaned = o.copy()
        cleaned["labels"] = labels
        cleaned = cleaned[labels != -1].copy()
        n_removed = (labels == -1).sum()
        if n_removed:
            logger.debug(f"DBSCAN removed {n_removed} noise echoes")
        return cleaned.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Amplitude gate
    # ------------------------------------------------------------------

    def _amplitude_gate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop echoes below ``self.min_amp_dB``; no-op when threshold is 0.

        Updates and returns the filtered DataFrame.  Logs a warning when
        the result is empty so callers can decide how to handle it.
        """
        if self.min_amp_dB <= 0 or "amplitude_db" not in df.columns:
            return df
        n_before = len(df)
        out = df[df["amplitude_db"] >= self.min_amp_dB].copy()
        n_removed = n_before - len(out)
        if n_removed:
            logger.info(
                f"Amplitude gate {self.min_amp_dB:.1f} dB: "
                f"{n_before} → {len(out)} echoes (removed {n_removed})"
            )
        if out.empty:
            logger.warning(
                f"All echoes below amplitude threshold "
                f"({self.min_amp_dB:.1f} dB) — cleaned_ will be empty"
            )
        return out.reset_index(drop=True)

    # ------------------------------------------------------------------
    # Shared cleaning pipeline
    # ------------------------------------------------------------------

    def clean(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        """Normalise → O-mode filter → DBSCAN → amplitude gate.

        Stores result in ``self.cleaned_`` and returns it.
        Raises nothing — returns empty DataFrame on failure.
        """
        df = self._normalise(raw_df)
        o = self._select_o_mode(df)
        if len(o) < self.min_echoes:
            logger.warning(
                f"Only {len(o)} O-mode echoes above noise floor — "
                "cleaned_ will be empty"
            )
            self.cleaned_ = pd.DataFrame()
            return self.cleaned_
        dbscaned = self._dbscan_clean(o)
        self.cleaned_ = self._amplitude_gate(dbscaned)
        logger.info(
            f"clean(): {len(raw_df)} raw → {len(o)} O-mode → "
            f"{len(dbscaned)} DBSCAN → {len(self.cleaned_)} after amp gate"
        )
        return self.cleaned_

    # ------------------------------------------------------------------
    # Layer band selection
    # ------------------------------------------------------------------

    @staticmethod
    def _band(
        df: pd.DataFrame,
        h_lo: float,
        h_hi: float,
        f_lo: float = 0.0,
        f_hi: float = 99.0,
    ) -> pd.DataFrame:
        return df[
            df["height_km"].between(h_lo, h_hi)
            & df["frequency_mhz"].between(f_lo, f_hi)
        ].copy()

    # ------------------------------------------------------------------
    # Parameter extraction
    # ------------------------------------------------------------------

    @staticmethod
    def _fo(echoes: pd.DataFrame, h_rise_km: float = 80.0) -> float:
        """Critical frequency — last f before the trace rises > h_rise_km."""
        if echoes.empty:
            return np.nan
        grp = (
            echoes.groupby("frequency_mhz")["height_km"]
            .min()
            .reset_index()
            .sort_values("frequency_mhz")
        )
        if len(grp) < 3:
            return float(grp["frequency_mhz"].max())
        dh = np.diff(grp["height_km"].values)
        steep = np.where(dh > h_rise_km)[0]
        if steep.size:
            return float(grp["frequency_mhz"].values[int(steep[-1])])
        return float(grp["frequency_mhz"].values[-1])

    @staticmethod
    def _h_prime(echoes: pd.DataFrame) -> float:
        """Minimum virtual height of the layer (classical ionosonde definition)."""
        return float(echoes["height_km"].min()) if not echoes.empty else np.nan

    # ------------------------------------------------------------------
    # IonoscaleResult builder
    # ------------------------------------------------------------------

    @staticmethod
    def _build_result(
        layers: dict[str, pd.DataFrame],
        quality_flags: dict | None = None,
        trace_metrics: dict | None = None,
    ) -> IonoscaleResult:
        """Build IonoscaleResult from a dict of per-layer DataFrames.

        Each DataFrame must have columns ``frequency_mhz`` and ``height_km``.
        fo  = max frequency on the layer trace / inlier set.
        h'  = minimum virtual height of the layer (classical ionosonde definition).

        Parameters
        ----------
        layers : dict with keys ``"E"``, ``"Es"``, ``"F1"``, ``"F2"``
            (missing keys → NaN for that layer).
        quality_flags : optional override; auto-generated from emptiness
            when None.
        """

        def fo(df):
            return float(df["frequency_mhz"].max()) if not df.empty else np.nan

        def hp(df):
            return IonoscaleBase._h_prime(df)

        e = layers.get("E", pd.DataFrame())
        es = layers.get("Es", pd.DataFrame())
        f1 = layers.get("F1", pd.DataFrame())
        f2 = layers.get("F2", pd.DataFrame())

        flags = quality_flags or {
            "E_detected": not e.empty,
            "Es_detected": not es.empty,
            "F1_detected": not f1.empty,
            "F2_detected": not f2.empty,
        }
        return IonoscaleResult(
            foE_mhz=fo(e),
            h_prime_E_km=hp(e),
            foEs_mhz=fo(es),
            h_prime_Es_km=hp(es),
            foF1_mhz=fo(f1),
            h_prime_F1_km=hp(f1),
            foF2_mhz=fo(f2),
            h_prime_F2_km=hp(f2),
            quality_flags=flags,
            trace_metrics=trace_metrics or {},
        )


# ===========================================================================
# RANSAC-based auto-scaler
# ===========================================================================


class IonogramAutoScaler(IonoscaleBase):
    """Robust ionogram auto-scaler using DBSCAN + RANSAC.

    Inherits cleaning pipeline from ``IonoscaleBase``.  Adds RANSAC
    polynomial fitting per layer band to isolate the ionospheric trace
    from residual interference.

    Args:
    dbscan_eps, dbscan_min_samples, min_echoes, col_map
        Passed to ``IonoscaleBase``.
    ransac_residual_km : float
        RANSAC inlier threshold [km].  Default 20.
    ransac_poly_degree : int
        Polynomial degree for RANSAC trace fit.  Default 2.
    f1_threshold_mhz : float
        foF1 must exceed foE by at least this margin [MHz].  Default 0.3.
    """

    def __init__(
        self,
        dbscan_eps: float = 0.04,
        dbscan_min_samples: int = 4,
        min_echoes: int = 5,
        min_amp_dB: float = 0.0,
        mode: str | None = "O",
        ransac_residual_km: float = 20.0,
        ransac_poly_degree: int = 2,
        f1_threshold_mhz: float = 0.3,
        col_map: Optional[dict] = None,
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
        self.ransac_residual = ransac_residual_km
        self.ransac_degree = ransac_poly_degree
        self.f1_threshold = f1_threshold_mhz

    # ------------------------------------------------------------------
    # RANSAC inlier fitting
    # ------------------------------------------------------------------

    def _ransac_inliers(self, echoes: pd.DataFrame) -> pd.DataFrame:
        if len(echoes) < self.min_echoes:
            return echoes
        X = echoes["frequency_mhz"].to_numpy().reshape(-1, 1)
        y = echoes["height_km"].to_numpy()
        try:
            model = make_pipeline(
                PolynomialFeatures(self.ransac_degree),
                RANSACRegressor(
                    min_samples=max(0.3, self.min_echoes / len(echoes)),
                    residual_threshold=self.ransac_residual,
                    random_state=42,
                    max_trials=200,
                ),
            )
            model.fit(X, y)
            mask = model[-1].inlier_mask_
            n_out = (~mask).sum()
            if n_out:
                logger.debug(f"RANSAC removed {n_out} outliers")
            return echoes[mask].reset_index(drop=True)
        except Exception as exc:
            logger.warning(f"RANSAC failed ({exc}); using all echoes")
            return echoes

    # ------------------------------------------------------------------
    # F1 detection
    # ------------------------------------------------------------------

    def _detect_f1(
        self, cleaned: pd.DataFrame, fo_e: float, fo_f2: float
    ) -> Tuple[float, float]:
        if np.isnan(fo_e) or np.isnan(fo_f2):
            return np.nan, np.nan
        f_lo, f_hi = fo_e + self.f1_threshold, fo_f2 - self.f1_threshold
        if f_lo >= f_hi:
            return np.nan, np.nan
        window = self._band(cleaned, _F1_H_LO, _F1_H_HI, f_lo, f_hi)
        if len(window) < self.min_echoes:
            return np.nan, np.nan
        grp = (
            window.groupby("frequency_mhz")["height_km"]
            .min()
            .reset_index()
            .sort_values("frequency_mhz")
        )
        if len(grp) < 3:
            return np.nan, np.nan
        cusp = int(np.argmin(np.gradient(grp["height_km"].values)))
        return float(grp["frequency_mhz"].iloc[cusp]), float(
            grp["height_km"].iloc[cusp]
        )

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def fit(
        self,
        df: pd.DataFrame,
        already_cleaned: bool = False,
        layers: list[dict] = [
            dict(name="E", h_lo=_E_H_LO, h_hi=_E_H_HI, f_lo=1.0, f_hi=5.0),
            dict(name="F2", h_lo=_F2_H_LO, h_hi=_F2_H_HI, f_lo=0.0, f_hi=99.0),
        ],
    ) -> IonoscaleResult:
        """Clean (optional) + RANSAC-fit per layer → IonoscaleResult.

        Parameters
        ----------
        df : pd.DataFrame
            Raw data (``already_cleaned=False``) or pre-cleaned
            ``IonogramAutoScaler.cleaned_`` / ``IonoscaleBase.cleaned_``
            (``already_cleaned=True``).
        already_cleaned : bool
            Skip the cleaning pipeline when True.
        layers : list of dicts
            Per-layer band definitions with keys:
            ``name``, ``h_lo``, ``h_hi``, ``f_lo``, ``f_hi``.
            F1 is always detected automatically between foE and foF2.

        Returns:
        IonoscaleResult — cleaned data available via ``self.cleaned_``.
        """
        cleaned = df if already_cleaned else self.clean(df)
        if cleaned.empty:
            return IonoscaleResult(quality_flags={"insufficient_data": True})
        cleaned = self.run_rfi_filter(cleaned, pol_col="mode")
        self.cleaned_ = cleaned
        if cleaned.empty:
            return IonoscaleResult(quality_flags={"insufficient_data": True})

        # RANSAC per declared layer band
        inliers: dict[str, pd.DataFrame] = {}
        for layer in layers:
            band = self._band(
                cleaned,
                layer["h_lo"],
                layer["h_hi"],
                f_lo=layer.get("f_lo", 0.0),
                f_hi=layer.get("f_hi", 99.0),
            )
            inliers[layer["name"]] = self._ransac_inliers(band)

        # F1 always derived from foE / foF2 — not a free band
        fo_e = self._fo(inliers.get("E", pd.DataFrame()))
        fo_f2 = self._fo(inliers.get("F2", pd.DataFrame()))
        fo_f1, hp_f1 = self._detect_f1(cleaned, fo_e, fo_f2)
        inliers["F1"] = _make_f1_df(fo_f1, hp_f1)

        result = self._build_result(inliers)
        logger.info(f"RANSAC result — {result.summary()}")
        return result


def _make_f1_df(fo_f1: float, hp_f1: float) -> pd.DataFrame:
    """Wrap scalar F1 values into a minimal DataFrame for _build_result."""
    if np.isnan(fo_f1) or np.isnan(hp_f1):
        return pd.DataFrame()
    return pd.DataFrame({"frequency_mhz": [fo_f1], "height_km": [hp_f1]})


def scale_dataframe(
    df: pd.DataFrame,
    *,
    mode: str | None = "O",
    method: str = "viterbi",
    source_format: str = "auto",
    scaler_kwargs: dict | None = None,
    layers: list[dict] | None = None,
    return_details: bool = False,
) -> IonoscaleResult | tuple[IonoscaleResult, dict]:
    """Scale one ionogram/sounding represented as an echo DataFrame.

    Args:
        df: RSF, SBF, MMM, or canonical echo DataFrame for one timestamp.
        mode: Echo mode to scale, usually ``"O"`` or ``"X"``. Use None to keep
            all modes.
        method: ``"viterbi"`` or ``"ransac"`` or ``"ransac|viterbi"``.
        source_format: Parser schema passed to ``normalize_echo_dataframe``.
        scaler_kwargs: Keyword arguments for the selected scaler class.
        layers: Optional layer configuration list for the selected scaler.
        return_details: If True, return a details dictionary with normalized,
            cleaned, and trace/inlier DataFrames.

    Returns:
        ``IonoscaleResult`` by default. If ``return_details=True``, returns
        ``(result, details)``.
    """
    scaler_kwargs = dict(scaler_kwargs or {})
    norm = normalize_echo_dataframe(df, source_format=source_format, mode=mode)
    if norm.empty:
        result = IonoscaleResult(quality_flags={"insufficient_data": True})
        details = {
            "normalized": norm,
            "cleaned": pd.DataFrame(),
            "traces": {},
            "trace_metrics": {},
        }
        return (result, details) if return_details else result

    method = method.lower()
    if method == "ransac":
        scaler = IonogramAutoScaler(mode=mode, **scaler_kwargs)
        fit_kwargs = {"already_cleaned": False}
        if layers is not None:
            fit_kwargs["layers"] = layers
        result = scaler.fit(norm, **fit_kwargs)
        details = {
            "normalized": norm,
            "cleaned": scaler.cleaned_,
            "traces": {},
            "trace_metrics": {},
            "scaler": scaler,
        }
    elif method == "viterbi":
        from pynasonde.digisonde.scaler.viterbi import ViterbiTracer

        tracer = ViterbiTracer(mode=mode, **scaler_kwargs)
        fit_kwargs = {"already_cleaned": False}
        if layers is not None:
            fit_kwargs["layers"] = layers
        result, traces = tracer.fit(norm, **fit_kwargs)
        details = {
            "normalized": norm,
            "cleaned": tracer.cleaned_,
            "traces": traces,
            "trace_metrics": tracer.trace_metrics_,
            "scaler": tracer,
        }
    else:
        raise ValueError("method must be 'viterbi' or 'ransac'")

    return (result, details) if return_details else result


def _merge_preset_results(
    e_result: IonoscaleResult | None,
    f2_result: IonoscaleResult,
) -> IonoscaleResult:
    """Merge layer-specific scaler results into one output object."""
    result = IonoscaleResult(
        foE_mhz=e_result.foE_mhz if e_result is not None else np.nan,
        h_prime_E_km=e_result.h_prime_E_km if e_result is not None else np.nan,
        foEs_mhz=e_result.foEs_mhz if e_result is not None else np.nan,
        h_prime_Es_km=e_result.h_prime_Es_km if e_result is not None else np.nan,
        foF1_mhz=f2_result.foF1_mhz,
        h_prime_F1_km=f2_result.h_prime_F1_km,
        foF2_mhz=f2_result.foF2_mhz,
        h_prime_F2_km=f2_result.h_prime_F2_km,
    )
    quality_flags = {}
    if e_result is None:
        quality_flags["E_layer_disabled"] = True
    else:
        quality_flags.update({f"E_{k}": v for k, v in e_result.quality_flags.items()})
        result.trace_metrics.update(e_result.trace_metrics)
    quality_flags.update({f"F2_{k}": v for k, v in f2_result.quality_flags.items()})
    result.quality_flags = quality_flags
    result.trace_metrics.update(f2_result.trace_metrics)
    for attr in ("hmF2_km", "hmF2_shimazaki"):
        if hasattr(f2_result, attr):
            setattr(result, attr, getattr(f2_result, attr))
    return result


def _concat_cleaned_by_layer(layer_details: dict[str, dict]) -> pd.DataFrame:
    """Return one cleaned DataFrame with a ``scaler_layer`` column."""
    frames = []
    for name, details in layer_details.items():
        cleaned = details.get("cleaned", pd.DataFrame())
        if cleaned.empty:
            continue
        frame = cleaned.copy()
        frame["scaler_layer"] = name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def scale_dataframe_with_preset(
    df: pd.DataFrame,
    *,
    time=None,
    lat_deg: float | None = None,
    lon_deg: float | None = None,
    sza_deg: float | None = None,
    regime: str | None = None,
    tuning_preset: str = "balanced",
    e_layer_max_sza: float = 100.0,
    mode: str | None = "O",
    source_format: str = "auto",
    rfi_filter: object | None = None,
    return_details: bool = False,
) -> IonoscaleResult | tuple[IonoscaleResult, dict]:
    """Scale one ionogram using SZA-aware layer-specific Viterbi presets.

    Args:
        df: RSF, SBF, MMM, or canonical echo DataFrame for one timestamp.
        time: Sounding time used for SZA when ``sza_deg`` is not provided.
        lat_deg: Station latitude in degrees north.
        lon_deg: Station longitude in degrees east; west is negative.
        sza_deg: Optional precomputed solar zenith angle.
        regime: Optional explicit ``day``, ``twilight``, or ``night`` regime.
        tuning_preset: Preset modifier name.
        e_layer_max_sza: Skip normal E-layer scaling at or above this SZA.
        mode: Echo mode to scale, usually ``"O"``.
        source_format: Parser schema passed to ``normalize_echo_dataframe``.
        rfi_filter: Optional user-created RFI filter object.
        return_details: If True, also return details and per-layer traces.

    Returns:
        Merged :class:`IonoscaleResult` by default. If ``return_details=True``,
        returns ``(result, details)``.
    """
    from pynasonde.digisonde.scaler.presets import get_viterbi_preset

    normalized = normalize_echo_dataframe(df, source_format=source_format)
    if (
        time is None
        and "time" in normalized.columns
        and normalized["time"].notna().any()
    ):
        time = normalized["time"].dropna().iloc[0]

    preset = get_viterbi_preset(
        time=time,
        lat_deg=lat_deg,
        lon_deg=lon_deg,
        sza_deg=sza_deg,
        regime=regime,
        tuning_preset=tuning_preset,
        e_layer_max_sza=e_layer_max_sza,
        rfi_filter=rfi_filter,
    )

    layer_results: dict[str, IonoscaleResult | None] = {"E": None}
    layer_details: dict[str, dict] = {}
    traces: dict[str, pd.DataFrame] = {}

    if preset.enabled["E"]:
        e_result, e_details = scale_dataframe(
            df,
            mode=mode,
            method="viterbi",
            source_format=source_format,
            layers=preset.layer("E"),
            scaler_kwargs=preset.kwargs("E"),
            return_details=True,
        )
        layer_results["E"] = e_result
        layer_details["E"] = e_details
        traces["E"] = e_details["traces"].get("E", pd.DataFrame())

    f2_result, f2_details = scale_dataframe(
        df,
        mode=mode,
        method="viterbi",
        source_format=source_format,
        layers=preset.layer("F2"),
        scaler_kwargs=preset.kwargs("F2"),
        return_details=True,
    )
    layer_results["F2"] = f2_result
    layer_details["F2"] = f2_details
    traces["F2"] = f2_details["traces"].get("F2", pd.DataFrame())

    result = _merge_preset_results(layer_results["E"], f2_result)
    if return_details:
        details = {
            "normalized": normalized,
            "cleaned": _concat_cleaned_by_layer(layer_details),
            "traces": traces,
            "trace_metrics": result.trace_metrics,
            "preset": preset,
            "per_layer": layer_details,
        }
        return result, details
    return result


def scale_dataframes_with_preset(
    dataframes: list[pd.DataFrame],
    *,
    times: list | None = None,
    lat_deg: float | None = None,
    lon_deg: float | None = None,
    sza_deg: list[float] | None = None,
    regime: str | None = None,
    tuning_preset: str = "balanced",
    e_layer_max_sza: float = 100.0,
    mode: str | None = "O",
    source_format: str = "auto",
    rfi_filter: object | None = None,
) -> pd.DataFrame:
    """Scale multiple ionogram DataFrames with SZA-aware presets."""
    rows = []
    times = [None] * len(dataframes) if times is None else list(times)
    sza_values = [None] * len(dataframes) if sza_deg is None else list(sza_deg)
    if len(times) != len(dataframes):
        raise ValueError("times must have the same length as dataframes")
    if len(sza_values) != len(dataframes):
        raise ValueError("sza_deg must have the same length as dataframes")

    for df, time_value, sza_value in zip(dataframes, times, sza_values):
        result, details = scale_dataframe_with_preset(
            df,
            time=time_value,
            lat_deg=lat_deg,
            lon_deg=lon_deg,
            sza_deg=sza_value,
            regime=regime,
            tuning_preset=tuning_preset,
            e_layer_max_sza=e_layer_max_sza,
            mode=mode,
            source_format=source_format,
            rfi_filter=rfi_filter,
            return_details=True,
        )
        row = result.to_series()
        row["mode"] = mode
        row["method"] = "viterbi"
        row["sza_deg"] = details["preset"].sza_deg
        row["sza_regime"] = details["preset"].regime
        row["tuning_preset"] = details["preset"].tuning_preset
        row["e_layer_enabled"] = details["preset"].enabled["E"]
        for key, value in details["preset"].metadata.items():
            row[key] = value
        if time_value is not None:
            row["time"] = pd.to_datetime(time_value)
        rows.append(row)
    return pd.DataFrame(rows)


def scale_dataframes(
    dataframes: list[pd.DataFrame],
    *,
    mode: str | None = "O",
    method: str = "viterbi",
    source_format: str = "auto",
    scaler_kwargs: dict | None = None,
    layers: list[dict] | None = None,
) -> pd.DataFrame:
    """Scale a sequence of ionogram/sounding DataFrames.

    Args:
        dataframes: One DataFrame per timestamp.
        mode: Echo mode to scale.
        method: ``"viterbi"`` or ``"ransac"``.
        source_format: Parser schema or ``"auto"``.
        scaler_kwargs: Keyword arguments for the selected scaler class.
        layers: Optional layer configuration list.

    Returns:
        DataFrame with one scaled-parameter row per input DataFrame.
    """
    rows = []
    for df in dataframes:
        result, details = scale_dataframe(
            df,
            mode=mode,
            method=method,
            source_format=source_format,
            scaler_kwargs=scaler_kwargs,
            layers=layers,
            return_details=True,
        )
        row = result.to_series()
        norm = details["normalized"]
        if not norm.empty and "time" in norm.columns:
            row["time"] = (
                norm["time"].dropna().iloc[0] if norm["time"].notna().any() else pd.NaT
            )
        row["mode"] = mode
        row["method"] = method
        rows.append(row)
    return pd.DataFrame(rows)
