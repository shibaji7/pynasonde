"""polarization.py — O/X wave-mode separation via the PP polarization parameter.

The PP parameter (``polarization_deg`` in :class:`~pynasonde.vipir.riq.echo.Echo`)
measures the chirality of the reflected wavefront, estimated from the differential
phase between quasi-orthogonal antenna pairs.

For a vertically incident HF signal the ionosphere reflects two magneto-ionic
modes: the **ordinary (O) mode** and the **extraordinary (X) mode**.  Their
reflected polarizations have opposite chirality, which maps to opposite signs
of PP.  The exact sign that corresponds to O vs X depends on:

* The orientation of the local geomagnetic field (sign of the vertical component
  Bz), which flips between the northern and southern magnetic hemispheres.
* The physical layout and wiring of the receiver antenna array.

This module provides:

:class:`PolarizationClassifier`
    Processor — labels each echo "O", "X", "ambiguous", or "unknown" by
    thresholding PP and applying a configurable sign convention.

:class:`PolarizationResult`
    Output dataclass — holds the annotated DataFrame and summary statistics.

References
----------
Zabotin, N. A., Wright, J. W., & Zhbankov, G. A. (2006). NeXtYZ: Three-
dimensional electron density inversion for Dynasonde and ARTIST ionosondes.
*Radio Science*, 41, RS6S32.

Wright, J. W. & Pitteway, M. L. V. (1994). A numerical study of the estimation
of the wave polarization in the ionospheric HF reflection region. *Journal of
Atmospheric and Terrestrial Physics*, 56, 577-585.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_PP_COL_DEFAULT = "polarization_deg"
_MODE_COL = "mode"

_MODE_O = "O"
_MODE_X = "X"
_MODE_AMBIGUOUS = "ambiguous"
_MODE_UNKNOWN = "unknown"  # PP is NaN (< 3 receivers)


# ===========================================================================
# Output dataclass
# ===========================================================================


@dataclass
class PolarizationResult:
    """O/X classification results for one ionogram sounding.

    Parameters
    ----------
    annotated_df:
        Copy of the input echo DataFrame with a new ``"mode"`` column added.
        Values are one of ``"O"``, ``"X"``, ``"ambiguous"``, ``"unknown"``.
    o_mode_count:
        Number of echoes labelled O-mode.
    x_mode_count:
        Number of echoes labelled X-mode.
    ambiguous_count:
        Number of echoes whose |PP| fell below ``pp_ambiguous_threshold_deg``
        (near-linear polarization; mode cannot be determined reliably).
    unknown_count:
        Number of echoes with NaN PP (fewer than ``min_rx_for_direction``
        receivers used in the spatial fit).
    o_mode_sign:
        Sign convention used: ``+1`` means positive PP → O-mode,
        ``-1`` means negative PP → O-mode.
    pp_ambiguous_threshold_deg:
        |PP| threshold below which echoes were labelled ``"ambiguous"``.
    """

    annotated_df: pd.DataFrame
    o_mode_count: int
    x_mode_count: int
    ambiguous_count: int
    unknown_count: int
    o_mode_sign: int
    pp_ambiguous_threshold_deg: float

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def to_dataframe(self) -> pd.DataFrame:
        """Return the annotated echo DataFrame."""
        return self.annotated_df.copy()

    def o_mode_df(self) -> pd.DataFrame:
        """Return only O-mode echoes."""
        return self.annotated_df[self.annotated_df[_MODE_COL] == _MODE_O].copy()

    def x_mode_df(self) -> pd.DataFrame:
        """Return only X-mode echoes."""
        return self.annotated_df[self.annotated_df[_MODE_COL] == _MODE_X].copy()

    def summary(self) -> str:
        """One-line text summary of the classification."""
        total = len(self.annotated_df)
        return (
            f"PolarizationResult: total={total}  "
            f"O={self.o_mode_count}  X={self.x_mode_count}  "
            f"ambiguous={self.ambiguous_count}  unknown={self.unknown_count}  "
            f"o_mode_sign={'positive' if self.o_mode_sign == 1 else 'negative'} PP"
        )

    def plot(self, ax: Optional[plt.Axes] = None) -> plt.Axes:
        """Plot a PP histogram with O/X classification regions highlighted.

        Parameters
        ----------
        ax:
            Existing :class:`matplotlib.axes.Axes` to draw into.  A new
            figure is created when ``None`` (default).

        Returns
        -------
        matplotlib.axes.Axes
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(7, 4))

        pp = self.annotated_df[_PP_COL_DEFAULT].dropna()
        if pp.empty:
            logger.warning("PolarizationResult.plot: no valid PP values to plot.")
            return ax

        ax.hist(
            pp,
            bins=36,
            range=(-180, 180),
            color="steelblue",
            edgecolor="white",
            linewidth=0.4,
            alpha=0.8,
        )

        thr = self.pp_ambiguous_threshold_deg
        if self.o_mode_sign == -1:
            o_region = (-180, -thr)
            x_region = (thr, 180)
        else:
            o_region = (thr, 180)
            x_region = (-180, -thr)

        ax.axvspan(*o_region, alpha=0.15, color="tab:blue", label="O-mode region")
        ax.axvspan(*x_region, alpha=0.15, color="tab:orange", label="X-mode region")
        ax.axvspan(-thr, thr, alpha=0.15, color="grey", label="ambiguous")

        ax.set_xlabel("PP  (degrees)")
        ax.set_ylabel("Echo count")
        ax.set_title("PP distribution — O/X classification")
        ax.legend(fontsize=8)
        return ax


# ===========================================================================
# Processor class
# ===========================================================================


class PolarizationClassifier:
    """Classify ionospheric echoes as O-mode, X-mode, or ambiguous using PP.

    The sign convention (which PP sign maps to O-mode) is station-specific and
    must be supplied by the user or inferred from the geomagnetic dip angle.
    The default (``o_mode_sign=-1``) applies to northern-hemisphere VIPIR
    installations where negative PP corresponds to left-hand (ordinary) wave
    polarization.

    Parameters
    ----------
    o_mode_sign:
        ``-1`` — negative PP → O-mode (northern hemisphere default).
        ``+1`` — positive PP → O-mode (southern hemisphere, or reversed layout).
    pp_ambiguous_threshold_deg:
        Echoes with ``|PP| < threshold`` are labelled ``"ambiguous"``
        (near-linear polarization; neither mode dominates).  Default ``20.0``.
    pp_col:
        Name of the PP column in the input DataFrame.
        Default ``"polarization_deg"``.

    Examples
    --------
    >>> clf = PolarizationClassifier(o_mode_sign=-1)
    >>> result = clf.fit(extractor.to_dataframe())
    >>> o_df = result.o_mode_df()
    """

    def __init__(
        self,
        o_mode_sign: int = -1,
        pp_ambiguous_threshold_deg: float = 20.0,
        pp_col: str = _PP_COL_DEFAULT,
    ) -> None:
        if o_mode_sign not in (-1, 1):
            raise ValueError("o_mode_sign must be +1 or -1.")
        if pp_ambiguous_threshold_deg < 0:
            raise ValueError("pp_ambiguous_threshold_deg must be non-negative.")

        self.o_mode_sign = o_mode_sign
        self.pp_ambiguous_threshold_deg = pp_ambiguous_threshold_deg
        self.pp_col = pp_col

    # ------------------------------------------------------------------
    # Optional: infer o_mode_sign from station coordinates
    # ------------------------------------------------------------------

    @staticmethod
    def infer_o_mode_sign(station_lat: float) -> int:
        """Heuristic sign convention from station latitude.

        In the Dynasonde / VIPIR convention the O-mode PP sign follows the
        sign of the vertical geomagnetic field component Bz.  Bz is positive
        (downward) in the northern hemisphere and negative (upward) in the
        southern hemisphere.  A rigorous determination requires IGRF, but for
        most mid-latitude stations this approximation is sufficient.

        Parameters
        ----------
        station_lat:
            Geodetic latitude in decimal degrees (positive north).

        Returns
        -------
        int
            ``-1`` for northern hemisphere, ``+1`` for southern hemisphere.
        """
        sign = -1 if station_lat >= 0 else 1
        logger.info(f"Inferred o_mode_sign={sign} from station_lat={station_lat:.2f}°")
        return sign

    # ------------------------------------------------------------------
    # Main method
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> PolarizationResult:
        """Classify every echo in *df* by wave mode.

        Parameters
        ----------
        df:
            Echo DataFrame produced by
            :class:`~pynasonde.vipir.riq.echo.EchoExtractor` or
            :class:`~pynasonde.vipir.riq.parsers.filter.IonogramFilter`.
            Must contain a column named ``self.pp_col``.

        Returns
        -------
        PolarizationResult
            Annotated DataFrame and count summary.

        Raises
        ------
        KeyError
            If ``self.pp_col`` is not found in *df*.
        """
        if self.pp_col not in df.columns:
            raise KeyError(
                f"Column '{self.pp_col}' not found in DataFrame. "
                f"Available columns: {list(df.columns)}"
            )

        out = df.copy()
        pp = out[self.pp_col]
        thr = self.pp_ambiguous_threshold_deg

        mode = pd.array([_MODE_UNKNOWN] * len(out), dtype="object")

        nan_mask = pp.isna()
        ambiguous_mask = (~nan_mask) & (pp.abs() < thr)
        o_mask = (
            (~nan_mask)
            & (~ambiguous_mask)
            & ((pp < 0) if self.o_mode_sign == -1 else (pp > 0))
        )
        x_mask = (~nan_mask) & (~ambiguous_mask) & ~o_mask

        mode[nan_mask] = _MODE_UNKNOWN
        mode[ambiguous_mask] = _MODE_AMBIGUOUS
        mode[o_mask] = _MODE_O
        mode[x_mask] = _MODE_X

        out[_MODE_COL] = mode

        n_o = int(o_mask.sum())
        n_x = int(x_mask.sum())
        n_amb = int(ambiguous_mask.sum())
        n_unk = int(nan_mask.sum())

        logger.info(
            f"PolarizationClassifier: O={n_o}  X={n_x}  "
            f"ambiguous={n_amb}  unknown={n_unk}"
        )

        return PolarizationResult(
            annotated_df=out,
            o_mode_count=n_o,
            x_mode_count=n_x,
            ambiguous_count=n_amb,
            unknown_count=n_unk,
            o_mode_sign=self.o_mode_sign,
            pp_ambiguous_threshold_deg=thr,
        )
