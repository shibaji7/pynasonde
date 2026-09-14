"""Zoomed 18:00-21:00 UT ionospheric-parameter and skymap plots for the
2026-01-19 IP-shock event (shock onset 19:15 UT) only.

This is a standalone, narrower companion to ``plot_foF2_hpF_ip_shock_events.py``
for a single event: instead of a 24-hour view with a quiet-time baseline, it
zooms into the 3-hour window straddling the shock and adds the E/Es-layer
characteristics (foE/h'E, foEs/h'Es) alongside F2 (foF2/h'F), plus per-scan
skymap plots built from DriftBase's raw ``SKYDATA`` for the same window.

Recipe
------
1. ``STATIONS`` lists the CONUS Digisonde network to query (same list as
   ``plot_foF2_hpF_ip_shock_events.py``); combinations with no coverage for
   this event are skipped with a logged warning, not an error.
2. :func:`fetch_zoom_dataframe` pulls ``foF2``/``hF``/``foE``/``hE``/
   ``foEs``/``hEs`` for the 18:00-21:00 UT window via
   ``DidBaseConnector.fetch_scaled_params`` (DIDBase's ``PARAM``/
   ``PARAMCAT`` tables directly -- no blob download/parsing).
3. :func:`plot_zoom` renders a 3-panel dual-axis figure -- one panel per
   ionospheric layer (F2, E, Es), each with the layer's critical frequency
   on the left axis and virtual height on the right -- with the shock time
   marked, and saves it under ``tmp/ip/19_Jan_2026/``.
4. :func:`plot_station_skymaps` downloads raw skymap blobs for the same
   window via ``DriftBaseConnector.fetch_sky_files`` and parses each with
   :class:`pynasonde.digisonde.parsers.sky.SkyExtractor` (the raw
   ``SKYDATA`` format, confirmed compatible even for stations/dates where
   DriftBase's derived ``VELOCITY`` table is not yet populated), producing
   one skymap PNG per available scan under ``tmp/ip/19_Jan_2026/``.
5. :func:`run` loops over every station, doing both, and never lets one
   station's failure stop the rest.

DIDBase credentials are read from ``DIDBASE_USER``/``DIDBASE_PASSWORD``
(+ optional ``DIDBASE_URL``/``DIDBASE_ROLE``); DriftBase credentials from
``DRIFTBASE_USER``/``DRIFTBASE_PASSWORD`` (+ optional ``DRIFTBASE_URL``/
``DRIFTBASE_ROLE``) -- DriftBase has been observed to accept the same
username/password as DIDBase. See ``if __name__ == "__main__":`` below.
"""

from __future__ import annotations

import datetime as dt
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.dates as mdates
import pandas as pd
from loguru import logger

from pynasonde.digisonde.didbase import DidBaseConnector
from pynasonde.digisonde.driftbase import DriftBaseConnector
from pynasonde.digisonde.digi_plots import SaoSummaryPlots, SkySummaryPlots
from pynasonde.digisonde.digi_utils import setsize
from pynasonde.digisonde.parsers.sky import SkyExtractor

# ── DIDBase connection settings ─────────────────────────────────────────────
DIDBASE_URL = os.environ.get(
    "DIDBASE_URL", "jdbc:firebirdsql://didbase.giro.uml.edu/didb"
)
DIDBASE_USER = os.environ.get("DIDBASE_USER", "")
DIDBASE_PASSWORD = os.environ.get("DIDBASE_PASSWORD", "")
DIDBASE_ROLE = os.environ.get("DIDBASE_ROLE", "COMMON")

# ── DriftBase connection settings ────────────────────────────────────────────
DRIFTBASE_URL = os.environ.get(
    "DRIFTBASE_URL", "jdbc:firebirdsql://129.63.134.45/drift"
)
DRIFTBASE_USER = os.environ.get("DRIFTBASE_USER", "")
DRIFTBASE_PASSWORD = os.environ.get("DRIFTBASE_PASSWORD", "")
DRIFTBASE_ROLE = os.environ.get("DRIFTBASE_ROLE", "COMMON")

# ── CONUS Digisonde network ─────────────────────────────────────────────────
STATIONS: List[Dict] = [
    dict(ursi="AL945", name="Alpena, MI"),
    dict(ursi="IF843", name="Idaho National Lab, ID"),
    dict(ursi="MHJ45", name="Millstone Hill, MA"),
    dict(ursi="HAJ43", name="Hanscom AFB, MA"),
    dict(ursi="BC840", name="Boulder, CO"),
    dict(ursi="CS839", name="Colorado Springs, CO"),
    dict(ursi="KI939", name="Kent Island, MD"),
    dict(ursi="WP937", name="Wallops Island, VA"),
    dict(ursi="KR835", name="Kirtland AFB, NM"),
    dict(ursi="PA836", name="Pt. Arguello, CA"),
    dict(ursi="DS932", name="Dyess AFB, TX"),
    dict(ursi="EG931", name="Eglin AFB, FL"),
    dict(ursi="AU930", name="Austin, TX"),
    dict(ursi="ME929", name="Melrose, FL"),
]

# ── 2026-01-19 event window (this script covers only this one event) ───────
EVENT_DATE = dt.date(2026, 1, 19)
SHOCK_TIME = dt.datetime(2026, 1, 19, 19, 15)
ZOOM_START = dt.datetime(2026, 1, 19, 18, 0)
ZOOM_END = dt.datetime(2026, 1, 19, 21, 0)

PARAMS: List[str] = ["foF2", "hF", "foE", "foEs", "hE", "hEs"]
# (frequency, height) pairs per ionospheric layer -> one dual-axis panel each
LAYER_PAIRS: List[Tuple[str, str]] = [("foF2", "hF"), ("foE", "hE"), ("foEs", "hEs")]
LAYER_LABELS: Dict[str, str] = {
    "foF2": "foF2, MHz", "hF": "h'F, km",
    "foE": "foE, MHz", "hE": "h'E, km",
    "foEs": "foEs, MHz", "hEs": "h'Es, km",
}

OUTPUT_DIR = Path("tmp/ip/19_Jan_2026")


def _slugify(name: str) -> str:
    """Turn a station name like 'Boulder, CO' into 'boulder_co'."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def fetch_zoom_dataframe(
    connector: DidBaseConnector,
    station: Dict,
    params: List[str] = PARAMS,
    start: dt.datetime = ZOOM_START,
    end: dt.datetime = ZOOM_END,
) -> Optional[pd.DataFrame]:
    """Fetch scaled F2/E/Es-layer characteristics for the zoom window.

    Queries ``[start, end]`` via ``DidBaseConnector.fetch_scaled_params``.
    Any requested parameter that has zero coverage across the whole window
    (e.g. ``foEs``/``hEs`` when no sporadic-E was present) is still added
    as an all-NaN column, so :func:`plot_zoom` can always address it
    without a ``KeyError``.

    Args:
        connector: An already-connected ``DidBaseConnector``.
        station: One entry from ``STATIONS`` (must have a ``ursi`` key).
        params: ``PARAMCAT.NAME`` values to fetch.
        start: Window start (inclusive).
        end: Window end (inclusive).

    Returns:
        A DataFrame with ``datetime`` plus one column per requested
        parameter, or ``None`` if no usable data was found at all.
    """
    ursi = station["ursi"]
    try:
        df = connector.fetch_scaled_params(station=ursi, start=start, end=end, params=params)
    except Exception as exc:  # station unknown to DIDBase, network error, etc.
        logger.warning(f"[{ursi}] fetch_scaled_params failed for the zoom window: {exc}")
        return None

    if df is None or df.empty:
        logger.warning(f"[{ursi}] no scaled {params} coverage for the zoom window.")
        return None

    df = df.dropna(subset=params, how="all")
    if df.empty:
        logger.warning(f"[{ursi}] zoom-window {params} rows were all NaN.")
        return None

    for p in params:
        if p not in df.columns:
            df[p] = float("nan")

    return df.sort_values("datetime").reset_index(drop=True)


def plot_zoom(
    df: pd.DataFrame,
    station: Dict,
    out_path: Path,
    xlim: List[dt.datetime] = None,
    shock_time: dt.datetime = SHOCK_TIME,
    font_size: float = 14,
) -> None:
    """Render a 3-panel dual-axis (freq. left / height right) zoom figure.

    One panel per ionospheric layer: F2 (foF2/h'F), E (foE/h'E), and Es
    (foEs/h'Es), each spanning the 18:00-21:00 UT zoom window with the
    shock time marked as a dashed vertical line.

    Args:
        df: DataFrame from :func:`fetch_zoom_dataframe`.
        station: One entry from ``STATIONS``.
        out_path: PNG path to save the figure to.
        xlim: ``[start, end]`` datetimes for the x-axis. Defaults to
            ``[ZOOM_START, ZOOM_END]``.
        shock_time: UT of the IP shock, marked as a dashed vertical line
            on every panel.
        font_size: Base font size for the figure.
    """
    setsize(font_size)
    xlim = xlim or [ZOOM_START, ZOOM_END]
    title = f"{station['ursi']} {station['name']} / 2026-01-19 storm (18-21 UT)"
    sao_plot = SaoSummaryPlots(
        fig_title=title,
        nrows=len(LAYER_PAIRS),
        ncols=1,
        figsize=(8, 3),
        font_size=font_size,
    )
    major_locator = mdates.MinuteLocator(byminute=range(0, 60, 30))
    minor_locator = mdates.MinuteLocator(byminute=range(0, 60, 10))

    axes = []
    for i, (fparam, hparam) in enumerate(LAYER_PAIRS):
        is_last = i == len(LAYER_PAIRS) - 1
        ax, _tax = sao_plot.plot_TS(
            df,
            left_yparams=[fparam],
            right_yparams=[hparam],
            ylabels=[LAYER_LABELS[fparam], LAYER_LABELS[hparam]],
            xlabel="Time, UT" if is_last else "",
            left_ylim=[0, 15],
            right_ylim=[80, 400],
            xlim=xlim,
            major_locator=major_locator,
            minor_locator=minor_locator,
        )
        axes.append(ax)

    for ax in axes:
        ax.axvline(shock_time, color="r", ls="--", lw=1.2, zorder=5)
        ax.legend(loc=2, fontsize=max(font_size - 4, 6))
    y0, y1 = axes[0].get_ylim()
    axes[0].text(
        shock_time, y1 - 0.05 * (y1 - y0), " IP shock",
        rotation=90, va="top", ha="left", color="r", fontsize=max(font_size - 4, 6),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    sao_plot.save(str(out_path))
    sao_plot.close()


def plot_station_skymaps(
    drift_connector: DriftBaseConnector,
    station: Dict,
    out_dir: Path,
    start: dt.datetime = ZOOM_START,
    end: dt.datetime = ZOOM_END,
    font_size: float = 12,
) -> List[Path]:
    """Download raw skymap blobs for the zoom window and plot one PNG each.

    Uses ``DriftBaseConnector.fetch_sky_files`` (raw ``SKYHEADER``/
    ``SKYDATA``, the ``BESTEXPERT``-chosen expert per measurement) and
    parses each file with :class:`~pynasonde.digisonde.parsers.sky.
    SkyExtractor`. This works even when DriftBase's derived ``VELOCITY``
    table has no rows for the date, since it reads the raw skymap points
    directly rather than a pre-computed velocity.

    Args:
        drift_connector: An already-connected ``DriftBaseConnector``.
        station: One entry from ``STATIONS`` (must have a ``ursi`` key).
        out_dir: Directory to save skymap PNGs into (created if absent).
        start: Window start (inclusive).
        end: Window end (inclusive).
        font_size: Base font size for each skymap figure.

    Returns:
        List of PNG paths written (one per successfully parsed scan).
    """
    ursi = station["ursi"]
    raw_dir = out_dir / "sky_raw" / ursi
    try:
        paths = drift_connector.fetch_sky_files(
            station=ursi, start=start, end=end, out_dir=raw_dir
        )
    except Exception as exc:
        logger.warning(f"[{ursi}] fetch_sky_files failed for the zoom window: {exc}")
        return []

    if not paths:
        logger.warning(f"[{ursi}] no SKY coverage for the zoom window.")
        return []

    written: List[Path] = []
    for p in paths:
        try:
            ext = SkyExtractor(str(p), extract_time_from_name=True, extract_stn_from_name=True)
            ext.extract()
            sky_df = ext.to_pandas()
        except Exception as exc:
            logger.warning(f"[{ursi}] failed to parse {p.name}: {exc}")
            continue

        if sky_df is None or sky_df.empty:
            continue

        ts = pd.Timestamp(sky_df["datetime"].iloc[0])
        sky_plot = SkySummaryPlots(
            fig_title=f"{ursi} {station['name']} / {ts:%Y-%m-%d %H:%M} UT",
            font_size=font_size,
        )
        try:
            sky_plot.plot_skymap(sky_df, text=f"{ts:%H:%M} UT")
        except Exception as exc:
            logger.warning(f"[{ursi}] failed to plot skymap for {ts}: {exc}")
            sky_plot.close()
            continue

        out_path = out_dir / f"{ursi}_{ts:%Y%m%d%H%M%S}_skymap.png"
        sky_plot.save(str(out_path))
        sky_plot.close()
        written.append(out_path)

    logger.info(f"[{ursi}] wrote {len(written)} skymap(s) to {out_dir}")
    return written


def run(
    stations: List[Dict] = STATIONS,
    out_dir: Path = OUTPUT_DIR,
) -> Dict[str, List[Path]]:
    """Produce the zoom plot and skymaps for every station, for the single
    2026-01-19 event.

    Failures (no DIDBase/DriftBase coverage, malformed records, etc.) are
    logged as warnings for the offending station and skipped, not raised.

    Returns:
        ``{"zoom": [...], "skymaps": [...]}`` -- lists of PNG paths
        successfully written for each product.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: Dict[str, List[Path]] = {"zoom": [], "skymaps": []}

    with DidBaseConnector(
        DIDBASE_URL, DIDBASE_USER, DIDBASE_PASSWORD, DIDBASE_ROLE
    ) as connector:
        for station in stations:
            df = fetch_zoom_dataframe(connector, station)
            if df is None:
                continue
            slug = _slugify(station["name"])
            out_path = out_dir / f"{station['ursi']}_{slug}_20260119_zoom_1821UT.png"
            try:
                plot_zoom(df, station, out_path)
            except Exception as exc:
                logger.warning(f"[{station['ursi']}] zoom plotting failed: {exc}")
                continue
            logger.info(f"Saved {out_path}")
            written["zoom"].append(out_path)

    with DriftBaseConnector(
        DRIFTBASE_URL, DRIFTBASE_USER, DRIFTBASE_PASSWORD, DRIFTBASE_ROLE
    ) as drift_connector:
        for station in stations:
            written["skymaps"].extend(
                plot_station_skymaps(drift_connector, station, out_dir)
            )

    logger.info(
        f"Wrote {len(written['zoom'])} zoom figure(s) and "
        f"{len(written['skymaps'])} skymap(s) to {out_dir}"
    )
    return written


if __name__ == "__main__":
    # Requires DIDBASE_USER / DIDBASE_PASSWORD and DRIFTBASE_USER /
    # DRIFTBASE_PASSWORD environment variables set to valid GIRO
    # credentials (see https://ulcar.uml.edu/DIDBase/). Example: run only
    # two stations:
    #
    #   run(stations=[s for s in STATIONS if s["ursi"] in ("AL945", "MHJ45")])
    run()
