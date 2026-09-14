"""foF2 / h'F / drift-velocity response of the CONUS Digisonde network to
IP-shock storms.

This script queries GIRO DIDBase (via :class:`pynasonde.digisonde.didbase.
DidBaseConnector`) and GIRO DriftBase (via :class:`pynasonde.digisonde.
driftbase.DriftBaseConnector`) for every (station, event) combination
defined below, fetching scaled foF2/h'F characteristics and scaled
horizontal drift velocity directly via SQL --
:meth:`~pynasonde.digisonde.didbase.DidBaseConnector.fetch_scaled_params`
and :meth:`~pynasonde.digisonde.driftbase.DriftBaseConnector.
fetch_velocity` respectively -- and produces a three-panel, 24-hour-UT
time-series plot (foF2, h'F, horizontal drift speed V_H) for the
interplanetary (IP) shock / CME-driven storm day, overlaid with a
quiet-time baseline (median +/- 1 STD per time-of-day) computed from the
``n_baseline_days`` calendar days immediately preceding the event.

Recipe
------
1. ``STATIONS`` lists the CONUS Digisonde network (URSI code, name, and
   coordinates) to query.
2. ``EVENTS`` lists the storm/shock dates of interest (Dst minima from
   2010-2020). Only the shock/storm date is known precisely here (see the
   caveat below on exact shock-arrival UT).
3. :func:`fetch_event_day_dataframe` pulls ``foF2``/``hF`` for the single
   00:00-24:00 UT day of the event through ``DidBaseConnector.
   fetch_scaled_params``, which reads DIDBase's ``PARAM``/``PARAMCAT``
   tables directly via SQL — no ionogram or SAO blob download/parsing
   involved. :func:`fetch_event_day_velocity` does the same for ``VH``
   through ``DriftBaseConnector.fetch_velocity`` (DriftBase's ``VELOCITY``
   table) — DriftBase coverage is independent of DIDBase's and may be
   unavailable for a given station/date even when foF2/h'F are present.
4. :func:`fetch_quiet_baseline` pulls the same characteristics for the
   ``n_baseline_days`` days immediately before the event, bins them by
   time-of-day (15-minute bins by default), and computes the per-bin
   median and standard deviation — reprojected onto the event day's
   calendar date so it can be overlaid on the same x-axis. The same
   binning is reused for the velocity baseline.
5. :func:`plot_foF2_hpF` renders the three-panel figure (event-day markers
   plus each panel's baseline median line and +/-1 STD shaded band) and
   saves it under ``tmp/ip/``. The velocity panel is simply omitted (with
   a warning) when DriftBase has no coverage for a station/event.
6. :func:`run` loops over every station/event pair, skipping (with a
   logged warning) any combination that DIDBase has no scaled coverage
   for, or that otherwise fails to fetch, instead of crashing the run.

DIDBase credentials are read from the ``DIDBASE_USER``, ``DIDBASE_PASSWORD``
and (optionally) ``DIDBASE_URL``/``DIDBASE_ROLE`` environment variables —
register for DIDBase access at https://ulcar.uml.edu/DIDBase/ before running
this script. DriftBase credentials are read the same way via
``DRIFTBASE_USER``/``DRIFTBASE_PASSWORD``/``DRIFTBASE_URL``/
``DRIFTBASE_ROLE`` (DriftBase has been observed to accept the same
username/password as DIDBase). See ``if __name__ == "__main__":`` below
for a usage example.

Note: the shock/storm date given in ``EVENTS`` is the calendar UT day, not
an independently verified shock-arrival timestamp; the IP-shock marker on
the plot defaults to 00:00 UT of that date (the left edge of the 24-hour
window) unless a more precise time is supplied.
"""

from __future__ import annotations

import datetime as dt
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.dates as mdates
import pandas as pd
from loguru import logger

from pynasonde.digisonde.didbase import DidBaseConnector
from pynasonde.digisonde.driftbase import DriftBaseConnector
from pynasonde.digisonde.digi_plots import SaoSummaryPlots
from pynasonde.digisonde.digi_utils import setsize

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
    dict(ursi="AL945", name="Alpena, MI", lat=45.07, lon=-83.56),
    dict(ursi="IF843", name="Idaho National Lab, ID", lat=43.81, lon=-112.68),
    dict(ursi="MHJ45", name="Millstone Hill, MA", lat=42.60, lon=-71.50),
    dict(ursi="HAJ43", name="Hanscom AFB, MA", lat=42.50, lon=-71.30),
    dict(ursi="BC840", name="Boulder, CO", lat=40.00, lon=-105.30),
    dict(ursi="CS839", name="Colorado Springs, CO", lat=39.00, lon=-104.88),
    dict(ursi="KI939", name="Kent Island, MD", lat=38.96, lon=-76.28),
    dict(ursi="WP937", name="Wallops Island, VA", lat=37.90, lon=-75.50),
    dict(ursi="KR835", name="Kirtland AFB, NM", lat=35.00, lon=-106.53),
    dict(ursi="PA836", name="Pt. Arguello, CA", lat=34.80, lon=-120.50),
    dict(ursi="DS932", name="Dyess AFB, TX", lat=32.40, lon=-99.80),
    dict(ursi="EG931", name="Eglin AFB, FL", lat=30.50, lon=-86.50),
    dict(ursi="AU930", name="Austin, TX", lat=30.40, lon=-97.70),
    dict(ursi="ME929", name="Melrose, FL", lat=29.71, lon=-82.00),
]

# ── IP-shock / CME-driven storms, 2010-2020 (date = storm/shock date) ───────
EVENTS: List[Dict] = [
    dict(
        name="2026-01-19 storm", date=dt.date(2026, 1, 19),
        shock_time=dt.datetime(2026, 1, 19, 19, 15),
    ),
    dict(name="2015-03-17 St. Patrick's Day storm", date=dt.date(2015, 3, 17), dst=-234),
    dict(name="2015-06-22 double-CME storm", date=dt.date(2015, 6, 22), dst=-198),
    dict(name="2018-08-26 storm", date=dt.date(2018, 8, 26), dst=-175),
    dict(name="2015-12-19 storm", date=dt.date(2015, 12, 19), dst=-166),
    dict(
        name="2011-10-24 storm", date=dt.date(2011, 10, 24), dst=-147,
        shock_time=dt.datetime(2011, 10, 24, 18, 25),
    ),
    dict(name="2012-03-08 double-shock storm", date=dt.date(2012, 3, 8), dst=-145),
    dict(name="2012-07-14 storm", date=dt.date(2012, 7, 14), dst=-139),
    dict(name="2013-03-17 storm", date=dt.date(2013, 3, 17), dst=-132),
    dict(name="2015-10-07 storm", date=dt.date(2015, 10, 7), dst=-130),
    dict(name="2017-05-27 storm", date=dt.date(2017, 5, 27), dst=-125),
    dict(name="2013-06-01 storm", date=dt.date(2013, 6, 1), dst=-124),
    dict(name="2012-10-01 storm", date=dt.date(2012, 10, 1), dst=-122),
    dict(name="2017-09-07 post-X9.3-flare storm", date=dt.date(2017, 9, 7), dst=-122),
]

OUTPUT_DIR = Path("tmp/ip")


def _slugify(name: str) -> str:
    """Turn a station name like 'Boulder, CO' into 'boulder_co'."""
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return slug


PARAMS: List[str] = ["foF2", "hF"]


def fetch_event_day_dataframe(
    connector: DidBaseConnector,
    station: Dict,
    event_date: dt.date,
    params: List[str] = PARAMS,
) -> Optional[pd.DataFrame]:
    """Fetch scaled foF2/h'F characteristics for the 24-hour UT shock day.

    Queries ``[00:00 UT of event_date, 00:00 UT of event_date + 1 day]``
    via ``DidBaseConnector.fetch_scaled_params``, which reads DIDBase's
    ``PARAM``/``PARAMCAT`` tables directly — no ionogram or SAO blob
    download/parsing involved.

    Args:
        connector: An already-connected ``DidBaseConnector``.
        station: One entry from ``STATIONS`` (must have a ``ursi`` key).
        event_date: The storm/shock date (UT).
        params: ``PARAMCAT.NAME`` values to fetch.

    Returns:
        A DataFrame with ``datetime`` plus one column per requested
        parameter, or ``None`` if no usable data was found.
    """
    ursi = station["ursi"]
    start = dt.datetime.combine(event_date, dt.time.min)
    end = start + dt.timedelta(days=1)

    try:
        df = connector.fetch_scaled_params(station=ursi, start=start, end=end, params=params)
    except Exception as exc:  # station unknown to DIDBase, network error, etc.
        logger.warning(f"[{ursi}] fetch_scaled_params failed for {event_date}: {exc}")
        return None

    if df is None or df.empty:
        logger.warning(f"[{ursi}] no scaled {params} coverage for {event_date} in DIDBase.")
        return None

    df = df.dropna(subset=params, how="all")
    if df.empty:
        logger.warning(f"[{ursi}] {event_date}: {params} rows were all NaN.")
        return None

    return df.sort_values("datetime").reset_index(drop=True)


def _time_of_day_baseline(
    df: pd.DataFrame,
    params: List[str],
    event_date: dt.date,
    bin_minutes: int,
) -> pd.DataFrame:
    """Bin ``df`` by time-of-day and compute the per-bin median/STD/count.

    Shared by :func:`fetch_quiet_baseline` and
    :func:`fetch_velocity_baseline` — bins every sample by time-of-day
    (rounded to the nearest ``bin_minutes``) and computes the median/
    standard deviation/count across days for each bin. Bin timestamps are
    reprojected onto ``event_date`` so the result can be overlaid
    directly on that day's 24-hour x-axis.

    Args:
        df: DataFrame with a ``datetime`` column plus one column per
            entry in ``params``.
        params: Column names in ``df`` to summarize.
        event_date: Calendar date to reproject bin timestamps onto.
        bin_minutes: Time-of-day bin width in minutes.

    Returns:
        DataFrame with a ``datetime`` column and, per parameter,
        ``{param}_median``/``{param}_std``/``{param}_n`` columns.
    """
    bin_size = pd.Timedelta(minutes=bin_minutes)
    time_of_day = df["datetime"] - df["datetime"].dt.normalize()
    df = df.assign(time_of_day=(time_of_day / bin_size).round().astype(int) * bin_size)

    agg_kwargs = {}
    for p in params:
        agg_kwargs[f"{p}_median"] = (p, "median")
        agg_kwargs[f"{p}_std"] = (p, "std")
        agg_kwargs[f"{p}_n"] = (p, "count")
    baseline = df.groupby("time_of_day").agg(**agg_kwargs).reset_index()
    baseline["datetime"] = pd.Timestamp(event_date) + baseline["time_of_day"]
    return baseline.sort_values("datetime").reset_index(drop=True)


def fetch_quiet_baseline(
    connector: DidBaseConnector,
    station: Dict,
    event_date: dt.date,
    n_baseline_days: int = 27,
    params: List[str] = PARAMS,
    bin_minutes: int = 15,
) -> Optional[pd.DataFrame]:
    """Build a quiet-time per-time-of-day foF2/h'F median/STD baseline.

    Fetches ``n_baseline_days`` calendar days immediately before
    ``event_date`` (``[event_date - n_baseline_days, event_date)`` UT) and
    summarizes them with :func:`_time_of_day_baseline`.

    Args:
        connector: An already-connected ``DidBaseConnector``.
        station: One entry from ``STATIONS`` (must have a ``ursi`` key).
        event_date: The storm/shock date (UT); the baseline window ends
            at 00:00 UT of this date (the event day itself is excluded).
        n_baseline_days: Number of preceding calendar days to average.
        params: ``PARAMCAT.NAME`` values to fetch/summarize.
        bin_minutes: Time-of-day bin width in minutes.

    Returns:
        DataFrame with a ``datetime`` column (event-day timestamps) and,
        per requested parameter, ``{param}_median``/``{param}_std``/
        ``{param}_n`` columns, or ``None`` if no usable data was found.
    """
    ursi = station["ursi"]
    # GETMEASUREMENTS' end bound is inclusive (timeUT <= :et), and the
    # event day's own window starts at this same midnight timestamp
    # (timeUT >= :st, also inclusive) -- subtract 1s so a measurement at
    # exactly 00:00:00 UT on event_date is never double-counted in both
    # the quiet baseline and the event day.
    end = dt.datetime.combine(event_date, dt.time.min) - dt.timedelta(seconds=1)
    start = end - dt.timedelta(days=n_baseline_days)

    try:
        df = connector.fetch_scaled_params(station=ursi, start=start, end=end, params=params)
    except Exception as exc:
        logger.warning(f"[{ursi}] baseline fetch_scaled_params failed for {event_date}: {exc}")
        return None

    if df is None or df.empty:
        logger.warning(
            f"[{ursi}] no {n_baseline_days}-day quiet-baseline coverage before {event_date}."
        )
        return None

    df = df.dropna(subset=params, how="all")
    if df.empty:
        logger.warning(f"[{ursi}] {event_date}: baseline {params} rows were all NaN.")
        return None

    return _time_of_day_baseline(df, params, event_date, bin_minutes)


VEL_PARAMS: List[str] = ["VZ"]
VEL_YLIMS: Dict[str, List[float]] = {"VX": [-150, 150], "VY": [-150, 150], "VZ": [-20, 20]}


def fetch_event_day_velocity(
    connector: DriftBaseConnector,
    station: Dict,
    event_date: dt.date,
    params: List[str] = VEL_PARAMS,
) -> Optional[pd.DataFrame]:
    """Fetch scaled drift velocity components for the 24-hour UT shock day.

    Queries ``[00:00 UT of event_date, 00:00 UT of event_date + 1 day]``
    via ``DriftBaseConnector.fetch_velocity``, which reads DriftBase's
    ``VELOCITY`` table directly — no drift-ionogram blob download/parsing
    involved. DriftBase coverage is independent of DIDBase's and may be
    unavailable even when foF2/h'F are present.

    Args:
        connector: An already-connected ``DriftBaseConnector``.
        station: One entry from ``STATIONS`` (must have a ``ursi`` key).
        event_date: The storm/shock date (UT).
        params: ``VELOCITY`` columns to keep, e.g. ``["VX", "VY", "VZ"]``
            (geographic/geomagnetic drift components, m/s). ``VH``
            (horizontal speed) and ``AZ`` (azimuth) are also available
            from ``fetch_velocity`` if needed.

    Returns:
        A DataFrame with ``datetime`` plus one column per requested
        component, or ``None`` if no usable data was found.
    """
    ursi = station["ursi"]
    start = dt.datetime.combine(event_date, dt.time.min)
    end = start + dt.timedelta(days=1)

    try:
        df = connector.fetch_velocity(station=ursi, start=start, end=end)
    except Exception as exc:  # station unknown to DriftBase, network error, etc.
        logger.warning(f"[{ursi}] fetch_velocity failed for {event_date}: {exc}")
        return None

    if df is None or df.empty:
        logger.warning(f"[{ursi}] no DriftBase VELOCITY coverage for {event_date}.")
        return None

    df = df.dropna(subset=params, how="all")
    if df.empty:
        logger.warning(f"[{ursi}] {event_date}: {params} rows were all NaN.")
        return None

    return df[["datetime", *params]].sort_values("datetime").reset_index(drop=True)


def fetch_velocity_baseline(
    connector: DriftBaseConnector,
    station: Dict,
    event_date: dt.date,
    n_baseline_days: int = 27,
    params: List[str] = VEL_PARAMS,
    bin_minutes: int = 15,
) -> Optional[pd.DataFrame]:
    """Build a quiet-time per-time-of-day drift-velocity median/STD baseline.

    Same approach as :func:`fetch_quiet_baseline`, but against DriftBase's
    ``VELOCITY`` table instead of DIDBase's scaled characteristics.

    Args:
        connector: An already-connected ``DriftBaseConnector``.
        station: One entry from ``STATIONS`` (must have a ``ursi`` key).
        event_date: The storm/shock date (UT); the baseline window ends
            at 00:00 UT of this date (the event day itself is excluded).
        n_baseline_days: Number of preceding calendar days to average.
        params: ``VELOCITY`` columns to fetch/summarize.
        bin_minutes: Time-of-day bin width in minutes.

    Returns:
        DataFrame with a ``datetime`` column and, per requested
        component, ``{param}_median``/``{param}_std``/``{param}_n``
        columns, or ``None`` if no usable data was found.
    """
    ursi = station["ursi"]
    # GETMEASUREMENTS' end bound is inclusive (timeUT <= :et), and the
    # event day's own window starts at this same midnight timestamp
    # (timeUT >= :st, also inclusive) -- subtract 1s so a measurement at
    # exactly 00:00:00 UT on event_date is never double-counted in both
    # the quiet baseline and the event day.
    end = dt.datetime.combine(event_date, dt.time.min) - dt.timedelta(seconds=1)
    start = end - dt.timedelta(days=n_baseline_days)

    try:
        df = connector.fetch_velocity(station=ursi, start=start, end=end)
    except Exception as exc:
        logger.warning(f"[{ursi}] baseline fetch_velocity failed for {event_date}: {exc}")
        return None

    if df is None or df.empty:
        logger.warning(
            f"[{ursi}] no {n_baseline_days}-day velocity-baseline coverage before {event_date}."
        )
        return None

    df = df.dropna(subset=params, how="all")
    if df.empty:
        logger.warning(f"[{ursi}] {event_date}: baseline {params} rows were all NaN.")
        return None

    return _time_of_day_baseline(df, params, event_date, bin_minutes)


def _overlay_baseline(
    ax, baseline: pd.DataFrame, param: str, n_baseline_days: int, show_std: bool = False
) -> None:
    """Draw the quiet-time median line (and, if ``show_std``, the +/-1 STD
    shaded band) on ``ax``."""
    median_col, std_col = f"{param}_median", f"{param}_std"
    if median_col not in baseline.columns:
        return
    med = baseline[median_col]
    ax.plot(
        baseline["datetime"], med, color="k", lw=1.5, zorder=4,
        label=f"{n_baseline_days}-day median",
    )
    if show_std:
        std = baseline[std_col].fillna(0.0)
        ax.fill_between(
            baseline["datetime"], med - std, med + std,
            color="k", alpha=0.15, zorder=1, label="±1 STD",
        )


def plot_foF2_hpF(
    df: pd.DataFrame,
    station: Dict,
    event: Dict,
    xlim: List[dt.datetime],
    out_path: Path,
    baseline: Optional[pd.DataFrame] = None,
    n_baseline_days: int = 27,
    vel_df: Optional[pd.DataFrame] = None,
    vel_baseline: Optional[pd.DataFrame] = None,
    shock_time: Optional[dt.datetime] = None,
    font_size: float = 14,
) -> None:
    """Render a foF2 / h'F / drift-velocity time-series figure.

    The event day's actual values are drawn as markers; if a baseline is
    given for a panel, its quiet-time median (line) and +/-1 STD (shaded
    band) are overlaid for comparison. The drift-velocity panel is
    included only when ``vel_df`` is provided (DriftBase coverage is not
    guaranteed for every station/event even when DIDBase's is).

    Args:
        df: Event-day foF2/h'F DataFrame from
            :func:`fetch_event_day_dataframe`.
        station: One entry from ``STATIONS``.
        event: One entry from ``EVENTS``.
        xlim: ``[start, end]`` datetimes for the x-axis (the 24-hour UT
            event day).
        out_path: PNG path to save the figure to.
        baseline: Optional foF2/h'F quiet-time baseline from
            :func:`fetch_quiet_baseline`.
        n_baseline_days: Number of days the baselines were averaged over
            (only used for the legend labels).
        vel_df: Optional event-day drift-velocity-component DataFrame
            (``VX``/``VY``/``VZ``) from :func:`fetch_event_day_velocity`.
            When ``None``, no velocity panels are drawn.
        vel_baseline: Optional drift-velocity quiet-time baseline from
            :func:`fetch_velocity_baseline`.
        shock_time: UT of the IP shock/storm onset, marked as a dashed
            vertical line on every panel. Defaults to 00:00 UT of
            ``event["date"]`` (the shock/storm date; see the module
            docstring's caveat that exact shock-arrival UT is not
            independently verified for every event).
        font_size: Base font size for the figure.
    """
    setsize(font_size)
    shock_time = shock_time or dt.datetime.combine(event["date"], dt.time.min)
    title = f"{station['ursi']} {station['name']} / {event['name']}"
    vel_params = [p for p in VEL_PARAMS if vel_df is not None and p in vel_df.columns]
    has_velocity = vel_df is not None and not vel_df.empty and len(vel_params) > 0
    nrows = 2 + len(vel_params) if has_velocity else 2
    sao_plot = SaoSummaryPlots(
        fig_title=title,
        nrows=nrows,
        ncols=1,
        figsize=(8, 3),
        font_size=font_size,
    )
    major_locator = mdates.HourLocator(byhour=range(0, 24, 6))
    minor_locator = mdates.HourLocator(byhour=range(0, 24, 2))

    ax0, _ = sao_plot.plot_TS(
        df,
        left_yparams=["foF2"],
        right_yparams=[],
        ylabels=["foF2, MHz", ""],
        xlabel="",
        left_ylim=[1, 15],
        xlim=xlim,
        major_locator=major_locator,
        minor_locator=minor_locator,
    )
    ax1, _ = sao_plot.plot_TS(
        df,
        left_yparams=["hF"],
        right_yparams=[],
        ylabels=["h'F, km", ""],
        xlabel="Time, UT" if not has_velocity else "",
        left_ylim=[100, 400],
        xlim=xlim,
        major_locator=major_locator,
        minor_locator=minor_locator,
    )
    axes = [ax0, ax1]

    if has_velocity:
        for i, p in enumerate(vel_params):
            is_last = i == len(vel_params) - 1
            ax_v, _ = sao_plot.plot_TS(
                vel_df,
                left_yparams=[p],
                right_yparams=[],
                ylabels=[f"{p}, m/s", ""],
                xlabel="Time, UT" if is_last else "",
                left_ylim=VEL_YLIMS.get(p, [-150, 150]),
                xlim=xlim,
                major_locator=major_locator,
                minor_locator=minor_locator,
            )
            if vel_baseline is not None:
                _overlay_baseline(ax_v, vel_baseline, p, n_baseline_days)
            axes.append(ax_v)

    if baseline is not None:
        _overlay_baseline(ax0, baseline, "foF2", n_baseline_days)
        _overlay_baseline(ax1, baseline, "hF", n_baseline_days)
    for ax in axes:
        ax.axvline(shock_time, color="r", ls="--", lw=1.2, zorder=5)
        ax.legend(loc=2, fontsize=max(font_size - 4, 6))
    y0, y1 = ax0.get_ylim()
    ax0.text(
        shock_time,
        y1 - 0.05 * (y1 - y0),
        " IP shock",
        rotation=90,
        va="top",
        ha="left",
        color="r",
        fontsize=max(font_size - 4, 6),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sao_plot.save(str(out_path))
    sao_plot.close()


def run(
    stations: List[Dict] = STATIONS,
    events: List[Dict] = EVENTS,
    n_baseline_days: int = 27,
    out_dir: Path = OUTPUT_DIR,
    include_velocity: bool = True,
) -> List[Path]:
    """Fetch, plot and save foF2/h'F(/drift-velocity) figures for every
    station/event pair.

    Each figure spans the 24-hour UT shock day and overlays a quiet-time
    median/STD baseline computed from the ``n_baseline_days`` days
    immediately before it. Failures (no DIDBase coverage, all-NaN rows,
    etc.) are logged as warnings for the offending station/event
    combination and skipped, not raised. A missing foF2/h'F baseline does
    not block the plot -- it is simply omitted with a warning. DriftBase
    velocity coverage is optional and independent: when unavailable for a
    station/event, the plot is still produced with just the foF2/h'F
    panels.

    Args:
        stations: Stations to process.
        events: Events to process.
        n_baseline_days: Days of quiet-time history to average for the
            baseline overlays.
        out_dir: Directory to save figures into.
        include_velocity: If True, also connect to DriftBase and add a
            horizontal drift-velocity (``V_H``) panel when coverage
            exists. Set False to skip DriftBase entirely (e.g. if it's
            unreachable) and only produce the foF2/h'F panels.

    Returns:
        List of PNG paths that were successfully written.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []

    drift_connector = (
        DriftBaseConnector(
            DRIFTBASE_URL, DRIFTBASE_USER, DRIFTBASE_PASSWORD, DRIFTBASE_ROLE
        ).connect()
        if include_velocity
        else None
    )
    try:
        with DidBaseConnector(
            DIDBASE_URL, DIDBASE_USER, DIDBASE_PASSWORD, DIDBASE_ROLE
        ) as connector:
            for event in events:
                start = dt.datetime.combine(event["date"], dt.time.min)
                end = start + dt.timedelta(days=1)
                for station in stations:
                    df = fetch_event_day_dataframe(connector, station, event["date"])
                    if df is None:
                        continue

                    baseline = fetch_quiet_baseline(
                        connector, station, event["date"], n_baseline_days=n_baseline_days
                    )
                    if baseline is None:
                        logger.warning(
                            f"[{station['ursi']}] plotting {event['date']} without a "
                            f"{n_baseline_days}-day baseline (none available)."
                        )

                    vel_df = vel_baseline = None
                    if drift_connector is not None:
                        vel_df = fetch_event_day_velocity(
                            drift_connector, station, event["date"]
                        )
                        if vel_df is not None:
                            vel_baseline = fetch_velocity_baseline(
                                drift_connector, station, event["date"],
                                n_baseline_days=n_baseline_days,
                            )

                    slug = _slugify(station["name"])
                    out_path = out_dir / (
                        f"{station['ursi']}_{slug}_{event['date']:%Y%m%d}_foF2_hpF.png"
                    )
                    try:
                        plot_foF2_hpF(
                            df, station, event, [start, end], out_path,
                            baseline=baseline, n_baseline_days=n_baseline_days,
                            vel_df=vel_df, vel_baseline=vel_baseline,
                            shock_time=event.get("shock_time"),
                        )
                    except Exception as exc:
                        logger.warning(
                            f"[{station['ursi']}] plotting failed for {event['date']}: {exc}"
                        )
                        continue

                    logger.info(f"Saved {out_path}")
                    written.append(out_path)
    finally:
        if drift_connector is not None:
            drift_connector.close()

    logger.info(f"Wrote {len(written)} figure(s) to {out_dir}")
    return written


if __name__ == "__main__":
    # Requires DIDBASE_USER / DIDBASE_PASSWORD (and, for the velocity
    # panel, DRIFTBASE_USER / DRIFTBASE_PASSWORD) environment variables
    # set to valid GIRO credentials (see https://ulcar.uml.edu/DIDBase/).
    # Example: run only two stations for a single event, with a shorter
    # 3-day quiet baseline instead of the 27-day default:
    #
    #   run(
    #       stations=[s for s in STATIONS if s["ursi"] in ("AL945", "IF843")],
    #       events=[e for e in EVENTS if e["date"] == dt.date(2015, 3, 17)],
    #       n_baseline_days=3,
    #   )
    #
    # Or skip DriftBase entirely and only produce foF2/h'F panels:
    #
    #   run(include_velocity=False)
    run()
