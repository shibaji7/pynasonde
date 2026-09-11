"""foF2 / h'F response of the CONUS Digisonde network to IP-shock storms.

This script queries GIRO DIDBase (via :class:`pynasonde.digisonde.didbase.
DidBaseConnector`) for every (station, event) combination defined below,
fetching scaled foF2/h'F characteristics directly via
:meth:`~pynasonde.digisonde.didbase.DidBaseConnector.fetch_scaled_params`,
and produces a two-panel, 24-hour-UT time-series plot (foF2 on top, h'F on
the bottom) for the interplanetary (IP) shock / CME-driven storm day,
overlaid with a quiet-time baseline (median +/- 1 STD per time-of-day)
computed from the ``n_baseline_days`` calendar days immediately preceding
the event.

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
   involved.
4. :func:`fetch_quiet_baseline` pulls the same characteristics for the
   ``n_baseline_days`` days immediately before the event, bins them by
   time-of-day (15-minute bins by default), and computes the per-bin
   median and standard deviation — reprojected onto the event day's
   calendar date so it can be overlaid on the same x-axis.
5. :func:`plot_foF2_hpF` renders the two-panel figure (event-day markers
   plus the baseline median line and +/-1 STD shaded band) and saves it
   under ``tmp/ip/``.
6. :func:`run` loops over every station/event pair, skipping (with a
   logged warning) any combination that DIDBase has no scaled coverage
   for, or that otherwise fails to fetch, instead of crashing the run.

DIDBase credentials are read from the ``DIDBASE_USER``, ``DIDBASE_PASSWORD``
and (optionally) ``DIDBASE_URL``/``DIDBASE_ROLE`` environment variables —
register for DIDBase access at https://ulcar.uml.edu/DIDBase/ before running
this script. See ``if __name__ == "__main__":`` below for a usage example.

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
from pynasonde.digisonde.digi_plots import SaoSummaryPlots
from pynasonde.digisonde.digi_utils import setsize

# ── DIDBase connection settings ─────────────────────────────────────────────
DIDBASE_URL = os.environ.get(
    "DIDBASE_URL", "jdbc:firebirdsql://didbase.giro.uml.edu/didb"
)
DIDBASE_USER = os.environ.get("DIDBASE_USER", "")
DIDBASE_PASSWORD = os.environ.get("DIDBASE_PASSWORD", "")
DIDBASE_ROLE = os.environ.get("DIDBASE_ROLE", "COMMON")

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
    dict(name="2015-03-17 St. Patrick's Day storm", date=dt.date(2015, 3, 17), dst=-234),
    dict(name="2015-06-22 double-CME storm", date=dt.date(2015, 6, 22), dst=-198),
    dict(name="2018-08-26 storm", date=dt.date(2018, 8, 26), dst=-175),
    dict(name="2015-12-19 storm", date=dt.date(2015, 12, 19), dst=-166),
    dict(name="2011-10-24 storm", date=dt.date(2011, 10, 24), dst=-147),
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


def fetch_quiet_baseline(
    connector: DidBaseConnector,
    station: Dict,
    event_date: dt.date,
    n_baseline_days: int = 7,
    params: List[str] = PARAMS,
    bin_minutes: int = 15,
) -> Optional[pd.DataFrame]:
    """Build a quiet-time per-time-of-day median/STD baseline.

    Fetches ``n_baseline_days`` calendar days immediately before
    ``event_date`` (``[event_date - n_baseline_days, event_date)`` UT),
    bins every sample by time-of-day (rounded to the nearest
    ``bin_minutes``), and computes the median/standard deviation/count
    across those days for each bin. Bin timestamps are reprojected onto
    ``event_date`` itself so the result can be overlaid directly on the
    event day's 24-hour x-axis.

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
    end = dt.datetime.combine(event_date, dt.time.min)
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


def _overlay_baseline(ax, baseline: pd.DataFrame, param: str, n_baseline_days: int) -> None:
    """Draw the quiet-time median line and +/-1 STD shaded band on ``ax``."""
    median_col, std_col = f"{param}_median", f"{param}_std"
    if median_col not in baseline.columns:
        return
    med = baseline[median_col]
    std = baseline[std_col].fillna(0.0)
    ax.plot(
        baseline["datetime"], med, color="k", lw=1.5, zorder=4,
        label=f"{n_baseline_days}-day median",
    )
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
    n_baseline_days: int = 7,
    shock_time: Optional[dt.datetime] = None,
    font_size: float = 14,
) -> None:
    """Render a two-panel foF2 (top) / h'F (bottom) time-series figure.

    The event day's actual values are drawn as markers; if ``baseline`` is
    given, its quiet-time median (line) and +/-1 STD (shaded band) are
    overlaid on each panel for comparison.

    Args:
        df: Event-day DataFrame from :func:`fetch_event_day_dataframe`.
        station: One entry from ``STATIONS``.
        event: One entry from ``EVENTS``.
        xlim: ``[start, end]`` datetimes for the x-axis (the 24-hour UT
            event day).
        out_path: PNG path to save the figure to.
        baseline: Optional quiet-time baseline from
            :func:`fetch_quiet_baseline`.
        n_baseline_days: Number of days the baseline was averaged over
            (only used for the legend label).
        shock_time: UT of the IP shock/storm onset, marked as a dashed
            vertical line on both panels. Defaults to 00:00 UT of
            ``event["date"]`` (the shock/storm date; see the module
            docstring's caveat that exact shock-arrival UT is not
            independently verified for every event).
        font_size: Base font size for the figure.
    """
    setsize(font_size)
    shock_time = shock_time or dt.datetime.combine(event["date"], dt.time.min)
    title = f"{station['ursi']} {station['name']} / {event['name']}"
    sao_plot = SaoSummaryPlots(
        fig_title=title,
        nrows=2,
        ncols=1,
        figsize=(8, 3),
        font_size=font_size,
    )
    ax0, _ = sao_plot.plot_TS(
        df,
        left_yparams=["foF2"],
        right_yparams=[],
        ylabels=["foF2, MHz", ""],
        xlabel="",
        left_ylim=[1, 15],
        xlim=xlim,
        major_locator=mdates.HourLocator(byhour=range(0, 24, 6)),
        minor_locator=mdates.HourLocator(byhour=range(0, 24, 2)),
    )
    ax1, _ = sao_plot.plot_TS(
        df,
        left_yparams=["hF"],
        right_yparams=[],
        ylabels=["h'F, km", ""],
        xlabel="Time, UT",
        left_ylim=[100, 400],
        xlim=xlim,
        major_locator=mdates.HourLocator(byhour=range(0, 24, 6)),
        minor_locator=mdates.HourLocator(byhour=range(0, 24, 2)),
    )
    if baseline is not None:
        _overlay_baseline(ax0, baseline, "foF2", n_baseline_days)
        _overlay_baseline(ax1, baseline, "hF", n_baseline_days)
    for ax in (ax0, ax1):
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
    n_baseline_days: int = 7,
    out_dir: Path = OUTPUT_DIR,
) -> List[Path]:
    """Fetch, plot and save foF2/h'F figures for every station/event pair.

    Each figure spans the 24-hour UT shock day and overlays a quiet-time
    median/STD baseline computed from the ``n_baseline_days`` days
    immediately before it. Failures (no DIDBase coverage, all-NaN rows,
    etc.) are logged as warnings for the offending station/event
    combination and skipped, not raised. A missing baseline does not
    block the plot -- it is simply omitted with a warning.

    Returns:
        List of PNG paths that were successfully written.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: List[Path] = []

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

                slug = _slugify(station["name"])
                out_path = out_dir / (
                    f"{station['ursi']}_{slug}_{event['date']:%Y%m%d}_foF2_hpF.png"
                )
                try:
                    plot_foF2_hpF(
                        df, station, event, [start, end], out_path,
                        baseline=baseline, n_baseline_days=n_baseline_days,
                    )
                except Exception as exc:
                    logger.warning(
                        f"[{station['ursi']}] plotting failed for {event['date']}: {exc}"
                    )
                    continue

                logger.info(f"Saved {out_path}")
                written.append(out_path)

    logger.info(f"Wrote {len(written)} figure(s) to {out_dir}")
    return written


if __name__ == "__main__":
    # Requires DIDBASE_USER / DIDBASE_PASSWORD environment variables set to
    # valid GIRO DIDBase credentials (see https://ulcar.uml.edu/DIDBase/).
    # Example: run only two stations for a single event, with a shorter
    # 3-day quiet baseline instead of the 7-day default:
    #
    #   run(
    #       stations=[s for s in STATIONS if s["ursi"] in ("BC840", "AU930")],
    #       events=[e for e in EVENTS if e["date"] == dt.date(2015, 3, 17)],
    #       n_baseline_days=3,
    #   )
    run()
