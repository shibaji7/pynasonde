"""foF2 / h'F response of the CONUS Digisonde network to IP-shock storms.

This script queries GIRO DIDBase (via :class:`pynasonde.digisonde.didbase.
DidBaseConnector`) for every (station, event) combination defined below,
parses the returned SAO characteristics with :class:`pynasonde.digisonde.
parsers.sao.SaoExtractor`, and produces a two-panel time-series plot (foF2
on top, h'F on the bottom) spanning a configurable window around each
interplanetary (IP) shock / CME-driven storm onset.

Recipe
------
1. ``STATIONS`` lists the CONUS Digisonde network (URSI code, name, and
   coordinates) to query.
2. ``EVENTS`` lists the storm/shock dates of interest (Dst minima from
   2010-2020). Only the shock/storm date is known precisely here; the
   default query window is +/- 24 h around 00:00 UT of that date, via the
   ``pre_hours``/``post_hours`` parameters.
3. :func:`fetch_scaled_dataframe` pulls raw SAO ionogram files for one
   station/event window through ``DidBaseConnector.fetch_files`` and parses
   the scaled characteristics (``foF2``, ``hF``) with
   ``SaoExtractor.load_SAO_files``.
4. :func:`plot_foF2_hpF` renders the two-panel figure and saves it under
   ``tmp/ip/``.
5. :func:`run` loops over every station/event pair, skipping (with a
   logged warning) any combination that DIDBase has no coverage for, or
   that otherwise fails to download/parse, instead of crashing the run.

DIDBase credentials are read from the ``DIDBASE_USER``, ``DIDBASE_PASSWORD``
and (optionally) ``DIDBASE_URL``/``DIDBASE_ROLE`` environment variables —
register for DIDBase access at https://ulcar.uml.edu/DIDBase/ before running
this script. See ``if __name__ == "__main__":`` below for a usage example.
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
from pynasonde.digisonde.parsers.sao import SaoExtractor

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


def fetch_scaled_dataframe(
    connector: DidBaseConnector,
    station: Dict,
    event_date: dt.date,
    pre_hours: float = 24,
    post_hours: float = 24,
    raw_dir: Optional[Path] = None,
) -> Optional[pd.DataFrame]:
    """Fetch and parse scaled SAO characteristics for one station/event window.

    Downloads raw SAO ionogram files for ``station`` over
    ``[00:00 UT of event_date - pre_hours, 00:00 UT of event_date + post_hours]``
    via ``DidBaseConnector.fetch_files`` and parses them with
    ``SaoExtractor.load_SAO_files(func_name="scaled")``.

    Args:
        connector: An already-connected ``DidBaseConnector``.
        station: One entry from ``STATIONS`` (must have a ``ursi`` key).
        event_date: The storm/shock date (UT).
        pre_hours: Hours before 00:00 UT of ``event_date`` to include.
        post_hours: Hours after 00:00 UT of ``event_date`` to include.
        raw_dir: Directory to download raw SAO files into. Defaults to
            ``tmp/ip/raw/{URSI}_{YYYYMMDD}``.

    Returns:
        A DataFrame with (at least) ``datetime``, ``foF2`` and ``hF``
        columns, or ``None`` if no usable data was found/parsed.
    """
    ursi = station["ursi"]
    start = dt.datetime.combine(event_date, dt.time.min) - dt.timedelta(hours=pre_hours)
    end = dt.datetime.combine(event_date, dt.time.min) + dt.timedelta(hours=post_hours)
    raw_dir = raw_dir or OUTPUT_DIR / "raw" / f"{ursi}_{event_date:%Y%m%d}"

    try:
        paths = connector.fetch_files(
            station=ursi, start=start, end=end, fmt="SAO", out_dir=raw_dir
        )
    except Exception as exc:  # station unknown to DIDBase, network error, etc.
        logger.warning(f"[{ursi}] fetch_files failed for {event_date}: {exc}")
        return None

    if not paths:
        logger.warning(f"[{ursi}] no SAO coverage for {event_date} in DIDBase.")
        return None

    try:
        df = SaoExtractor.load_SAO_files(
            folders=[str(raw_dir)],
            func_name="scaled",
            extract_time_from_name=True,
            extract_stn_from_name=True,
        )
    except Exception as exc:
        logger.warning(f"[{ursi}] failed to parse SAO files for {event_date}: {exc}")
        return None

    if df is None or df.empty:
        logger.warning(f"[{ursi}] SAO parsing returned no rows for {event_date}.")
        return None

    missing = [c for c in ("foF2", "hF") if c not in df.columns or df[c].isna().all()]
    if missing:
        logger.warning(
            f"[{ursi}] {event_date}: no usable data for {missing} "
            "(malformed record or XML SAO variant without h'F)."
        )
        return None

    df = df.sort_values("datetime").reset_index(drop=True)
    return df


def plot_foF2_hpF(
    df: pd.DataFrame,
    station: Dict,
    event: Dict,
    xlim: List[dt.datetime],
    out_path: Path,
    font_size: float = 14,
) -> None:
    """Render a two-panel foF2 (top) / h'F (bottom) time-series figure.

    Args:
        df: DataFrame from :func:`fetch_scaled_dataframe`.
        station: One entry from ``STATIONS``.
        event: One entry from ``EVENTS``.
        xlim: ``[start, end]`` datetimes for the x-axis.
        out_path: PNG path to save the figure to.
        font_size: Base font size for the figure.
    """
    setsize(font_size)
    title = f"{station['ursi']} {station['name']} / {event['name']}"
    sao_plot = SaoSummaryPlots(
        fig_title=title,
        nrows=2,
        ncols=1,
        figsize=(8, 3),
        font_size=font_size,
    )
    sao_plot.plot_TS(
        df,
        left_yparams=["foF2"],
        right_yparams=[],
        ylabels=["foF2, MHz", ""],
        xlabel="",
        left_ylim=[1, 15],
        xlim=xlim,
        major_locator=mdates.HourLocator(byhour=range(0, 24, 12)),
        minor_locator=mdates.HourLocator(byhour=range(0, 24, 4)),
    )
    sao_plot.plot_TS(
        df,
        left_yparams=["hF"],
        right_yparams=[],
        ylabels=["h'F, km", ""],
        xlabel="Time, UT",
        left_ylim=[100, 400],
        xlim=xlim,
        major_locator=mdates.HourLocator(byhour=range(0, 24, 12)),
        minor_locator=mdates.HourLocator(byhour=range(0, 24, 4)),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sao_plot.save(str(out_path))
    sao_plot.close()


def run(
    stations: List[Dict] = STATIONS,
    events: List[Dict] = EVENTS,
    pre_hours: float = 24,
    post_hours: float = 24,
    out_dir: Path = OUTPUT_DIR,
) -> List[Path]:
    """Fetch, plot and save foF2/h'F figures for every station/event pair.

    Failures (no DIDBase coverage, malformed SAO records, missing
    characteristics, etc.) are logged as warnings for the offending
    station/event combination and skipped, not raised.

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
            start = dt.datetime.combine(event["date"], dt.time.min) - dt.timedelta(
                hours=pre_hours
            )
            end = dt.datetime.combine(event["date"], dt.time.min) + dt.timedelta(
                hours=post_hours
            )
            for station in stations:
                df = fetch_scaled_dataframe(
                    connector,
                    station,
                    event["date"],
                    pre_hours=pre_hours,
                    post_hours=post_hours,
                )
                if df is None:
                    continue

                slug = _slugify(station["name"])
                out_path = out_dir / (
                    f"{station['ursi']}_{slug}_{event['date']:%Y%m%d}_foF2_hpF.png"
                )
                try:
                    plot_foF2_hpF(df, station, event, [start, end], out_path)
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
    # Example: run only two stations for a single event, with a narrower
    # +/-12 h window:
    #
    #   run(
    #       stations=[s for s in STATIONS if s["ursi"] in ("BC840", "AU930")],
    #       events=[e for e in EVENTS if e["date"] == dt.date(2015, 3, 17)],
    #       pre_hours=12,
    #       post_hours=12,
    #   )
    run()
