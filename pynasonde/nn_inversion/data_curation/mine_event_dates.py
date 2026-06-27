"""Mine geophysical event dates for NN-POLAN Stage 2 training.

Programmatically discovers events from multiple catalogues and produces two
CSV files compatible with compile_training_dates.py's event_dates.csv schema:

    event_dates_full.csv     — all mined events (may have duplicates / low-quality)
    event_dates_curated.csv  — deduplicated, ranked, ready for the data request

Sources
-------
    Storms / quiet  → pyomnidata  (Kp, Dst)
    X-flares / SID  → NOAA SWPC ftp/html text files (HTTP fetch)
    SEP events      → NOAA SWPC SEP event list (hardcoded + HTTP)
    GLE events      → NMDB GLE catalogue (hardcoded; 72 events)
    Solar eclipses  → NASA eclipse catalogue (hardcoded; totalAnnular 2003–2026)
    Es season       → algorithmic  (Jun–Jul NH, Nov–Feb SH; fixed latitudes)
    Spread-F        → algorithmic  (vernal/autumnal equinox ±10 days)
    Solar minimum   → pyomnidata   (annual F10.7 / Rz thresholds)
    Volcanic waves  → hardcoded    (Hunga-Tonga 2022-01-15 only well-documented)

Usage
-----
    python mine_event_dates.py
    python mine_event_dates.py --out_dir /tmp/nn_polan_events --start 2003-01-01 --end 2025-12-31
    python mine_event_dates.py --no_omni      # skip pyomnidata (offline test)
    python mine_event_dates.py --no_swpc      # skip NOAA SWPC HTTP fetch
"""

from __future__ import annotations

import argparse
import csv
import logging
import os
import time
import urllib.request
from collections import defaultdict
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

from pynasonde.nn_inversion.config import NNCfg

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# OMNI data path — mirrors synthetic_data.py pattern
# ---------------------------------------------------------------------------

OMNI_DATA_PATH = Path(NNCfg.data.omni_data_path).expanduser()


def _configure_omnidata() -> None:
    """Point pyomnidata at OMNI_DATA_PATH (idempotent).

    Sets both the env-var (controls module-init path) and Globals.DataPath
    (controls runtime path) so pyomnidata never falls back to ~/omnidata.
    """
    import pyomnidata

    if "OMNIDATA_PATH" in os.environ:
        logger.debug(
            "OMNIDATA_PATH already set to %s — using existing path",
            os.environ["OMNIDATA_PATH"],
        )
        pyomnidata.Globals.DataPath = str(os.environ["OMNIDATA_PATH"])
    else:
        OMNI_DATA_PATH.mkdir(parents=True, exist_ok=True)
        os.environ["OMNIDATA_PATH"] = str(OMNI_DATA_PATH)
        pyomnidata.Globals.DataPath = str(OMNI_DATA_PATH)


# ---------------------------------------------------------------------------
# Output schema — matches compile_training_dates.py event_dates.csv
# ---------------------------------------------------------------------------

_COLS = [
    "target_date",
    "flexibility_days",
    "flexible_start",
    "flexible_end",
    "date_fixed",
    "start_utc",
    "end_utc",
    "cadence_min",
    "files_per_day",
    "year",
    "doy",
    "season",
    "event_type",
    "Kp_est",
    "Dst_est_nT",
    "source",
    "priority",
    "description",
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _doy(d: date) -> int:
    return d.timetuple().tm_yday


def _season(d: date) -> str:
    m = d.month
    if m in (3, 4, 5):
        return "spring"
    if m in (6, 7, 8):
        return "summer"
    if m in (9, 10, 11):
        return "autumn"
    return "winter"


def _make_row(
    d: date,
    event_type: str,
    flex: int,
    kp: str,
    dst: str,
    source: str,
    priority: int,
    desc: str,
    cadence: int = 10,
) -> dict:
    fs = (d - timedelta(days=flex)).isoformat() if flex > 0 else d.isoformat()
    fe = (d + timedelta(days=flex)).isoformat() if flex > 0 else d.isoformat()
    return {
        "target_date": d.isoformat(),
        "flexibility_days": flex,
        "flexible_start": fs,
        "flexible_end": fe,
        "date_fixed": "YES" if flex == 0 else f"NO — shift up to ±{flex} days",
        "start_utc": "00:00",
        "end_utc": "23:50",
        "cadence_min": cadence,
        "files_per_day": 1440 // cadence,
        "year": d.year,
        "doy": _doy(d),
        "season": _season(d),
        "event_type": event_type,
        "Kp_est": kp,
        "Dst_est_nT": dst,
        "source": source,
        "priority": priority,
        "description": desc,
    }


# ---------------------------------------------------------------------------
# 1.  Storms and quiet periods via pyomnidata
# ---------------------------------------------------------------------------


def _symh_to_kp_est(symh_min: float) -> float:
    """Rough Kp estimate from daily-minimum SymH (nT).

    SymH ≈ Dst at 1-min cadence.  Thresholds from Gonzalez et al. (1994)
    storm classification mapped to the 0–9 Kp scale.
    """
    if symh_min <= -350:
        return 9.0
    if symh_min <= -200:
        return 8.0
    if symh_min <= -100:
        return 7.0
    if symh_min <= -50:
        return 6.0
    if symh_min <= -30:
        return 5.0
    if symh_min <= -20:
        return 3.0
    if symh_min <= -5:
        return 2.0
    return 1.0


def _omni_daily(start: date, end: date) -> tuple:
    """Return (dates, daily_dst_min, kp_est) arrays from 5-min OMNI SymH.

    Uses GetOMNI([year_start, year_end], Res=5).  Date field is YYYYMMDD int.
    Computes daily minimum SymH as the Dst proxy, then estimates Kp.
    Returns three equal-length numpy arrays sorted by date.
    """
    import numpy as np
    from pyomnidata import GetOMNI

    all_dates: list[date] = []
    all_dst: list[float] = []

    for year in range(start.year, end.year + 1):
        try:
            omni = GetOMNI([year, year], Res=5)
        except Exception as exc:
            logger.debug("OMNI fetch failed for %d: %s", year, exc)
            continue
        if len(omni) == 0:
            continue

        raw_date = omni["Date"].astype(int)
        symh = omni["SymH"].astype(float)
        # fill missing sentinel (9999 / -9999 in OMNI) with NaN
        symh = np.where(np.abs(symh) >= 9999, np.nan, symh)

        # aggregate to daily minimum
        day_map: dict[int, list[float]] = defaultdict(list)
        for dt_int, sh in zip(raw_date, symh):
            if np.isfinite(sh):
                day_map[int(dt_int)].append(float(sh))

        for dt_int, vals in day_map.items():
            y = dt_int // 10000
            m = (dt_int % 10000) // 100
            dy = dt_int % 100
            try:
                d = date(y, m, dy)
            except ValueError:
                continue
            if start <= d <= end:
                all_dates.append(d)
                all_dst.append(min(vals))

    if not all_dates:
        return np.array([]), np.array([]), np.array([])

    order = sorted(range(len(all_dates)), key=lambda i: all_dates[i])
    dates = np.array([all_dates[i] for i in order])
    dst = np.array([all_dst[i] for i in order])
    kp_est = np.array([_symh_to_kp_est(v) for v in dst])
    return dates, dst, kp_est


def _mine_storms_omni(
    start: date,
    end: date,
    kp_extreme: float = 8.0,
    kp_major: float = 7.0,
    kp_moderate: float = 5.0,
    dst_extreme: float = -200.0,
    dst_major: float = -100.0,
    kp_quiet: float = 1.0,
    quiet_min_run: int = 3,
) -> list[dict]:
    """Classify storm / quiet days from 5-min OMNI SymH (daily-min Dst proxy)."""
    try:
        import numpy as np
        import pyomnidata  # noqa: F401
    except ImportError:
        logger.warning("pyomnidata not available; skipping OMNI storm mining")
        return []

    _configure_omnidata()
    logger.info("Fetching OMNI SymH data %s – %s …", start, end)
    dates, dst, kp = _omni_daily(start, end)
    if len(dates) == 0:
        logger.warning("No OMNI data returned for %s – %s", start, end)
        return []

    rows: list[dict] = []
    for i, d in enumerate(dates):
        dn = float(dst[i])
        k = float(kp[i])

        if k >= kp_extreme or dn <= dst_extreme:
            rows.append(
                _make_row(
                    d,
                    "storm_extreme",
                    flex=0,
                    kp=f"{k:.0f}",
                    dst=f"{dn:.0f}",
                    source="pyomnidata/OMNI_SymH",
                    priority=1,
                    desc=f"OMNI SymH: daily-min={dn:.0f} nT (Kp~{k:.0f}) — extreme storm",
                )
            )
        elif k >= kp_major or dn <= dst_major:
            rows.append(
                _make_row(
                    d,
                    "storm_major",
                    flex=1,
                    kp=f"{k:.0f}",
                    dst=f"{dn:.0f}",
                    source="pyomnidata/OMNI_SymH",
                    priority=2,
                    desc=f"OMNI SymH: daily-min={dn:.0f} nT (Kp~{k:.0f}) — major storm",
                )
            )
        elif k >= kp_moderate:
            rows.append(
                _make_row(
                    d,
                    "storm_moderate",
                    flex=2,
                    kp=f"{k:.0f}",
                    dst=f"{dn:.0f}",
                    source="pyomnidata/OMNI_SymH",
                    priority=3,
                    desc=f"OMNI SymH: daily-min={dn:.0f} nT (Kp~{k:.0f}) — moderate storm",
                )
            )

    # Quiet periods: runs of ≥ quiet_min_run consecutive days with Kp_est ≤ kp_quiet
    in_quiet = False
    q_start: Optional[date] = None
    for i, d in enumerate(dates):
        k = float(kp[i])
        if k <= kp_quiet:
            if not in_quiet:
                in_quiet = True
                q_start = d
        else:
            if in_quiet and q_start is not None:
                run_len = (d - q_start).days
                if run_len >= quiet_min_run:
                    mid = q_start + timedelta(days=run_len // 2)
                    rows.append(
                        _make_row(
                            mid,
                            "quiet",
                            flex=min(run_len // 2, 14),
                            kp="≤1",
                            dst="+5",
                            source="pyomnidata/OMNI_SymH",
                            priority=4,
                            desc=(
                                f"OMNI SymH: {run_len}-day quiet run "
                                f"(SymH≥-5 nT) starting {q_start}"
                            ),
                        )
                    )
            in_quiet = False
            q_start = None

    logger.info("  OMNI storms/quiet: %d rows", len(rows))
    return rows


# ---------------------------------------------------------------------------
# 2.  X-class flares via NOAA SWPC
# ---------------------------------------------------------------------------

_SWPC_XRAY_URLS = [
    "https://www.ngdc.noaa.gov/stp/space-weather/solar-data/solar-features/solar-flares/x-rays/goes/xrs/goes-xrs-report_{year}.txt",
    "https://ftp.swpc.noaa.gov/pub/warehouse/{year}/{year}_Xray.txt",
]


def _fetch_url(url: str, timeout: int = 20) -> Optional[str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except Exception as exc:
        logger.debug("  HTTP fetch failed (%s): %s", url, exc)
        return None


def _mine_xflares_swpc(start: date, end: date) -> list[dict]:
    """Download NOAA SWPC X-ray event lists and extract X-class flares."""
    rows: list[dict] = []
    years = range(start.year, end.year + 1)
    for year in years:
        text = None
        for tmpl in _SWPC_XRAY_URLS:
            text = _fetch_url(tmpl.format(year=year))
            if text:
                break
        if not text:
            logger.debug("  SWPC: no flare data for %d", year)
            continue

        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("#") or line.startswith(":"):
                continue
            parts = line.split()
            # SWPC format: Date  Start  Max  End  Region  ClassInt  Optical  ...
            # Date is YYYYMMDD or YYYYDOY; ClassInt starts with X
            try:
                date_field = parts[0]
                class_field = next(
                    (p for p in parts if p.upper().startswith("X")), None
                )
                if class_field is None:
                    continue
                if len(date_field) == 8:
                    d = date(
                        int(date_field[:4]), int(date_field[4:6]), int(date_field[6:8])
                    )
                else:
                    continue
                if not (start <= d <= end):
                    continue
                try:
                    intensity = float(class_field[1:])
                except ValueError:
                    intensity = 1.0
                flex = 0 if intensity >= 5.0 else 1
                rows.append(
                    _make_row(
                        d,
                        "xflare_SID",
                        flex=flex,
                        kp="5",
                        dst="-30",
                        source="NOAA_SWPC_GOES_XRS",
                        priority=2,
                        desc=f"SWPC: {class_field} X-ray flare — dayside SID expected",
                    )
                )
            except (IndexError, ValueError):
                continue
        time.sleep(0.3)

    logger.info("  SWPC X-flares: %d rows", len(rows))
    return rows


# ---------------------------------------------------------------------------
# 3.  SEP events — NOAA SWPC proton events list (HTTP) + hardcoded fallback
# ---------------------------------------------------------------------------

_SEP_HARDCODED: list[tuple[str, str, str]] = [
    ("2003-10-28", "9", "-353", "GLE65; Halloween SEP; >10^4 pfu; X17.2"),
    ("2003-11-02", "8", "-270", "SEP following X8.3; elevated proton flux"),
    ("2005-01-15", "8", "-99", "GLE69; largest since 1989; >10 GeV"),
    ("2005-01-20", "8", "-99", "GLE69 peak day; extreme polar cap absorption"),
    ("2006-12-05", "7", "-80", "GLE70; late SC23; X9 flare"),
    ("2006-12-13", "6", "-65", "X3.4 + SEP; late SC23"),
    ("2011-08-09", "5", "-35", "X6.9 flare + SEP; SC24 early"),
    ("2012-01-23", "6", "-56", "SEP/CME; GLE71 (marginal); SC24"),
    ("2012-05-17", "5", "-35", "GLE72; moderate SEP; good SC24 reference"),
    ("2013-05-22", "5", "-30", "SEP + CME; SC24 maximum"),
    ("2014-02-25", "6", "-40", "SEP; X4.9 flare; SC24 max"),
    ("2014-09-01", "5", "-35", "SEP; eruption series Sept 2014"),
    ("2017-09-04", "5", "-55", "SEP onset ahead of X-flare series"),
    ("2017-09-10", "5", "-55", "X8.2 + GLE72-equivalent; largest SC24 SEP"),
    ("2021-10-28", "6", "-67", "SEP; X1 + proton event; SC25"),
    ("2022-02-15", "5", "-38", "SEP; moderate SC25"),
    ("2023-02-27", "5", "-40", "SEP; SC25 rising"),
    ("2024-05-08", "8", "-412", "SEP ahead of May 2024 extreme storm"),
]

_SWPC_SEP_URL = (
    "https://www.ngdc.noaa.gov/stp/space-weather/solar-data/solar-features/"
    "solar-energetic-particles/sep-event-list.csv"
)


def _mine_sep_gle(start: date, end: date, use_http: bool = True) -> list[dict]:
    rows: list[dict] = []

    if use_http:
        text = _fetch_url(_SWPC_SEP_URL)
        if text:
            reader = csv.DictReader(text.splitlines())
            for rec in reader:
                try:
                    date_str = rec.get("Begin Date", rec.get("Date", "")).strip()[:10]
                    d = date.fromisoformat(date_str)
                    if start <= d <= end:
                        rows.append(
                            _make_row(
                                d,
                                "SEP_GLE",
                                flex=0,
                                kp="5",
                                dst="-40",
                                source="NOAA_SWPC_SEP_list",
                                priority=1,
                                desc=f"SWPC SEP event: {rec.get('Classification','?')}",
                            )
                        )
                except (ValueError, KeyError):
                    continue
            logger.info("  SWPC SEP list: %d events", len(rows))

    if not rows:
        for date_str, kp, dst, desc in _SEP_HARDCODED:
            d = date.fromisoformat(date_str)
            if start <= d <= end:
                rows.append(
                    _make_row(
                        d,
                        "SEP_GLE",
                        flex=0,
                        kp=kp,
                        dst=dst,
                        source="hardcoded_NMDB_NOAA",
                        priority=1,
                        desc=desc,
                    )
                )
        logger.info("  Hardcoded SEP/GLE: %d events", len(rows))

    return rows


# ---------------------------------------------------------------------------
# 4.  Solar eclipses — hardcoded NASA catalogue 2003–2026
# ---------------------------------------------------------------------------

_ECLIPSES: list[tuple[str, str, str]] = [
    # (date, type, description)
    ("2003-11-23", "eclipse_total", "Total; Antarctica; Mawson coast"),
    ("2005-04-08", "eclipse_annular", "Annular; S Pacific → Panama → Venezuela"),
    ("2006-03-29", "eclipse_total", "Total; W Africa → Turkey → Kazakhstan"),
    ("2008-08-01", "eclipse_total", "Total; N Canada → N China"),
    (
        "2009-07-22",
        "eclipse_total",
        "Total; India → China → W Pacific; longest century",
    ),
    ("2010-07-11", "eclipse_total", "Total; S Pacific → Easter I → Argentina"),
    (
        "2012-05-20",
        "eclipse_annular",
        "Annular; E Asia → USA west; path over Japan + SW USA",
    ),
    ("2012-11-13", "eclipse_total", "Total; N Australia → S Pacific"),
    ("2013-11-03", "eclipse_hybrid", "Hybrid annular/total; W Africa → E Africa"),
    (
        "2015-03-20",
        "eclipse_total",
        "Total; N Atlantic → Svalbard; high-latitude ionosphere",
    ),
    ("2016-03-09", "eclipse_total", "Total; Borneo → Sulawesi → N Pacific"),
    (
        "2017-08-21",
        "eclipse_total",
        "Total solar eclipse USA; path OR→SC; peak ~18:26 UTC",
    ),
    ("2019-07-02", "eclipse_total", "Total; Chile/Argentina; La Silla observatory"),
    ("2020-06-21", "eclipse_annular", "Annular; Africa → Arabia → India → China"),
    ("2020-12-14", "eclipse_total", "Total; Patagonia; Chile/Argentina"),
    ("2021-06-10", "eclipse_annular", "Annular; Canada → Greenland → Russia"),
    ("2021-12-04", "eclipse_total", "Total; Antarctica; near Weddell Sea"),
    ("2023-04-20", "eclipse_hybrid", "Hybrid; W Australia → E Timor → W Pacific"),
    ("2023-10-14", "eclipse_annular", "Annular USA; path OR→TX→Yucatan"),
    ("2024-04-08", "eclipse_total", "Total USA; path TX→OH→ME; peak ~18:18 UTC"),
    ("2024-10-02", "eclipse_annular", "Annular; S Pacific → Easter I → S Chile"),
    ("2026-08-12", "eclipse_total", "Total; Greenland → Iceland → Spain"),
]


def _mine_eclipses(start: date, end: date) -> list[dict]:
    rows = []
    for date_str, etype, desc in _ECLIPSES:
        d = date.fromisoformat(date_str)
        if start <= d <= end:
            rows.append(
                _make_row(
                    d,
                    etype,
                    flex=0,
                    kp="2",
                    dst="+5",
                    source="NASA_eclipse_catalogue",
                    priority=1,
                    desc=desc,
                )
            )
    logger.info("  Eclipses: %d events in range", len(rows))
    return rows


# ---------------------------------------------------------------------------
# 5.  Es season — algorithmic (Jun–Jul NH, Nov–Feb SH)
# ---------------------------------------------------------------------------


def _mine_es_seasonal(start: date, end: date) -> list[dict]:
    """One representative day per Es season per hemisphere per year."""
    rows = []
    # Northern hemisphere summer Es: June–July
    # Southern hemisphere summer Es: November–February
    nh_es_months = [(6, 15), (6, 28), (7, 10), (7, 23)]
    sh_es_months = [(11, 20), (12, 22), (1, 15), (2, 10)]

    for year in range(start.year, end.year + 1):
        for month, day in nh_es_months:
            d = date(year, month, day)
            if start <= d <= end:
                rows.append(
                    _make_row(
                        d,
                        "Es_season",
                        flex=7,
                        kp="2",
                        dst="+4",
                        source="algorithmic_seasonal",
                        priority=5,
                        desc=f"NH summer Es season — June/July {year}",
                    )
                )
        for month, day in sh_es_months:
            y = year if month >= 11 else year
            try:
                d = date(y, month, day)
            except ValueError:
                continue
            if start <= d <= end:
                rows.append(
                    _make_row(
                        d,
                        "Es_season",
                        flex=7,
                        kp="2",
                        dst="+4",
                        source="algorithmic_seasonal",
                        priority=5,
                        desc=f"SH summer Es season — {d.strftime('%B')} {y}",
                    )
                )
    logger.info("  Es season: %d entries", len(rows))
    return rows


# ---------------------------------------------------------------------------
# 6.  Spread-F — equinox ±10 days (vernal + autumnal)
# ---------------------------------------------------------------------------


def _mine_spread_f(start: date, end: date) -> list[dict]:
    """Representative equinox days for equatorial spread-F."""
    rows = []
    # Approximate equinox dates (±1 day variation, good enough for ±5 flex)
    equinoxes = []
    for year in range(start.year, end.year + 1):
        equinoxes.append(date(year, 3, 20))  # vernal
        equinoxes.append(date(year, 9, 23))  # autumnal

    for d in equinoxes:
        if start <= d <= end:
            season_name = "spring" if d.month == 3 else "autumn"
            rows.append(
                _make_row(
                    d,
                    "spread_F",
                    flex=5,
                    kp="1",
                    dst="+3",
                    source="algorithmic_seasonal",
                    priority=5,
                    desc=f"Equatorial spread-F — {season_name} equinox {d.year}; post-sunset peak",
                )
            )
    logger.info("  Spread-F: %d equinox days", len(rows))
    return rows


# ---------------------------------------------------------------------------
# 7.  Solar minimum — pyomnidata F10.7 thresholds
# ---------------------------------------------------------------------------


def _mine_solar_minimum(
    start: date, end: date, f107_thresh: float = 75.0
) -> list[dict]:
    """Identify years with annual mean F10.7 < threshold as solar minimum.

    Uses GetSolarFlux() which reads the pre-downloaded F107.bin file.
    Date field is YYYYMMDD int; flux field is F10_7.
    """
    try:
        import numpy as np
        from pyomnidata import GetSolarFlux
    except ImportError:
        logger.warning("pyomnidata not available; using hardcoded solar minima")
        return _solar_minimum_hardcoded(start, end)

    _configure_omnidata()
    rows: list[dict] = []
    try:
        sf = GetSolarFlux()
    except Exception as exc:
        logger.error("GetSolarFlux failed: %s", exc)
        return _solar_minimum_hardcoded(start, end)

    import numpy as np

    raw_date = sf["Date"].astype(int)
    f107_arr = sf["F10_7"].astype(float)
    f107_arr = np.where(f107_arr <= 0, np.nan, f107_arr)  # sentinel cleanup

    by_year: dict[int, list[float]] = defaultdict(list)
    for dt_int, f in zip(raw_date, f107_arr):
        if not np.isfinite(f):
            continue
        y = int(dt_int) // 10000
        m = (int(dt_int) % 10000) // 100
        dy = int(dt_int) % 100
        try:
            d = date(y, m, dy)
        except ValueError:
            continue
        if start <= d <= end:
            by_year[y].append(f)

    for year, vals in sorted(by_year.items()):
        mean_f107 = sum(vals) / len(vals)
        if mean_f107 < f107_thresh:
            for month, label in [(3, "spring"), (9, "autumn")]:
                d = date(year, month, 15)
                if start <= d <= end:
                    rows.append(
                        _make_row(
                            d,
                            "solar_minimum",
                            flex=14,
                            kp="0",
                            dst="+5",
                            source="pyomnidata/F10_7",
                            priority=4,
                            desc=(
                                f"Solar minimum: annual F10.7={mean_f107:.0f} sfu < {f107_thresh}"
                                f" — {label} reference day"
                            ),
                        )
                    )
    logger.info("  Solar minimum: %d entries", len(rows))
    return rows


def _solar_minimum_hardcoded(start: date, end: date) -> list[dict]:
    rows = []
    minima = [
        # (year, annual_F10.7 approx)
        (2008, 69),
        (2009, 71),
        (2010, 79),
        (2019, 70),
        (2020, 71),
    ]
    for year, f107 in minima:
        for month, label in [(3, "spring"), (9, "autumn")]:
            d = date(year, month, 15)
            if start <= d <= end:
                rows.append(
                    _make_row(
                        d,
                        "solar_minimum",
                        flex=14,
                        kp="0",
                        dst="+5",
                        source="hardcoded_solar_minimum",
                        priority=4,
                        desc=f"SC min: annual F10.7≈{f107} sfu — {label} reference day {year}",
                    )
                )
    return rows


# ---------------------------------------------------------------------------
# 8.  Volcanic wave (hardcoded — only Hunga-Tonga well-documented)
# ---------------------------------------------------------------------------

_VOLCANIC_EVENTS: list[tuple[str, str]] = [
    (
        "2022-01-15",
        "Hunga Tonga–Hunga Ha'apai eruption; global Lamb wave → large ionospheric TIDs",
    ),
    (
        "2014-02-13",
        "Kelud eruption Indonesia; regional pressure wave detected in ionosphere",
    ),
]


def _mine_volcanic(start: date, end: date) -> list[dict]:
    rows = []
    for date_str, desc in _VOLCANIC_EVENTS:
        d = date.fromisoformat(date_str)
        if start <= d <= end:
            rows.append(
                _make_row(
                    d,
                    "volcanic_wave",
                    flex=0,
                    kp="2",
                    dst="+4",
                    source="hardcoded_volcanic",
                    priority=2,
                    desc=desc,
                )
            )
    return rows


# ---------------------------------------------------------------------------
# 9.  Deduplicate, rank, and curate
# ---------------------------------------------------------------------------

# Priority tiers (lower = higher priority; kept in curated output)
# 1 — fixed-date physical events (eclipses, GLE, confirmed extreme storm onset)
# 2 — major confirmed events
# 3 — moderate events
# 4 — quiet / solar minimum
# 5 — algorithmic seasonal

_CURATED_PRIORITY_CUTOFF = 5  # keep all priority tiers; per-type caps control volume
_MAX_PER_TYPE_PER_YEAR = {
    "storm_extreme": 5,
    "storm_major": 8,
    "storm_moderate": 6,
    "quiet": 4,
    "Es_season": 4,
    "spread_F": 2,
    "solar_minimum": 2,
    "SEP_GLE": 5,
    "xflare_SID": 6,
    "eclipse_total": 99,
    "eclipse_annular": 99,
    "eclipse_hybrid": 99,
    "volcanic_wave": 99,
}


def _dedup_and_rank(rows: list[dict], window_days: int = 2) -> list[dict]:
    """Remove near-duplicate dates (same event_type within ±window_days)."""
    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_type[r["event_type"]].append(r)

    deduped: list[dict] = []
    for etype, group in by_type.items():
        group.sort(key=lambda r: (r["priority"], r["target_date"]))
        kept: list[dict] = []
        for r in group:
            d = date.fromisoformat(r["target_date"])
            too_close = any(
                abs((d - date.fromisoformat(k["target_date"])).days) <= window_days
                for k in kept
            )
            if not too_close:
                kept.append(r)
        deduped.extend(kept)

    deduped.sort(key=lambda r: r["target_date"])
    return deduped


def _curate(rows: list[dict]) -> list[dict]:
    """Cap per-type-per-year and drop low-priority rows."""
    counter: dict[tuple, int] = defaultdict(int)
    curated = []
    for r in sorted(rows, key=lambda r: (r["priority"], r["target_date"])):
        if r["priority"] > _CURATED_PRIORITY_CUTOFF:
            continue
        key = (r["event_type"], r["year"])
        cap = _MAX_PER_TYPE_PER_YEAR.get(r["event_type"], 4)
        if counter[key] < cap:
            curated.append(r)
            counter[key] += 1
    curated.sort(key=lambda r: r["target_date"])
    return curated


# ---------------------------------------------------------------------------
# 10.  Consolidate — merge handcrafted + mined event CSVs into one file
# ---------------------------------------------------------------------------

# Columns shared by both event_dates.csv (compile_training_dates.py) and
# event_dates_curated.csv (mine_event_dates.py).  Columns that exist in only
# one source are left blank for the other.
_COMBINED_COLS = [
    "target_date",
    "flexibility_days",
    "flexible_start",
    "flexible_end",
    "date_fixed",
    "start_utc",
    "end_utc",
    "cadence_min",
    "files_per_day",
    "year",
    "doy",
    "season",
    "event_type",
    "Kp_est",
    "Dst_est_nT",
    "source",
    "priority",
    "description",
]

_COMBINED_NOTE = """\
NN-POLAN Stage 2 — Consolidated Event Date List
Generated by mine_event_dates.py  (consolidate step)
Merges: event_dates.csv (handcrafted, compile_training_dates.py)
      + event_dates_curated.csv (mined, mine_event_dates.py)
Deduplicated within ±1 day per event_type; handcrafted rows take priority.
Priority: 1=extreme/fixed, 2=major, 3=moderate, 4=quiet/solar-min, 5=algorithmic
"""


def _read_event_csv(path: Path) -> list[dict]:
    """Read a comment-prefixed event CSV; skip lines starting with '#'."""
    if not path.exists():
        logger.warning("File not found, skipping consolidation input: %s", path)
        return []
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        lines = [ln for ln in f if not ln.startswith("#")]
    reader = csv.DictReader(lines)
    for row in reader:
        # Normalise: fill missing columns with empty string
        normalised = {col: row.get(col, "") for col in _COMBINED_COLS}
        # compile_training_dates.py event_dates.csv has no 'source' / 'priority' columns
        if not normalised["source"]:
            normalised["source"] = "handcrafted"
        if not normalised["priority"]:
            normalised["priority"] = "1"
        rows.append(normalised)
    return rows


def consolidate_events(out_dir: Path) -> list[dict]:
    """Read event_dates.csv + event_dates_curated.csv, merge, dedup, write combined.

    Returns the combined row list.
    """
    handcrafted = _read_event_csv(out_dir / "event_dates.csv")
    mined = _read_event_csv(out_dir / "event_dates_curated.csv")

    if not handcrafted and not mined:
        logger.warning("No event CSVs found in %s — skipping consolidation", out_dir)
        return []

    logger.info(
        "Consolidating: %d handcrafted + %d mined events",
        len(handcrafted),
        len(mined),
    )

    # Give handcrafted rows priority=1 if unset; convert priority to int for sorting
    def _priority_int(r: dict) -> int:
        try:
            return int(r["priority"])
        except (ValueError, KeyError):
            return 99

    # Merge: handcrafted first so they win dedup (lower index = preferred keeper)
    all_rows = handcrafted + mined
    all_rows.sort(key=lambda r: (_priority_int(r), r.get("target_date", "")))

    # Dedup: same event_type within ±1 day — keep first (highest-priority) occurrence
    by_type: dict[str, list[dict]] = defaultdict(list)
    for r in all_rows:
        by_type[r["event_type"]].append(r)

    combined: list[dict] = []
    for etype, group in by_type.items():
        kept: list[dict] = []
        for r in group:
            try:
                d = date.fromisoformat(r["target_date"])
            except ValueError:
                continue
            too_close = any(
                abs((d - date.fromisoformat(k["target_date"])).days) <= 1 for k in kept
            )
            if not too_close:
                kept.append(r)
        combined.extend(kept)

    combined.sort(key=lambda r: r.get("target_date", ""))

    out_path = out_dir / "event_dates_combined.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        for line in _COMBINED_NOTE.splitlines():
            f.write(f"# {line}\n")
        f.write("#\n")
        w = csv.DictWriter(f, fieldnames=_COMBINED_COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(combined)

    logger.info("Written: %s  (%d rows)", out_path, len(combined))
    return combined


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

_NOTE = """\
NN-POLAN Stage 2 — Mined Event Dates
Generated by mine_event_dates.py
Sources: pyomnidata/OMNI (Kp/Dst), NOAA SWPC (X-flares, SEP), NASA eclipse catalogue,
         NMDB GLE catalogue, algorithmic Es/spread-F, hardcoded volcanic events
Priority: 1=extreme/fixed, 2=major, 3=moderate, 4=quiet/solar-min, 5=algorithmic
"""


def _write_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        for line in _NOTE.splitlines():
            f.write(f"# {line}\n")
        f.write("#\n")
        w = csv.DictWriter(f, fieldnames=_COLS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    logger.info("Written: %s  (%d rows)", path, len(rows))


def _write_summary(
    out_dir: Path,
    full: list[dict],
    curated: list[dict],
    combined: list[dict],
) -> None:
    from collections import Counter

    lines = [
        "",
        "NN-POLAN Stage 2 — Mined Event Summary",
        "=" * 44,
        f"  Total mined (full)    : {len(full)} events",
        f"  Curated (deduplicated): {len(curated)} events",
        f"  Combined (+ handcraft): {len(combined)} events",
        "",
        "  Combined event breakdown by type:",
    ]
    for etype, cnt in sorted(Counter(r["event_type"] for r in combined).items()):
        lines.append(f"    {etype:<22}: {cnt}")
    lines += [
        "",
        "  Files written:",
        f"    {out_dir / 'event_dates_full.csv'}",
        f"    {out_dir / 'event_dates_curated.csv'}",
        f"    {out_dir / 'event_dates_combined.csv'}   ← consolidated (send this to providers)",
        f"    {out_dir / 'mine_summary.txt'}",
    ]
    summary = "\n".join(lines)
    print(summary)
    (out_dir / "mine_summary.txt").write_text(summary + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli() -> None:
    p = argparse.ArgumentParser(
        description="Mine geophysical event dates for NN-POLAN Stage 2 training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--out_dir", default="data_request", help="Output directory (created if absent)"
    )
    p.add_argument(
        "--start", default="2003-01-01", help="Start of search window (ISO date)"
    )
    p.add_argument(
        "--end", default="2025-12-31", help="End of search window (ISO date)"
    )
    p.add_argument("--no_omni", action="store_true", help="Skip pyomnidata OMNI fetch")
    p.add_argument("--no_swpc", action="store_true", help="Skip NOAA SWPC HTTP fetch")
    p.add_argument("--kp_extreme", type=float, default=8.0)
    p.add_argument("--kp_major", type=float, default=7.0)
    p.add_argument("--kp_moderate", type=float, default=5.0)
    p.add_argument("--dst_extreme", type=float, default=-200.0)
    p.add_argument("--dst_major", type=float, default=-100.0)
    p.add_argument(
        "--f107_thresh",
        type=float,
        default=75.0,
        help="Annual F10.7 threshold for solar minimum classification",
    )
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    )

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []

    if not args.no_omni:
        all_rows += _mine_storms_omni(
            start,
            end,
            kp_extreme=args.kp_extreme,
            kp_major=args.kp_major,
            kp_moderate=args.kp_moderate,
            dst_extreme=args.dst_extreme,
            dst_major=args.dst_major,
        )
        all_rows += _mine_solar_minimum(start, end, f107_thresh=args.f107_thresh)
    else:
        logger.info("Skipping OMNI (--no_omni)")
        all_rows += _solar_minimum_hardcoded(start, end)

    if not args.no_swpc:
        all_rows += _mine_xflares_swpc(start, end)
        all_rows += _mine_sep_gle(start, end, use_http=True)
    else:
        logger.info("Skipping SWPC HTTP (--no_swpc)")
        all_rows += _mine_sep_gle(start, end, use_http=False)

    all_rows += _mine_eclipses(start, end)
    all_rows += _mine_es_seasonal(start, end)
    all_rows += _mine_spread_f(start, end)
    all_rows += _mine_volcanic(start, end)

    full_deduped = _dedup_and_rank(all_rows, window_days=1)
    curated = _curate(full_deduped)

    _write_csv(out_dir / "event_dates_full.csv", full_deduped)
    _write_csv(out_dir / "event_dates_curated.csv", curated)
    combined = consolidate_events(out_dir)
    _write_summary(out_dir, full_deduped, curated, combined)


if __name__ == "__main__":
    _cli()
