"""Compile ionosonde data-request date lists for NN-POLAN Stage 2 training.

Produces two CSV files designed for a flexible, provider-friendly data request:

    baseline_dates.csv
    ------------------
    Stratified opportunistic sampling across solar cycles 24–25.
    The dataset is divided into ~45-day windows.  For each window the
    provider supplies ANY 1 good-quality day they have available —
    the exact date is entirely up to them.  This removes the burden of
    checking whether the instrument was operational on a specific calendar
    date, while still ensuring broad coverage of diurnal, seasonal, and
    solar-cycle variability.
    Cadence: 30 min (48 files/day) | ~98 windows → ~4 700 files

    event_dates.csv
    ---------------
    Targeted event days where the date is either fixed by nature (eclipses,
    known storms) or has a stated flexibility window (Es season, spread-F).
    For fixed events the provider shares what they have; if the instrument
    was not running that day they skip it.  For flexible events they may
    shift ±N days within the stated window.
    Cadence: 10 min (144 files/day) | ~55 entries → ~7 900 files

    Grand total ≈ 12 600 files

Key message to data providers
------------------------------
"We do not require data on specific dates.  For the baseline we only ask
that you cover as many 45-day windows as possible — one good day per
window, your choice.  For events, exact dates are listed but instrument
availability takes priority; please share whatever you have nearest to
the listed date within the stated flexibility window."

Usage
-----
    python compile_training_dates.py
    python compile_training_dates.py --out_dir /tmp/nn_polan_dates
    python compile_training_dates.py --baseline_start 2010-01-01 --step_days 45
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _doy(d: date) -> int:
    return d.timetuple().tm_yday


def _solar_cycle_phase(d: date) -> str:
    if d < date(2013, 1, 1):
        return "SC24_rising"
    elif d < date(2014, 4, 1):
        return "SC24_maximum"
    elif d < date(2019, 12, 1):
        return "SC24_declining_to_minimum"
    elif d < date(2022, 6, 1):
        return "SC25_rising"
    else:
        return "SC25_maximum"


def _season(d: date) -> str:
    m = d.month
    if m in (3, 4, 5):
        return "spring"
    if m in (6, 7, 8):
        return "summer"
    if m in (9, 10, 11):
        return "autumn"
    return "winter"


# ---------------------------------------------------------------------------
# TABLE 1  Baseline — coverage windows (provider picks the actual day)
# ---------------------------------------------------------------------------


def build_baseline(
    start: date = date(2013, 1, 1),
    end: date = date(2024, 12, 31),
    step: int = 45,
) -> list[dict]:
    """Generate one 45-day coverage window row per step interval.

    The provider picks ANY available good-quality day inside [window_start,
    window_end].  Exact date is their choice; coverage of the window is
    what matters.
    """
    rows = []
    window_id = 1
    current = start
    while current <= end:
        window_end = min(current + timedelta(days=step - 1), end)
        mid = current + timedelta(days=(window_end - current).days // 2)
        rows.append(
            {
                "window_id": window_id,
                "window_start": current.isoformat(),
                "window_end": window_end.isoformat(),
                "days_requested": 1,
                "date_flexible": "YES — any day within window",
                "cadence_min": 30,
                "files_per_day": 48,
                "mid_window_year": mid.year,
                "mid_window_doy": _doy(mid),
                "season": _season(mid),
                "solar_cycle_phase": _solar_cycle_phase(mid),
                "notes": (
                    "Provider selects any 1 good-quality day within window_start–window_end. "
                    "Skip this window entirely if no data available."
                ),
            }
        )
        current += timedelta(days=step)
        window_id += 1
    return rows


# ---------------------------------------------------------------------------
# TABLE 2  Event-specific dates
# ---------------------------------------------------------------------------

# Each entry:
#   (date_str, event_type, flexibility_days, kp_est, dst_est_nT, description)
#
# flexibility_days:
#   0  → fixed by nature (eclipse, confirmed storm onset) — exact date required
#   N  → provider may shift ±N days if instrument was not running on that day
#
_EVENTS: list[tuple[str, str, int, str, str, str]] = [
    # ── Extreme Geomagnetic Storms (Dst < −200, Kp ≥ 8) ──────────────────
    (
        "2003-10-29",
        "storm_extreme",
        0,
        "9",
        "-383",
        "Halloween storm day 1; Dst −383 nT; main phase onset",
    ),
    (
        "2003-10-30",
        "storm_extreme",
        0,
        "9",
        "-353",
        "Halloween storm day 2; continued main phase",
    ),
    (
        "2003-11-20",
        "storm_extreme",
        0,
        "9",
        "-422",
        "Post-Halloween; Dst −422 nT; SC23 cycle peak",
    ),
    (
        "2015-03-17",
        "storm_extreme",
        0,
        "8",
        "-223",
        "St. Patrick's Day 2015; largest SC24 storm (to date)",
    ),
    (
        "2015-06-22",
        "storm_extreme",
        0,
        "8",
        "-204",
        "June 2015 storm day 1; two-step main phase",
    ),
    (
        "2015-06-23",
        "storm_extreme",
        1,
        "7",
        "-191",
        "June 2015 storm day 2; recovery (±1 day OK)",
    ),
    (
        "2017-09-07",
        "storm_extreme",
        0,
        "8",
        "-142",
        "September 2017; X9.3-associated CME arrival",
    ),
    (
        "2024-05-10",
        "storm_extreme",
        0,
        "9",
        "-412",
        "May 2024 extreme event; Kp=9; globally visible aurora",
    ),
    (
        "2024-05-11",
        "storm_extreme",
        1,
        "8",
        "-351",
        "May 2024 extreme event day 2; main/recovery (±1 day OK)",
    ),
    # ── Major Geomagnetic Storms (Dst −100 to −200, Kp 7–8) ───────────────
    ("2005-01-17", "storm_major", 0, "8", "-103", "January 2005 storm day 1"),
    (
        "2005-01-21",
        "storm_major",
        1,
        "8",
        "-99",
        "January 2005 storm day 5; GLE period (±1 day OK)",
    ),
    (
        "2006-12-14",
        "storm_major",
        0,
        "8",
        "-146",
        "December 2006; X-flare + CME arrival",
    ),
    ("2012-03-07", "storm_major", 0, "7", "-131", "March 2012; X5.4 flare + CME"),
    (
        "2012-07-15",
        "storm_major",
        1,
        "7",
        "-133",
        "July 2012; near-miss CME (±1 day OK)",
    ),
    (
        "2013-03-17",
        "storm_major",
        0,
        "7",
        "-132",
        "St. Patrick's Day 2013; SC24 rising-phase reference",
    ),
    (
        "2017-09-08",
        "storm_major",
        1,
        "7",
        "-124",
        "September 2017 recovery + second CME (±1 day OK)",
    ),
    (
        "2023-03-24",
        "storm_major",
        1,
        "7",
        "-130",
        "March 2023; SC25 rising (±1 day OK)",
    ),
    # ── Moderate Storms (Kp 5–6, fills mid-range) ─────────────────────────
    (
        "2014-02-19",
        "storm_moderate",
        2,
        "6",
        "-74",
        "February 2014 moderate; SC24 near maximum (±2 days OK)",
    ),
    (
        "2019-08-31",
        "storm_moderate",
        2,
        "5",
        "-47",
        "August 2019 moderate; deep solar minimum (±2 days OK)",
    ),
    (
        "2020-05-30",
        "storm_moderate",
        2,
        "6",
        "-66",
        "May 2020; SC25 onset (±2 days OK)",
    ),
    (
        "2021-11-04",
        "storm_moderate",
        2,
        "7",
        "-92",
        "November 2021; SC25 early rising (±2 days OK)",
    ),
    (
        "2022-11-03",
        "storm_moderate",
        2,
        "7",
        "-108",
        "November 2022; SC25 rising (±2 days OK)",
    ),
    # ── Quiet-Time Reference (Kp 0–1) ─────────────────────────────────────
    (
        "2019-05-01",
        "quiet",
        7,
        "0",
        "+5",
        "Deep solar min; Kp~0; cleanest IRI baseline (±7 days OK)",
    ),
    (
        "2019-10-14",
        "quiet",
        7,
        "1",
        "+3",
        "Deep solar min; autumn contrast (±7 days OK)",
    ),
    ("2020-04-15", "quiet", 7, "0", "+4", "SC25 onset; Kp~0 reference (±7 days OK)"),
    # ── Solar Eclipses (date fixed by orbital mechanics) ──────────────────
    (
        "2017-08-21",
        "eclipse_total",
        0,
        "2",
        "+5",
        "Total solar eclipse USA; path OR→SC; peak ~18:26 UTC",
    ),
    (
        "2019-07-02",
        "eclipse_total",
        0,
        "1",
        "+3",
        "Total solar eclipse South America; Chile/Argentina",
    ),
    (
        "2020-12-14",
        "eclipse_total",
        0,
        "2",
        "+4",
        "Total solar eclipse South America; Patagonia",
    ),
    (
        "2023-10-14",
        "eclipse_annular",
        0,
        "2",
        "+5",
        "Annular solar eclipse USA; path OR→TX→Yucatan",
    ),
    (
        "2024-04-08",
        "eclipse_total",
        0,
        "3",
        "-8",
        "Total solar eclipse USA; path TX→OH→ME; peak ~18:18 UTC",
    ),
    # ── SEP / GLE Events (date fixed — particle flux onset is abrupt) ──────
    (
        "2003-10-28",
        "SEP_GLE",
        0,
        "9",
        "-353",
        "X17.2 flare + Halloween SEP; GLE65; >10^4 pfu",
    ),
    (
        "2005-01-20",
        "SEP_GLE",
        0,
        "8",
        "-99",
        "GLE69; largest since 1989; >10 GeV protons; extreme HF absorption",
    ),
    ("2006-12-05", "SEP_GLE", 0, "7", "-80", "X9 flare + SEP; GLE70; late SC23"),
    (
        "2012-05-17",
        "SEP_GLE",
        0,
        "5",
        "-35",
        "GLE72; moderate SEP; good SC24 mid-cycle example",
    ),
    (
        "2017-09-10",
        "SEP_GLE",
        0,
        "5",
        "-55",
        "X8.2 flare + SEP; largest SC24 SEP event",
    ),
    # ── X-class Flares / SID (short-wave fade on dayside) ─────────────────
    (
        "2003-11-04",
        "xflare_SID",
        0,
        "5",
        "-69",
        "X28 mega-flare (largest on record); intense dayside SID",
    ),
    (
        "2006-12-05",
        "xflare_SID",
        0,
        "7",
        "-80",
        "X9.0 flare; strong dayside SID (same day as GLE70)",
    ),
    (
        "2014-09-10",
        "xflare_SID",
        1,
        "5",
        "-40",
        "X1.6 flare; well-documented ionospheric response (±1 day OK)",
    ),
    (
        "2017-09-06",
        "xflare_SID",
        0,
        "5",
        "-48",
        "X9.3 flare; largest SC24; strong SID on sunlit hemisphere",
    ),
    (
        "2024-02-22",
        "xflare_SID",
        2,
        "4",
        "-25",
        "X6.3 flare; SC25; good modern reference (±2 days OK)",
    ),
    # ── Sporadic-E Season (mid-latitude, summer; ±7 days flexibility) ──────
    (
        "2018-06-21",
        "Es_season",
        7,
        "2",
        "+4",
        "Summer solstice Es peak; northern mid-latitude (±7 days OK)",
    ),
    (
        "2019-06-21",
        "Es_season",
        7,
        "1",
        "+3",
        "Solstice Es; deep solar min — low background (±7 days OK)",
    ),
    ("2020-07-01", "Es_season", 7, "2", "+4", "July Es peak; SC25 onset (±7 days OK)"),
    ("2021-06-15", "Es_season", 7, "3", "+2", "Es season SC25 rising (±7 days OK)"),
    ("2022-07-04", "Es_season", 7, "2", "+5", "July Es peak; SC25 (±7 days OK)"),
    (
        "2023-06-21",
        "Es_season",
        7,
        "2",
        "+3",
        "Solstice Es 2023; SC25 near maximum (±7 days OK)",
    ),
    # ── Equatorial Spread-F (equinox post-sunset; ±5 days flexibility) ────
    (
        "2013-09-22",
        "spread_F",
        5,
        "1",
        "+3",
        "Autumn equinox 2013; equatorial spread-F 21–03 LT (±5 days OK)",
    ),
    (
        "2015-03-20",
        "spread_F",
        5,
        "2",
        "+5",
        "Spring equinox 2015; spread-F + storm build-up (±5 days OK)",
    ),
    (
        "2018-03-20",
        "spread_F",
        5,
        "1",
        "+4",
        "Spring equinox 2018; SC24 declining (±5 days OK)",
    ),
    (
        "2020-03-20",
        "spread_F",
        5,
        "1",
        "+3",
        "Spring equinox 2020; deep solar min — clean spread-F (±5 days OK)",
    ),
    (
        "2022-09-23",
        "spread_F",
        5,
        "3",
        "+2",
        "Autumn equinox 2022; SC25 rising (±5 days OK)",
    ),
    # ── Tonga Volcanic Eruption (unique; date fixed) ───────────────────────
    (
        "2022-01-15",
        "volcanic_wave",
        0,
        "2",
        "+4",
        "Hunga Tonga eruption; global Lamb wave → large ionospheric TIDs",
    ),
    # ── Solar Minimum Reference (extended quiet; ±14 days flexibility) ─────
    (
        "2008-12-01",
        "solar_minimum",
        14,
        "0",
        "+5",
        "SC23/24 deep minimum; very low NmF2 (±14 days OK)",
    ),
    (
        "2009-06-01",
        "solar_minimum",
        14,
        "0",
        "+4",
        "SC23/24 deep minimum; summer (±14 days OK)",
    ),
]


def build_events() -> list[dict]:
    rows = []
    for date_str, etype, flex, kp, dst, desc in _EVENTS:
        d = date.fromisoformat(date_str)
        flex_start = (d - timedelta(days=flex)).isoformat() if flex > 0 else date_str
        flex_end = (d + timedelta(days=flex)).isoformat() if flex > 0 else date_str
        rows.append(
            {
                "target_date": date_str,
                "flexibility_days": flex,
                "flexible_start": flex_start,
                "flexible_end": flex_end,
                "date_fixed": "YES" if flex == 0 else f"NO — shift up to ±{flex} days",
                "start_utc": "00:00",
                "end_utc": "23:50",
                "cadence_min": 10,
                "files_per_day": 144,
                "year": d.year,
                "doy": _doy(d),
                "season": _season(d),
                "event_type": etype,
                "Kp_est": kp,
                "Dst_est_nT": dst,
                "description": desc,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# CSV output
# ---------------------------------------------------------------------------

_BASELINE_NOTE = """\
TABLE 1 — BASELINE COVERAGE  (stratified opportunistic sampling)
Cadence requested : 30 minutes = 48 files/day
Date flexibility  : HIGH — provider selects any 1 good-quality day inside
                    [window_start, window_end].  Exact date is entirely at
                    the provider's discretion.  Skip any window where data
                    are unavailable; partial coverage is fully acceptable.
Coverage goal     : broad diurnal + seasonal + solar-cycle variability
Period            : SC24 rising through SC25 maximum (2013–2024)
Approximate files : ~98 windows × 1 day × 48 files ≈ 4 700 files
"""

_EVENT_NOTE = """\
TABLE 2 — EVENT-SPECIFIC COVERAGE  (targeted geophysical conditions)
Cadence requested : 10 minutes = 144 files/day
Date flexibility  : VARIABLE — see flexibility_days and date_fixed columns.
                    Events with flexibility_days=0 are fixed by nature
                    (eclipse geometry, confirmed storm onset, SEP/GLE onset).
                    Events with flexibility_days > 0 allow a shift of ±N days
                    within [flexible_start, flexible_end]; provider chooses
                    whichever day they have clean data closest to the target.
                    Skip any event the instrument did not cover.
Coverage goal     : tail-of-distribution events unlikely in the random baseline
Event categories  : storm_extreme, storm_major, storm_moderate, quiet,
                    eclipse_total, eclipse_annular, SEP_GLE, xflare_SID,
                    Es_season, spread_F, volcanic_wave, solar_minimum
Approximate files : ~55 entries × 144 files ≈ 7 900 files
"""

_BASELINE_COLS = [
    "window_id",
    "window_start",
    "window_end",
    "days_requested",
    "date_flexible",
    "cadence_min",
    "files_per_day",
    "mid_window_year",
    "mid_window_doy",
    "season",
    "solar_cycle_phase",
    "notes",
]

_EVENT_COLS = [
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
    "description",
]


def _write_csv(path: Path, note: str, cols: list[str], rows: list[dict]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        for line in note.splitlines():
            f.write(f"# {line}\n")
        f.write("#\n")
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"  Written: {path}  ({len(rows)} rows)")


def _write_summary(out_dir: Path, baseline: list[dict], events: list[dict]) -> None:
    n_b = len(baseline)
    n_e = len(events)
    total = n_b * 48 + n_e * 144
    fixed = sum(1 for r in events if r["flexibility_days"] == 0)
    flex = n_e - fixed
    lines = [
        "",
        "NN-POLAN Stage 2 — Data Request Summary",
        "=" * 44,
        f"  Baseline windows : {n_b:>3}  × 1 day × 48  files (30-min) = {n_b*48:>5} files",
        f"  Event days       : {n_e:>3}  × 144 files (10-min)          = {n_e*144:>5} files",
        f"  Grand total                                        ≈ {total:>5} files",
        "",
        f"  Event date flexibility:",
        f"    Fixed (flexibility_days=0) : {fixed} entries  (eclipses, storm onsets, SEPs)",
        f"    Flexible (±N days window)  : {flex} entries  (Es season, spread-F, quiet, etc.)",
        "",
        "  Event breakdown by type:",
    ]
    for etype, cnt in sorted(Counter(r["event_type"] for r in events).items()):
        lines.append(f"    {etype:<22}: {cnt}")
    lines += ["", "  Files written:"]
    for fname in ("baseline_dates.csv", "event_dates.csv", "data_request_summary.txt"):
        lines.append(f"    {out_dir / fname}")
    summary = "\n".join(lines)
    print(summary)
    (out_dir / "data_request_summary.txt").write_text(summary + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli() -> None:
    p = argparse.ArgumentParser(
        description="Compile NN-POLAN Stage 2 data-request date lists.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--out_dir", default="data_request")
    p.add_argument("--baseline_start", default="2013-01-01")
    p.add_argument("--baseline_end", default="2024-12-31")
    p.add_argument(
        "--step_days",
        default=45,
        type=int,
        help="Coverage window width / baseline sampling step (days)",
    )
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    baseline = build_baseline(
        start=date.fromisoformat(args.baseline_start),
        end=date.fromisoformat(args.baseline_end),
        step=args.step_days,
    )
    events = build_events()

    _write_csv(out_dir / "baseline_dates.csv", _BASELINE_NOTE, _BASELINE_COLS, baseline)
    _write_csv(out_dir / "event_dates.csv", _EVENT_NOTE, _EVENT_COLS, events)
    _write_summary(out_dir, baseline, events)


if __name__ == "__main__":
    _cli()
