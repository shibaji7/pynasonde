"""Solar geometry helpers for Digisonde autoscaling presets."""

from __future__ import annotations

import datetime as dt
import math

SZA_DAY_MAX_DEG = 80.0
SZA_TWILIGHT_MAX_DEG = 100.0


def _as_utc_naive(time: dt.datetime) -> dt.datetime:
    """Return a timezone-naive UTC datetime."""
    if time.tzinfo is None:
        return time
    return time.astimezone(dt.timezone.utc).replace(tzinfo=None)


def solar_zenith_angle(
    time: dt.datetime,
    lat_deg: float,
    lon_deg: float,
) -> float:
    """Return approximate solar zenith angle in degrees.

    Args:
        time: Sounding datetime. Naive values are treated as UTC.
        lat_deg: Station geodetic latitude in degrees north.
        lon_deg: Station longitude in degrees east; west is negative.

    Returns:
        Solar zenith angle in degrees. Zero is overhead; ninety is the
        geometric horizon.
    """
    utc = _as_utc_naive(time)
    doy = utc.timetuple().tm_yday
    ut_h = utc.hour + utc.minute / 60.0 + utc.second / 3600.0
    decl = math.radians(23.45 * math.sin(math.radians(360.0 / 365.0 * (doy - 81))))
    hour_angle = math.radians(15.0 * (ut_h - 12.0) + lon_deg)
    lat_r = math.radians(lat_deg)
    cos_sza = math.sin(lat_r) * math.sin(decl) + math.cos(lat_r) * math.cos(
        decl
    ) * math.cos(hour_angle)
    return math.degrees(math.acos(max(-1.0, min(1.0, cos_sza))))


def classify_solar_regime(
    sza_deg: float,
    *,
    day_max_deg: float = SZA_DAY_MAX_DEG,
    twilight_max_deg: float = SZA_TWILIGHT_MAX_DEG,
) -> str:
    """Classify SZA into ``day``, ``twilight``, or ``night``.

    Args:
        sza_deg: Solar zenith angle in degrees.
        day_max_deg: Maximum SZA for daytime.
        twilight_max_deg: Maximum SZA for twilight.

    Returns:
        One of ``"day"``, ``"twilight"``, or ``"night"``.
    """
    if sza_deg < day_max_deg:
        return "day"
    if sza_deg < twilight_max_deg:
        return "twilight"
    return "night"
