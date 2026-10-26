"""Timezone helpers shared by the Digisonde and VIPIR subpackages.

Requires the optional ``time`` extra (``pytz`` and ``timezonefinder``) only
when :class:`TimeZoneConversion` is instantiated.
"""

import datetime as dt

import pandas as pd
from loguru import logger


class TimeZoneConversion:
    """Convert timestamps between UTC and a station's local timezone."""

    def __init__(
        self,
        local_tz: str = None,
        lat: float = 37.8815,
        long: float = -75.4374,
    ):
        """Initialize the converter with either a timezone name or coordinates.

        Args:
            local_tz: Explicit timezone name; inferred from lat/lon when None.
            lat: Station latitude used when deriving the timezone.
            long: Station longitude used when deriving the timezone.
        """
        import pytz
        from timezonefinder import TimezoneFinder

        self.local_tz = local_tz
        tf = TimezoneFinder()
        if (long is not None) and (lat is not None):
            self.local_tz = tf.timezone_at(lng=long, lat=lat)
        logger.info(f"Local Time: {self.local_tz}")
        self.utc_zone, self.local_zone = pytz.timezone("UTC"), pytz.timezone(
            self.local_tz
        )
        return

    def utc_to_local_time(self, dates):
        """Translate an iterable of UTC datetimes to the configured local zone.

        Args:
            dates: Iterable of naive or timezone-aware datetime objects.

        Returns:
            List of localized datetimes adjusted to `self.local_zone`.
        """
        date = pd.to_datetime(dates[0])
        tdiff_hr = (
            self.local_zone.localize(date)
            - self.utc_zone.localize(date).astimezone(self.local_zone)
        ).seconds / 3600
        dates = [d - dt.timedelta(hours=tdiff_hr) for d in dates]
        return dates
