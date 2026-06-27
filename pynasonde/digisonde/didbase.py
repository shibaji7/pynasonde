"""DIDBase connector — pull ionogram data from GIRO DIDBase (Firebird SQL).

Connection details
------------------
host     : didbase.giro.uml.edu
database : didb
role     : COMMON  ← required; grants SELECT/EXECUTE to ionogram data
driver   : fdb  (native Python Firebird client, no Java required)

Install::

    pip install fdb
    sudo apt-get install libfbclient2

Typical usage::

    from pynasonde.digisonde.didbase import DidBaseConnector

    with DidBaseConnector(url, user, password, role) as db:

        # List available stations
        stations = db.stations()   # → pd.DataFrame  (IDENT, URSI, NAME, ...)

        # Download raw ionogram files to disk
        paths = db.fetch_files(
            station="AU930",
            start="2017-05-27",
            end="2017-05-27",
            fmt="MMM",            # None → all formats
            out_dir="/tmp/au930/",
        )

        # Parse directly into a DataFrame
        df = db.fetch_dataframe(
            station="AU930",
            start="2017-05-27",
            end="2017-05-27",
            fmt="MMM",
        )

Query flow
----------
1. :meth:`_loc_id` — look up numeric location IDENT for a URSI station code
   via ``STATIONSLIST`` (cached after first call).
2. ``SELECT * FROM GETMEASUREMENTS(loc_id, start, end)`` — returns one row
   per ionogram (IDENT, TIMEUT, …).
3. ``SELECT DATA, ZIPPED, FORMATID FROM IONOGRAM WHERE MEASUREMENTID = ?``
   — returns the raw binary blob for each measurement.
4. If ``ZIPPED = 1`` the blob is zlib-decompressed before being written or
   parsed.
"""

from __future__ import annotations

import datetime as dt
import io
import os
import re
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Union

import pandas as pd
from loguru import logger

# JDBC URL pattern: jdbc:firebirdsql://host[:port]/database
_JDBC_RE = re.compile(
    r"^jdbc:firebirdsql://(?P<host>[^/:]+)(?::(?P<port>\d+))?/(?P<db>.+)$",
    re.IGNORECASE,
)

DateLike = Union[str, dt.date, dt.datetime]


def _parse_jdbc_url(url: str) -> tuple[str, str]:
    """Parse a JDBC Firebird URL into (host, database).

    Accepts ``jdbc:firebirdsql://host/db`` or ``jdbc:firebirdsql://host:port/db``.

    Raises:
        ValueError: if the URL does not match the expected pattern.
    """
    m = _JDBC_RE.match(url.strip())
    if not m:
        raise ValueError(
            f"Cannot parse JDBC URL: {url!r}\n"
            "Expected format: jdbc:firebirdsql://host[:port]/database"
        )
    return m.group("host"), m.group("db")


# ── connector ─────────────────────────────────────────────────────────────────

class DidBaseConnector:
    """Read ionograms from GIRO DIDBase over a native Firebird connection.

    Args:
        url:      JDBC URL, e.g. ``"jdbc:firebirdsql://didbase.giro.uml.edu/didb"``.
        user:     Database username.
        password: Database password.
        role:     Firebird role (e.g. ``"COMMON"`` — required for data access).
    """

    def __init__(
        self,
        url: str,
        user: str,
        password: str,
        role: str,
    ):
        """Create a DIDBase connector without opening the database connection.

        Args:
            url: JDBC Firebird URL.
            user: Database username.
            password: Database password.
            role: Firebird role used for data access.
        """
        self.url      = url
        self.host, self.database = _parse_jdbc_url(url)
        self.user     = user
        self.role     = role
        self._password = password
        self._con      = None
        self._stations_cache: Optional[pd.DataFrame] = None
        self._format_cache:   Optional[Dict[int, str]] = None

    # ── setup validation ──────────────────────────────────────────────────────

    @classmethod
    def ping(cls, url: str, user: str, password: str, role: str) -> None:
        """Validate the full connection stack and raise a descriptive exception
        at the first failure.

        Checks (in order):
        1. ``fdb`` package is importable.
        2. Firebird native client library (``libfbclient``) is found by fdb.
        3. JDBC URL parses correctly.
        4. TCP connection + authentication succeeds.
        5. The specified role is granted to the user.
        6. Role grants SELECT/EXECUTE access (queries ``STATIONSLIST``).

        Args:
            url:      JDBC URL, e.g. ``"jdbc:firebirdsql://host/db"``.
            user:     Database username.
            password: Database password.
            role:     Firebird role (default ``"COMMON"``).

        Raises:
            RuntimeError: with a human-readable message describing what failed
                and how to fix it.
        """
        # 1 ── fdb importable?
        try:
            import fdb as _fdb
        except ImportError:
            raise RuntimeError(
                "fdb package not installed.\n"
                "Fix: pip install fdb"
            )

        # 2 ── native client library present?
        try:
            _fdb.load_api()
        except Exception as e:
            raise RuntimeError(
                f"Firebird native client library not found: {e}\n"
                "Fix: sudo apt-get install libfbclient2   "
                "(or set FDB_HOME / LD_LIBRARY_PATH to the library location)"
            )

        # 3 ── URL parses?
        try:
            host, database = _parse_jdbc_url(url)
        except ValueError as e:
            raise RuntimeError(str(e))

        # 4 ── connect + authenticate
        try:
            con = _fdb.connect(
                host=host,
                database=database,
                user=user,
                password=password,
                role=role,
            )
        except _fdb.fbcore.DatabaseError as e:
            msg = str(e)
            if "335544472" in msg or "password" in msg.lower() or "login" in msg.lower():
                raise RuntimeError(
                    f"Authentication failed for user '{user}'.\n"
                    f"Check username and password.\nFirebird error: {e}"
                )
            if "335544578" in msg or "unavailable" in msg.lower() or "connection" in msg.lower():
                raise RuntimeError(
                    f"Cannot reach {host}/{database}.\n"
                    f"Check host, database path, and network access.\nFirebird error: {e}"
                )
            raise RuntimeError(f"Connection failed: {e}")

        # 5 ── role granted to user?
        try:
            cur = con.cursor()
            cur.execute(
                "SELECT RDB$RELATION_NAME FROM RDB$USER_PRIVILEGES "
                "WHERE RDB$USER = ? AND RDB$PRIVILEGE = 'M'",
                (user.upper(),),
            )
            granted_roles = [r[0].strip() for r in cur.fetchall()]
            if role.upper() not in [r.upper() for r in granted_roles]:
                con.close()
                raise RuntimeError(
                    f"Role '{role}' is not granted to user '{user}'.\n"
                    f"Roles available to this user: {granted_roles or ['(none)']}\n"
                    "Ask the DIDBase admin to grant the appropriate role."
                )
        except RuntimeError:
            raise
        except Exception as e:
            con.close()
            raise RuntimeError(f"Could not verify role membership: {e}")

        # 6 ── role actually grants data access?
        try:
            cur.execute("SELECT FIRST 1 IDENT FROM STATIONSLIST")
            cur.fetchone()
        except Exception as e:
            con.close()
            raise RuntimeError(
                f"Role '{role}' does not grant EXECUTE on STATIONSLIST.\n"
                f"The role exists but may lack data permissions.\nFirebird error: {e}"
            )

        con.close()
        logger.info(
            f"ping OK — {user}@{host}/{database} (role={role}) — "
            "fdb, libfbclient, auth, role, and data access all verified."
        )

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def connect(self) -> "DidBaseConnector":
        """Open the connection.  Returns ``self`` for chaining."""
        import fdb
        try:
            self._con = fdb.connect(
                host=self.host,
                database=self.database,
                user=self.user,
                password=self._password,
                role=self.role,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to connect to {self.host}/{self.database}: {e}\n"
                "Run DidBaseConnector.ping(url, user, password) for a full diagnostic."
            ) from e
        logger.info(
            f"Connected to {self.host}/{self.database} "
            f"as {self.user} (role={self.role})"
        )
        return self

    def close(self):
        """Close the active Firebird connection, if any."""
        if self._con:
            self._con.close()
            self._con = None

    def __enter__(self):
        """Open the connection when entering a context manager."""
        return self.connect()

    def __exit__(self, *_):
        """Close the connection when leaving a context manager."""
        self.close()

    # ── helpers ───────────────────────────────────────────────────────────────

    def _cur(self):
        """Return a database cursor for the active connection."""
        if self._con is None:
            raise RuntimeError("Not connected — call connect() first.")
        return self._con.cursor()

    # ── station metadata ──────────────────────────────────────────────────────

    def stations(self) -> pd.DataFrame:
        """Return all stations as a DataFrame (IDENT, URSI, NAME, MINTIME, MAXTIME).

        Result is cached after the first call.
        """
        if self._stations_cache is not None:
            return self._stations_cache
        cur = self._cur()
        logger.info("Fetching station list from DIDBase …")
        cur.execute("SELECT * FROM STATIONSLIST")
        cols = [d[0] for d in cur.description]
        rows = []
        while True:
            batch = cur.fetchmany(100)
            if not batch:
                break
            rows.extend(batch)
        df = pd.DataFrame(rows, columns=cols)
        df["URSI"] = df["URSI"].str.strip()
        df["NAME"] = df["NAME"].str.strip()
        self._stations_cache = df
        logger.info(f"Station list: {len(df)} stations")
        return df

    def _loc_id(self, station: str) -> int:
        """Return the numeric IDENT for a URSI station code."""
        stn = self.stations()
        match = stn[stn["URSI"] == station.strip().upper()]
        if match.empty:
            raise ValueError(
                f"Station '{station}' not found in DIDBase. "
                f"Call stations() to browse available stations."
            )
        return int(match.iloc[0]["IDENT"])

    def _formats(self) -> Dict[int, str]:
        """Return {IDENT: name} for all ionogram formats (cached)."""
        if self._format_cache is not None:
            return self._format_cache
        cur = self._cur()
        cur.execute("SELECT IDENT, NAME FROM FORMAT")
        self._format_cache = {r[0]: r[1].strip() for r in cur.fetchall()}
        return self._format_cache

    # ── measurement index ─────────────────────────────────────────────────────

    def _get_measurements(
        self,
        station: str,
        start: DateLike,
        end: DateLike,
    ) -> List[tuple]:
        """Return [(meas_id, timeut, format_id), …] for station + time range.

        Uses ``GETMEASUREMENTS(loc_id, start, end)`` → joins against IONOGRAM
        view to add format_id.
        """
        loc_id   = self._loc_id(station)
        start_dt = _to_datetime(start)
        end_dt   = _to_datetime(end, end_of_day=True)

        cur = self._cur()
        logger.info(
            f"Querying measurements for {station} "
            f"{start_dt.date()} – {end_dt.date()} …"
        )
        cur.execute(
            "SELECT IDENT, TIMEUT FROM GETMEASUREMENTS(?, ?, ?)",
            (loc_id, start_dt, end_dt),
        )
        meas_rows = cur.fetchall()  # [(ident, timeut), …]

        if not meas_rows:
            logger.warning(f"No measurements found for {station} in range.")
            return []

        logger.info(f"Found {len(meas_rows)} measurements.")
        return meas_rows   # [(meas_id, timeut), …]

    # ── blob retrieval ────────────────────────────────────────────────────────

    def _fetch_blob(self, meas_id: int) -> Optional[tuple]:
        """Return (data_bytes, format_name) for one measurement ID, or None."""
        cur = self._cur()
        cur.execute(
            "SELECT DATA, ZIPPED, FORMATID FROM IONOGRAM WHERE MEASUREMENTID = ?",
            (meas_id,),
        )
        row = cur.fetchone()
        if row is None or row[0] is None:
            return None

        data, zipped, fmt_id = row
        raw = bytes(data)
        if zipped:
            with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                raw = zf.read(zf.namelist()[0])

        fmt_name = self._formats().get(fmt_id, "UNKNOWN")
        return raw, fmt_name

    # ── public API ────────────────────────────────────────────────────────────

    def fetch_files(
        self,
        station: str,
        start: DateLike,
        end: DateLike,
        fmt: Optional[str] = None,
        out_dir: Union[str, Path] = ".",
    ) -> List[Path]:
        """Download ionograms to individual files and return their paths.

        Args:
            station: URSI station code (e.g. ``"AU930"``).
            start:   Start date/datetime (inclusive).
            end:     End date/datetime (inclusive).
            fmt:     Restrict to one format (``"MMM"``, ``"RSF"``, ``"SBF"``).
                     ``None`` → all formats.
            out_dir: Directory to write files into (created if absent).

        Returns:
            List of :class:`~pathlib.Path` for every file written.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        meas = self._get_measurements(station, start, end)
        paths = []
        for meas_id, timeut in meas:
            result = self._fetch_blob(meas_id)
            if result is None:
                continue
            data, fmt_name = result
            if fmt and fmt.upper() != fmt_name.upper():
                continue
            fname = out_dir / f"{station}_{timeut.strftime('%Y%j%H%M%S')}.{fmt_name}"
            fname.write_bytes(data)
            paths.append(fname)
            logger.debug(f"Saved {fname.name}  ({len(data):,} bytes)")

        logger.info(f"fetch_files → {len(paths)} files written to {out_dir}")
        return paths

    def fetch_dataframe(
        self,
        station: str,
        start: DateLike,
        end: DateLike,
        fmt: str = "MMM",
    ) -> pd.DataFrame:
        """Download and parse ionograms, returning a single combined DataFrame.

        Args:
            station: URSI station code.
            start:   Start date/datetime (inclusive).
            end:     End date/datetime (inclusive).
            fmt:     Ionogram format (``"MMM"``, ``"RSF"``, ``"SBF"``).

        Returns:
            Combined :class:`~pandas.DataFrame`.  Empty if nothing found.
        """
        from pynasonde.digisonde.parsers.grm import _EXTRACTOR

        cls = _EXTRACTOR.get(fmt.upper())
        if cls is None:
            raise ValueError(f"Unknown format '{fmt}'. Choose MMM, RSF, or SBF.")

        meas = self._get_measurements(station, start, end)
        frames = []
        for meas_id, timeut in meas:
            result = self._fetch_blob(meas_id)
            if result is None:
                continue
            data, fmt_name = result
            if fmt_name.upper() != fmt.upper():
                continue

            suffix = f".{fmt.upper()}"
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                    tmp.write(data)
                    tmp_path = tmp.name
                ext = cls(tmp_path)
                ext.extract()
                df = ext.to_pandas()
                if not df.empty:
                    frames.append(df)
            except Exception as exc:
                logger.warning(f"Failed to parse {station} @ {timeut}: {exc}")
            finally:
                if tmp_path:
                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass

        if not frames:
            logger.warning("No ionograms parsed.")
            return pd.DataFrame()

        result = pd.concat(frames, ignore_index=True)
        logger.info(
            f"fetch_dataframe → {len(frames)} ionograms, "
            f"{len(result):,} rows  [{station} {fmt}]"
        )
        return result

    # ── convenience ───────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        """Return a compact connector status string."""
        status = "connected" if self._con else "disconnected"
        return f"<DidBaseConnector {self.host}/{self.database} [{status}]>"


# ── helpers ───────────────────────────────────────────────────────────────────

def _to_datetime(val: DateLike, end_of_day: bool = False) -> dt.datetime:
    """Convert supported date-like values to ``datetime``.

    Args:
        val: ISO string, ``date``, or ``datetime`` value.
        end_of_day: If True and the value has no explicit time, return
            23:59:59 for that date.

    Returns:
        Converted ``datetime`` instance.
    """
    if isinstance(val, dt.datetime):
        return val
    if isinstance(val, dt.date):
        base = dt.datetime(val.year, val.month, val.day)
    else:
        base = dt.datetime.fromisoformat(str(val))
    if end_of_day and base.hour == 0 and base.minute == 0 and base.second == 0:
        return base.replace(hour=23, minute=59, second=59)
    return base
