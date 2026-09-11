"""DriftBase connector — pull drift-velocity/skymap data from GIRO DriftBase
(Firebird SQL).

Connection details
------------------
host     : 129.63.134.45
database : drift  ← a Firebird *alias*, not a filesystem path (unlike a
           raw path such as ``/ext2/db/ib/dfdb``, which does not exist on
           the server — the registered alias is ``drift``)
role     : COMMON  ← required; grants SELECT/EXECUTE to drift data
driver   : fdb  (native Python Firebird client, no Java required)

Install::

    pip install fdb
    sudo apt-get install libfbclient2
    sudo apt-get install firebird3.0-utils

Typical usage::

    from pynasonde.digisonde.driftbase import DriftBaseConnector

    with DriftBaseConnector(url, user, password, role) as db:

        # List available stations
        stations = db.stations()   # → pd.DataFrame  (IDENT, URSI, NAME, ...)

        # Browse the "expert" (scaler) catalog — analogous to didbase.py's
        # SCALER table; multiple experts can independently scale the same
        # measurement.
        catalog = db.experts()

        # Fetch scaled drift velocities directly via SQL — no blob
        # download/parsing involved. One row per measurement (the
        # official BESTEXPERT's row, by default).
        vel = db.fetch_velocity(station="AL945", start="2015-03-17", end="2015-03-18")

        # Download raw drift-ionogram files (DRIFTHEADER+DRIFTDATA)
        paths = db.fetch_drift_files(
            station="AL945", start="2015-03-17", end="2015-03-18",
            out_dir="/tmp/al945_drift/",
        )

        # Download raw skymap files (SKYHEADER+SKYDATA), one per measurement
        paths = db.fetch_sky_files(
            station="AL945", start="2015-03-17", end="2015-03-18",
            out_dir="/tmp/al945_sky/",
        )

Query flow (raw drift/sky blobs — fetch_drift_files / fetch_sky_files)
------------------------------------------------------------------------
1. ``GETMEASUREMENTSALT(ursi, start, end)`` — measurement lookup for a
   station + time range in one call (returns ``IDENT``, ``TIMEUT``,
   ``LOCATIONID``, ``EQUIPMENTID``); unlike DIDBase's ``GETMEASUREMENTS``,
   this takes the URSI code directly, no separate ``STATIONSLIST`` lookup
   needed for the common case.
2. ``DRIFTHEADER``/``DRIFTDATA`` (or the ``DRIFT`` view that joins them) —
   the raw drift-ionogram blob for each measurement, keyed by
   ``MEASUREMENTID`` alone.
3. ``SKYHEADER``/``SKYDATA`` — the raw skymap blob, keyed by
   ``(MEASUREMENTID, EXPERTID)`` since skymap derivation can be redone by
   different experts (see ``EXPERT``/``BESTEXPERT``).
4. If ``ZIPPED = 1`` the blob is zip-decompressed before being written or
   parsed (mirrors :mod:`pynasonde.digisonde.didbase`'s handling, and the
   same fdb ``BlobReader`` caveat applies — read via ``.read()``, never
   ``bytes(...)``, on a streamed blob object).

Query flow (scaled velocities — fetch_velocity)
--------------------------------------------------
Scaled drift velocities are stored relationally in ``VELOCITY``,
independent of the raw drift/sky blobs, so no blob download/parsing is
needed:

1. ``GETMEASUREMENTSALT(ursi, start, end)`` — same measurement lookup.
2. ``SELECT * FROM VELOCITY WHERE MEASUREMENTID IN (...)`` — one row per
   (measurement, expert) with ``VX``/``VY``/``VZ``/``VH``/``AZ`` (+ error
   columns), the ``COORD`` system, and the height/frequency range used.
3. A measurement can be scaled by more than one expert; :meth:`fetch_velocity`
   joins against ``BESTEXPERT`` — DriftBase's own "official expert per
   measurement" view — to pick exactly one row per measurement by default
   (mirrors :meth:`~pynasonde.digisonde.didbase.DidBaseConnector.
   fetch_scaled_params`'s scaler disambiguation, but DriftBase already
   provides the "which one is authoritative" answer directly).

Loading raw DRIFTDATA/SKYDATA blobs into DataFrames
------------------------------------------------------
Confirmed live: DriftBase's raw blobs are the *same* binary formats
pynasonde already parses elsewhere — no new parser needed::

    from pynasonde.digisonde.parsers.dft import DftExtractor
    from pynasonde.digisonde.parsers.sky import SkyExtractor

    # DRIFTHEADER/DRIFTDATA blobs -> DftExtractor (fixed 4096-byte blocks,
    # same format as pynasonde's existing DFT/Doppler-spectra files).
    drift_paths = db.fetch_drift_files(station="AL945", start="2015-03-16",
                                        end="2015-03-16", out_dir="/tmp/al945_drift/")
    ext = DftExtractor(str(drift_paths[0]), extract_time_from_name=True,
                        extract_stn_from_name=True)
    ext.extract()
    drift_df = ext.to_pandas()  # columns: block_idx, subcase_idx, height_km,
                                 # doppler_bin, amplitude, phase, frequency_hz, date

    # SKYHEADER/SKYDATA blobs -> SkyExtractor (same skymap format used by
    # SkySummaryPlots.plot_skymap elsewhere in pynasonde).
    sky_paths = db.fetch_sky_files(station="AL945", start="2015-03-16",
                                    end="2015-03-16", out_dir="/tmp/al945_sky/")
    ext = SkyExtractor(str(sky_paths[0]), extract_time_from_name=True,
                        extract_stn_from_name=True)
    ext.extract()
    sky_df = ext.to_pandas()  # columns: x_coord, y_coord, spect_amp, spect_dop,
                               # zenith_angle, n_sources, rms_error, datetime, ...

Verified against real AL945 (2015-03-16) blobs: a 393,216-byte ``.DFT``
file (96 x 4096-byte blocks) parsed into 196,608 Doppler-spectra rows,
and a ``.SKY`` file parsed into 1,781 skymap-point rows — both with
sensible physical values. ``fetch_sky_files``'s default filenames
(``{station}_{timestamp}.SKY``, one per measurement via ``BESTEXPERT``)
are required for ``extract_time_from_name``/``extract_stn_from_name`` to
work; the ``_expert{N}``-suffixed filenames written when
``prefer_best=False`` break that filename parsing (see
:meth:`fetch_sky_files`).
"""

from __future__ import annotations

import datetime as dt
import io
import re
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
    ``db`` may be a bare Firebird alias (e.g. ``"drift"``) registered on the
    server rather than a filesystem path.

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


# ── connector ─────────────────────────────────────────────────────────────────

class DriftBaseConnector:
    """Read drift-velocity/skymap data from GIRO DriftBase over a native
    Firebird connection.

    Args:
        url:      JDBC URL, e.g. ``"jdbc:firebirdsql://129.63.134.45/drift"``.
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
        """Create a DriftBase connector without opening the database connection.

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
        self._expert_cache:   Optional[pd.DataFrame] = None

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
        try:
            import fdb as _fdb
        except ImportError:
            raise RuntimeError(
                "fdb package not installed.\n"
                "Fix: pip install fdb"
            )

        try:
            _fdb.load_api()
        except Exception as e:
            raise RuntimeError(
                f"Firebird native client library not found: {e}\n"
                "Fix: sudo apt-get install libfbclient2   "
                "(or set FDB_HOME / LD_LIBRARY_PATH to the library location)"
            )

        try:
            host, database = _parse_jdbc_url(url)
        except ValueError as e:
            raise RuntimeError(str(e))

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
            if "335544344" in msg or "no such file" in msg.lower():
                raise RuntimeError(
                    f"Database '{database}' not found on {host}.\n"
                    "If this is meant to be a Firebird alias (not a filesystem "
                    "path), double-check the alias name with the DriftBase "
                    "administrator.\nFirebird error: {e}"
                )
            if "335544578" in msg or "unavailable" in msg.lower() or "connection" in msg.lower():
                raise RuntimeError(
                    f"Cannot reach {host}/{database}.\n"
                    f"Check host, database path, and network access.\nFirebird error: {e}"
                )
            raise RuntimeError(f"Connection failed: {e}")

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
                    "Ask the DriftBase admin to grant the appropriate role."
                )
        except RuntimeError:
            raise
        except Exception as e:
            con.close()
            raise RuntimeError(f"Could not verify role membership: {e}")

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

    def connect(self) -> "DriftBaseConnector":
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
                "Run DriftBaseConnector.ping(url, user, password, role) for a "
                "full diagnostic."
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

    @staticmethod
    def _chunks(seq: list, chunk_size: int):
        """Yield ``seq`` in slices of at most ``chunk_size`` — keeps bound
        Firebird IN(...) parameter lists under the driver's limit."""
        for i in range(0, len(seq), chunk_size):
            yield seq[i : i + chunk_size]

    # ── station / expert metadata ────────────────────────────────────────────

    def stations(self) -> pd.DataFrame:
        """Return all stations as a DataFrame (IDENT, URSI, NAME, MINTIME, MAXTIME).

        Result is cached after the first call. Note: this call has been
        observed to be slow (comparable to or slower than DIDBase's
        equivalent) — prefer :meth:`_get_measurements`
        (``GETMEASUREMENTSALT``) for regular per-station queries, which
        takes a URSI code directly and does not require this cache.
        """
        if self._stations_cache is not None:
            return self._stations_cache
        cur = self._cur()
        logger.info("Fetching station list from DriftBase …")
        cur.execute("SELECT IDENT, URSI, NAME, MINTIME, MAXTIME FROM STATIONSLIST")
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

    def experts(self) -> pd.DataFrame:
        """Return the catalog of drift "experts" (scalers) DriftBase knows.

        Analogous to :meth:`~pynasonde.digisonde.didbase.
        DidBaseConnector.scaled_params_catalog` — ``EXPERT.RANK`` mirrors
        DIDBase's ``SCALER.RANK``, but there is no human/auto flag here
        (see :meth:`fetch_velocity`'s ``prefer_best`` for how DriftBase's
        own ``BESTEXPERT`` view resolves multi-expert measurements).

        Result is cached after the first call.
        """
        if self._expert_cache is not None:
            return self._expert_cache
        cur = self._cur()
        cur.execute("SELECT IDENT, LOGNAME, FNAME, LNAME, RANK FROM EXPERT")
        cols = [d[0] for d in cur.description]
        df = pd.DataFrame(cur.fetchall(), columns=cols)
        for c in ("LOGNAME", "FNAME", "LNAME"):
            df[c] = df[c].str.strip()
        self._expert_cache = df
        logger.info(f"experts: {len(df)} known expert(s)")
        return df

    # ── measurement index ─────────────────────────────────────────────────────

    def _get_measurements(
        self,
        station: str,
        start: DateLike,
        end: DateLike,
    ) -> List[tuple]:
        """Return [(meas_id, timeut, location_id, equipment_id), …] for a
        station + time range.

        Uses ``GETMEASUREMENTSALT(ursi, start, end)`` — takes the URSI
        code directly, unlike DIDBase's ``GETMEASUREMENTS`` which needs a
        numeric location IDENT first.
        """
        start_dt = _to_datetime(start)
        end_dt   = _to_datetime(end, end_of_day=True)

        cur = self._cur()
        logger.info(
            f"Querying measurements for {station} "
            f"{start_dt.date()} – {end_dt.date()} …"
        )
        cur.execute(
            "SELECT IDENT, TIMEUT, LOCATIONID, EQUIPMENTID "
            "FROM GETMEASUREMENTSALT(?, ?, ?)",
            (station.strip().upper(), start_dt, end_dt),
        )
        rows = cur.fetchall()

        if not rows:
            logger.warning(f"No measurements found for {station} in range.")
            return []

        logger.info(f"Found {len(rows)} measurements.")
        return rows

    def _best_experts(self, meas_ids: List[int], chunk_size: int = 400) -> Dict[int, int]:
        """Return {MEASUREMENTID: EXPERTID} using DriftBase's own
        ``BESTEXPERT`` view — the officially designated expert per
        measurement, when one has been chosen."""
        if not meas_ids:
            return {}
        cur = self._cur()
        best: Dict[int, int] = {}
        for batch in self._chunks(meas_ids, chunk_size):
            placeholders = ",".join("?" for _ in batch)
            cur.execute(
                f"SELECT MEASID, EXPERTID FROM BESTEXPERT WHERE MEASID IN ({placeholders})",
                tuple(batch),
            )
            best.update(dict(cur.fetchall()))
        return best

    # ── blob retrieval ────────────────────────────────────────────────────────

    @staticmethod
    def _read_blob(data) -> bytes:
        """Return raw bytes from an fdb column value.

        fdb returns a streaming ``BlobReader`` for BLOB columns; calling
        ``bytes(reader)`` forces Python to iterate it and trips a
        str/bytes bug in fdb's own ``BlobReader.readline()`` (see
        :mod:`pynasonde.digisonde.didbase`'s ``_fetch_blob`` for the same
        fix). Read the reader directly instead.
        """
        if hasattr(data, "read"):
            raw = data.read()
            data.close()
            return raw
        return bytes(data)

    def _fetch_drift_blob(self, meas_id: int) -> Optional[bytes]:
        """Return the raw drift-ionogram blob for one measurement, or None."""
        cur = self._cur()
        cur.execute(
            "SELECT DATA, ZIPPED FROM DRIFT WHERE MEASUREMENTID = ?",
            (meas_id,),
        )
        row = cur.fetchone()
        if row is None or row[0] is None:
            return None
        data, zipped = row
        raw = self._read_blob(data)
        if zipped:
            with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                raw = zf.read(zf.namelist()[0])
        return raw

    def _fetch_sky_blob(self, meas_id: int, expert_id: Optional[int]) -> Optional[tuple]:
        """Return (data_bytes, expert_id) for one measurement's skymap
        blob, or None. If ``expert_id`` is None, the first available
        expert's blob for this measurement is returned."""
        cur = self._cur()
        if expert_id is not None:
            cur.execute(
                "SELECT sh.EXPERTID, sd.DATA, sh.ZIPPED FROM SKYHEADER sh "
                "JOIN SKYDATA sd ON sd.MEASUREMENTID = sh.MEASUREMENTID "
                "AND sd.EXPERTID = sh.EXPERTID "
                "WHERE sh.MEASUREMENTID = ? AND sh.EXPERTID = ?",
                (meas_id, expert_id),
            )
        else:
            cur.execute(
                "SELECT sh.EXPERTID, sd.DATA, sh.ZIPPED FROM SKYHEADER sh "
                "JOIN SKYDATA sd ON sd.MEASUREMENTID = sh.MEASUREMENTID "
                "AND sd.EXPERTID = sh.EXPERTID "
                "WHERE sh.MEASUREMENTID = ? ROWS 1",
                (meas_id,),
            )
        row = cur.fetchone()
        if row is None or row[1] is None:
            return None
        eid, data, zipped = row
        raw = self._read_blob(data)
        if zipped:
            with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                raw = zf.read(zf.namelist()[0])
        return raw, eid

    # ── public API ────────────────────────────────────────────────────────────

    def fetch_drift_files(
        self,
        station: str,
        start: DateLike,
        end: DateLike,
        out_dir: Union[str, Path] = ".",
    ) -> List[Path]:
        """Download raw drift-ionogram blobs to individual files.

        Args:
            station: URSI station code (e.g. ``"WI937"``).
            start:   Start date/datetime (inclusive).
            end:     End date/datetime (inclusive).
            out_dir: Directory to write files into (created if absent).

        Returns:
            List of :class:`~pathlib.Path` for every file written.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        meas = self._get_measurements(station, start, end)
        paths = []
        for meas_id, timeut, *_ in meas:
            raw = self._fetch_drift_blob(meas_id)
            if raw is None:
                continue
            fname = out_dir / f"{station}_{timeut.strftime('%Y%j%H%M%S')}.DFT"
            fname.write_bytes(raw)
            paths.append(fname)
            logger.debug(f"Saved {fname.name}  ({len(raw):,} bytes)")

        logger.info(f"fetch_drift_files → {len(paths)} files written to {out_dir}")
        return paths

    def fetch_sky_files(
        self,
        station: str,
        start: DateLike,
        end: DateLike,
        out_dir: Union[str, Path] = ".",
        prefer_best: bool = True,
    ) -> List[Path]:
        """Download raw skymap blobs to individual files.

        Args:
            station: URSI station code.
            start:   Start date/datetime (inclusive).
            end:     End date/datetime (inclusive).
            out_dir: Directory to write files into (created if absent).
            prefer_best: If True (default), download only the
                ``BESTEXPERT``-chosen expert's skymap per measurement —
                exactly one file per measurement, named
                ``{station}_{timestamp}.SKY`` so
                ``pynasonde.digisonde.parsers.sky.SkyExtractor``'s
                filename-based datetime/station parsing (
                ``extract_time_from_name``/``extract_stn_from_name``)
                works unmodified. If False, every available expert's
                skymap is downloaded, disambiguated with an
                ``_expert{N}`` filename suffix (breaks
                ``extract_time_from_name`` — pass the known measurement
                time separately in that case).

        Returns:
            List of :class:`~pathlib.Path` for every file written.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        meas = self._get_measurements(station, start, end)
        if not meas:
            return []

        best = self._best_experts([m[0] for m in meas]) if prefer_best else {}

        paths = []
        for meas_id, timeut, *_ in meas:
            expert_id = best.get(meas_id) if prefer_best else None
            result = self._fetch_sky_blob(meas_id, expert_id)
            if result is None:
                continue
            raw, eid = result
            suffix = "" if prefer_best else f"_expert{eid}"
            fname = out_dir / f"{station}_{timeut.strftime('%Y%j%H%M%S')}{suffix}.SKY"
            fname.write_bytes(raw)
            paths.append(fname)
            logger.debug(f"Saved {fname.name}  ({len(raw):,} bytes)")

        logger.info(f"fetch_sky_files → {len(paths)} files written to {out_dir}")
        return paths

    def fetch_velocity(
        self,
        station: str,
        start: DateLike,
        end: DateLike,
        prefer_best: bool = True,
        chunk_size: int = 400,
    ) -> pd.DataFrame:
        """Fetch scaled drift velocities directly via SQL.

        Unlike :meth:`fetch_drift_files`/:meth:`fetch_sky_files` (which
        download raw blobs), this reads scaled velocity components
        straight out of DriftBase's ``VELOCITY`` table — no blob
        download or parsing involved.

        Args:
            station: URSI station code (e.g. ``"WI937"``).
            start:   Start date/datetime (inclusive).
            end:     End date/datetime (inclusive).
            prefer_best: A measurement can be scaled by more than one
                expert. If True, keep only the row whose ``EXPERTID``
                matches DriftBase's own ``BESTEXPERT`` view for that
                measurement (one row per measurement). If False, return
                every expert's row (multiple rows per measurement may
                result).
            chunk_size: Maximum measurement IDs per SQL query, to stay
                well under Firebird's bound-parameter limit for windows
                with many measurements.

        Returns:
            DataFrame with all ``VELOCITY`` columns (``VX``, ``VY``,
            ``VZ``, ``VH``, ``AZ`` + ``_ERR`` variants, ``COORD``,
            height/frequency range, ...) plus a ``datetime`` column.
            Empty if nothing found.
        """
        meas = self._get_measurements(station, start, end)
        if not meas:
            return pd.DataFrame()

        meas_time = {meas_id: timeut for meas_id, timeut, *_ in meas}
        meas_ids = list(meas_time.keys())
        cur = self._cur()

        rows = []
        cols = None
        for batch in self._chunks(meas_ids, chunk_size):
            placeholders = ",".join("?" for _ in batch)
            cur.execute(
                f"SELECT * FROM VELOCITY WHERE MEASUREMENTID IN ({placeholders})",
                tuple(batch),
            )
            if cols is None:
                cols = [d[0] for d in cur.description]
            rows.extend(cur.fetchall())

        if not rows:
            logger.warning(f"No scaled velocities found for {station} in range.")
            return pd.DataFrame()

        df = pd.DataFrame(rows, columns=cols)

        if prefer_best:
            best = self._best_experts(meas_ids, chunk_size=chunk_size)
            if best:
                before = len(df)
                df = df[df["EXPERTID"] == df["MEASUREMENTID"].map(best)]
                dropped = before - len(df)
                if dropped:
                    logger.info(
                        f"[{station}] kept {len(df)}/{before} row(s) via "
                        f"BESTEXPERT (dropped {dropped} non-best-expert row(s))."
                    )

        df["datetime"] = df["MEASUREMENTID"].map(meas_time)
        df = df.sort_values("datetime").reset_index(drop=True)
        logger.info(f"fetch_velocity → {len(df)} scaled record(s)  [{station}]")
        return df

    # ── convenience ───────────────────────────────────────────────────────────

    def __repr__(self) -> str:
        """Return a compact connector status string."""
        status = "connected" if self._con else "disconnected"
        return f"<DriftBaseConnector {self.host}/{self.database} [{status}]>"
