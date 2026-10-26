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
    sudo apt-get install firebird3.0-utils

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

        # See what scaled characteristic names DIDBase actually has
        # (foF2, hF, hF2, foE, hmF2, ... one row per PARAMCAT entry).
        catalog = db.scaled_params_catalog()

        # Fetch scaled characteristics (foF2, hF, ...) directly via SQL --
        # no blob download/parsing, works whenever DIDBase has scaled the
        # measurement, independent of SAO blob availability.
        scaled = db.fetch_scaled_params(
            station="AU930",
            start="2017-05-27",
            end="2017-05-27",
            params=["foF2", "hF"],
        )

Query flow (raw ionograms — fetch_files / fetch_dataframe)
------------------------------------------------------------
1. :meth:`_loc_id` — look up numeric location IDENT for a URSI station code
   via ``STATIONSLIST`` (cached after first call).
2. ``SELECT * FROM GETMEASUREMENTS(loc_id, start, end)`` — returns one row
   per ionogram (IDENT, TIMEUT, …).
3. ``SELECT DATA, ZIPPED, FORMATID FROM IONOGRAM WHERE MEASUREMENTID = ?``
   — returns the raw binary blob for each measurement.
4. If ``ZIPPED = 1`` the blob is zlib-decompressed before being written or
   parsed.

Query flow (scaled characteristics — fetch_scaled_params)
-------------------------------------------------------------
Scaled characteristics (foF2, hF, hmF2, foE, ...) are stored relationally
in DIDBase, independent of whether a raw ionogram or a SAO blob exists for
the measurement, so no blob download/parsing is needed:

1. :meth:`_loc_id` + ``GETMEASUREMENTS`` — same measurement lookup as above.
2. ``PARAMCAT`` — the catalog of every scaled characteristic name DIDBase
   knows about (``IDENT``, ``NAME`` e.g. ``"foF2"``/``"hF"``, ``DESCRIPTION``,
   units, valid range). Browse it with :meth:`scaled_params_catalog`.
3. ``SELECT p.MEASUREMENTID, pc.NAME, p.VAL FROM PARAM p JOIN PARAMCAT pc
   ON p.PARAMCATID = pc.IDENT WHERE p.MEASUREMENTID IN (...) AND pc.NAME
   IN (...)`` — one numeric ``VAL`` row per (measurement, parameter).
   :meth:`fetch_scaled_params` pivots these into one DataFrame row per
   measurement with a ``datetime`` column plus one column per requested
   parameter name.
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
        self.url = url
        self.host, self.database = _parse_jdbc_url(url)
        self.user = user
        self.role = role
        self._password = password
        self._con = None
        self._stations_cache: Optional[pd.DataFrame] = None
        self._format_cache: Optional[Dict[int, str]] = None
        self._paramcat_cache: Optional[pd.DataFrame] = None

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
            raise RuntimeError("fdb package not installed.\n" "Fix: pip install fdb")

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
            if (
                "335544472" in msg
                or "password" in msg.lower()
                or "login" in msg.lower()
            ):
                raise RuntimeError(
                    f"Authentication failed for user '{user}'.\n"
                    f"Check username and password.\nFirebird error: {e}"
                )
            if (
                "335544578" in msg
                or "unavailable" in msg.lower()
                or "connection" in msg.lower()
            ):
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

    def close(self) -> None:
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

    def scaled_params_catalog(self) -> pd.DataFrame:
        """Return the catalog of every scaled characteristic DIDBase knows.

        Each row is one ``PARAMCAT`` entry — its ``NAME`` (e.g. ``"foF2"``,
        ``"hF"``, ``"hF2"``, ``"foE"``, ``"hmF2"``, ...) is exactly what can
        be passed in :meth:`fetch_scaled_params`'s ``params`` argument.
        Browse this first if you don't already know which characteristic
        names DIDBase supports.

        Result is cached after the first call.

        Returns:
            DataFrame with columns ``IDENT``, ``NAME``, ``DESCRIPTION``,
            ``WIDTH``, ``FRACTION``, ``EXPONENT``, ``UNITSID``,
            ``MINVALUE``, ``MAXVALUE``, ``FILLVALUE``.
        """
        if self._paramcat_cache is not None:
            return self._paramcat_cache
        cur = self._cur()
        cur.execute(
            "SELECT IDENT, NAME, DESCRIPTION, WIDTH, FRACTION, EXPONENT, "
            "UNITSID, MINVALUE, MAXVALUE, FILLVALUE FROM PARAMCAT "
            "ORDER BY IDENT"
        )
        cols = [d[0] for d in cur.description]
        df = pd.DataFrame(cur.fetchall(), columns=cols)
        df["NAME"] = df["NAME"].str.strip()
        df["DESCRIPTION"] = df["DESCRIPTION"].str.strip()
        self._paramcat_cache = df
        logger.info(f"scaled_params_catalog: {len(df)} known characteristic(s)")
        return df

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
        loc_id = self._loc_id(station)
        start_dt = _to_datetime(start)
        end_dt = _to_datetime(end, end_of_day=True)

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
        return meas_rows  # [(meas_id, timeut), …]

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
        if hasattr(data, "read"):
            # fdb returns a streaming BlobReader for this column; bytes(data)
            # trips a str/bytes bug in fdb's own BlobReader.readline(), so
            # read the reader directly instead.
            raw = data.read()
            data.close()
        else:
            raw = bytes(data)
        if zipped:
            with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                raw = zf.read(zf.namelist()[0])

        fmt_name = self._formats().get(fmt_id, "UNKNOWN")
        return raw, fmt_name

    def _format_id(self, fmt: str) -> Optional[int]:
        """Reverse-lookup ``FORMAT.IDENT`` for a format name (e.g. ``"SAO"``)."""
        fmt_upper = fmt.strip().upper()
        for ident, name in self._formats().items():
            if name.upper() == fmt_upper:
                return ident
        return None

    def _measurement_ids_with_format(
        self,
        meas_ids: List[int],
        fmt_id: int,
        chunk_size: int = 400,
    ) -> set:
        """Return the subset of ``meas_ids`` whose ``IONOGRAMHEADER.FORMATID``
        matches ``fmt_id`` — a metadata-only query with no blob transfer.

        Chunked to stay under Firebird's bound-parameter limit for windows
        with many measurements.
        """
        cur = self._cur()
        matching: set = set()
        for i in range(0, len(meas_ids), chunk_size):
            batch = meas_ids[i : i + chunk_size]
            placeholders = ",".join("?" for _ in batch)
            cur.execute(
                f"SELECT MEASUREMENTID FROM IONOGRAMHEADER "
                f"WHERE MEASUREMENTID IN ({placeholders}) AND FORMATID = ?",
                (*batch, fmt_id),
            )
            matching.update(r[0] for r in cur.fetchall())
        return matching

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
            fmt:     Restrict to one format (``"MMM"``, ``"RSF"``, ``"SBF"``,
                     ``"SAO"``, ...). ``None`` → all formats. When given,
                     measurements are pre-filtered via ``IONOGRAMHEADER``
                     (metadata only) so non-matching blobs are never
                     downloaded — the previous implementation fetched every
                     blob in the window before filtering client-side, which
                     could take minutes even for zero matches.
            out_dir: Directory to write files into (created if absent).

        Returns:
            List of :class:`~pathlib.Path` for every file written.
        """
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        meas = self._get_measurements(station, start, end)
        if not meas:
            return []

        meas_ids = [meas_id for meas_id, _ in meas]
        if fmt:
            fmt_id = self._format_id(fmt)
            if fmt_id is None:
                logger.warning(f"Unknown format '{fmt}' — no matching FORMAT entry.")
                return []
            wanted = self._measurement_ids_with_format(meas_ids, fmt_id)
            meas = [(mid, t) for mid, t in meas if mid in wanted]

        paths = []
        for meas_id, timeut in meas:
            result = self._fetch_blob(meas_id)
            if result is None:
                continue
            data, fmt_name = result
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
                     Measurements are pre-filtered via ``IONOGRAMHEADER``
                     (metadata only) so non-matching blobs are never
                     downloaded.

        Returns:
            Combined :class:`~pandas.DataFrame`.  Empty if nothing found.
        """
        from pynasonde.digisonde.parsers.grm import _EXTRACTOR

        cls = _EXTRACTOR.get(fmt.upper())
        if cls is None:
            raise ValueError(f"Unknown format '{fmt}'. Choose MMM, RSF, or SBF.")

        meas = self._get_measurements(station, start, end)
        if not meas:
            return pd.DataFrame()

        meas_ids = [meas_id for meas_id, _ in meas]
        fmt_id = self._format_id(fmt)
        if fmt_id is None:
            logger.warning(f"Unknown format '{fmt}' — no matching FORMAT entry.")
            return pd.DataFrame()
        wanted = self._measurement_ids_with_format(meas_ids, fmt_id)
        meas = [(mid, t) for mid, t in meas if mid in wanted]

        frames = []
        for meas_id, timeut in meas:
            result = self._fetch_blob(meas_id)
            if result is None:
                continue
            data, _fmt_name = result

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

    def _scaler_info(self, scaler_ids: List[int]) -> Dict[int, tuple]:
        """Return {SCALERID: (is_human, rank)} for the given scaler IDs."""
        if not scaler_ids:
            return {}
        cur = self._cur()
        placeholders = ",".join("?" for _ in scaler_ids)
        cur.execute(
            f"SELECT IDENT, HUMAN, RANK FROM SCALER WHERE IDENT IN ({placeholders})",
            tuple(scaler_ids),
        )
        return {r[0]: (bool(r[1]), r[2]) for r in cur.fetchall()}

    @staticmethod
    def _choose_scaler(
        candidate_ids: List[int],
        scaler_info: Dict[int, tuple],
        prefer_human: bool,
    ) -> int:
        """Pick one ``SCALERID`` when a measurement has more than one.

        If ``prefer_human`` and at least one candidate is a human scaler,
        pick the human with the highest ``RANK``; otherwise pick the
        highest-``RANK`` candidate overall (ties broken arbitrarily).
        """
        if len(candidate_ids) == 1:
            return candidate_ids[0]
        if prefer_human:
            human_ids = [i for i in candidate_ids if scaler_info.get(i, (False, 0))[0]]
            if human_ids:
                return max(human_ids, key=lambda i: scaler_info.get(i, (False, 0))[1])
        return max(candidate_ids, key=lambda i: scaler_info.get(i, (False, 0))[1])

    def fetch_scaled_params(
        self,
        station: str,
        start: DateLike,
        end: DateLike,
        params: List[str] = ["foF2", "hF"],
        prefer_human: bool = True,
        include_quality: bool = True,
        include_provenance: bool = False,
        chunk_size: int = 400,
    ) -> pd.DataFrame:
        """Fetch scaled ionosonde characteristics directly via SQL.

        Unlike :meth:`fetch_files`/:meth:`fetch_dataframe` (which download
        and parse raw ionogram or SAO blobs), this reads scaled
        characteristics straight out of DIDBase's ``PARAM``/``PARAMCAT``
        tables — one numeric value per (measurement, parameter), no blob
        download or parsing involved. This is available for any
        measurement DIDBase has scaled, independent of whether a ``SAO``
        blob was ever uploaded for it.

        A measurement can be scaled more than once (e.g. once by the
        ARTIST autoscaler, once by a human reviewer) — ``PARAM`` is keyed
        by ``(SCALERID, MEASUREMENTID, PARAMCATID)``. When a measurement
        has multiple scalers, exactly one is chosen per :meth:`_choose_scaler`
        (see ``prefer_human``) and all returned columns for that row come
        from that single scaler, so values stay internally consistent.

        Args:
            station: URSI station code (e.g. ``"AL945"``).
            start:   Start date/datetime (inclusive).
            end:     End date/datetime (inclusive).
            params:  ``PARAMCAT.NAME`` values to fetch, e.g. ``["foF2",
                "hF"]``. Call :meth:`scaled_params_catalog` to browse every
                name DIDBase supports (``foF2``, ``hF``, ``hF2``, ``foE``,
                ``foEs``, ``hmF1``, ...) instead of guessing.
            prefer_human: When a measurement was scaled by more than one
                scaler, prefer a human-reviewed value (``SCALER.HUMAN=1``)
                over an autoscaler's when both exist. If False, always
                pick the highest-``SCALER.RANK`` scaler regardless of
                human/auto.
            include_quality: If True, add ``{param}_qletter``/
                ``{param}_dletter`` (per-value URSI quality-flag letters
                from ``PARAM``) and ``scaling_quality``/
                ``scaling_confidence`` (per-measurement overall scores
                from ``SCALINGHEADER``) columns. Note: for autoscaler
                (non-human) rows, ``scaling_quality`` has been observed to
                just echo that scaler's ``SCALER.RANK`` rather than vary
                per measurement — ``scaling_confidence`` is the field
                that actually carries per-measurement signal.
            include_provenance: If True, add ``equipment_id``/
                ``program_id`` columns from ``MEASUREMENT`` — useful for
                spotting instrument/program changes across a long time
                series that could otherwise look like real variability.
            chunk_size: Maximum measurement IDs per SQL query, to stay
                well under Firebird's bound-parameter limit for windows
                with many measurements.

        Returns:
            DataFrame with a ``datetime`` column, a ``scaler_id`` column,
            one column per requested parameter name, and the optional
            quality/provenance columns described above. Empty if nothing
            found.

        Raises:
            ValueError: if any ``params`` entry is longer than
                ``PARAMCAT.NAME``'s column width (15 chars) — such a name
                can never match and would otherwise fail with a cryptic
                low-level ``fdb`` bind error instead.
        """
        too_long = [p for p in params if len(p) > 15]
        if too_long:
            raise ValueError(
                f"Invalid parameter name(s) {too_long}: longer than "
                "PARAMCAT.NAME's 15-char column width. Call "
                "scaled_params_catalog() to see valid names."
            )

        meas = self._get_measurements(station, start, end)
        if not meas:
            return pd.DataFrame()

        meas_time = {meas_id: timeut for meas_id, timeut in meas}
        meas_ids = list(meas_time.keys())
        cur = self._cur()

        def _chunks(seq):
            for i in range(0, len(seq), chunk_size):
                yield seq[i : i + chunk_size]

        # ── 1/2. Raw (measurement, scaler, param) rows + quality letters ───────
        param_placeholders = ",".join("?" for _ in params)
        raw_rows = []
        for batch in _chunks(meas_ids):
            batch_placeholders = ",".join("?" for _ in batch)
            cur.execute(
                f"""
                SELECT p.MEASUREMENTID, p.SCALERID, pc.NAME, p.VAL,
                       p.QLETTER, p.DLETTER
                FROM PARAM p
                JOIN PARAMCAT pc ON p.PARAMCATID = pc.IDENT
                WHERE p.MEASUREMENTID IN ({batch_placeholders})
                  AND pc.NAME IN ({param_placeholders})
                """,
                (*batch, *params),
            )
            raw_rows.extend(cur.fetchall())

        if not raw_rows:
            logger.warning(
                f"No scaled parameters {params} found for {station} in range."
            )
            return pd.DataFrame()

        by_meas: Dict[int, list] = {}
        for meas_id, scaler_id, name, val, qletter, dletter in raw_rows:
            by_meas.setdefault(meas_id, []).append(
                (scaler_id, name.strip(), val, qletter, dletter)
            )

        # ── 1. Disambiguate multi-scaler measurements ──────────────────────────
        scaler_ids = sorted({r[1] for r in raw_rows})
        scaler_info = self._scaler_info(scaler_ids)
        chosen_scaler: Dict[int, int] = {}
        collisions = 0
        for meas_id, rows in by_meas.items():
            candidates = sorted({r[0] for r in rows})
            if len(candidates) > 1:
                collisions += 1
            chosen_scaler[meas_id] = self._choose_scaler(
                candidates, scaler_info, prefer_human
            )
        if collisions:
            logger.info(
                f"[{station}] {collisions} measurement(s) had multiple scalers; "
                f"{'preferring human' if prefer_human else 'using highest-rank'} "
                "scaler per measurement."
            )

        # ── 3. Per-measurement scaling quality/confidence ──────────────────────
        quality_map: Dict[tuple, tuple] = {}
        if include_quality:
            for batch in _chunks(meas_ids):
                batch_placeholders = ",".join("?" for _ in batch)
                cur.execute(
                    f"SELECT MEASUREMENTID, SCALERID, QUALITY, CONFIDENCE "
                    f"FROM SCALINGHEADER WHERE MEASUREMENTID IN ({batch_placeholders})",
                    tuple(batch),
                )
                for mid, sid, quality, confidence in cur.fetchall():
                    quality_map[(mid, sid)] = (quality, confidence)

        # ── 4. Equipment/program provenance ─────────────────────────────────────
        provenance_map: Dict[int, tuple] = {}
        if include_provenance:
            for batch in _chunks(meas_ids):
                batch_placeholders = ",".join("?" for _ in batch)
                cur.execute(
                    f"SELECT IDENT, EQUIPMENTID, PROGRAMID FROM MEASUREMENT "
                    f"WHERE IDENT IN ({batch_placeholders})",
                    tuple(batch),
                )
                for mid, equipment_id, program_id in cur.fetchall():
                    provenance_map[mid] = (equipment_id, program_id)

        # ── Assemble one row per measurement from its chosen scaler only ──────
        records = []
        for meas_id, rows in by_meas.items():
            sid = chosen_scaler[meas_id]
            row = {"datetime": meas_time[meas_id], "scaler_id": sid}
            for scaler_id, name, val, qletter, dletter in rows:
                if scaler_id != sid:
                    continue
                row[name] = val
                if include_quality:
                    row[f"{name}_qletter"] = (qletter or "").strip() or None
                    row[f"{name}_dletter"] = (dletter or "").strip() or None
            if include_quality:
                quality, confidence = quality_map.get((meas_id, sid), (None, None))
                row["scaling_quality"] = quality
                row["scaling_confidence"] = confidence
            if include_provenance:
                equipment_id, program_id = provenance_map.get(meas_id, (None, None))
                row["equipment_id"] = equipment_id
                row["program_id"] = program_id
            records.append(row)

        df = pd.DataFrame(records).sort_values("datetime").reset_index(drop=True)
        logger.info(
            f"fetch_scaled_params → {len(df)} scaled record(s)  [{station} {params}]"
            + (f", {collisions} multi-scaler" if collisions else "")
        )
        return df

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
