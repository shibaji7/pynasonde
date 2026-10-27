"""A tiny in-memory stand-in for the ``fdb`` Firebird driver.

The DIDBase and DriftBase connectors only use ``fdb.connect``, ``fdb.load_api`` and
cursors with ``execute``/``fetchone``/``fetchmany``/``fetchall``/``description``.
:class:`FakeDatabase` answers each SQL statement from a list of ``(needle, result)``
routes (first route whose needle occurs in the SQL wins), records every call, and
lets a test inject errors. No Firebird client library or network is needed.
"""

from __future__ import annotations

import sys
import types
from typing import Callable, List, Sequence, Tuple, Union


class DatabaseError(Exception):
    """Stand-in for ``fdb.fbcore.DatabaseError``."""


class FakeBlob:
    """Streaming blob reader like fdb's ``BlobReader`` (``read`` and ``close``)."""

    def __init__(self, payload: bytes):
        self.payload = payload
        self.closed = False

    def read(self) -> bytes:
        return self.payload

    def close(self) -> None:
        self.closed = True


Result = Tuple[Sequence[str], Sequence[tuple]]
Handler = Union[Result, Exception, Callable[[tuple], Union[Result, Exception]]]


class FakeCursor:
    def __init__(self, db: "FakeDatabase"):
        self._db = db
        self._rows: List[tuple] = []
        self.description: List[tuple] = []

    def execute(self, sql: str, params: tuple = ()) -> None:
        self._db.calls.append((sql, tuple(params)))
        for needle, handler in self._db.routes:
            if needle in sql:
                result = handler(tuple(params)) if callable(handler) else handler
                break
        else:
            raise AssertionError(f"FakeDatabase has no route for SQL: {sql[:100]!r}")
        if isinstance(result, Exception):
            raise result
        cols, rows = result
        self.description = [(c,) for c in cols]
        self._rows = list(rows)

    def fetchone(self):
        return self._rows.pop(0) if self._rows else None

    def fetchmany(self, size: int = 1):
        batch, self._rows = self._rows[:size], self._rows[size:]
        return batch

    def fetchall(self):
        rows, self._rows = self._rows, []
        return rows


class FakeConnection:
    def __init__(self, db: "FakeDatabase"):
        self._db = db
        self.closed = False

    def cursor(self) -> FakeCursor:
        return FakeCursor(self._db)

    def close(self) -> None:
        self.closed = True


class FakeDatabase:
    """Routes SQL to canned results and records the calls made."""

    def __init__(self, routes: Sequence[Tuple[str, Handler]] = ()):
        self.routes = list(routes)
        self.calls: List[Tuple[str, tuple]] = []
        self.connect_kwargs: List[dict] = []
        self.connect_error: Union[Exception, None] = None
        self.load_api_error: Union[Exception, None] = None
        self.connections: List[FakeConnection] = []

    def sql_containing(self, needle: str) -> List[Tuple[str, tuple]]:
        return [c for c in self.calls if needle in c[0]]

    def module(self) -> types.ModuleType:
        """Build a module object that quacks like ``fdb``."""
        mod = types.ModuleType("fdb")
        mod.fbcore = types.SimpleNamespace(DatabaseError=DatabaseError)
        mod.DatabaseError = DatabaseError

        def load_api(*_a, **_k):
            if self.load_api_error is not None:
                raise self.load_api_error

        def connect(**kwargs):
            self.connect_kwargs.append(kwargs)
            if self.connect_error is not None:
                raise self.connect_error
            con = FakeConnection(self)
            self.connections.append(con)
            return con

        mod.load_api = load_api
        mod.connect = connect
        return mod

    def install(self, monkeypatch) -> "FakeDatabase":
        """Make ``import fdb`` return this fake for the duration of a test."""
        monkeypatch.setitem(sys.modules, "fdb", self.module())
        return self
