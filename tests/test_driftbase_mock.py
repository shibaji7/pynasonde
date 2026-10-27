"""DriftBase connector against an in-memory fake of the ``fdb`` driver.

No Firebird client, network or credentials are needed. The raw blobs returned by the
fake are the DFT and SKY sample files in ``examples/data/``, so the downloaded files
are checked byte for byte. ``test_driftbase_live.py`` is the opt-in check against the
real database.
"""

import datetime as dt
import io
import sys
import zipfile
from pathlib import Path

import pytest
from fakefdb import DatabaseError, FakeBlob, FakeDatabase

from pynasonde.digisonde.driftbase import (
    DriftBaseConnector,
    _parse_jdbc_url,
    _to_datetime,
)

URL = "jdbc:firebirdsql://drift.example.org/drift"
DATA = Path(__file__).resolve().parents[1] / "examples/data"
SAMPLE_DFT = DATA / "KR835_2023287000915.DFT"
SAMPLE_SKY = DATA / "KR835_2024099160913.SKY"

T0 = dt.datetime(2023, 10, 14, 0, 0, 0)
T1 = dt.datetime(2023, 10, 14, 0, 15, 0)

STATIONS = (
    ["IDENT", "URSI", "NAME", "MINTIME", "MAXTIME"],
    [(3, "WI937 ", "Millstone Hill ", T0, T1), (4, "AL945", "Alpena", T0, T1)],
)
EXPERTS = (
    ["IDENT", "LOGNAME", "FNAME", "LNAME", "RANK"],
    [(1, "auto ", "Auto ", "Scaler ", 1), (2, "human", "Ann ", "Lee ", 5)],
)


def zipped(payload: bytes) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("inner.bin", payload)
    return buf.getvalue()


def make_db(**extra_routes) -> FakeDatabase:
    return FakeDatabase(list(extra_routes.items()))


def measurements(rows):
    return {
        "FROM GETMEASUREMENTSALT": (
            ["IDENT", "TIMEUT", "LOCATIONID", "EQUIPMENTID"],
            rows,
        )
    }


ROWS = [(100, T0, 3, 1), (101, T1, 3, 1)]


def open_conn(monkeypatch, **routes):
    db = make_db(**routes).install(monkeypatch)
    return DriftBaseConnector(URL, "u", "p", "COMMON"), db


# ── URL and date helpers ──────────────────────────────────────────────────────


def test_parse_jdbc_url():
    assert _parse_jdbc_url(URL) == ("drift.example.org", "drift")
    assert _parse_jdbc_url("jdbc:firebirdsql://h:3050/alias") == ("h", "alias")
    with pytest.raises(ValueError, match="Cannot parse JDBC URL"):
        _parse_jdbc_url("http://h/db")


def test_to_datetime_forms():
    assert _to_datetime(T1) is T1
    assert _to_datetime(dt.date(2023, 10, 14)) == T0
    assert _to_datetime("2023-10-14") == T0
    assert _to_datetime("2023-10-14", end_of_day=True).hour == 23
    assert _to_datetime("2023-10-14T05:00:00", end_of_day=True).hour == 5


def test_chunks_helper():
    assert list(DriftBaseConnector._chunks([1, 2, 3, 4, 5], 2)) == [[1, 2], [3, 4], [5]]
    assert list(DriftBaseConnector._chunks([], 2)) == []


# ── lifecycle ─────────────────────────────────────────────────────────────────


def test_not_connected_until_connect():
    conn = DriftBaseConnector(URL, "u", "p", "COMMON")
    assert repr(conn) == "<DriftBaseConnector drift.example.org/drift [disconnected]>"
    with pytest.raises(RuntimeError, match="Not connected"):
        conn.stations()


def test_connect_close_and_context_manager(monkeypatch):
    conn, db = open_conn(monkeypatch)
    assert conn.connect() is conn
    assert "[connected]" in repr(conn)
    assert db.connect_kwargs[0]["role"] == "COMMON"
    assert db.connect_kwargs[0]["database"] == "drift"
    conn.close()
    assert db.connections[0].closed
    conn.close()
    with conn:
        assert not db.connections[1].closed
    assert db.connections[1].closed


def test_connect_failure_is_wrapped(monkeypatch):
    conn, db = open_conn(monkeypatch)
    db.connect_error = DatabaseError("refused")
    with pytest.raises(RuntimeError, match="Failed to connect to drift.example.org"):
        conn.connect()


# ── ping ──────────────────────────────────────────────────────────────────────


def ping_db(monkeypatch, granted=("COMMON",), **attrs):
    db = FakeDatabase(
        [
            (
                "RDB$USER_PRIVILEGES",
                (["R"], [(r + "  ",) for r in granted]),
            ),
            ("FIRST 1 IDENT", (["IDENT"], [(1,)])),
        ]
    )
    for key, value in attrs.items():
        setattr(db, key, value)
    return db.install(monkeypatch)


def ping(url=URL):
    DriftBaseConnector.ping(url, "alice", "pw", "common")


def test_ping_ok(monkeypatch):
    db = ping_db(monkeypatch)
    ping()
    assert db.connections[0].closed
    assert db.calls[0][1] == ("ALICE",)


def test_ping_fdb_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "fdb", None)
    with pytest.raises(RuntimeError, match="fdb package not installed"):
        ping()


def test_ping_client_library_missing(monkeypatch):
    ping_db(monkeypatch, load_api_error=OSError("no lib"))
    with pytest.raises(RuntimeError, match="native client library not found"):
        ping()


def test_ping_bad_url(monkeypatch):
    ping_db(monkeypatch)
    with pytest.raises(RuntimeError, match="Cannot parse JDBC URL"):
        ping("junk")


@pytest.mark.parametrize(
    "error, expected",
    [
        ("login failed", "Authentication failed"),
        ("335544472", "Authentication failed"),
        ("I/O error: No such file or directory", "not found on drift.example.org"),
        ("335544344", "not found on drift.example.org"),
        ("connection reset", "Cannot reach"),
        ("335544578", "Cannot reach"),
        ("weird", "Connection failed"),
    ],
)
def test_ping_connect_errors(monkeypatch, error, expected):
    ping_db(monkeypatch, connect_error=DatabaseError(error))
    with pytest.raises(RuntimeError, match=expected):
        ping()


def test_ping_role_not_granted(monkeypatch):
    db = ping_db(monkeypatch, granted=("READONLY",))
    with pytest.raises(RuntimeError, match="not granted to user"):
        ping()
    assert db.connections[0].closed


def test_ping_role_lookup_failure(monkeypatch):
    db = FakeDatabase([("RDB$USER_PRIVILEGES", DatabaseError("denied"))])
    db.install(monkeypatch)
    with pytest.raises(RuntimeError, match="Could not verify role membership"):
        ping()
    assert db.connections[0].closed


def test_ping_no_data_access(monkeypatch):
    db = FakeDatabase(
        [
            ("RDB$USER_PRIVILEGES", (["R"], [("COMMON",)])),
            ("FIRST 1 IDENT", DatabaseError("no execute")),
        ]
    ).install(monkeypatch)
    with pytest.raises(RuntimeError, match="does not grant EXECUTE"):
        ping()
    assert db.connections[0].closed


# ── metadata ──────────────────────────────────────────────────────────────────


def test_stations_stripped_and_cached(monkeypatch):
    conn, db = open_conn(monkeypatch, **{"FROM STATIONSLIST": STATIONS})
    with conn:
        df = conn.stations()
        assert list(df["URSI"]) == ["WI937", "AL945"]
        assert list(df["NAME"]) == ["Millstone Hill", "Alpena"]
        assert conn.stations() is df
    assert len(db.sql_containing("STATIONSLIST")) == 1


def test_experts_stripped_and_cached(monkeypatch):
    conn, db = open_conn(monkeypatch, **{"FROM EXPERT": EXPERTS})
    with conn:
        df = conn.experts()
        assert list(df["LOGNAME"]) == ["auto", "human"]
        assert list(df["LNAME"]) == ["Scaler", "Lee"]
        assert conn.experts() is df
    assert len(db.sql_containing("FROM EXPERT")) == 1


def test_get_measurements_uses_ursi_code_and_day_range(monkeypatch):
    conn, db = open_conn(monkeypatch, **measurements(ROWS))
    with conn:
        assert conn._get_measurements(" wi937 ", "2023-10-14", "2023-10-14") == ROWS
    _, params = db.sql_containing("GETMEASUREMENTSALT")[0]
    assert params == ("WI937", T0, dt.datetime(2023, 10, 14, 23, 59, 59))


def test_get_measurements_empty(monkeypatch):
    conn, _ = open_conn(monkeypatch, **measurements([]))
    with conn:
        assert conn._get_measurements("WI937", "2023-10-14", "2023-10-14") == []


def test_best_experts_chunked(monkeypatch):
    def best(params):
        return (["MEASID", "EXPERTID"], [(m, m + 1000) for m in params])

    conn, db = open_conn(monkeypatch, **{"FROM BESTEXPERT": best})
    with conn:
        assert conn._best_experts([]) == {}
        got = conn._best_experts([1, 2, 3], chunk_size=2)
    assert got == {1: 1001, 2: 1002, 3: 1003}
    assert len(db.sql_containing("BESTEXPERT")) == 2


# ── blobs ─────────────────────────────────────────────────────────────────────


def test_read_blob_bytes_and_reader():
    assert DriftBaseConnector._read_blob(bytearray(b"xyz")) == b"xyz"
    reader = FakeBlob(b"stream")
    assert DriftBaseConnector._read_blob(reader) == b"stream"
    assert reader.closed


def drift_route(row):
    return {"FROM DRIFT WHERE": (["DATA", "ZIPPED"], row)}


def test_fetch_drift_blob_variants(monkeypatch):
    conn, _ = open_conn(monkeypatch, **drift_route([]))
    with conn:
        assert conn._fetch_drift_blob(1) is None
    conn, _ = open_conn(monkeypatch, **drift_route([(None, 0)]))
    with conn:
        assert conn._fetch_drift_blob(1) is None
    conn, _ = open_conn(monkeypatch, **drift_route([(zipped(b"dft bytes"), 1)]))
    with conn:
        assert conn._fetch_drift_blob(1) == b"dft bytes"
    conn, _ = open_conn(monkeypatch, **drift_route([(FakeBlob(b"plain"), 0)]))
    with conn:
        assert conn._fetch_drift_blob(1) == b"plain"


def sky_route(handler):
    return {"FROM SKYHEADER": handler}


def test_fetch_sky_blob_with_and_without_expert(monkeypatch):
    seen = []

    def sky(params):
        seen.append(params)
        return (["E", "D", "Z"], [(params[-1] if len(params) > 1 else 7, b"sky", 0)])

    conn, db = open_conn(monkeypatch, **sky_route(sky))
    with conn:
        assert conn._fetch_sky_blob(10, 4) == (b"sky", 4)
        assert conn._fetch_sky_blob(10, None) == (b"sky", 7)
    assert seen == [(10, 4), (10,)]
    assert "ROWS 1" in db.calls[1][0]


def test_fetch_sky_blob_missing_and_zipped(monkeypatch):
    conn, _ = open_conn(monkeypatch, **sky_route((["E", "D", "Z"], [])))
    with conn:
        assert conn._fetch_sky_blob(1, None) is None
    conn, _ = open_conn(monkeypatch, **sky_route((["E", "D", "Z"], [(1, None, 0)])))
    with conn:
        assert conn._fetch_sky_blob(1, 1) is None
    row = (2, zipped(b"unzipped sky"), 1)
    conn, _ = open_conn(monkeypatch, **sky_route((["E", "D", "Z"], [row])))
    with conn:
        assert conn._fetch_sky_blob(1, 2) == (b"unzipped sky", 2)


# ── fetch_drift_files / fetch_sky_files ───────────────────────────────────────


def test_fetch_drift_files_writes_sample_dft_byte_for_byte(monkeypatch, tmp_path):
    dft = SAMPLE_DFT.read_bytes()

    def drift(params):
        return (["DATA", "ZIPPED"], [] if params[0] == 101 else [(dft, 0)])

    conn, _ = open_conn(
        monkeypatch, **measurements(ROWS), **{"FROM DRIFT WHERE": drift}
    )
    with conn:
        paths = conn.fetch_drift_files(
            "WI937", "2023-10-14", "2023-10-14", out_dir=tmp_path / "drift"
        )
    assert [p.name for p in paths] == ["WI937_2023287000000.DFT"]  # 101 had no blob
    assert paths[0].read_bytes() == dft


def test_fetch_drift_files_no_measurements(monkeypatch, tmp_path):
    conn, _ = open_conn(monkeypatch, **measurements([]))
    with conn:
        assert (
            conn.fetch_drift_files(
                "WI937", "2023-10-14", "2023-10-14", out_dir=tmp_path
            )
            == []
        )


def sky_db_routes(sky_bytes):
    def sky(params):
        expert = params[1] if len(params) > 1 else 2
        return (["E", "D", "Z"], [(expert, sky_bytes, 0)])

    return {
        **measurements(ROWS),
        "FROM BESTEXPERT": lambda p: (["MEASID", "EXPERTID"], [(100, 1), (101, 2)]),
        "FROM SKYHEADER": sky,
    }


def test_fetch_sky_files_prefers_best_expert(monkeypatch, tmp_path):
    sky = SAMPLE_SKY.read_bytes()
    conn, db = open_conn(monkeypatch, **sky_db_routes(sky))
    with conn:
        paths = conn.fetch_sky_files(
            "WI937", "2023-10-14", "2023-10-14", out_dir=tmp_path
        )
    assert [p.name for p in paths] == [
        "WI937_2023287000000.SKY",
        "WI937_2023287001500.SKY",
    ]
    assert paths[0].read_bytes() == sky
    # one query per measurement, each pinned to that measurement's best expert
    assert [p for _, p in db.sql_containing("FROM SKYHEADER")] == [(100, 1), (101, 2)]


def test_fetch_sky_files_all_experts_adds_suffix(monkeypatch, tmp_path):
    conn, db = open_conn(monkeypatch, **sky_db_routes(b"sky"))
    with conn:
        paths = conn.fetch_sky_files(
            "WI937", "2023-10-14", "2023-10-14", out_dir=tmp_path, prefer_best=False
        )
    assert [p.name for p in paths] == [
        "WI937_2023287000000_expert2.SKY",
        "WI937_2023287001500_expert2.SKY",
    ]
    assert not db.sql_containing("BESTEXPERT")


def test_fetch_sky_files_skips_measurements_without_blob_and_empty_window(
    monkeypatch, tmp_path
):
    routes = sky_db_routes(b"sky")
    routes["FROM SKYHEADER"] = (["E", "D", "Z"], [])
    conn, _ = open_conn(monkeypatch, **routes)
    with conn:
        assert (
            conn.fetch_sky_files("WI937", "2023-10-14", "2023-10-14", out_dir=tmp_path)
            == []
        )
    conn, _ = open_conn(monkeypatch, **measurements([]))
    with conn:
        assert (
            conn.fetch_sky_files("WI937", "2023-10-14", "2023-10-14", out_dir=tmp_path)
            == []
        )


def test_sample_blobs_round_trip_through_the_real_parsers(monkeypatch, tmp_path):
    """What the connector writes is what DftExtractor and SkyExtractor expect."""
    from pynasonde.digisonde.parsers.dft import DftExtractor
    from pynasonde.digisonde.parsers.sky import SkyExtractor

    dft = SAMPLE_DFT.read_bytes()
    conn, _ = open_conn(
        monkeypatch,
        **measurements([(100, T0, 3, 1)]),
        **{"FROM DRIFT WHERE": (["DATA", "ZIPPED"], [(dft, 0)])},
    )
    with conn:
        (path,) = conn.fetch_drift_files(
            "WI937", "2023-10-14", "2023-10-14", out_dir=tmp_path
        )
    ext = DftExtractor(
        str(path), extract_time_from_name=True, extract_stn_from_name=True
    )
    ext.extract()
    assert len(ext.to_pandas()) > 0

    sky = SAMPLE_SKY.read_bytes()
    conn, _ = open_conn(monkeypatch, **sky_db_routes(sky))
    with conn:
        paths = conn.fetch_sky_files(
            "WI937", "2023-10-14", "2023-10-14", out_dir=tmp_path
        )
    ext = SkyExtractor(
        str(paths[0]), extract_time_from_name=True, extract_stn_from_name=True
    )
    ext.extract()
    assert len(ext.to_pandas()) > 0


# ── fetch_velocity ────────────────────────────────────────────────────────────

VEL_COLS = ["MEASUREMENTID", "EXPERTID", "VX", "VY"]
VEL_ROWS = [
    (101, 2, 10.0, 11.0),
    (100, 1, 1.0, 2.0),
    (100, 2, 3.0, 4.0),  # a second expert for measurement 100
]


def velocity_routes(best=((100, 1), (101, 2))):
    return {
        **measurements(ROWS),
        "FROM VELOCITY": (VEL_COLS, VEL_ROWS),
        "FROM BESTEXPERT": (["MEASID", "EXPERTID"], list(best)),
    }


def test_fetch_velocity_keeps_best_expert_and_sorts_by_time(monkeypatch):
    conn, _ = open_conn(monkeypatch, **velocity_routes())
    with conn:
        df = conn.fetch_velocity("WI937", "2023-10-14", "2023-10-14")
    assert list(df["MEASUREMENTID"]) == [100, 101]
    assert list(df["EXPERTID"]) == [1, 2]
    assert list(df["datetime"]) == [T0, T1]


def test_fetch_velocity_all_experts(monkeypatch):
    conn, db = open_conn(monkeypatch, **velocity_routes())
    with conn:
        df = conn.fetch_velocity("WI937", "2023-10-14", "2023-10-14", prefer_best=False)
    assert len(df) == 3
    assert not db.sql_containing("BESTEXPERT")


def test_fetch_velocity_without_best_expert_view_keeps_everything(monkeypatch):
    conn, _ = open_conn(monkeypatch, **velocity_routes(best=()))
    with conn:
        df = conn.fetch_velocity("WI937", "2023-10-14", "2023-10-14")
    assert len(df) == 3


def test_fetch_velocity_chunks_measurement_ids(monkeypatch):
    conn, db = open_conn(monkeypatch, **velocity_routes())
    with conn:
        conn.fetch_velocity("WI937", "2023-10-14", "2023-10-14", chunk_size=1)
    assert len(db.sql_containing("FROM VELOCITY")) == 2


def test_fetch_velocity_empty_cases(monkeypatch):
    conn, _ = open_conn(monkeypatch, **measurements([]))
    with conn:
        assert conn.fetch_velocity("WI937", "2023-10-14", "2023-10-14").empty
    routes = velocity_routes()
    routes["FROM VELOCITY"] = (VEL_COLS, [])
    conn, _ = open_conn(monkeypatch, **routes)
    with conn:
        assert conn.fetch_velocity("WI937", "2023-10-14", "2023-10-14").empty
