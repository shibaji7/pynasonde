"""DIDBase connector against an in-memory fake of the ``fdb`` driver.

These tests need no Firebird client, network or credentials. They check the SQL the
connector sends, how it parses rows, caching, chunking and its error messages.
The opt-in ``test_didbase_live.py`` is the check against the real database.
"""

import datetime as dt
import io
import sys
import zipfile
from pathlib import Path

import pandas as pd
import pytest
from fakefdb import DatabaseError, FakeBlob, FakeDatabase

from pynasonde.digisonde.didbase import DidBaseConnector, _parse_jdbc_url, _to_datetime

URL = "jdbc:firebirdsql://db.example.org/didb"
SAMPLE_RSF = (
    Path(__file__).resolve().parents[1] / "examples/data/WS833_2023287000000.RSF"
)

T0 = dt.datetime(2023, 10, 14, 0, 0, 0)
T1 = dt.datetime(2023, 10, 14, 0, 15, 0)

STATIONS = (
    ["IDENT", "URSI", "NAME", "MINTIME", "MAXTIME"],
    [
        (7, "WS833  ", "Wallops Island ", T0, T1),
        (9, "AU930", "Austin  ", T0, T1),
    ],
)
FORMATS = (["IDENT", "NAME"], [(1, "SAO "), (2, "RSF "), (3, "MMM")])


def zipped(payload: bytes) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr("inner.bin", payload)
    return buf.getvalue()


def make_db(**extra_routes) -> FakeDatabase:
    """A fake database that answers the metadata queries every method needs."""
    routes = [
        ("FROM STATIONSLIST", STATIONS),
        ("FROM FORMAT", FORMATS),
    ]
    routes += list(extra_routes.items())
    return FakeDatabase(routes)


@pytest.fixture
def connector(monkeypatch):
    db = make_db()
    db.install(monkeypatch)
    with DidBaseConnector(URL, "user", "secret", "COMMON") as conn:
        yield conn, db


# ── URL and date helpers ──────────────────────────────────────────────────────


def test_parse_jdbc_url_with_and_without_port():
    assert _parse_jdbc_url(URL) == ("db.example.org", "didb")
    assert _parse_jdbc_url("jdbc:firebirdsql://h:3050/db/alias") == ("h", "db/alias")


def test_parse_jdbc_url_rejects_bad_url():
    with pytest.raises(ValueError, match="Cannot parse JDBC URL"):
        _parse_jdbc_url("postgres://h/db")


def test_to_datetime_forms():
    assert _to_datetime(T1) is T1
    assert _to_datetime(dt.date(2023, 10, 14)) == T0
    assert _to_datetime("2023-10-14") == T0
    assert _to_datetime("2023-10-14", end_of_day=True) == dt.datetime(
        2023, 10, 14, 23, 59, 59
    )
    # an explicit time is never overridden
    assert _to_datetime("2023-10-14T06:30:00", end_of_day=True).hour == 6
    assert _to_datetime(dt.date(2023, 10, 14), end_of_day=True).hour == 23


def test_constructor_does_not_connect():
    conn = DidBaseConnector(URL, "u", "p", "COMMON")
    assert conn.host == "db.example.org"
    assert conn.database == "didb"
    assert repr(conn) == "<DidBaseConnector db.example.org/didb [disconnected]>"
    with pytest.raises(RuntimeError, match="Not connected"):
        conn.stations()


# ── lifecycle ─────────────────────────────────────────────────────────────────


def test_connect_passes_credentials_and_closes(monkeypatch):
    db = make_db().install(monkeypatch)
    conn = DidBaseConnector(URL, "alice", "pw", "COMMON")
    assert conn.connect() is conn
    assert "connected" in repr(conn)
    assert db.connect_kwargs == [
        {
            "host": "db.example.org",
            "database": "didb",
            "user": "alice",
            "password": "pw",
            "role": "COMMON",
        }
    ]
    conn.close()
    assert db.connections[0].closed
    assert "disconnected" in repr(conn)
    conn.close()  # closing twice is harmless


def test_context_manager_closes_connection(monkeypatch):
    db = make_db().install(monkeypatch)
    with DidBaseConnector(URL, "u", "p", "COMMON"):
        assert not db.connections[0].closed
    assert db.connections[0].closed


def test_connect_failure_is_wrapped(monkeypatch):
    db = make_db().install(monkeypatch)
    db.connect_error = DatabaseError("boom")
    with pytest.raises(RuntimeError, match="Failed to connect to db.example.org/didb"):
        DidBaseConnector(URL, "u", "p", "COMMON").connect()


# ── ping ──────────────────────────────────────────────────────────────────────


def ping_db(monkeypatch, granted=("COMMON",), **extra):
    db = FakeDatabase(
        [
            (
                "RDB$USER_PRIVILEGES",
                (["RDB$RELATION_NAME"], [(r + "  ",) for r in granted]),
            ),
            ("FIRST 1 IDENT", (["IDENT"], [(1,)])),
        ]
    )
    for k, v in extra.items():
        setattr(db, k, v)
    return db.install(monkeypatch)


def ping():
    DidBaseConnector.ping(URL, "alice", "pw", "common")


def test_ping_ok(monkeypatch):
    db = ping_db(monkeypatch)
    ping()
    assert db.connections[0].closed
    # the role lookup is upper-cased like Firebird stores it
    assert db.calls[0][1] == ("ALICE",)


def test_ping_fdb_missing(monkeypatch):
    monkeypatch.setitem(sys.modules, "fdb", None)
    with pytest.raises(RuntimeError, match="fdb package not installed"):
        ping()


def test_ping_client_library_missing(monkeypatch):
    ping_db(monkeypatch, load_api_error=OSError("no libfbclient"))
    with pytest.raises(RuntimeError, match="native client library not found"):
        ping()


def test_ping_bad_url(monkeypatch):
    ping_db(monkeypatch)
    with pytest.raises(RuntimeError, match="Cannot parse JDBC URL"):
        DidBaseConnector.ping("nonsense", "u", "p", "COMMON")


@pytest.mark.parametrize(
    "error, expected",
    [
        ("Your user name and password are not defined", "Authentication failed"),
        ("error 335544472", "Authentication failed"),
        ("connection refused", "Cannot reach"),
        ("335544578 unavailable database", "Cannot reach"),
        ("something else entirely", "Connection failed"),
    ],
)
def test_ping_connect_errors(monkeypatch, error, expected):
    ping_db(monkeypatch, connect_error=DatabaseError(error))
    with pytest.raises(RuntimeError, match=expected):
        ping()


def test_ping_role_not_granted(monkeypatch):
    db = ping_db(monkeypatch, granted=("OTHER",))
    with pytest.raises(RuntimeError, match="not granted to user"):
        ping()
    assert db.connections[0].closed


def test_ping_role_lookup_failure(monkeypatch):
    db = FakeDatabase([("RDB$USER_PRIVILEGES", DatabaseError("denied"))]).install(
        monkeypatch
    )
    with pytest.raises(RuntimeError, match="Could not verify role membership"):
        ping()
    assert db.connections[0].closed


def test_ping_no_data_access(monkeypatch):
    db = FakeDatabase(
        [
            ("RDB$USER_PRIVILEGES", (["R"], [("COMMON",)])),
            ("FIRST 1 IDENT", DatabaseError("no permission")),
        ]
    ).install(monkeypatch)
    with pytest.raises(RuntimeError, match="does not grant EXECUTE"):
        ping()
    assert db.connections[0].closed


# ── station and catalogue metadata ────────────────────────────────────────────


def test_stations_strips_text_and_is_cached(connector):
    conn, db = connector
    df = conn.stations()
    assert list(df["URSI"]) == ["WS833", "AU930"]
    assert list(df["NAME"]) == ["Wallops Island", "Austin"]
    assert conn.stations() is df
    assert len(db.sql_containing("STATIONSLIST")) == 1


def test_loc_id_lookup(connector):
    conn, _ = connector
    assert conn._loc_id(" ws833 ") == 7
    with pytest.raises(ValueError, match="not found in DIDBase"):
        conn._loc_id("XX000")


def test_formats_and_reverse_lookup(connector):
    conn, db = connector
    assert conn._formats() == {1: "SAO", 2: "RSF", 3: "MMM"}
    assert conn._format_id("rsf") == 2
    assert conn._format_id("NOPE") is None
    assert len(db.sql_containing("FROM FORMAT")) == 1


def test_scaled_params_catalog_cached(monkeypatch):
    cols = [
        "IDENT",
        "NAME",
        "DESCRIPTION",
        "WIDTH",
        "FRACTION",
        "EXPONENT",
        "UNITSID",
        "MINVALUE",
        "MAXVALUE",
        "FILLVALUE",
    ]
    rows = [(1, "foF2   ", "F2 critical frequency ", 5, 2, 0, 1, 0, 30, 9999)]
    db = make_db(**{"FROM PARAMCAT ORDER BY": (cols, rows)}).install(monkeypatch)
    with DidBaseConnector(URL, "u", "p", "COMMON") as conn:
        df = conn.scaled_params_catalog()
        assert list(df["NAME"]) == ["foF2"]
        assert df.loc[0, "DESCRIPTION"] == "F2 critical frequency"
        assert conn.scaled_params_catalog() is df
    assert len(db.sql_containing("FROM PARAMCAT")) == 1


# ── measurements ──────────────────────────────────────────────────────────────


def with_measurements(monkeypatch, rows, **extra):
    db = make_db(
        **{"FROM GETMEASUREMENTS": (["IDENT", "TIMEUT"], rows)},
        **extra,
    ).install(monkeypatch)
    return db


def test_get_measurements_sends_station_id_and_day_range(monkeypatch):
    db = with_measurements(monkeypatch, [(100, T0), (101, T1)])
    with DidBaseConnector(URL, "u", "p", "COMMON") as conn:
        assert conn._get_measurements("WS833", "2023-10-14", "2023-10-14") == [
            (100, T0),
            (101, T1),
        ]
    _, params = db.sql_containing("GETMEASUREMENTS")[0]
    assert params == (7, T0, dt.datetime(2023, 10, 14, 23, 59, 59))


def test_get_measurements_empty(monkeypatch):
    with_measurements(monkeypatch, [])
    with DidBaseConnector(URL, "u", "p", "COMMON") as conn:
        assert conn._get_measurements("WS833", "2023-10-14", "2023-10-15") == []


def test_measurement_ids_with_format_chunks(monkeypatch):
    seen = []

    def header(params):
        *ids, fmt_id = params
        seen.append((len(ids), fmt_id))
        return (["MEASUREMENTID"], [(i,) for i in ids if i % 2 == 0])

    make_db(**{"FROM IONOGRAMHEADER": header}).install(monkeypatch)
    with DidBaseConnector(URL, "u", "p", "COMMON") as conn:
        got = conn._measurement_ids_with_format(list(range(1, 11)), 2, chunk_size=4)
    assert got == {2, 4, 6, 8, 10}
    assert seen == [(4, 2), (4, 2), (2, 2)]


# ── blobs ─────────────────────────────────────────────────────────────────────


def blob_db(monkeypatch, row):
    return make_db(
        **{"FROM IONOGRAM WHERE": (["DATA", "ZIPPED", "FORMATID"], row)}
    ).install(monkeypatch)


def test_fetch_blob_missing_row_or_data(monkeypatch):
    blob_db(monkeypatch, [])
    with DidBaseConnector(URL, "u", "p", "COMMON") as conn:
        assert conn._fetch_blob(1) is None
    blob_db(monkeypatch, [(None, 0, 1)])
    with DidBaseConnector(URL, "u", "p", "COMMON") as conn:
        assert conn._fetch_blob(1) is None


def test_fetch_blob_plain_bytes(monkeypatch):
    blob_db(monkeypatch, [(bytearray(b"abc"), 0, 2)])
    with DidBaseConnector(URL, "u", "p", "COMMON") as conn:
        assert conn._fetch_blob(5) == (b"abc", "RSF")


def test_fetch_blob_reader_is_read_and_closed(monkeypatch):
    reader = FakeBlob(b"streamed")
    blob_db(monkeypatch, [(reader, 0, 99)])
    with DidBaseConnector(URL, "u", "p", "COMMON") as conn:
        assert conn._fetch_blob(5) == (b"streamed", "UNKNOWN")
    assert reader.closed


def test_fetch_blob_unzips_first_member(monkeypatch):
    blob_db(monkeypatch, [(zipped(b"inner payload"), 1, 1)])
    with DidBaseConnector(URL, "u", "p", "COMMON") as conn:
        assert conn._fetch_blob(5) == (b"inner payload", "SAO")


# ── fetch_files ───────────────────────────────────────────────────────────────


def files_db(monkeypatch):
    def blob(params):
        meas_id = params[0]
        if meas_id == 102:
            return (["DATA", "ZIPPED", "FORMATID"], [])
        return (
            ["DATA", "ZIPPED", "FORMATID"],
            [(f"payload-{meas_id}".encode(), 0, 2)],
        )

    return with_measurements(
        monkeypatch,
        [(100, T0), (101, T1), (102, T1)],
        **{
            "FROM IONOGRAMHEADER": lambda p: (["MEASUREMENTID"], [(100,), (102,)]),
            "FROM IONOGRAM WHERE": blob,
        },
    )


def test_fetch_files_all_formats_skips_missing_blobs(monkeypatch, tmp_path):
    files_db(monkeypatch)
    with DidBaseConnector(URL, "u", "p", "COMMON") as conn:
        paths = conn.fetch_files(
            "WS833", "2023-10-14", "2023-10-14", out_dir=tmp_path / "o"
        )
    assert [p.name for p in paths] == [
        "WS833_2023287000000.RSF",
        "WS833_2023287001500.RSF",
    ]
    assert paths[0].read_bytes() == b"payload-100"


def test_fetch_files_filters_by_format_without_downloading_others(
    monkeypatch, tmp_path
):
    db = files_db(monkeypatch)
    with DidBaseConnector(URL, "u", "p", "COMMON") as conn:
        paths = conn.fetch_files(
            "WS833", "2023-10-14", "2023-10-14", fmt="rsf", out_dir=tmp_path
        )
    assert [p.read_bytes() for p in paths] == [b"payload-100"]
    # 101 is not an RSF header row, so its blob is never requested
    asked = [p[0] for _, p in db.sql_containing("FROM IONOGRAM WHERE")]
    assert asked == [100, 102]


def test_fetch_files_unknown_format_and_no_measurements(monkeypatch, tmp_path):
    files_db(monkeypatch)
    with DidBaseConnector(URL, "u", "p", "COMMON") as conn:
        assert (
            conn.fetch_files(
                "WS833", "2023-10-14", "2023-10-14", fmt="XYZ", out_dir=tmp_path
            )
            == []
        )
    with_measurements(monkeypatch, [])
    with DidBaseConnector(URL, "u", "p", "COMMON") as conn:
        assert (
            conn.fetch_files("WS833", "2023-10-14", "2023-10-14", out_dir=tmp_path)
            == []
        )


# ── fetch_dataframe ───────────────────────────────────────────────────────────


def dataframe_db(monkeypatch, payloads, header_ids=None):
    def blob(params):
        data = payloads.get(params[0])
        return (["DATA", "ZIPPED", "FORMATID"], [] if data is None else [(data, 0, 2)])

    ids = list(payloads) if header_ids is None else header_ids
    return with_measurements(
        monkeypatch,
        [(100, T0), (101, T1)],
        **{
            "FROM IONOGRAMHEADER": lambda p: (["MEASUREMENTID"], [(i,) for i in ids]),
            "FROM IONOGRAM WHERE": blob,
        },
    )


def test_fetch_dataframe_parses_real_rsf_blob(monkeypatch):
    dataframe_db(monkeypatch, {100: SAMPLE_RSF.read_bytes()})
    with DidBaseConnector(URL, "u", "p", "COMMON") as conn:
        df = conn.fetch_dataframe("WS833", "2023-10-14", "2023-10-14", fmt="RSF")
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


def test_fetch_dataframe_unparseable_blob_gives_empty_frame(monkeypatch):
    dataframe_db(monkeypatch, {100: b"not an ionogram", 101: None})
    with DidBaseConnector(URL, "u", "p", "COMMON") as conn:
        df = conn.fetch_dataframe("WS833", "2023-10-14", "2023-10-14", fmt="RSF")
    assert df.empty


def test_fetch_dataframe_rejects_unknown_format(connector):
    conn, _ = connector
    with pytest.raises(ValueError, match="Unknown format"):
        conn.fetch_dataframe("WS833", "2023-10-14", "2023-10-14", fmt="SAO")


def test_fetch_dataframe_no_measurements(monkeypatch):
    with_measurements(monkeypatch, [])
    with DidBaseConnector(URL, "u", "p", "COMMON") as conn:
        assert conn.fetch_dataframe(
            "WS833", "2023-10-14", "2023-10-14", fmt="RSF"
        ).empty


def test_fetch_dataframe_format_missing_from_database(monkeypatch):
    # FORMAT table has no SBF entry
    with_measurements(monkeypatch, [(100, T0)])
    with DidBaseConnector(URL, "u", "p", "COMMON") as conn:
        assert conn.fetch_dataframe(
            "WS833", "2023-10-14", "2023-10-14", fmt="SBF"
        ).empty


# ── scaled parameters ─────────────────────────────────────────────────────────

PARAM_ROWS = [
    # (measurement, scaler, name, value, qletter, dletter)
    (100, 1, "foF2   ", 5.5, "A", " "),
    (100, 1, "hF     ", 250.0, None, None),
    (100, 2, "foF2   ", 5.7, "B", "C"),
    (100, 2, "hF     ", 255.0, "", ""),
    (101, 1, "foF2   ", 6.1, "A", " "),
]


def scaled_db(monkeypatch, param_rows=PARAM_ROWS):
    return with_measurements(
        monkeypatch,
        [(100, T0), (101, T1)],
        **{
            "JOIN PARAMCAT": (["M", "S", "N", "V", "Q", "D"], param_rows),
            "FROM SCALER WHERE": (["IDENT", "HUMAN", "RANK"], [(1, 0, 5), (2, 1, 3)]),
            "FROM SCALINGHEADER": (
                ["M", "S", "Q", "C"],
                [(100, 1, 0.5, 0.6), (100, 2, 0.9, 0.95), (101, 1, 0.4, 0.3)],
            ),
            "FROM MEASUREMENT WHERE": (["I", "E", "P"], [(100, 11, 21), (101, 12, 22)]),
        },
    )


def test_scaled_params_rejects_long_names(connector):
    conn, _ = connector
    with pytest.raises(ValueError, match="longer than"):
        conn.fetch_scaled_params("WS833", "2023-10-14", "2023-10-14", params=["x" * 16])


def test_scaled_params_prefers_human_scaler(monkeypatch):
    scaled_db(monkeypatch)
    with DidBaseConnector(URL, "u", "p", "COMMON") as conn:
        df = conn.fetch_scaled_params("WS833", "2023-10-14", "2023-10-14")
    assert list(df["datetime"]) == [T0, T1]
    first = df.iloc[0]
    assert first["scaler_id"] == 2  # human beats the higher-ranked autoscaler
    assert first["foF2"] == 5.7 and first["hF"] == 255.0
    assert first["foF2_qletter"] == "B" and first["foF2_dletter"] == "C"
    assert pd.isna(first["hF_qletter"])  # blank letters become missing
    assert first["scaling_quality"] == 0.9 and first["scaling_confidence"] == 0.95
    assert df.iloc[1]["scaler_id"] == 1
    assert "equipment_id" not in df.columns


def test_scaled_params_highest_rank_when_not_preferring_human(monkeypatch):
    scaled_db(monkeypatch)
    with DidBaseConnector(URL, "u", "p", "COMMON") as conn:
        df = conn.fetch_scaled_params(
            "WS833", "2023-10-14", "2023-10-14", prefer_human=False
        )
    assert df.iloc[0]["scaler_id"] == 1 and df.iloc[0]["foF2"] == 5.5


def test_scaled_params_optional_columns(monkeypatch):
    scaled_db(monkeypatch)
    with DidBaseConnector(URL, "u", "p", "COMMON") as conn:
        df = conn.fetch_scaled_params(
            "WS833",
            "2023-10-14",
            "2023-10-14",
            include_quality=False,
            include_provenance=True,
        )
    assert list(df["equipment_id"]) == [11, 12]
    assert list(df["program_id"]) == [21, 22]
    assert not any(c.endswith("_qletter") for c in df.columns)
    assert "scaling_quality" not in df.columns


def test_scaled_params_chunks_and_binds_parameter_names(monkeypatch):
    db = scaled_db(monkeypatch)
    with DidBaseConnector(URL, "u", "p", "COMMON") as conn:
        conn.fetch_scaled_params(
            "WS833", "2023-10-14", "2023-10-14", params=["foF2"], chunk_size=1
        )
    calls = db.sql_containing("JOIN PARAMCAT")
    assert len(calls) == 2  # one per measurement chunk
    assert calls[0][1] == (100, "foF2")


def test_scaled_params_empty_results(monkeypatch):
    with_measurements(monkeypatch, [])
    with DidBaseConnector(URL, "u", "p", "COMMON") as conn:
        assert conn.fetch_scaled_params("WS833", "2023-10-14", "2023-10-14").empty
    scaled_db(monkeypatch, param_rows=[])
    with DidBaseConnector(URL, "u", "p", "COMMON") as conn:
        assert conn.fetch_scaled_params("WS833", "2023-10-14", "2023-10-14").empty


def test_scaler_info_and_choice():
    info = {1: (False, 9), 2: (True, 1), 3: (True, 4)}
    pick = DidBaseConnector._choose_scaler
    assert pick([1], info, True) == 1
    assert pick([1, 2, 3], info, True) == 3
    assert pick([1, 2, 3], info, False) == 1
    assert pick([1], {}, False) == 1
    # no human among the candidates: fall back to the highest rank
    assert pick([1, 4], {1: (False, 2), 4: (False, 7)}, True) == 4


def test_scaler_info_with_no_ids(connector):
    conn, db = connector
    assert conn._scaler_info([]) == {}
    assert not db.sql_containing("FROM SCALER")
