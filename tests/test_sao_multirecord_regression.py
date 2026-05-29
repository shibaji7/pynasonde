"""Regression tests for multi-record SAO parsing modes."""

import datetime as dt

import pandas as pd

from pynasonde.digisonde.parsers.sao import SaoExtractor


def _fake_multi_lines():
    ff_lines = [
        "FF202429910250003040120126101600007520000000001107301000080202560000820040000",
        "FF202429910250008040120126101600007520000000001107301000080202560000820040000",
        "FF202429910250013040120126101600007520000000001107301000080202560000820040000",
    ]
    lines = []
    for ff in ff_lines:
        lines.extend(
            [
                "HDR1",
                "HDR2",
                ff,
                "DATA1",
                "DATA2",
                "DATA3",
                "DATA4",
                "DATA5",
                "DATA6",
                "DATA7",
            ]
        )
    return lines


def _fake_structs():
    return [
        {
            "Scaled": {"foF2": 5.1, "hmF2": 250.0},
            "TH": [100.0, 120.0],
            "PF": [3.0, 3.5],
            "ED": [1.0e10, 1.1e10],
        },
        {
            "Scaled": {"foF2": 5.2, "hmF2": 260.0},
            "TH": [110.0, 130.0],
            "PF": [3.2, 3.7],
            "ED": [1.2e10, 1.3e10],
        },
        {
            "Scaled": {"foF2": 5.3, "hmF2": 270.0},
            "TH": [115.0, 140.0],
            "PF": [3.4, 3.9],
            "ED": [1.4e10, 1.5e10],
        },
    ]


def test_extract_auto_multi_scaled_and_height(monkeypatch):
    ex = SaoExtractor("KR835_20241025(299).SAO", extract_time_from_name=True)
    lines = _fake_multi_lines()
    structs = iter(_fake_structs())

    monkeypatch.setattr(ex, "read_file", lambda: lines)
    monkeypatch.setattr(ex, "_find_record_starts", lambda _lines: [0, 10, 20])
    monkeypatch.setattr(ex, "_extract_record_struct", lambda _section: next(structs))

    ex.extract(mode="auto")
    assert len(ex.sao_records) == 3
    assert ex.sao_records[0]["record_datetime"] == dt.datetime(2024, 10, 25, 0, 3, 0)
    assert ex.sao_records[-1]["record_datetime"] == dt.datetime(2024, 10, 25, 0, 13, 0)

    scaled = ex.get_scaled_datasets()
    assert isinstance(scaled, pd.DataFrame)
    assert len(scaled) == 3
    assert scaled["record_index"].tolist() == [0, 1, 2]
    assert "source_file" in scaled.columns

    hp = ex.get_height_profile()
    assert isinstance(hp, pd.DataFrame)
    assert hp["record_index"].nunique() == 3
    assert "source_file" in hp.columns


def test_extract_single_record_index_on_multi(monkeypatch):
    ex = SaoExtractor("KR835_20241025(299).SAO", extract_time_from_name=True)
    lines = _fake_multi_lines()
    structs = iter(_fake_structs())

    monkeypatch.setattr(ex, "read_file", lambda: lines)
    monkeypatch.setattr(ex, "_find_record_starts", lambda _lines: [0, 10, 20])
    monkeypatch.setattr(ex, "_extract_record_struct", lambda _section: next(structs))

    ex.extract(mode="single", record_index=1)
    assert len(ex.sao_records) == 1
    assert ex.sao_records[0]["record_datetime"] == dt.datetime(2024, 10, 25, 0, 8, 0)

    scaled = ex.get_scaled_datasets()
    assert len(scaled) == 1
    assert scaled["datetime"].iloc[0] == dt.datetime(2024, 10, 25, 0, 8, 0)


def test_extract_single_negative_index_on_multi(monkeypatch):
    ex = SaoExtractor("KR835_20241025(299).SAO", extract_time_from_name=True)
    lines = _fake_multi_lines()
    structs = iter(_fake_structs())

    monkeypatch.setattr(ex, "read_file", lambda: lines)
    monkeypatch.setattr(ex, "_find_record_starts", lambda _lines: [0, 10, 20])
    monkeypatch.setattr(ex, "_extract_record_struct", lambda _section: next(structs))

    ex.extract(mode="single", record_index=-1)
    assert len(ex.sao_records) == 1
    assert ex.sao_records[0]["record_datetime"] == dt.datetime(2024, 10, 25, 0, 13, 0)


class _DummyPool:
    def __init__(self, n_procs):
        self.n_procs = n_procs

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def imap(self, fn, items):
        for item in items:
            yield fn(item)


def test_load_sao_files_propagates_mode_and_record_index(monkeypatch):
    import pynasonde.digisonde.parsers.sao as sao_module

    calls = []
    fake_files = ["a.SAO", "b.SAO"]

    def fake_extract(file, **kwargs):
        calls.append((file, kwargs))
        return pd.DataFrame(
            {
                "source_file": [file],
                "record_index": [kwargs.get("record_index", -1)],
                "mode": [kwargs.get("mode", "")],
            }
        )

    monkeypatch.setattr(sao_module, "Pool", _DummyPool)
    monkeypatch.setattr(
        sao_module.glob, "glob", lambda *_args, **_kwargs: list(fake_files)
    )
    monkeypatch.setattr(sao_module, "tqdm", lambda x, total=None: x)
    monkeypatch.setattr(SaoExtractor, "extract_SAO", staticmethod(fake_extract))

    out = SaoExtractor.load_SAO_files(
        folders=["/tmp/demo"],
        ext="*.SAO",
        n_procs=2,
        extract_time_from_name=False,
        extract_stn_from_name=False,
        func_name="scaled",
        mode="single",
        record_index=5,
    )

    assert isinstance(out, pd.DataFrame)
    assert len(out) == 2
    assert out["record_index"].tolist() == [5, 5]
    assert out["mode"].tolist() == ["single", "single"]
    assert [c[0] for c in calls] == fake_files
