"""Regression tests for shared Digisonde batch-loader plumbing."""

from pathlib import Path

import pandas as pd

from pynasonde.digisonde.cadi.extractor import CadiExtractor
from pynasonde.digisonde.parsers.dvl import DvlExtractor
from pynasonde.digisonde.parsers.edp import EdpExtractor
from pynasonde.digisonde.parsers.sao import SaoExtractor


def _touch(folder: Path, names: list[str]) -> None:
    folder.mkdir()
    for name in names:
        (folder / name).write_text("")


def test_migrated_sao_and_xml_loaders_use_shared_batch_path(tmp_path, monkeypatch):
    folder = tmp_path / "sao"
    _touch(folder, ["A_20240101000000.SAO", "B_20240101000000.XML"])

    def fake_extract(file, **kwargs):
        return pd.DataFrame(
            {
                "source_file": [Path(file).name],
                "func_name": [kwargs["func_name"]],
                "mode": [kwargs["mode"]],
            }
        )

    monkeypatch.setattr(SaoExtractor, "extract_SAO", staticmethod(fake_extract))

    sao = SaoExtractor.load_SAO_files(
        folders=[str(folder)],
        ext="*.SAO",
        n_procs=1,
        func_name="scaled",
        mode="single",
    )
    xml = SaoExtractor.load_XML_files(
        folders=[str(folder)],
        ext="*.XML",
        n_procs=1,
        func_name="height_profile",
        mode="auto",
    )

    assert sao["source_file"].tolist() == ["A_20240101000000.SAO"]
    assert sao["func_name"].tolist() == ["scaled"]
    assert sao["mode"].tolist() == ["single"]
    assert xml["source_file"].tolist() == ["B_20240101000000.XML"]


def test_migrated_dvl_loader_uses_shared_batch_path(tmp_path, monkeypatch):
    folder = tmp_path / "dvl"
    _touch(folder, ["A_20240101000000.DVL", "B_20240101000000.DVL"])

    def fake_extract(file, **kwargs):
        return pd.DataFrame({"source_file": [Path(file).name]})

    monkeypatch.setattr(DvlExtractor, "extract_DVL_pandas", staticmethod(fake_extract))

    out = DvlExtractor.load_DVL_files(folders=[str(folder)], ext="*.DVL", n_procs=1)

    assert out["source_file"].tolist() == [
        "A_20240101000000.DVL",
        "B_20240101000000.DVL",
    ]


def test_migrated_edp_loader_is_implemented_with_shared_batch_path(
    tmp_path, monkeypatch
):
    folder = tmp_path / "edp"
    _touch(folder, ["A_20240101000000.EDP"])

    def fake_extract(file, **kwargs):
        return pd.DataFrame(
            {
                "source_file": [Path(file).name],
                "func_name": [kwargs["func_name"]],
            }
        )

    monkeypatch.setattr(EdpExtractor, "extract_EDP", staticmethod(fake_extract))

    out = EdpExtractor.load_EDP_files(
        folders=[str(folder)],
        ext="*.EDP",
        n_procs=1,
        func_name="scaled",
    )

    assert out["source_file"].tolist() == ["A_20240101000000.EDP"]
    assert out["func_name"].tolist() == ["scaled"]


def test_migrated_cadi_loader_uses_shared_batch_path(tmp_path, monkeypatch):
    folder = tmp_path / "cadi"
    _touch(folder, ["A.md4", "B.md2"])

    def fake_extract(file, **kwargs):
        return pd.DataFrame(
            {
                "source_file": [Path(file).name],
                "product": [kwargs["product"]],
                "dheight_km": [kwargs["dheight_km"]],
            }
        )

    monkeypatch.setattr(CadiExtractor, "extract_CADI", staticmethod(fake_extract))

    out = CadiExtractor.load_CADI_files(
        folders=[str(folder)],
        exts=("*.md2", "*.md4"),
        n_procs=1,
        dheight_km=2.5,
        product="products",
    )

    assert out["source_file"].tolist() == ["A.md4", "B.md2"]
    assert out["product"].tolist() == ["products", "products"]
    assert out["dheight_km"].tolist() == [2.5, 2.5]
