"""Execute digisonde example scripts with lightweight stubs for coverage."""

import datetime as dt
import runpy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT_ROOT / "docs/examples/figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def _cleanup(paths):
    for path in paths:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


@pytest.mark.skipif(
    not (PROJECT_ROOT / "examples/data/KR835_2024099160913.SKY").exists(),
    reason="Sample SKY file not available",
)
def test_run_sky_example(monkeypatch):
    pytest.importorskip("matplotlib")

    outputs = [
        FIG_DIR / "single_skymap.png",
        FIG_DIR / "panel_skymaps.png",
    ]

    monkeypatch.chdir(PROJECT_ROOT)
    runpy.run_path(str(PROJECT_ROOT / "examples/digisonde/sky.py"), run_name="__main__")

    for output in outputs:
        assert output.exists() and output.stat().st_size > 0

    _cleanup(outputs)


def test_run_sao_example(monkeypatch, tmp_path):
    pytest.importorskip("matplotlib")

    base_time = dt.datetime(2023, 10, 14)
    hp_df = pd.DataFrame(
        {
            "datetime": pd.date_range(base_time, periods=4, freq="6H"),
            "th": np.linspace(100, 200, 4),
            "ed": np.linspace(0.2, 0.8, 4),
        }
    )
    scaled_df = pd.DataFrame(
        {
            "datetime": pd.date_range(base_time, periods=4, freq="6H"),
            "hmF2": np.linspace(220, 260, 4),
            "foF2": np.linspace(4, 8, 4),
        }
    )

    from pynasonde.digisonde.parsers import sao as sao_module

    def fake_load(
        folders=None,
        ext="*.SAO",
        n_procs=4,
        extract_time_from_name=True,
        extract_stn_from_name=True,
        func_name="height_profile",
    ):
        return hp_df if func_name == "height_profile" else scaled_df

    monkeypatch.setattr(
        sao_module.SaoExtractor,
        "load_SAO_files",
        staticmethod(fake_load),
    )

    outputs = [
        FIG_DIR / "stack_sao_ne.png",
        FIG_DIR / "stack_sao_F2.png",
    ]

    monkeypatch.chdir(PROJECT_ROOT)
    runpy.run_path(str(PROJECT_ROOT / "examples/digisonde/sao.py"), run_name="__main__")

    for output in outputs:
        assert output.exists() and output.stat().st_size > 0

    _cleanup(outputs)


def test_run_dvl_example(monkeypatch):
    pytest.importorskip("matplotlib")

    from pynasonde.digisonde.parsers import dvl as dvl_module

    base_time = dt.datetime(2023, 10, 14)
    dvl_df = pd.DataFrame(
        {
            "datetime": pd.date_range(base_time, periods=4, freq="6H"),
            "Hb": np.linspace(250, 280, 4),
            "Ht": np.linspace(270, 300, 4),
            "Vx": np.linspace(-10, 10, 4),
            "Vy": np.linspace(-5, 5, 4),
            "Vz": np.linspace(-2, 2, 4),
            "Vx_err": np.ones(4),
            "Vy_err": np.ones(4),
            "Vz_err": np.ones(4),
            "Cord": ["GEO"] * 4,
        }
    )

    monkeypatch.setattr(
        dvl_module.DvlExtractor,
        "load_DVL_files",
        staticmethod(lambda *args, **kwargs: dvl_df),
    )

    output = FIG_DIR / "stackplots_dvl.png"

    monkeypatch.chdir(PROJECT_ROOT)
    runpy.run_path(str(PROJECT_ROOT / "examples/digisonde/dvl.py"), run_name="__main__")

    assert output.exists() and output.stat().st_size > 0
    _cleanup([output])


def test_run_rsf_example(monkeypatch):
    from pynasonde.digisonde.parsers import rsf as rsf_module

    class DummyUnit:
        header = {"foo": "bar"}
        frequency_groups = ["grp"]

    class DummyExtractor:
        def __init__(self, *args, **kwargs):
            self.rsf_data = SimpleNamespace(rsf_data_units=[DummyUnit()])

        def extract(self):
            return self

    monkeypatch.setattr(rsf_module, "RsfExtractor", DummyExtractor)

    monkeypatch.chdir(PROJECT_ROOT)
    runpy.run_path(str(PROJECT_ROOT / "examples/digisonde/rsf.py"), run_name="__main__")
