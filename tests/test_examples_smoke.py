"""Smoke-run example scripts that ship with data, writing outputs to a temp dir."""

import runpy
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "examples" / "data"


@pytest.mark.skipif(
    not list(DATA_DIR.glob("WS833_*.RSF")), reason="Sample RSF files not available"
)
def test_rsf_scaler_example_runs(monkeypatch, tmp_path):
    """Run examples/digisonde/rsf_scaler.py on the bundled WS833 RSF files.

    The script writes relative paths (``tmp/`` and ``docs/examples/figures/``),
    so run it from a temporary directory with the data symlinked in; this keeps
    the tracked figure in the repository untouched.
    """
    pytest.importorskip("sklearn", reason="needs pynasonde[digisonde-scaler]")
    pytest.importorskip("joblib", reason="needs pynasonde[digisonde-scaler]")
    pytest.importorskip("matplotlib")

    (tmp_path / "examples").mkdir()
    (tmp_path / "examples" / "data").symlink_to(DATA_DIR, target_is_directory=True)
    monkeypatch.chdir(tmp_path)

    runpy.run_path(
        str(PROJECT_ROOT / "examples/digisonde/rsf_scaler.py"), run_name="__main__"
    )

    assert (tmp_path / "tmp" / "rsf_scaled_example.csv").stat().st_size > 0
    assert (
        tmp_path / "docs" / "examples" / "figures" / "rsf_scaler_viterbi.png"
    ).stat().st_size > 0


@pytest.mark.skipif(
    not list(DATA_DIR.glob("WS833_*.RSF")), reason="Sample RSF files not available"
)
def test_rsf_direction_ionogram_example_runs(monkeypatch, tmp_path):
    """Run examples/digisonde/rsf_direction_ionogram.py on the bundled WS833 files."""
    pytest.importorskip("joblib", reason="needs pynasonde[digisonde]")
    pytest.importorskip("matplotlib")

    (tmp_path / "examples").mkdir()
    (tmp_path / "examples" / "data").symlink_to(DATA_DIR, target_is_directory=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RSF_EXAMPLE_STN", raising=False)
    monkeypatch.delenv("RSF_EXAMPLE_DIR", raising=False)

    runpy.run_path(
        str(PROJECT_ROOT / "examples/digisonde/rsf_direction_ionogram.py"),
        run_name="__main__",
    )

    figs = tmp_path / "docs" / "examples" / "figures"
    for name in ("rsf_direction_ionogram_WS833.png", "rsf_directogram_WS833_daily.png"):
        assert (figs / name).stat().st_size > 0


@pytest.mark.skipif(
    not (DATA_DIR / "WI937_2022233235902.RIQ").exists(),
    reason="Sample WI937 RIQ file not available",
)
def test_ionogram_filter_wi937_example_runs(monkeypatch, tmp_path):
    """Run examples/vipir/ionogram_filter_wi937.py on the bundled WI937 RIQ file."""
    pytest.importorskip("lxml", reason="example also imports the Digisonde helpers")
    pytest.importorskip("sklearn", reason="needs pynasonde[vipir]")
    pytest.importorskip("matplotlib")

    (tmp_path / "examples").mkdir()
    (tmp_path / "examples" / "data").symlink_to(DATA_DIR, target_is_directory=True)
    (tmp_path / "docs" / "examples" / "figures").mkdir(parents=True)
    monkeypatch.chdir(tmp_path)

    runpy.run_path(
        str(PROJECT_ROOT / "examples/vipir/ionogram_filter_wi937.py"),
        run_name="__main__",
    )

    figs = tmp_path / "docs" / "examples" / "figures"
    for name in ("ionogram_filter_wi937.png", "ionogram_filter_wi937_riq_power.png"):
        assert (figs / name).stat().st_size > 0
