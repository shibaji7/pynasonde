"""Execute the VIPIR RIQ example script to ensure it runs end-to-end."""

import runpy
from pathlib import Path

import pytest


@pytest.mark.skipif(
    not (
        Path(__file__).resolve().parents[1] / "examples/data/PL407_2024058061501.RIQ"
    ).exists(),
    reason="Sample RIQ file not available",
)
def test_proc_riq_script_execution(monkeypatch):
    pytest.importorskip("matplotlib")

    project_root = Path(__file__).resolve().parents[1]
    script_path = project_root / "examples/vipir/proc_riq.py"
    output = project_root / "docs/examples/figures/ionogram_from_riq.png"

    monkeypatch.chdir(project_root)
    runpy.run_path(str(script_path), run_name="__main__")

    assert output.exists() and output.stat().st_size > 0
    output.unlink()
