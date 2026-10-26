"""``import pynasonde`` must stay light, while every 1.x public path still resolves."""

import subprocess
import sys

import pytest

import pynasonde

HEAVY = ["cv2", "sklearn", "skimage", "pytesseract", "timezonefinder", "fdb", "pysolar"]


def test_bare_import_does_not_load_heavy_modules():
    code = (
        "import sys, pynasonde;"
        f"print([m for m in {HEAVY!r} if m in sys.modules])"
    )
    out = subprocess.run(
        [sys.executable, "-W", "ignore", "-c", code], capture_output=True, text=True
    )
    assert out.stdout.strip().splitlines()[-1] == "[]", out.stderr[-500:]


@pytest.mark.parametrize("name", pynasonde.__all__)
def test_all_1x_names_resolve(name):
    try:
        assert hasattr(pynasonde, name)
    except ModuleNotFoundError as exc:  # name needs an optional extra
        pytest.skip(f"{name} needs optional dependency: {exc.name}")


def test_timezone_conversion_backcompat_paths():
    pytest.importorskip("pytz")
    from pynasonde import TimeZoneConversion as top
    from pynasonde.timeutils import TimeZoneConversion as new
    from pynasonde.vipir.ngi.utils import TimeZoneConversion as old

    assert top is new is old


def test_unknown_attribute_raises():
    with pytest.raises(AttributeError):
        pynasonde.DoesNotExist
