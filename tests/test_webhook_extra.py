"""Additional webhook tests to cover remaining branches (lines 74-96 etc.)."""

import datetime as dt
import os
from pathlib import Path

import pytest

from pynasonde.webhook import Webhook


def _mock_resp(status=200, hrefs=None, content=b"data"):
    class R:
        def __init__(self):
            self.status_code = status
            self.text = "".join(f'<a href="{h}">link</a>' for h in (hrefs or []))
            self.content = content

    return R()


# ---------------------------------------------------------------------------
# download() — ngi=False, riq=False branch (lines 29->32, 32->26)
# ---------------------------------------------------------------------------


def test_download_ngi_false_riq_false(monkeypatch, tmp_path):
    """When ngi=False and riq=False, no requests are made."""
    calls = []

    def fake_get(url):
        calls.append(url)
        return _mock_resp()

    monkeypatch.setattr("pynasonde.webhook.requests.get", fake_get)
    monkeypatch.setattr("pynasonde.webhook.tqdm", lambda x, total=None: x)

    wh = Webhook(dates=[dt.datetime(2024, 9, 20)])
    dest = wh.download(source=str(tmp_path), ngi=False, riq=False)
    assert len(calls) == 0


def test_download_riq_only(monkeypatch, tmp_path):
    """ngi=False, riq=True exercises the riq branch (line 32->34)."""

    def fake_get(url):
        return _mock_resp(hrefs=["WI937_2024264000000.RIQ"])

    monkeypatch.setattr("pynasonde.webhook.requests.get", fake_get)
    monkeypatch.setattr("pynasonde.webhook.tqdm", lambda x, total=None: x)

    wh = Webhook(dates=[dt.datetime(2024, 9, 20)])
    dest = wh.download(source=str(tmp_path), ngi=False, riq=True)
    assert Path(dest).exists()


# ---------------------------------------------------------------------------
# __dump_files__ — href doesn't match (line 46->44)
# ---------------------------------------------------------------------------


def test_dump_files_href_not_matching(monkeypatch, tmp_path):
    """hrefs that don't match URSI or extension are silently skipped."""

    def fake_get(url):
        return _mock_resp(hrefs=["OTHER_file.ngi", "WI937_other.txt"])

    monkeypatch.setattr("pynasonde.webhook.requests.get", fake_get)
    monkeypatch.setattr("pynasonde.webhook.tqdm", lambda x, total=None: x)

    wh = Webhook()
    wh.__dump_files__("http://x.com/", str(tmp_path), ".ngi", itr=1)
    assert list(tmp_path.iterdir()) == []


# ---------------------------------------------------------------------------
# __dump_files__ — file already exists (line 50->61)
# ---------------------------------------------------------------------------


def test_dump_files_file_already_exists(monkeypatch, tmp_path):
    """When file exists, download is skipped but itr counter still increments."""
    existing = tmp_path / "WI937_2024264000000.ngi"
    existing.write_bytes(b"existing content")
    get_calls = []

    def fake_get(url):
        get_calls.append(url)
        return _mock_resp(hrefs=["WI937_2024264000000.ngi"])

    monkeypatch.setattr("pynasonde.webhook.requests.get", fake_get)
    monkeypatch.setattr("pynasonde.webhook.tqdm", lambda x, total=None: x)

    wh = Webhook()
    wh.__dump_files__("http://x.com/", str(tmp_path), ".ngi", itr=1)
    # Original file content unchanged (no second download)
    assert existing.read_bytes() == b"existing content"


# ---------------------------------------------------------------------------
# __dump_files__ — inner download fails (line 58)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# __check_all_sub_folders__ — 200 path and 404 path (lines 74-96)
# ---------------------------------------------------------------------------


def test_check_all_sub_folders_200(monkeypatch, tmp_path):
    """Status 200 with matching href → file written (lines 74-91)."""
    call_num = [0]

    def fake_get(url):
        call_num[0] += 1
        if call_num[0] == 1:
            return _mock_resp(status=200, hrefs=["data.ngi"])
        return _mock_resp(status=200, content=b"file_content")

    monkeypatch.setattr("pynasonde.webhook.requests.get", fake_get)
    monkeypatch.setattr("pynasonde.webhook.tqdm", lambda x, total=None: x)

    wh = Webhook()
    wh.__check_all_sub_folders__("http://x.com/", str(tmp_path), ["ngi"])
    assert (tmp_path / "data.ngi").exists()


def test_check_all_sub_folders_404(monkeypatch, tmp_path):
    """Non-200 status → body not parsed (lines 74-78 only)."""

    def fake_get(url):
        return _mock_resp(status=404)

    monkeypatch.setattr("pynasonde.webhook.requests.get", fake_get)
    monkeypatch.setattr("pynasonde.webhook.tqdm", lambda x, total=None: x)

    wh = Webhook()
    wh.__check_all_sub_folders__("http://x.com/", str(tmp_path), ["ngi"])
    assert list(tmp_path.iterdir()) == []
