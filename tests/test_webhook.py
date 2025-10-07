"""Tests for the Webhook downloader utilities."""

import datetime as dt
from pathlib import Path

import pytest

from pynasonde.webhook import Webhook


def _mock_response(status=200, hrefs=None, content=b"data"):
    class MockResponse:
        def __init__(self):
            self.status_code = status
            self.text = "".join(f'<a href="{href}">link</a>' for href in (hrefs or []))
            self.content = content

    return MockResponse()


def test_download_creates_structure(monkeypatch, tmp_path):
    calls = {}

    def fake_get(url):
        calls.setdefault(url, 0)
        calls[url] += 1
        if url.endswith("/raw/"):
            return _mock_response(hrefs=["WI937_2024264000000.RIQ"])
        if url.endswith("/ionogram/"):
            return _mock_response(hrefs=["WI937_2024264000000.ngi"])
        return _mock_response()

    monkeypatch.setattr("pynasonde.webhook.requests.get", fake_get)
    monkeypatch.setattr("pynasonde.webhook.tqdm", lambda x, total=None: x)

    wh = Webhook(dates=[dt.datetime(2024, 9, 20)])
    dest = wh.download(source=str(tmp_path), itr=1, ngi=True, riq=True)

    assert Path(dest).exists()
    assert any(f.suffix.lower() == ".ngi" for f in Path(dest).iterdir())
    assert any(f.suffix.upper() == ".RIQ" for f in Path(dest).iterdir())


def test_dump_files_handles_failure(monkeypatch, tmp_path):
    def fake_get(url):
        return _mock_response(status=404)

    monkeypatch.setattr("pynasonde.webhook.requests.get", fake_get)
    monkeypatch.setattr("pynasonde.webhook.tqdm", lambda x, total=None: x)

    wh = Webhook()
    wh.__dump_files__("http://example.com", str(tmp_path), ".ngi", itr=1)
