"""Live DIDBase login check.

Credentials come from the environment or from the git-ignored ``.env`` file in the
project root (``tests/conftest.py`` loads it; see ``.env.example``):
``DIDBASE_URL``, ``DIDBASE_USER``, ``DIDBASE_PASSWORD`` and ``DIDBASE_ROLE``.

It talks to the real database (about a minute), so it is opt-in: run
``make test-live`` or set ``RUN_LIVE_TESTS=1``. It is also skipped when credentials
are missing or when the environment cannot reach the database (no ``fdb`` driver,
no Firebird client library, no network). A rejected login or a missing role fails.
"""

import os

import pytest

from pynasonde.digisonde.didbase import DidBaseConnector

CREDENTIAL_KEYS = ("DIDBASE_URL", "DIDBASE_USER", "DIDBASE_PASSWORD", "DIDBASE_ROLE")

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        os.environ.get("RUN_LIVE_TESTS") != "1",
        reason="live test: run `make test-live` or set RUN_LIVE_TESTS=1",
    ),
    pytest.mark.skipif(
        not all(os.environ.get(k) for k in CREDENTIAL_KEYS),
        reason="DIDBASE_* credentials not set (copy .env.example to .env)",
    ),
]

# ping() reports environment problems with these messages; anything else is a
# real failure (bad login, missing role, no data access).
ENVIRONMENT_PROBLEMS = (
    "fdb package not installed",
    "Firebird native client library not found",
    "Cannot reach",
)


def test_didbase_login_and_stations():
    url, user, password, role = (os.environ[k] for k in CREDENTIAL_KEYS)

    try:
        DidBaseConnector.ping(url, user, password, role)
    except RuntimeError as exc:
        if any(msg in str(exc) for msg in ENVIRONMENT_PROBLEMS):
            pytest.skip(f"DIDBase not reachable from this environment: {exc}")
        raise

    with DidBaseConnector(url, user, password, role) as db:
        stations = db.stations()
    assert len(stations) > 0
    assert "URSI" in stations.columns
