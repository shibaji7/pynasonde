"""Live DriftBase login check.

Credentials come from the environment or from the git-ignored ``.env`` file in the
project root (``tests/conftest.py`` loads it; see ``.env.example``):
``DRIFTBASE_URL``, ``DRIFTBASE_USER``, ``DRIFTBASE_PASSWORD`` and ``DRIFTBASE_ROLE``.

It talks to the real database, so it is opt-in: run ``make test-live`` or set
``RUN_LIVE_TESTS=1``. It is also skipped when credentials are missing or when the
environment cannot reach the database (no ``fdb`` driver, no Firebird client library,
no network). A rejected login or a missing role fails the test.
"""

import os

import pytest

from pynasonde.digisonde.driftbase import DriftBaseConnector

CREDENTIAL_KEYS = (
    "DRIFTBASE_URL",
    "DRIFTBASE_USER",
    "DRIFTBASE_PASSWORD",
    "DRIFTBASE_ROLE",
)

pytestmark = [
    pytest.mark.live,
    pytest.mark.skipif(
        os.environ.get("RUN_LIVE_TESTS") != "1",
        reason="live test: run `make test-live` or set RUN_LIVE_TESTS=1",
    ),
    pytest.mark.skipif(
        not all(os.environ.get(k) for k in CREDENTIAL_KEYS),
        reason="DRIFTBASE_* credentials not set (copy .env.example to .env)",
    ),
]

# ping() reports environment problems with these messages; anything else is a
# real failure (bad login, missing role, no data access).
ENVIRONMENT_PROBLEMS = (
    "fdb package not installed",
    "Firebird native client library not found",
    "Cannot reach",
)


def test_driftbase_login():
    """ping() checks the driver, client library, login, role and station-table access."""
    url, user, password, role = (os.environ[k] for k in CREDENTIAL_KEYS)

    try:
        DriftBaseConnector.ping(url, user, password, role)
    except RuntimeError as exc:
        if any(msg in str(exc) for msg in ENVIRONMENT_PROBLEMS):
            pytest.skip(f"DriftBase not reachable from this environment: {exc}")
        raise
