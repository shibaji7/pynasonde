"""List the stations available in the DIDBase (GIRO) database.

Credentials are read from the environment or from a ``.env`` file in the project
root (git-ignored; copy ``.env.example`` to ``.env`` and fill it in):
``DIDBASE_URL``, ``DIDBASE_USER``, ``DIDBASE_PASSWORD`` and ``DIDBASE_ROLE``.
Requires ``pip install "pynasonde[digisonde-db]"`` and the Firebird client library.
"""

import os
from pathlib import Path

from pynasonde.digisonde.didbase import DidBaseConnector


def load_dotenv(path: Path) -> None:
    """Load ``KEY=VALUE`` lines from ``path`` without overriding the environment."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip().removeprefix("export ").strip()
        value = value.strip().strip("\"'")
        if key and value:
            os.environ.setdefault(key, value)


# Look for .env in the current directory, then in the project root.
for candidate in (Path.cwd() / ".env", Path(__file__).resolve().parents[2] / ".env"):
    load_dotenv(candidate)

url = os.environ.get("DIDBASE_URL", "jdbc:firebirdsql://didbase.giro.uml.edu/didb")
user = os.environ.get("DIDBASE_USER", "")
password = os.environ.get("DIDBASE_PASSWORD", "")
role = os.environ.get("DIDBASE_ROLE", "COMMON")

if not (user and password):
    raise SystemExit(
        "Set DIDBASE_USER and DIDBASE_PASSWORD (environment or .env; see .env.example)."
    )

with DidBaseConnector(url, user, password, role) as db:
    # List available stations
    stations = db.stations()  # -> pd.DataFrame  (IDENT, URSI, NAME, ...)
    print(stations.head())
