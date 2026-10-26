"""Test configuration ensuring local package is imported and plots use Agg.

Also loads ``.env`` from the project root (git-ignored; see ``.env.example``) so
tests that need credentials, such as the live DIDBase check, can find them.
"""

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _load_dotenv(path: Path) -> None:
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


_load_dotenv(PROJECT_ROOT / ".env")
