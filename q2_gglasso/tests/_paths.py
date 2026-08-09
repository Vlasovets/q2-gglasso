"""Location of the shared test fixtures.

Several tests used to reference ``data/...`` as a bare relative path, so they
only passed when pytest happened to be invoked from the repository root and
silently failed (or skipped) otherwise. Resolve it from ``__file__`` instead.
"""

from pathlib import Path

# q2_gglasso/tests/_paths.py -> q2_gglasso/tests -> q2_gglasso -> <repo root>
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"


def fixture(name: str) -> str:
    """Absolute path to a fixture in the repo-root ``data/`` directory."""
    return str(DATA_DIR / name)
