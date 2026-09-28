"""Repository locations and the pinned upstream CoLaKG checkout."""

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = ROOT / "vendor/CoLaKG"
DATA = UPSTREAM / "data/ml-1m"
REVISION = "be3aa19b59419d197dd4e20f44db737041821f58"
ARTIFACTS = ROOT / "artifacts"
MANIFEST = ARTIFACTS / "EXPERIMENT_MANIFEST.json"
MODEL_CACHE = ARTIFACTS / "model_cache"


def upstream_revision() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=UPSTREAM, text=True
    ).strip()


def upstream_is_pinned() -> bool:
    """True when the vendored CoLaKG checkout is the pinned, unmodified revision."""
    return upstream_revision() == REVISION and not subprocess.check_output(
        ["git", "diff", "HEAD", "--"], cwd=UPSTREAM
    )
