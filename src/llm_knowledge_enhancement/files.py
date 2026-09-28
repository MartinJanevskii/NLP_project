"""JSON, hashing and upstream data-file helpers."""

import hashlib
import json
from pathlib import Path
from typing import Any

from llm_knowledge_enhancement.paths import DATA


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def file_hash(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def read_item_map(path: Path = DATA / "item_map.txt") -> list[tuple[int, int]]:
    """(raw MovieID, item id) pairs ordered by item id."""
    return sorted(
        (tuple(map(int, line.split())) for line in path.read_text().splitlines()),
        key=lambda pair: pair[1],
    )


def load_manifest(path: Path) -> dict:
    """Read the global manifest, or start a fresh one; never replaces an existing file."""
    if path.exists():
        return json.loads(path.read_text())
    return {
        "baseline": {"status": "NOT_RUN"},
        "validations": {"subset_20": "NOT_RUN", "subset_100": "NOT_RUN"},
        "experiments": {},
        "analysis": "NOT_RUN",
        "paper": "NOT_RUN",
    }
