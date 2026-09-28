"""Run the unmodified upstream CoLaKG MovieLens training (2,000 epochs, seed 2020).

A run with --timeout-seconds is a diagnostic pilot and never counts as the baseline.
"""

import argparse
import json
import os
import signal
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from llm_knowledge_enhancement.baseline import completed_metrics
from llm_knowledge_enhancement.files import atomic_json, load_manifest
from llm_knowledge_enhancement.paths import (
    ARTIFACTS,
    MANIFEST,
    ROOT,
    UPSTREAM,
    upstream_is_pinned,
    upstream_revision,
)


def verify_saved(manifest: dict) -> None:
    saved = json.loads((ROOT / manifest["baseline"]["result"]).read_text())
    if (
        saved.get("status") != "COMPLETE"
        or saved.get("pilot")
        or completed_metrics((ROOT / saved["log"]).read_text()) != saved.get("metrics")
    ):
        raise SystemExit("Saved baseline result is inconsistent")


def stop(process: subprocess.Popen) -> None:
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        help="Time limit for a diagnostic pilot",
    )
    args = parser.parse_args()
    if args.timeout_seconds is not None and args.timeout_seconds <= 0:
        parser.error("timeout must be positive")
    if not upstream_is_pinned():
        raise SystemExit("vendor/CoLaKG is not the pinned, unmodified checkout")
    manifest = load_manifest(MANIFEST)
    if manifest["baseline"]["status"] == "COMPLETE":
        verify_saved(manifest)
        print("Baseline already complete; see", manifest["baseline"]["result"])
        return
    run = ARTIFACTS / "baseline" / datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    run.mkdir(parents=True)
    (UPSTREAM / "logs").mkdir(exist_ok=True)
    env = dict(
        os.environ,
        PATH=str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", ""),
        PYTHONUNBUFFERED="1",
    )
    log_path = run / "stdout.log"
    record = {
        "status": "RUNNING",
        "upstream_revision": upstream_revision(),
        "seed": 2020,
        "python": sys.executable,
        "pilot": args.timeout_seconds is not None,
        "command": ["sh", "train_movielens.sh"],
        "metrics": None,
        "log": str(log_path.relative_to(ROOT)),
    }

    def save() -> None:
        (run / "result.json").write_text(json.dumps(record, indent=2) + "\n")
        manifest["baseline"] = dict(
            record, result=str((run / "result.json").relative_to(ROOT))
        )
        atomic_json(MANIFEST, manifest)

    save()
    print("Log:", log_path, flush=True)
    with log_path.open("w") as log:
        process = subprocess.Popen(
            record["command"],
            cwd=UPSTREAM / "rec_code",
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            record["returncode"] = process.wait(timeout=args.timeout_seconds)
            record["status"] = "FAILED" if record["returncode"] else "INCOMPLETE"
        except (subprocess.TimeoutExpired, KeyboardInterrupt):
            stop(process)
            record["status"] = "INCOMPLETE"
        if record.get("returncode") == 0 and not record["pilot"]:
            try:
                record["metrics"] = completed_metrics(log_path.read_text())
                record["status"] = "COMPLETE"
            except ValueError as error:
                record["error"] = str(error)
        save()
    print(json.dumps(record, indent=2))
    if record["status"] != "COMPLETE":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
