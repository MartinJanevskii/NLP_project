"""Run the unmodified upstream MovieLens script; never count a pilot as a baseline."""

import argparse
import json
import math
import os
import re
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / "vendor/CoLaKG"
REVISION = "be3aa19b59419d197dd4e20f44db737041821f58"


def load_manifest(path):
    """Start fresh without local run artifacts; never replace an existing manifest."""
    if path.exists():
        return json.loads(path.read_text())
    return {
        "baseline": {"status": "NOT_RUN"},
        "validations": {"subset_20": "NOT_RUN", "subset_100": "NOT_RUN"},
        "experiments": {},
        "analysis": "NOT_RUN",
        "paper": "NOT_RUN",
    }


def completed_metrics(log):
    """Require all training epochs and the final scheduled upstream evaluation."""
    if not re.search(r"^EPOCH\[2000/2000\]", log, re.MULTILINE):
        raise ValueError("Original 2,000-epoch training has not completed")
    match = re.search(r"TEST RESULTS at EPOCH\[1996/2000\]: (.*)", log)
    if not match:
        raise ValueError("Final scheduled evaluation is missing")
    result = {}
    for name in ("recall", "ndcg"):
        values = re.search(r"'" + name + r"': array\(\[([^]]+)\]\)", match[1])
        if values is None:
            raise ValueError(f"Missing {name}")
        numbers = [float(value) for value in values[1].split(",")]
        if len(numbers) != 2 or not all(
            math.isfinite(x) and 0 <= x <= 1 for x in numbers
        ):
            raise ValueError(f"Invalid {name}")
        result.update({f"{name}@{k}": value for k, value in zip((10, 20), numbers)})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        help="Bound a diagnostic pilot; never opens the baseline gate",
    )
    args = parser.parse_args()
    if args.timeout_seconds is not None and args.timeout_seconds <= 0:
        parser.error("timeout must be positive")
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=UPSTREAM, text=True
    ).strip()
    if revision != REVISION or subprocess.check_output(
        ["git", "diff", "HEAD", "--"], cwd=UPSTREAM
    ):
        raise SystemExit("Expected the pinned, unmodified upstream checkout")
    manifest_path = ROOT / "artifacts/EXPERIMENT_MANIFEST.json"
    manifest = load_manifest(manifest_path)
    if manifest["baseline"]["status"] == "COMPLETE":
        saved = json.loads((ROOT / manifest["baseline"]["result"]).read_text())
        if (
            saved.get("status") != "COMPLETE"
            or saved.get("pilot")
            or completed_metrics((ROOT / saved["log"]).read_text())
            != saved.get("metrics")
        ):
            raise SystemExit(
                "Saved baseline completion is inconsistent; inspect its artifacts"
            )
        print("Baseline already complete; see", manifest["baseline"]["result"])
        return
    run = (
        ROOT
        / "artifacts/baseline"
        / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    )
    run.mkdir(parents=True)
    (UPSTREAM / "logs").mkdir(exist_ok=True)
    env = dict(
        os.environ,
        PATH=str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", ""),
        PYTHONUNBUFFERED="1",
    )
    record = {
        "status": "RUNNING",
        "upstream_revision": revision,
        "seed": 2020,
        "python": sys.executable,
        "pilot": args.timeout_seconds is not None,
        "command": ["sh", "train_movielens.sh"],
        "metrics": None,
        "log": str((run / "stdout.log").relative_to(ROOT)),
    }

    def save():
        (run / "result.json").write_text(json.dumps(record, indent=2) + "\n")
        manifest["baseline"] = dict(
            record, result=str((run / "result.json").relative_to(ROOT))
        )
        temporary = manifest_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(manifest, indent=2) + "\n")
        temporary.replace(manifest_path)

    save()
    print("Log:", run / "stdout.log", flush=True)
    with (run / "stdout.log").open("w") as log:
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
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            record["status"] = "INCOMPLETE"
        if record.get("returncode") == 0 and not record["pilot"]:
            try:
                record["metrics"] = completed_metrics((run / "stdout.log").read_text())
                record["status"] = "COMPLETE"
            except ValueError as error:
                record["error"] = str(error)
        save()
    print(json.dumps(record, indent=2))
    if record["status"] != "COMPLETE":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
