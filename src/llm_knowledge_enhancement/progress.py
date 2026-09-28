"""Progress log and heartbeat for long runs, readable from another terminal."""

import json
import statistics
import time
from contextlib import contextmanager
from pathlib import Path

from llm_knowledge_enhancement.files import atomic_json, read_json

KEPT_EPOCH_TIMES = 200
STALE_SECONDS = 900
TRACKED = ("current", "log", "runs_done", "runs_total", "epochs_per_run", "last_error")


class Progress:
    """Appends events to progress.jsonl and keeps status.json current, with an ETA."""

    def __init__(self, run: Path, runs_total: int, epochs_per_run: int):
        self.run = run
        self.path = run / "status.json"
        previous = read_json(self.path) if self.path.exists() else {}
        self.status = {
            "stage": "starting",
            "runs_done": 0,
            "runs_total": runs_total,
            "epochs_per_run": epochs_per_run,
            "epochs_done": 0,
            "epoch_seconds": previous.get("epoch_seconds", [])[-KEPT_EPOCH_TIMES:],
            "current": None,
            "log": None,
            "last_error": None,
            "started": previous.get("started", time.time()),
            "updated": time.time(),
            "eta_hours": None,
        }

    def event(self, stage: str, **fields) -> None:
        fields = {k: v for k, v in fields.items() if v is not None}
        with (self.run / "progress.jsonl").open("a") as file:
            file.write(
                json.dumps({"time": time.time(), "stage": stage, **fields}) + "\n"
            )
        s = self.status
        s.update(stage=stage, updated=time.time())
        s.update({key: fields[key] for key in TRACKED if key in fields})
        if "config" in fields:
            s["current"] = f"{fields['config']} seed {fields.get('seed')}"
        if "seconds" in fields:
            s["epoch_seconds"] = (s["epoch_seconds"] + [fields["seconds"]])[
                -KEPT_EPOCH_TIMES:
            ]
            s["epochs_done"] = fields.get("epoch", s["epochs_done"])
        if s["epoch_seconds"]:
            remaining = (s["runs_total"] - s["runs_done"]) * s["epochs_per_run"] - s[
                "epochs_done"
            ]
            s["eta_hours"] = (
                max(0, remaining) * statistics.mean(s["epoch_seconds"]) / 3600
            )
        atomic_json(self.path, s)

    @contextmanager
    def stage(self, name: str, **fields):
        self.event(name, **fields)
        try:
            yield
        except BaseException as error:
            self.event("FAILED", last_error=f"{name}: {type(error).__name__}: {error}")
            raise


def status_text(run: Path) -> str:
    if not (run / "status.json").exists():
        return f"No status yet in {run}"
    s = read_json(run / "status.json")
    age = time.time() - s["updated"]
    lines = [
        f"Run:      {run}",
        f"Stage:    {s['stage']}" + (f"  ({s['current']})" if s.get("current") else ""),
        f"Training: {s['runs_done']}/{s['runs_total']} runs, epoch {s['epochs_done']}/{s['epochs_per_run']} of current",
        f"ETA:      {s['eta_hours']:.1f} h (training only)"
        if s.get("eta_hours") is not None
        else "ETA:      not yet measured",
        f"Elapsed:  {(time.time() - s['started']) / 3600:.1f} h; last update {age / 60:.0f} min ago",
    ]
    if age > STALE_SECONDS and s["stage"] not in ("COMPLETE", "FAILED"):
        lines.append(
            "WARNING:  STALE heartbeat (> 15 min). Is the process still running / did the Mac sleep?"
        )
    if s.get("last_error"):
        lines.append(f"Error:    {s['last_error']}")
    if s.get("log") and Path(s["log"]).exists():
        tail = Path(s["log"]).read_text(errors="replace").splitlines()[-5:]
        lines += ["Log tail:"] + ["  " + line for line in tail]
    return "\n".join(lines)
