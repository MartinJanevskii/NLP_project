"""Run with: python scripts/test_baseline.py (no training dependencies)."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from run_baseline import completed_metrics, load_manifest

with TemporaryDirectory() as directory:
    path = Path(directory) / "manifest.json"
    fresh = load_manifest(path)
    assert fresh["baseline"]["status"] == "NOT_RUN"
    assert fresh["validations"]["subset_20"] == "NOT_RUN"
    assert not path.exists()
    fresh["baseline"]["status"] = "INCOMPLETE"
    path.write_text(json.dumps(fresh))
    assert load_manifest(path) == fresh
    path.write_text("not JSON")
    try:
        load_manifest(path)
    except json.JSONDecodeError:
        pass
    else:
        raise AssertionError("A corrupt manifest must not silently reset progress")

log = "TEST RESULTS at EPOCH[1996/2000]: {'recall': array([0.1, 0.2]), 'ndcg': array([0.3, 0.4])}\n"
try:
    completed_metrics(log)
except ValueError:
    pass
else:
    raise AssertionError("An incomplete run must not pass the baseline gate")
log += "EPOCH[2000/2000] loss0.1 - Time: 1 seconds\n"
assert completed_metrics(log) == {
    "recall@10": 0.1,
    "recall@20": 0.2,
    "ndcg@10": 0.3,
    "ndcg@20": 0.4,
}
for invalid in (
    log.replace("0.1, 0.2", "nan, 0.2"),
    log.replace("1996/2000", "1/2000"),
):
    try:
        completed_metrics(invalid)
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid evaluation accepted")
print("Baseline completion checks passed")
