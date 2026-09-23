"""Offline checks for the overnight reduced run: metrics, data, statistics, progress."""

import math
import sys

import numpy as np
from experiment_train import parse_loss, rank_metrics
from run_baseline import UPSTREAM

assert parse_loss("loss0.388-{'Sample': 1.2}") == 0.388
assert math.isnan(parse_loss("lossnan-{}"))
# Per-user ranking metrics equal upstream's batch metrics.
m = rank_metrics([5, 1, 9], {1, 7}, ks=(2, 3))
assert m["recall@2"] == 0.5
assert math.isclose(m["ndcg@2"], (1 / math.log2(3)) / (1 + 1 / math.log2(3)))
sys.path.insert(0, str(UPSTREAM / "rec_code"))
sys.argv = ["test_overnight"]
import utils

rng = np.random.default_rng(0)
tops = [rng.choice(40, 20, replace=False).tolist() for _ in range(50)]
truths = [
    set(rng.choice(40, int(rng.integers(1, 30)), replace=False).tolist())
    for _ in range(50)
]
r = utils.getLabel([list(t) for t in truths], np.array(tops))
for k in (10, 20):
    ours = [rank_metrics(t, g)[f"ndcg@{k}"] for t, g in zip(tops, truths)]
    assert math.isclose(sum(ours), utils.NDCGatK_r([list(t) for t in truths], r, k))
    ours = [rank_metrics(t, g)[f"recall@{k}"] for t, g in zip(tops, truths)]
    upstream = utils.RecallPrecision_ATk([list(t) for t in truths], r, k)["recall"]
    assert math.isclose(sum(ours), upstream)


# ---- dataset derivation -------------------------------------------------------
import json
import statistics
import tempfile
import time
from pathlib import Path

import overnight as ov
from experiment_report import contrasts

train = {0: [0, 1, 2], 1: [3, 4], 2: [5], 3: [0, 5]}
test = {0: [5], 1: [4], 2: [2], 3: [3]}
new_train, new_test, users, items = ov.derive(train, test, keep=[0, 1, 2, 5])
# User 1 keeps no train items; user 3's only test item (3) is not kept.
assert users == [0, 2] and items == [0, 1, 2, 5]
assert new_train == {0: [0, 1, 2], 1: [3]} and new_test == {0: [3], 1: [2]}
assert all(not set(new_train[u]) & set(new_test[u]) for u in new_train)
# Every item index appears in train, so the upstream loader sees all rows.
assert sorted({i for v in new_train.values() for i in v}) == list(range(len(items)))
assert ov.sample_items(range(100), 10, 2026) == ov.sample_items(range(100), 10, 2026)
assert len(set(ov.sample_items(range(100), 10, 2026))) == 10

big = {u: list(range(u, u + 20)) for u in range(5)} | {9: [7]}
fit, val = ov.holdout(big, 0.1, 2026)
assert all(len(val[u]) == 2 and len(fit[u]) == 18 for u in range(5))
assert 9 not in val and fit[9] == [7]
assert all(not set(fit[u]) & set(val.get(u, [])) for u in fit)
assert (fit, val) == ov.holdout(big, 0.1, 2026)
with tempfile.TemporaryDirectory() as folder:
    ov.write_upstream(Path(folder) / "t.txt", new_train)
    assert ov.read_upstream(Path(folder) / "t.txt") == new_train

# ---- epoch budget ---------------------------------------------------------------
curve = [
    {"epoch": e, "ndcg@20": v}
    for e, v in [(10, 0.1), (20, 0.2), (30, 0.297), (40, 0.3)]
]
assert ov.choose_epochs(curve, low=10, high=400) == 30
assert ov.choose_epochs(curve, low=50, high=400) == 50

# ---- statistics -----------------------------------------------------------------
assert ov.holm([0.01, 0.04, 0.03]) == [0.03, 0.06, 0.06]
assert ov.verdict(0.01, 0.2, [0.1, 0.3, 0.2]) == "supported"
assert ov.verdict(0.01, -0.2, [-0.1, -0.3, -0.2]) == "opposite"
assert ov.verdict(0.01, 0.2, [0.1, -0.3, 0.2]) == "inconclusive"
assert ov.verdict(0.2, 0.2, [0.1, 0.3, 0.2]) == "inconclusive"
assert ov.wilcoxon_p(np.zeros(10)) == 1.0
toy = [
    {"configuration": c, "seed": s, **{m: rng.random() for m in ov.METRICS}}
    for c in ov.CONFIGS
    for s in (42, 123, 2026)
]
theirs = {(r["comparison"], r["metric"]): r["mean_difference"] for r in contrasts(toy)}
for name, terms in ov.CONTRASTS.items():
    ours = statistics.mean(
        sum(
            w
            * next(
                r["ndcg@20"] for r in toy if r["configuration"] == c and r["seed"] == s
            )
            for c, w in terms
        )
        for s in (42, 123, 2026)
    )
    assert math.isclose(ours, theirs[name, "ndcg@20"]), name
assert list(ov.CONTRASTS) == list(
    dict.fromkeys(r["comparison"] for r in contrasts(toy))
)

# ---- progress and status --------------------------------------------------------
with tempfile.TemporaryDirectory() as folder:
    run = Path(folder)
    progress = ov.Progress(run, runs_total=4, epochs_per_run=10)
    progress.event("train", config="H1P1", seed=42, epoch=1, seconds=2.0)
    status = json.loads((run / "status.json").read_text())
    assert status["stage"] == "train" and math.isclose(
        status["eta_hours"], (4 * 10 - 1) * 2.0 / 3600
    )
    assert len((run / "progress.jsonl").read_text().splitlines()) == 1
    try:
        with progress.stage("generate"):
            raise RuntimeError("HTTP Error 500")
    except RuntimeError:
        pass
    status = json.loads((run / "status.json").read_text())
    assert status["stage"] == "FAILED" and "HTTP Error 500" in status["last_error"]
    status["updated"] = time.time() - 3600
    status["stage"] = "train"
    (run / "status.json").write_text(json.dumps(status))
    assert "STALE" in ov.status_text(run)
print("overnight data, statistics and progress checks passed")

# ---- robust generation against a local fake provider -------------------------------
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

replies = []


class Fake(BaseHTTPRequestHandler):
    def do_POST(self):
        self.rfile.read(int(self.headers["Content-Length"]))
        code, finish, text = replies.pop(0)
        body = json.dumps(
            {
                "choices": [{"finish_reason": finish, "message": {"content": text}}],
                "model": "fake",
                "usage": {},
            }
        ).encode()
        self.send_response(code)
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


server = HTTPServer(("127.0.0.1", 0), Fake)
threading.Thread(target=server.serve_forever, daemon=True).start()
endpoint = f"http://127.0.0.1:{server.server_port}/v1/chat/completions"
with tempfile.TemporaryDirectory() as folder:
    cache = Path(folder)
    payload = {"model": "m", "messages": [{"role": "user", "content": "x"}]}
    replies[:] = [(500, "stop", ""), (200, "stop", ""), (200, "stop", "fine")]
    assert ov.generate_robust(endpoint, payload, "k", cache, pause=0)["text"] == "fine"
    payload2 = {"model": "m", "messages": [{"role": "user", "content": "y"}]}
    replies[:] = [(200, "length", "cut off")] * 5
    saved = ov.generate_robust(endpoint, payload2, "k", cache, pause=0)
    assert saved["text"] == "cut off" and saved["truncated"] is True
    replies[:] = []  # cached now: no request may be sent
    assert (
        ov.generate_robust(endpoint, payload2, "k", cache, pause=0)["truncated"] is True
    )
    payload3 = {"model": "m", "messages": [{"role": "user", "content": "z"}]}
    replies[:] = [(500, "stop", "")] * 5
    try:
        ov.generate_robust(endpoint, payload3, "k", cache, pause=0)
        raise AssertionError("persistent failures must raise")
    except RuntimeError as error:
        assert "after 4 attempts" in str(error)
server.shutdown()
print("robust generation checks passed")
