"""Offline checks for the overnight reduced run: metrics, data, statistics, progress."""

import math
import sys

import numpy as np
from experiment_train import rank_metrics
from run_baseline import UPSTREAM

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
print("overnight checks passed")
