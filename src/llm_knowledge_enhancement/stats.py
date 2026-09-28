"""Pre-registered per-user significance tests for the reduced run."""

import csv
import json
import statistics
from pathlib import Path

from llm_knowledge_enhancement.design import CONFIGS, CONTRASTS, METRICS, SEEDS
from llm_knowledge_enhancement.files import atomic_json, read_json

ALPHA = 0.05
PRIMARY_METRIC = "ndcg@20"
TESTED_METRICS = ("ndcg@20", "recall@20")
TEXT_COLUMNS = (
    "triples",
    "response_words",
    "entity_coverage",
    "unsupported_known_entities",
    "input_tokens",
    "output_tokens",
)


def holm(pvalues: list[float]) -> list[float]:
    """Holm step-down adjusted p-values, in input order."""
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adjusted, running = [0.0] * len(pvalues), 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[index]))
        adjusted[index] = round(running, 12)
    return adjusted


def verdict(p_holm: float, delta: float, seed_deltas: list[float]) -> str:
    """Supported only when significant and every seed agrees on the sign."""
    if p_holm < ALPHA and delta > 0 and all(d > 0 for d in seed_deltas):
        return "supported"
    if p_holm < ALPHA and delta < 0 and all(d < 0 for d in seed_deltas):
        return "opposite"
    return "inconclusive"


def wilcoxon_p(differences) -> float:
    import numpy as np
    from scipy.stats import wilcoxon

    if not np.any(differences):
        return 1.0
    return float(wilcoxon(differences).pvalue)


def bootstrap_ci(differences, resamples: int = 2000, seed: int = 0):
    import numpy as np

    rng = np.random.default_rng(seed)
    means = [
        differences[rng.integers(0, len(differences), len(differences))].mean()
        for _ in range(resamples)
    ]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def load_per_user(run: Path) -> dict:
    import numpy as np

    per_seed, users = {}, None
    for config in [*CONFIGS, "reference"]:
        for seed in SEEDS:
            data = np.load(run / config / f"seed_{seed}_users.npz")
            if users is None:
                users = data["users"]
            if not np.array_equal(users, data["users"]):
                raise ValueError(f"User order differs: {config} seed {seed}")
            per_seed[config, seed] = {m: data[m] for m in METRICS}
    return per_seed


def contrast_tests(per_seed: dict, metric: str) -> list[dict]:
    import numpy as np

    found = []
    for name, terms in CONTRASTS.items():
        by_seed = [sum(w * per_seed[c, s][metric] for c, w in terms) for s in SEEDS]
        d = np.mean(by_seed, axis=0)
        low, high = bootstrap_ci(d)
        found.append(
            {
                "comparison": name,
                "metric": metric,
                "mean_difference": float(d.mean()),
                "ci_low": low,
                "ci_high": high,
                "seed_differences": [float(x.mean()) for x in by_seed],
                "p": wilcoxon_p(d),
                "users": len(d),
            }
        )
    for row, adjusted in zip(found, holm([r["p"] for r in found]), strict=True):
        row["p_holm"] = adjusted
        row["verdict"] = (
            verdict(adjusted, row["mean_difference"], row["seed_differences"])
            if metric == PRIMARY_METRIC
            else "secondary"
        )
    return found


def write_significance(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                dict(row, seed_differences=json.dumps(row["seed_differences"]))
            )


def metric_summary(per_seed: dict) -> dict:
    def means(config, metric):
        return [float(per_seed[config, s][metric].mean()) for s in SEEDS]

    return {
        config: {
            m: {
                "mean": statistics.mean(means(config, m)),
                "std": statistics.stdev(means(config, m)),
            }
            for m in METRICS
        }
        for config in [*CONFIGS, "reference"]
    }


def text_summary(run: Path) -> dict:
    by_config = {}
    with (run / "text_analysis.csv").open() as file:
        for row in csv.DictReader(file):
            by_config.setdefault(row["configuration"], []).append(row)
    return {
        config: {k: statistics.mean(float(r[k]) for r in rows) for k in TEXT_COLUMNS}
        for config, rows in by_config.items()
    }


def analyse(run: Path) -> dict:
    """Write significance.csv and findings.json for a completed reduced run."""
    per_seed = load_per_user(run)
    rows = [row for m in TESTED_METRICS for row in contrast_tests(per_seed, m)]
    write_significance(run / "significance.csv", rows)
    findings = {
        "dataset": read_json(run / "data/dataset.json"),
        "epoch_budget": read_json(run / "epoch_budget.json"),
        "metrics": metric_summary(per_seed),
        "contrasts": rows,
        "rq5_text": text_summary(run),
        "truncated_responses": {
            c: sum(
                bool(r.get("truncated")) for r in read_json(run / c / "responses.json")
            )
            for c in CONFIGS
        },
        "status": read_json(run / "status.json"),
    }
    atomic_json(run / "findings.json", findings)
    return findings
