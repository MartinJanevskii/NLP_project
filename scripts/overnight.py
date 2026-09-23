"""Overnight reduced experiment on the Mac: 1,000-item subsample, validation-chosen
epoch budget, reference + 9 configurations x 3 seeds, per-user significance tests.

Commands (run from the repository root):
  prepare    build the subsample, contexts and requests (no API, no training)
  run        everything, resumable: prepare -> generate -> encode -> budget -> train -> analyse
  status     human-readable progress (safe to run any time, from another terminal)
  check-key  one tiny API request to confirm the key works

Overnight: caffeinate -i uv run --no-sync --env-file .env python scripts/overnight.py run
"""

import argparse
import csv
import json
import os
import random
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from urllib.request import Request, urlopen

from experiment import (
    COMMON,
    DATA,
    PROMPTS,
    SEEDS,
    build_kg,
    context,
    digest,
    encode,
    file_hash,
    make_payload,
    read_json,
)
from llm_subset import ENCODER, ENCODER_REVISION, atomic_json, generate, request_id
from run_baseline import REVISION, ROOT, UPSTREAM

N_ITEMS = 1000
SAMPLE_SEED = 2026
VAL_FRACTION = 0.1
BUDGET_MAX, BUDGET_EVERY, BUDGET_PATIENCE, BUDGET_SEED = 600, 10, 10, 2026
E_MIN, E_MAX = 50, 400
CURVE_EVERY = 50
WORKERS = 8
CONFIGS = [h + p for h in ("H1", "H2", "H3") for p in PROMPTS]
METRICS = ("recall@10", "recall@20", "ndcg@10", "ndcg@20")
MODEL = os.environ.get("LLM_MODEL", "deepseek-chat")
ENDPOINT = os.environ.get(
    "LLM_ENDPOINT", "https://api.deepseek.com/v1/chat/completions"
)
SETTINGS = {
    "version": 1,
    "items": N_ITEMS,
    "sample_seed": SAMPLE_SEED,
    "val_fraction": VAL_FRACTION,
    "budget": [BUDGET_MAX, BUDGET_EVERY, BUDGET_PATIENCE, BUDGET_SEED, E_MIN, E_MAX],
    "curve_every": CURVE_EVERY,
    "seeds": SEEDS,
}
# Same weights as experiment_report.contrasts(), applied per user (test_overnight checks this).
CONTRASTS = {
    "RQ1_H2_minus_H1": [("H2" + p, 1 / 3) for p in ("P1", "P2", "P3")]
    + [("H1" + p, -1 / 3) for p in ("P1", "P2", "P3")],
}
for _p in ("P2", "P3"):
    CONTRASTS[f"RQ2_{_p}_minus_P1_over_H1_H2"] = [
        (h + _p, 0.5) for h in ("H1", "H2")
    ] + [(h + "P1", -0.5) for h in ("H1", "H2")]
    CONTRASTS[f"RQ3_depth_effect_{_p}_minus_P1"] = [
        ("H2" + _p, 1),
        ("H1" + _p, -1),
        ("H2P1", -1),
        ("H1P1", 1),
    ]
for _p in ("P1", "P2", "P3"):
    CONTRASTS[f"RQ4_H3_minus_H2_{_p}"] = [("H3" + _p, 1), ("H2" + _p, -1)]


# ---- dataset derivation -----------------------------------------------------------


def read_upstream(path):
    rows = {}
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if parts:
            rows[int(parts[0])] = [int(i) for i in parts[1:]]
    return rows


def write_upstream(path, rows):
    Path(path).write_text(
        "".join(f"{u} {' '.join(map(str, rows[u]))}\n" for u in sorted(rows))
    )


def sample_items(item_ids, n, seed):
    return sorted(random.Random(seed).sample(list(item_ids), n))


def derive(train, test, keep):
    """Restrict the supplied split to kept items and re-index users and items.

    Iterates to a fixed point: every user keeps >= 1 train and >= 1 test item, and
    every item appears in train (the upstream loader sizes the catalogue from the files).
    """
    items = set(keep)
    while True:
        tr = {u: [i for i in v if i in items] for u, v in train.items()}
        te = {u: [i for i in test.get(u, []) if i in items] for u in tr}
        users = sorted(u for u in tr if tr[u] and te[u])
        in_train = {i for u in users for i in tr[u]}
        if in_train == items:
            break
        items = in_train
    item_list, user_map = sorted(items), {u: n for n, u in enumerate(users)}
    item_map = {i: n for n, i in enumerate(item_list)}
    new_train = {user_map[u]: sorted(item_map[i] for i in tr[u]) for u in users}
    new_test = {user_map[u]: sorted(item_map[i] for i in te[u]) for u in users}
    return new_train, new_test, users, item_list


def holdout(train, fraction, seed):
    """Validation split from training data only; users with one item keep it."""
    rng, fit, val = random.Random(seed), {}, {}
    for user in sorted(train):
        items = train[user]
        if len(items) < 2:
            fit[user] = list(items)
            continue
        chosen = set(rng.sample(items, max(1, round(len(items) * fraction))))
        fit[user] = [i for i in items if i not in chosen]
        val[user] = sorted(chosen)
    return fit, val


def choose_epochs(curve, low=E_MIN, high=E_MAX):
    """First evaluated epoch within 1% of the best validation NDCG@20, clamped."""
    best = max(point["ndcg@20"] for point in curve)
    first = next(p["epoch"] for p in curve if p["ndcg@20"] >= 0.99 * best)
    return min(max(first, low), high)


# ---- statistics --------------------------------------------------------------------


def holm(pvalues):
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    adjusted, running = [0.0] * len(pvalues), 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (len(pvalues) - rank) * pvalues[index]))
        adjusted[index] = round(running, 12)
    return adjusted


def verdict(p_holm, delta, seed_deltas):
    if p_holm < 0.05 and delta > 0 and all(d > 0 for d in seed_deltas):
        return "supported"
    if p_holm < 0.05 and delta < 0 and all(d < 0 for d in seed_deltas):
        return "opposite"
    return "inconclusive"


def wilcoxon_p(differences):
    import numpy as np
    from scipy.stats import wilcoxon

    if not np.any(differences):
        return 1.0
    return float(wilcoxon(differences).pvalue)


def bootstrap_ci(differences, resamples=2000, seed=0):
    import numpy as np

    rng = np.random.default_rng(seed)
    means = [
        differences[rng.integers(0, len(differences), len(differences))].mean()
        for _ in range(resamples)
    ]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def analyse(run):
    """Pre-registered per-user tests; writes significance.csv and findings.json."""
    import numpy as np

    per_seed, users = {}, None
    for config in CONFIGS + ["reference"]:
        for seed in SEEDS:
            data = np.load(run / config / f"seed_{seed}_users.npz")
            if users is None:
                users = data["users"]
            if not np.array_equal(users, data["users"]):
                raise ValueError(f"User order differs: {config} seed {seed}")
            per_seed[config, seed] = {m: data[m] for m in METRICS}
    rows = []
    for metric in ("ndcg@20", "recall@20"):
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
        for row, adjusted in zip(found, holm([r["p"] for r in found])):
            row["p_holm"] = adjusted
            row["verdict"] = (
                verdict(adjusted, row["mean_difference"], row["seed_differences"])
                if metric == "ndcg@20"
                else "secondary"
            )
        rows += found
    with (run / "significance.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow(
                dict(row, seed_differences=json.dumps(row["seed_differences"]))
            )
    summary = {
        config: {
            m: {
                "mean": statistics.mean(
                    float(per_seed[config, s][m].mean()) for s in SEEDS
                ),
                "std": statistics.stdev(
                    float(per_seed[config, s][m].mean()) for s in SEEDS
                ),
            }
            for m in METRICS
        }
        for config in CONFIGS + ["reference"]
    }
    text = {}
    for row in csv.DictReader((run / "text_analysis.csv").open()):
        text.setdefault(row["configuration"], []).append(row)
    findings = {
        "dataset": read_json(run / "data/dataset.json"),
        "epoch_budget": read_json(run / "epoch_budget.json"),
        "metrics": summary,
        "contrasts": rows,
        "rq5_text": {
            c: {
                k: statistics.mean(float(r[k]) for r in v)
                for k in (
                    "triples",
                    "response_words",
                    "entity_coverage",
                    "unsupported_known_entities",
                    "input_tokens",
                    "output_tokens",
                )
            }
            for c, v in text.items()
        },
        "status": read_json(run / "status.json"),
    }
    atomic_json(run / "findings.json", findings)
    return findings


# ---- progress and status -------------------------------------------------------------


class Progress:
    """progress.jsonl (one event per line) and status.json (heartbeat with ETA)."""

    def __init__(self, run, runs_total, epochs_per_run):
        self.run = run
        self.path = run / "status.json"
        previous = read_json(self.path) if self.path.exists() else {}
        self.status = {
            "stage": "starting",
            "runs_done": 0,
            "runs_total": runs_total,
            "epochs_per_run": epochs_per_run,
            "epochs_done": 0,
            "epoch_seconds": [],
            "current": None,
            "log": None,
            "last_error": None,
            "started": previous.get("started", time.time()),
            "updated": time.time(),
            "eta_hours": None,
        }
        self.status["epoch_seconds"] = previous.get("epoch_seconds", [])[-200:]

    def event(self, stage, **fields):
        fields = {k: v for k, v in fields.items() if v is not None}
        with (self.run / "progress.jsonl").open("a") as file:
            file.write(
                json.dumps({"time": time.time(), "stage": stage, **fields}) + "\n"
            )
        s = self.status
        s.update(stage=stage, updated=time.time())
        for key in (
            "current",
            "log",
            "runs_done",
            "runs_total",
            "epochs_per_run",
            "last_error",
        ):
            if key in fields:
                s[key] = fields[key]
        if "config" in fields:
            s["current"] = f"{fields['config']} seed {fields.get('seed')}"
        if "seconds" in fields:
            s["epoch_seconds"] = (s["epoch_seconds"] + [fields["seconds"]])[-200:]
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
    def stage(self, name, **fields):
        self.event(name, **fields)
        try:
            yield
        except BaseException as error:
            self.event("FAILED", last_error=f"{name}: {type(error).__name__}: {error}")
            raise


def status_text(run):
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
    if age > 900 and s["stage"] not in ("COMPLETE", "FAILED"):
        lines.append(
            "WARNING:  STALE heartbeat (> 15 min). Is the process still running / did the Mac sleep?"
        )
    if s.get("last_error"):
        lines.append(f"Error:    {s['last_error']}")
    if s.get("log") and Path(s["log"]).exists():
        tail = Path(s["log"]).read_text(errors="replace").splitlines()[-5:]
        lines += ["Log tail:"] + ["  " + line for line in tail]
    return "\n".join(lines)


# ---- stages ------------------------------------------------------------------------


def experiment_base():
    protocol = {
        "version": 1,
        "upstream": REVISION,
        "prompts": PROMPTS,
        "common": COMMON,
        "encoder": [ENCODER, ENCODER_REVISION],
        "model": MODEL,
        "endpoint": ENDPOINT,
        "context_seed": 2026,
        "second_hop_per_entity": 10,
        "input_sha256": {
            p.name: file_hash(p)
            for p in sorted(DATA.iterdir())
            if p.suffix in (".txt", ".csv", ".pt")
        },
    }
    base = ROOT / "artifacts/experiments" / digest(protocol)[:16]
    if not (base / "protocol.json").exists() or read_json(
        base / "protocol.json"
    ) != json.loads(json.dumps(protocol)):
        raise SystemExit(f"Experiment protocol not found or changed: {base}")
    return base


def run_dir():
    return experiment_base() / f"overnight_{digest(SETTINGS)[:8]}"


def check_vendor():
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=UPSTREAM, text=True
    ).strip()
    if head != REVISION or subprocess.check_output(
        ["git", "diff", "HEAD", "--"], cwd=UPSTREAM
    ):
        raise SystemExit("Use the pinned, unmodified CoLaKG checkout")


def write_if_changed(path, value):
    if path.exists() and read_json(path) != json.loads(json.dumps(value)):
        raise SystemExit(
            f"Definition changed: {path}; bump SETTINGS['version'] instead of overwriting"
        )
    atomic_json(path, value)


def prepare(run):
    import torch

    check_vendor()
    run.mkdir(exist_ok=True)
    mapping = sorted(
        (
            tuple(map(int, line.split()))
            for line in (DATA / "item_map.txt").read_text().splitlines()
        ),
        key=lambda pair: pair[1],
    )
    raw_of = {item: raw for raw, item in mapping}
    keep = sample_items([item for _, item in mapping], N_ITEMS, SAMPLE_SEED)
    train, test, users, items = derive(
        read_upstream(DATA / "train.txt"), read_upstream(DATA / "test.txt"), keep
    )
    fit, val = holdout(train, VAL_FRACTION, SAMPLE_SEED)
    item_emb = torch.load(
        DATA / "movie_embeddings_simcse_kg.pt", map_location="cpu", weights_only=True
    )[items]
    user_emb = torch.load(
        DATA / "movie_embeddings_simcse_kg_user.pt",
        map_location="cpu",
        weights_only=True,
    )[users]
    item_map = {
        str(n): {"raw_movie_id": raw_of[old], "old_item_id": old}
        for n, old in enumerate(items)
    }
    for name, (tr, te) in {"data": (train, test), "data_val": (fit, val)}.items():
        folder = run / name
        folder.mkdir(exist_ok=True)
        write_upstream(folder / "train.txt", tr)
        write_upstream(folder / "test.txt", te)
        write_if_changed(folder / "item_map.json", item_map)
        write_if_changed(
            folder / "user_map.json", {str(n): old for n, old in enumerate(users)}
        )
        for file, tensor in (
            ("item_emb_published.pt", item_emb),
            ("user_emb_published.pt", user_emb),
        ):
            if not (folder / file).exists():
                torch.save(tensor.clone(), folder / file)
    interactions = sum(map(len, train.values())) + sum(map(len, test.values()))
    write_if_changed(
        run / "data/dataset.json",
        {
            "items": len(items),
            "sampled_items": N_ITEMS,
            "users": len(users),
            "train_interactions": sum(map(len, train.values())),
            "test_interactions": sum(map(len, test.values())),
            "density": interactions / (len(users) * len(items)),
            "val_interactions": sum(map(len, val.values())),
            "sample_seed": SAMPLE_SEED,
            "source_sha256": {
                f: file_hash(DATA / f)
                for f in ("train.txt", "test.txt", "item_map.txt")
            },
        },
    )
    rows = [{"raw_movie_id": raw_of[old], "item_id": n} for n, old in enumerate(items)]
    for name in ("reference", "budget"):
        folder = run / name
        folder.mkdir(exist_ok=True)
        if not (folder / "embeddings.pt").exists():
            torch.save(item_emb.clone(), folder / "embeddings.pt")
        write_if_changed(folder / "embedding_rows.json", rows)
    with (DATA / "ml1m_extended_movie.csv").open() as file:
        labels, adjacency = build_kg(list(csv.DictReader(file)))
    base, total, cached = run.parent, 0, 0
    for config in CONFIGS:
        folder = run / config
        folder.mkdir(exist_ok=True)
        contexts = [
            dict(row, **context(row["raw_movie_id"], config[:2], adjacency))
            for row in rows
        ]
        requests = [
            make_payload(MODEL, config[2:], r["raw_movie_id"], f, labels)
            for r, f in zip(rows, contexts)
        ]
        write_if_changed(folder / "contexts.json", contexts)
        write_if_changed(folder / "requests.json", requests)
        total += len(requests)
        cached += sum(
            (base / "responses" / f"{request_id(ENDPOINT, p)}.json").exists()
            for p in requests
        )
    atomic_json(
        run / "scale.json",
        {
            "requests": total,
            "cached": cached,
            "to_generate": total - cached,
            "workers": WORKERS,
        },
    )
    atomic_json(
        run / "prepared.json",
        {"settings": SETTINGS, "settings_sha256": digest(SETTINGS)},
    )
    print(
        f"Prepared {run}\n  {len(items)} items, {len(users)} users, {total} requests ({cached} cached)"
    )
    return rows


def train_run(run, progress, folder, seed, epochs, data, runs_done, extra=()):
    """Stream one experiment_train subprocess into its log and the progress files."""
    log = folder / f"seed_{seed}.log"
    command = [
        sys.executable,
        str(ROOT / "scripts/experiment_train.py"),
        str(folder),
        "--seed",
        str(seed),
        "--epochs",
        str(epochs),
        "--reduced",
        "--data-dir",
        str(data),
        *extra,
    ]
    progress.event(
        "train",
        config=folder.name,
        seed=seed,
        log=str(log),
        runs_done=runs_done,
        epoch=0,
    )
    with (
        log.open("a") as out,
        subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        ) as child,
    ):
        for line in child.stdout:
            out.write(line)
            out.flush()
            if line.startswith("EPOCH "):
                epoch = int(line.split()[1].split("/")[0])
                seconds = float(line.rsplit("seconds=", 1)[1])
                progress.event(
                    "train", config=folder.name, seed=seed, epoch=epoch, seconds=seconds
                )
            elif line.startswith("CURVE "):
                progress.event(
                    "curve", config=folder.name, seed=seed, **json.loads(line[6:])
                )
    if child.returncode:
        raise RuntimeError(f"Training failed ({folder.name}, seed {seed}); see {log}")
    return read_json(folder / f"seed_{seed}.json")


def run_all(run, key):
    rows = prepare(run)
    base = run.parent
    total_runs = (len(CONFIGS) + 1) * len(SEEDS)
    budget_file = run / "epoch_budget.json"
    progress = Progress(
        run,
        total_runs,
        read_json(budget_file)["epochs"] if budget_file.exists() else E_MAX,
    )
    manifest = {
        "purpose": "reduced",
        "status": "RUNNING",
        "items": len(rows),
        "seeds": SEEDS,
        "epochs": None,
        "configurations": {c: {"status": "NOT_RUN"} for c in CONFIGS},
        "analysis": "NOT_RUN",
    }
    if (run / "manifest.json").exists():
        manifest = read_json(run / "manifest.json")
    atomic_json(run / "manifest.json", manifest)
    (base / "responses").mkdir(exist_ok=True)
    (base / "vectors").mkdir(exist_ok=True)
    for config in CONFIGS:
        folder = run / config
        requests = read_json(folder / "requests.json")
        with progress.stage("generate", current=f"{config}: {len(requests)} requests"):
            unique = {request_id(ENDPOINT, p): p for p in requests}
            with ThreadPoolExecutor(max_workers=WORKERS) as pool:
                done = dict(
                    zip(
                        unique,
                        pool.map(
                            lambda p: generate(ENDPOINT, p, key, base / "responses"),
                            unique.values(),
                        ),
                    )
                )
            records = [done[request_id(ENDPOINT, p)] for p in requests]
            atomic_json(
                folder / "responses.json", [dict(r, **x) for r, x in zip(rows, records)]
            )
        with progress.stage("encode", current=config):
            manifest["configurations"][config]["embedding_sha256"] = encode(
                records, rows, folder, base / "vectors"
            )
            manifest["configurations"][config]["status"] = "ENCODED"
            atomic_json(run / "manifest.json", manifest)
    with progress.stage("budget", current="reference on validation split"):
        if not budget_file.exists():
            result = train_run(
                run,
                progress,
                run / "budget",
                BUDGET_SEED,
                BUDGET_MAX,
                run / "data_val",
                0,
                ["--eval-every", str(BUDGET_EVERY), "--patience", str(BUDGET_PATIENCE)],
            )
            atomic_json(
                budget_file,
                {
                    "epochs": choose_epochs(result["curve"]),
                    "rule": "first validation epoch with NDCG@20 >= 0.99 x best, clamped to [50, 400]",
                    "best": max(result["curve"], key=lambda p: p["ndcg@20"]),
                    "curve": result["curve"],
                },
            )
    epochs = read_json(budget_file)["epochs"]
    manifest["epochs"] = epochs
    progress.status["epochs_per_run"] = epochs
    done = 0
    for config in ["reference"] + CONFIGS:
        for seed in SEEDS:
            with progress.stage("train", config=config, seed=seed):
                train_run(
                    run,
                    progress,
                    run / config,
                    seed,
                    epochs,
                    run / "data",
                    done,
                    ["--eval-every", str(CURVE_EVERY)],
                )
            done += 1
            progress.event("run-complete", runs_done=done, config=config, seed=seed)
        if config != "reference":
            manifest["configurations"][config]["status"] = "COMPLETE"
            manifest["configurations"][config]["seeds"] = {
                str(s): "COMPLETE" for s in SEEDS
            }
            atomic_json(run / "manifest.json", manifest)
    with progress.stage("analyse"):
        from experiment_report import report

        report(run)
        manifest.update(status="COMPLETE", analysis="COMPLETE")
        atomic_json(run / "manifest.json", manifest)
        progress.event("COMPLETE", runs_done=total_runs)
        findings = analyse(run)
    print(f"COMPLETE. Findings: {run / 'findings.json'}")
    for row in findings["contrasts"]:
        if row["metric"] == "ndcg@20":
            print(
                f"  {row['comparison']:32s} {row['mean_difference']:+.5f}  p_holm={row['p_holm']:.3g}  {row['verdict']}"
            )


def check_key(key):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Reply with OK."}],
        "max_tokens": 5,
        "temperature": 0,
    }
    request = Request(
        ENDPOINT,
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    with urlopen(request, timeout=60) as response:
        result = json.load(response)
    print("Key OK; served model:", result.get("model"))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("command", choices=("prepare", "run", "status", "check-key"))
    args = parser.parse_args()
    if args.command == "status":
        print(status_text(run_dir()))
        return
    key = os.environ.get("LLM_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")
    if args.command in ("run", "check-key") and not key:
        parser.error(
            "No API key: fill DEEPSEEK_API_KEY in .env and use 'uv run --no-sync --env-file .env'"
        )
    if args.command == "prepare":
        prepare(run_dir())
    elif args.command == "check-key":
        check_key(key)
    else:
        check_vendor()
        run_all(run_dir(), key)


if __name__ == "__main__":
    main()
