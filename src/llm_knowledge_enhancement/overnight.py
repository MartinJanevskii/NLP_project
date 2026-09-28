"""The reduced overnight study: 1,000 sampled items, validation-chosen epoch budget,
the CoLaKG reference plus nine configurations, three seeds each.

Every stage is resumable: finished work is detected from the files it wrote.
"""

import csv
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from llm_knowledge_enhancement.design import CONFIGS, SEEDS
from llm_knowledge_enhancement.encoder import encode
from llm_knowledge_enhancement.files import (
    atomic_json,
    digest,
    file_hash,
    read_item_map,
    read_json,
)
from llm_knowledge_enhancement.kg import build_kg, context
from llm_knowledge_enhancement.llm import (
    DEFAULT_ENDPOINT,
    DEFAULT_MODEL,
    generate_robust,
    post,
    request_id,
)
from llm_knowledge_enhancement.paths import DATA, ROOT, upstream_is_pinned
from llm_knowledge_enhancement.progress import Progress
from llm_knowledge_enhancement.prompts import make_payload
from llm_knowledge_enhancement.protocol import build_protocol, experiment_folder
from llm_knowledge_enhancement.reduced import (
    EPOCHS_MAX,
    EPOCHS_MIN,
    choose_epochs,
    derive,
    holdout,
    read_upstream,
    sample_items,
    write_upstream,
)

N_ITEMS = 1000
SAMPLE_SEED = 2026
VAL_FRACTION = 0.1
BUDGET_MAX, BUDGET_EVERY, BUDGET_PATIENCE, BUDGET_SEED = 600, 10, 10, 2026
CURVE_EVERY = 50
WORKERS = 8
MODEL = DEFAULT_MODEL
ENDPOINT = DEFAULT_ENDPOINT
SETTINGS = {
    "version": 1,
    "items": N_ITEMS,
    "sample_seed": SAMPLE_SEED,
    "val_fraction": VAL_FRACTION,
    "budget": [
        BUDGET_MAX,
        BUDGET_EVERY,
        BUDGET_PATIENCE,
        BUDGET_SEED,
        EPOCHS_MIN,
        EPOCHS_MAX,
    ],
    "curve_every": CURVE_EVERY,
    "seeds": SEEDS,
}
TRAIN_SCRIPT = ROOT / "scripts/experiment_train.py"


def experiment_base() -> Path:
    """The protocol folder written by `scripts/experiment.py`; it must already exist."""
    protocol = build_protocol(MODEL, ENDPOINT)
    base = experiment_folder(protocol)
    if not (base / "protocol.json").exists() or read_json(
        base / "protocol.json"
    ) != json.loads(json.dumps(protocol)):
        raise SystemExit(f"Experiment protocol not found or changed: {base}")
    return base


def run_dir() -> Path:
    return experiment_base() / f"overnight_{digest(SETTINGS)[:8]}"


def check_vendor() -> None:
    if not upstream_is_pinned():
        raise SystemExit("vendor/CoLaKG is not the pinned, unmodified checkout")


def write_if_changed(path: Path, value) -> None:
    """Write a definition file, refusing to silently replace a different one."""
    if path.exists() and read_json(path) != json.loads(json.dumps(value)):
        raise SystemExit(f"Definition changed: {path}")
    atomic_json(path, value)


def load_embeddings(name: str):
    import torch

    return torch.load(DATA / name, map_location="cpu", weights_only=True)


def prepare_data(run: Path) -> list[dict]:
    """Sample items, derive the train/test and validation splits, return embedding rows."""
    import torch

    mapping = read_item_map()
    raw_of = {item: raw for raw, item in mapping}
    keep = sample_items([item for _, item in mapping], N_ITEMS, SAMPLE_SEED)
    train, test, users, items = derive(
        read_upstream(DATA / "train.txt"), read_upstream(DATA / "test.txt"), keep
    )
    fit, val = holdout(train, VAL_FRACTION, SAMPLE_SEED)
    item_emb = load_embeddings("movie_embeddings_simcse_kg.pt")[items]
    user_emb = load_embeddings("movie_embeddings_simcse_kg_user.pt")[users]
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
    train_count = sum(map(len, train.values()))
    test_count = sum(map(len, test.values()))
    write_if_changed(
        run / "data/dataset.json",
        {
            "items": len(items),
            "sampled_items": N_ITEMS,
            "users": len(users),
            "train_interactions": train_count,
            "test_interactions": test_count,
            "density": (train_count + test_count) / (len(users) * len(items)),
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
    return rows


def prepare_requests(run: Path, rows: list[dict]) -> tuple[int, int]:
    """Write each configuration's contexts and requests; return (total, cached)."""
    with (DATA / "ml1m_extended_movie.csv").open() as file:
        labels, adjacency = build_kg(list(csv.DictReader(file)))
    responses, total, cached = run.parent / "responses", 0, 0
    for config in CONFIGS:
        folder = run / config
        folder.mkdir(exist_ok=True)
        contexts = [
            dict(row, **context(row["raw_movie_id"], config[:2], adjacency))
            for row in rows
        ]
        requests = [
            make_payload(MODEL, config[2:], r["raw_movie_id"], f, labels)
            for r, f in zip(rows, contexts, strict=True)
        ]
        write_if_changed(folder / "contexts.json", contexts)
        write_if_changed(folder / "requests.json", requests)
        total += len(requests)
        cached += sum(
            (responses / f"{request_id(ENDPOINT, p)}.json").exists() for p in requests
        )
    return total, cached


def prepare(run: Path) -> list[dict]:
    """Build the subsample, contexts and requests. No API calls, no training."""
    check_vendor()
    run.mkdir(exist_ok=True)
    rows = prepare_data(run)
    total, cached = prepare_requests(run, rows)
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
    users = read_json(run / "data/dataset.json")["users"]
    print(
        f"Prepared {run}\n  {len(rows)} items, {users} users, {total} requests ({cached} cached)"
    )
    return rows


def train_run(
    progress: Progress,
    folder: Path,
    seed: int,
    epochs: int,
    data: Path,
    runs_done: int,
    extra=(),
) -> dict:
    """Run one training subprocess, streaming its output into the log and progress files."""
    log = folder / f"seed_{seed}.log"
    command = [
        sys.executable,
        str(TRAIN_SCRIPT),
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
        raise RuntimeError(f"Training failed ({folder.name}, seed {seed}): {log}")
    return read_json(folder / f"seed_{seed}.json")


def generate_and_encode(
    run: Path, rows: list[dict], key: str, progress: Progress, manifest: dict
) -> None:
    base = run.parent
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
                            lambda p: generate_robust(
                                ENDPOINT, p, key, base / "responses"
                            ),
                            unique.values(),
                        ),
                        strict=True,
                    )
                )
            records = [done[request_id(ENDPOINT, p)] for p in requests]
            atomic_json(
                folder / "responses.json",
                [dict(r, **x) for r, x in zip(rows, records, strict=True)],
            )
        with progress.stage("encode", current=config):
            state = manifest["configurations"][config]
            state["embedding_sha256"] = encode(records, rows, folder, base / "vectors")
            state["status"] = "ENCODED"
            atomic_json(run / "manifest.json", manifest)


def epoch_budget(run: Path, progress: Progress) -> int:
    """Train the reference on the validation split once and pick the epoch count."""
    budget_file = run / "epoch_budget.json"
    with progress.stage("budget", current="reference on validation split"):
        if not budget_file.exists():
            result = train_run(
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
    return read_json(budget_file)["epochs"]


def train_all(run: Path, progress: Progress, manifest: dict, epochs: int) -> None:
    done = 0
    for config in ["reference", *CONFIGS]:
        for seed in SEEDS:
            with progress.stage("train", config=config, seed=seed):
                train_run(
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
            state = manifest["configurations"][config]
            state["status"] = "COMPLETE"
            state["seeds"] = {str(s): "COMPLETE" for s in SEEDS}
            atomic_json(run / "manifest.json", manifest)


def run_all(run: Path, key: str) -> None:
    """prepare -> generate -> encode -> epoch budget -> train -> analyse."""
    from llm_knowledge_enhancement.report import report
    from llm_knowledge_enhancement.stats import analyse

    rows = prepare(run)
    total_runs = (len(CONFIGS) + 1) * len(SEEDS)
    budget_file = run / "epoch_budget.json"
    progress = Progress(
        run,
        total_runs,
        read_json(budget_file)["epochs"] if budget_file.exists() else EPOCHS_MAX,
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
    (run.parent / "responses").mkdir(exist_ok=True)
    (run.parent / "vectors").mkdir(exist_ok=True)
    generate_and_encode(run, rows, key, progress, manifest)
    epochs = epoch_budget(run, progress)
    manifest["epochs"] = epochs
    progress.status["epochs_per_run"] = epochs
    train_all(run, progress, manifest, epochs)
    with progress.stage("analyse"):
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


def check_key(key: str) -> None:
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": "Reply with OK."}],
        "max_tokens": 5,
        "temperature": 0,
    }
    result = post(ENDPOINT, payload, key, timeout=60)
    print("Key OK; served model:", result.get("model"))
