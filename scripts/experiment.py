"""MovieLens experiment runner for the pilots and the full protocol.

Stages: prepare (contexts and requests, no API calls), run (generate, encode,
train three seeds per configuration) and report. Defaults to preparing the
20-item engineering pilot, which also writes the experiment's protocol.json.
"""

import argparse
import csv
import json
import re
import subprocess
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

from llm_knowledge_enhancement.baseline import completed_metrics
from llm_knowledge_enhancement.design import CONTEXTS, SEEDS
from llm_knowledge_enhancement.encoder import encode
from llm_knowledge_enhancement.files import (
    atomic_json,
    digest,
    load_manifest,
    read_item_map,
    read_json,
)
from llm_knowledge_enhancement.kg import build_kg, context
from llm_knowledge_enhancement.llm import (
    DEFAULT_ENDPOINT,
    DEFAULT_MODEL,
    api_key,
    generate,
    request_id,
)
from llm_knowledge_enhancement.paths import DATA, MANIFEST, ROOT, upstream_is_pinned
from llm_knowledge_enhancement.prompts import PROMPTS, make_payload
from llm_knowledge_enhancement.protocol import build_protocol, experiment_folder

TRAIN_SCRIPT = ROOT / "scripts/experiment_train.py"
FULL_EPOCHS = 2000


def parse_args() -> tuple[argparse.ArgumentParser, argparse.Namespace]:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--items", choices=("20", "100", "full"), default="20")
    parser.add_argument(
        "--contexts", nargs="+", choices=CONTEXTS, default=list(CONTEXTS)
    )
    parser.add_argument(
        "--stage", choices=("prepare", "run", "report"), default="prepare"
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--pilot-epochs", type=int, default=2)
    parser.add_argument(
        "--run-tag",
        default="",
        help="Suffix for the run directory",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8 or not 1 <= args.pilot_epochs <= 10:
        parser.error("--workers must be 1–8 and --pilot-epochs 1–10")
    if args.run_tag and not re.fullmatch(r"[A-Za-z0-9_-]+", args.run_tag):
        parser.error(
            "Run tags may contain only letters, digits, hyphens and underscores"
        )
    return parser, args


def kg_inspection(labels: dict, adjacency: dict) -> dict:
    return {
        "entities": len(labels),
        "triples": sum(map(len, adjacency.values())),
        "relation_counts": dict(
            Counter(r for edges in adjacency.values() for _, r, _ in edges)
        ),
        "highest_degree_attributes": sorted(
            [
                (labels[e], len(edges))
                for e, edges in adjacency.items()
                if e.startswith("entity:")
            ],
            key=lambda pair: (-pair[1], pair[0]),
        )[:20],
    }


class Experiment:
    def __init__(self, parser: argparse.ArgumentParser, args: argparse.Namespace):
        self.parser, self.args = parser, args
        self.full = args.items == "full"
        mapping = read_item_map()
        if [item for _, item in mapping] != list(range(len(mapping))):
            parser.error("Item mapping is not contiguous")
        with (DATA / "ml1m_extended_movie.csv").open() as file:
            self.labels, self.adjacency = build_kg(list(csv.DictReader(file)))
        self.protocol = build_protocol(args.model, args.endpoint)
        self.base = experiment_folder(self.protocol)
        self.suffix = "_" + args.run_tag if args.run_tag else ""
        name = "full" if self.full else f"pilot{args.items}_e{args.pilot_epochs}"
        self.run_dir = self.base / (name + self.suffix)
        self.rows = [
            {"raw_movie_id": raw, "item_id": item}
            for raw, item in (mapping if self.full else mapping[: int(args.items)])
        ]
        self.configs = [h + p for h in dict.fromkeys(args.contexts) for p in PROMPTS]
        self.manifest_path = self.run_dir / "manifest.json"
        self.cache = self.base / "responses"
        self.vector_cache = self.base / "vectors"

    def new_manifest(self) -> dict:
        return {
            "purpose": "research" if self.full else "engineering_only",
            "status": "NOT_RUN",
            "items": len(self.rows),
            "protocol_sha256": digest(self.protocol),
            "epochs": FULL_EPOCHS if self.full else self.args.pilot_epochs,
            "seeds": SEEDS,
            "configurations": {},
            "analysis": "NOT_RUN",
        }

    def save(self) -> None:
        """Write the run manifest and mirror its status into the global manifest."""
        manifest = self.manifest
        atomic_json(self.manifest_path, manifest)
        overall = load_manifest(MANIFEST)
        overall.setdefault("pipeline_runs", {})[str(self.run_dir.relative_to(ROOT))] = {
            "status": manifest["status"],
            "purpose": manifest["purpose"],
            "manifest": str(self.manifest_path.relative_to(ROOT)),
        }
        if self.full:
            overall["experiments"].update(manifest["configurations"])
            overall["analysis"] = manifest["analysis"]
            overall["paper"] = (
                "COMPLETE" if (self.run_dir / "paper_draft.md").exists() else "NOT_RUN"
            )
        elif manifest["status"] == "COMPLETE" and len(manifest["configurations"]) == 9:
            overall["validations"]["subset_" + self.args.items] = "COMPLETE"
        atomic_json(MANIFEST, overall)

    def check_prerequisites(self) -> None:
        """Full runs need the baseline; each size needs the smaller one validated first."""
        if self.full:
            baseline = load_manifest(MANIFEST)["baseline"]
            if (
                baseline["status"] != "COMPLETE"
                or completed_metrics((ROOT / baseline["log"]).read_text())
                != baseline["metrics"]
            ):
                self.parser.error(
                    "Full research runs require the completed original baseline"
                )
        if not (self.full or self.args.items == "100"):
            return
        size = "100" if self.full else "20"
        previous = (
            self.base
            / (f"pilot{size}_e{self.args.pilot_epochs}" + self.suffix)
            / "manifest.json"
        )
        configs = read_json(previous)["configurations"] if previous.exists() else {}
        if any(configs.get(c, {}).get("status") != "COMPLETE" for c in self.configs):
            self.parser.error(f"The matching {size}-item validation is not complete")

    def prepare(self) -> None:
        self.requests = {}
        for config in self.configs:
            folder = self.run_dir / config
            folder.mkdir(exist_ok=True)
            contexts = [
                dict(row, **context(row["raw_movie_id"], config[:2], self.adjacency))
                for row in self.rows
            ]
            self.requests[config] = [
                make_payload(
                    self.args.model, config[2:], row["raw_movie_id"], facts, self.labels
                )
                for row, facts in zip(self.rows, contexts, strict=True)
            ]
            for name, value in (
                ("contexts.json", contexts),
                ("requests.json", self.requests[config]),
            ):
                path = folder / name
                if path.exists() and read_json(path) != json.loads(json.dumps(value)):
                    raise ValueError(f"Definition changed: {path}")
                atomic_json(path, value)
            self.manifest["configurations"].setdefault(
                config,
                {
                    "status": "NOT_RUN",
                    "contexts": "COMPLETE",
                    "responses": "NOT_RUN",
                    "embeddings": "NOT_RUN",
                    "seeds": {str(seed): "NOT_RUN" for seed in SEEDS},
                },
            )

    def write_scale(self) -> None:
        payloads = [p for ps in self.requests.values() for p in ps]
        chars = sum(len(m["content"]) for p in payloads for m in p["messages"])
        scale = {
            "generations": len(payloads),
            "cached_files": sum(
                (self.cache / f"{request_id(self.args.endpoint, p)}.json").exists()
                for p in payloads
            ),
            "model": self.args.model,
            "prompt_characters": chars,
            "rough_input_tokens": (chars + 3) // 4,
            "max_output_tokens_per_request": 512,
            "training_runs": len(self.configs) * 3,
        }
        atomic_json(self.run_dir / "scale.json", scale)
        print("Run:", self.run_dir, "\n", json.dumps(scale, indent=2), flush=True)

    def run_config(self, config: str, key: str) -> None:
        endpoint = self.args.endpoint
        folder = self.run_dir / config
        state = self.manifest["configurations"][config]
        state["status"] = "RUNNING"
        self.save()
        unique = {request_id(endpoint, p): p for p in self.requests[config]}
        with ThreadPoolExecutor(max_workers=self.args.workers) as pool:
            generated = list(
                pool.map(
                    lambda p: generate(endpoint, p, key, self.cache), unique.values()
                )
            )
        by_id = dict(zip(unique, generated, strict=True))
        records = [by_id[request_id(endpoint, p)] for p in self.requests[config]]
        atomic_json(
            folder / "responses.json",
            [
                dict(row, **record)
                for row, record in zip(self.rows, records, strict=True)
            ],
        )
        state["responses"] = "COMPLETE"
        self.save()
        state["embedding_sha256"] = encode(
            records, self.rows, folder, self.vector_cache
        )
        state["embeddings"] = "COMPLETE"
        self.save()
        for seed in SEEDS:
            command = [
                sys.executable,
                str(TRAIN_SCRIPT),
                str(folder),
                "--seed",
                str(seed),
                "--epochs",
                str(self.manifest["epochs"]),
            ]
            if not self.full:
                command.append("--pilot")
            with (folder / f"seed_{seed}.log").open("a") as log:
                subprocess.run(
                    command, stdout=log, stderr=subprocess.STDOUT, check=True
                )
            state["seeds"][str(seed)] = "COMPLETE"
            self.save()
        state["status"] = "COMPLETE"
        self.save()
        print(config, "COMPLETE", flush=True)

    def h1_h2_complete(self) -> bool:
        return all(
            self.manifest["configurations"].get(h + p, {}).get("status") == "COMPLETE"
            for h in ("H1", "H2")
            for p in PROMPTS
        )

    def run_all(self) -> None:
        key = api_key()
        if not key:
            self.parser.error("LLM_API_KEY or DEEPSEEK_API_KEY is not set")
        self.manifest["status"] = "RUNNING"
        self.save()
        try:
            for config in self.configs:
                if config.startswith("H3") and not self.h1_h2_complete():
                    raise ValueError(
                        "H3 runs require the six H1/H2 configurations to complete first"
                    )
                self.run_config(config, key)
        except (Exception, KeyboardInterrupt):
            self.manifest["configurations"][config]["status"] = "INCOMPLETE"
            self.manifest["status"] = "INCOMPLETE"
            self.save()
            raise

    def report(self) -> None:
        from llm_knowledge_enhancement.report import report

        report(self.run_dir)
        self.manifest["analysis"] = "COMPLETE"
        self.manifest["status"] = (
            "COMPLETE"
            if all(
                s["status"] == "COMPLETE"
                for s in self.manifest["configurations"].values()
            )
            else "INCOMPLETE"
        )
        self.save()

    def main(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(self.base / "protocol.json", self.protocol)
        atomic_json(
            self.base / "kg_inspection.json", kg_inspection(self.labels, self.adjacency)
        )
        atomic_json(self.base / "entity_labels.json", self.labels)
        self.manifest = (
            read_json(self.manifest_path)
            if self.manifest_path.exists()
            else self.new_manifest()
        )
        if self.args.stage == "run":
            self.check_prerequisites()
        self.prepare()
        self.cache.mkdir(exist_ok=True)
        self.vector_cache.mkdir(exist_ok=True)
        self.write_scale()
        self.save()
        if self.args.stage == "prepare":
            return
        if self.args.stage == "run":
            self.run_all()
        self.report()


def main() -> None:
    parser, args = parse_args()
    if not upstream_is_pinned():
        parser.error("vendor/CoLaKG is not the pinned, unmodified checkout")
    Experiment(parser, args).main()


if __name__ == "__main__":
    main()
