"""KG extraction, context sampling and prompt checks; optionally resume against a real pilot."""

import argparse
import math
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from llm_knowledge_enhancement.files import read_json
from llm_knowledge_enhancement.kg import build_kg, context
from llm_knowledge_enhancement.paths import ROOT
from llm_knowledge_enhancement.prompts import PROMPTS, make_payload
from llm_knowledge_enhancement.report import contrasts, mentions

rows = [
    dict(
        MovieID=str(i),
        Title=f"Movie {i}",
        genres="Drama|Comedy",
        Genres="",
        director="Director",
        actors="Actor|Second|Third|Fourth",
    )
    for i in range(1, 15)
]
labels, graph = build_kg(rows)
h1, h2 = context(1, "H1", graph), context(1, "H2", graph)
assert h1["first_hop"] == h2["first_hop"] and not h1["second_hop"]
assert h2 == context(1, "H2", graph)
h3 = context(1, "H3", graph)
assert h3["first_hop"] == h1["first_hop"]
assert set(h3["second_hop"]) <= set(h2["second_hop"])
assert len({e[2] for e in h3["second_hop"]}) == len(h3["second_hop"])
assert all(
    e[1] in ("directed", "starred in") and e[0] != "entity:unknown"
    for e in h3["second_hop"]
)
assert all(
    sum(e[0] == entity for e in h3["second_hop"]) <= 3
    for _, _, entity in h1["first_hop"]
)
assert len(h1["first_hop"]) == 5 and "entity:Fourth" not in labels
assert all(edge in graph[edge[0]] and edge[2] != "movie:1" for edge in h2["second_hop"])
assert all(
    sum(edge[0] == entity for edge in h2["second_hop"]) <= 10
    for _, _, entity in h1["first_hop"]
)
payloads = [make_payload("deepseek-chat", prompt, 1, h2, labels) for prompt in PROMPTS]
assert len({p["messages"][1]["content"] for p in payloads}) == 1
assert len({p["messages"][0]["content"] for p in payloads}) == 3
assert mentions(
    "An actor and ActorTwo, not an actorish name", {"Actor", "ActorTwo", "Unknown"}
) == {"Actor", "ActorTwo"}
print(
    "KG extraction, deterministic sampling, prompt isolation and mention checks passed"
)

toy = [
    {
        "configuration": h + p,
        "seed": seed,
        **{
            metric: value + offset
            for metric in ("recall@10", "recall@20", "ndcg@10", "ndcg@20")
        },
    }
    for h, value in (("H1", 0.1), ("H2", 0.2), ("H3", 0.15))
    for p, offset in (("P1", 0), ("P2", 0.01), ("P3", 0.02))
    for seed in (42, 123, 2026)
]
for result in contrasts(toy):
    if result["comparison"].startswith("RQ1"):
        assert math.isclose(result["mean_difference"], 0.1)
    if result["comparison"].startswith("RQ3"):
        assert abs(result["mean_difference"]) < 1e-12
    if result["comparison"].startswith("RQ4"):
        assert math.isclose(result["mean_difference"], -0.05)
    assert result["sample_std"] == 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume-from", type=Path)
    args = parser.parse_args()
    if args.resume_from:
        import torch

        reference = args.resume_from.resolve()
        with TemporaryDirectory() as directory:
            folder = Path(directory)
            for name in ("embeddings.pt", "embedding_rows.json"):
                shutil.copy2(reference / name, folder / name)
            command = [
                sys.executable,
                str(ROOT / "scripts/experiment_train.py"),
                str(folder),
                "--seed",
                "42",
                "--epochs",
                "2",
                "--pilot",
            ]
            stopped = subprocess.run(
                command + ["--stop-after-epoch", "1"], capture_output=True, text=True
            )
            assert stopped.returncode == 75, stopped.stdout + stopped.stderr
            subprocess.run(command, check=True, capture_output=True)
            resumed = read_json(folder / "seed_42.json")
            original = read_json(reference / "seed_42.json")
            assert resumed["metrics"] == original["metrics"]
            a = torch.load(
                folder / "seed_42.pt", map_location="cpu", weights_only=False
            )
            b = torch.load(
                reference / "seed_42.pt", map_location="cpu", weights_only=False
            )
            for name in a["model"]:
                torch.testing.assert_close(
                    a["model"][name], b["model"][name], rtol=1e-5, atol=1e-6
                )
            print(
                "Maximum absolute weight difference:",
                max(
                    float((a["model"][k] - b["model"][k]).abs().max())
                    for k in a["model"]
                ),
            )
            print(
                "Resume matches uninterrupted metrics exactly and model weights within floating-point tolerance"
            )
