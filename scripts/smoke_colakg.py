"""Check cached MovieLens artifacts and one original-model optimizer step on CPU.

The target-item batch is small; the original graph and candidate catalog stay
intact. This is NOT a baseline or a new H/P experiment, and makes no API calls.
"""

import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path

from run_baseline import ROOT, UPSTREAM, load_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", type=int, choices=(20, 100), default=20)
    parser.add_argument(
        "--semantic-subset",
        type=Path,
        help="Use freshly generated subset embeddings and their row mapping",
    )
    args = parser.parse_args()
    smoke_root = ROOT / "artifacts/smoke"
    if args.semantic_subset:
        smoke_root /= hashlib.sha256(
            str(args.semantic_subset.resolve().parent).encode()
        ).hexdigest()[:16]
    output = smoke_root / str(args.items)
    output.mkdir(parents=True, exist_ok=True)
    data = UPSTREAM / "data/ml-1m"
    if args.items == 100:
        previous = smoke_root / "20/result.json"
        if (
            not previous.exists()
            or json.loads(previous.read_text())["status"] != "PASS"
        ):
            parser.error("Run the 20-item smoke check successfully first")

    # These settings match train_movielens.sh; CPU/tensorboard are diagnostic only.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    sys.path.insert(0, str(UPSTREAM / "rec_code"))
    sys.argv = [
        "smoke_colakg",
        "--bpr_batch",
        "4096",
        "--decay",
        "0.0001",
        "--lr",
        "0.001",
        "--layer",
        "3",
        "--seed",
        "2020",
        "--dataset",
        "ml-1m",
        "--topks",
        "[10,20]",
        "--recdim",
        "64",
        "--use_drop_edge",
        "0",
        "--keepprob",
        "1.0",
        "--neighbor_k",
        "30",
        "--tensorboard",
        "0",
    ]
    import numpy as np
    import torch
    import utils
    import world
    from dataloader import Loader
    from model import CoLaKG
    from Procedure import test_one_batch
    from sklearn.metrics.pairwise import cosine_similarity

    world.device = torch.device("cpu")
    utils.set_seed(2020)
    mapping = dict(
        tuple(map(int, line.split()))
        for line in (data / "item_map.txt").read_text().splitlines()
    )
    assert sorted(mapping.values()) == list(range(3260))
    with (data / "ml1m_extended_movie.csv").open() as file:
        metadata = {int(row["MovieID"]): row for row in csv.DictReader(file)}
    prompts = json.loads((data / "llm_input_item.json").read_text())
    responses = json.loads((data / "llm_response_item.json").read_text())
    item_embeddings = torch.load(
        data / "movie_embeddings_simcse_kg.pt", map_location="cpu", weights_only=True
    )
    user_embeddings = torch.load(
        data / "movie_embeddings_simcse_kg_user.pt",
        map_location="cpu",
        weights_only=True,
    )
    assert item_embeddings.shape == (3260, 1024) and user_embeddings.shape == (
        6040,
        1024,
    )
    assert (
        torch.isfinite(item_embeddings).all() and torch.isfinite(user_embeddings).all()
    )
    selected = sorted(mapping, key=mapping.get)[: args.items]
    subset_hashes = {}
    if args.semantic_subset:
        subset = args.semantic_subset
        validation = json.loads((subset / "validation.json").read_text())
        assert (
            validation["live_generation"] == "COMPLETE"
            and validation["encoding"] == "PASS"
        )
        rows = json.loads((subset / "embedding_rows.json").read_text())
        assert rows == [
            {"item_id": mapping[raw], "raw_movie_id": raw} for raw in selected
        ]
        fresh = torch.load(
            subset / "embeddings.pt", map_location="cpu", weights_only=True
        )
        assert fresh.shape == (args.items, 1024) and torch.isfinite(fresh).all()
        subset_hashes = {
            name: hashlib.sha256((subset / name).read_bytes()).hexdigest()
            for name in ("embeddings.pt", "embedding_rows.json", "responses.json")
        }
        assert subset_hashes["embeddings.pt"] == validation["embedding_sha256"]
        generated = json.loads((subset / "responses.json").read_text())
        assert [
            {"item_id": row["item_id"], "raw_movie_id": row["raw_movie_id"]}
            for row in generated
        ] == rows
        assert (
            hashlib.sha256(
                json.dumps([row["text"] for row in generated]).encode()
            ).hexdigest()
            == validation["response_sha256"]
        )
        responses.update({str(row["raw_movie_id"]): row["text"] for row in generated})
        item_embeddings[[row["item_id"] for row in rows]] = fresh
    examples = []
    for raw_id in selected:
        item_id = mapping[raw_id]
        title = metadata[raw_id]["Title"]
        prompt, response = prompts[str(item_id)], responses[str(raw_id)]
        assert title in prompt, (raw_id, item_id, title)
        assert isinstance(response, str) and response.strip()
        assert "first-degree" in prompt and "second-order" in prompt
        examples.append(
            {
                "raw_movie_id": raw_id,
                "item_id": item_id,
                "title": title,
                "prompt": prompt,
                "response": response,
                "prompt_characters": len(prompt),
                "response_words": len(response.split()),
            }
        )
    (output / "examples.json").write_text(json.dumps(examples, indent=2) + "\n")

    dataset = Loader(path=str(data))
    similarities = cosine_similarity(item_embeddings.numpy())
    neighbors = torch.tensor(np.argsort(-similarities, axis=1)[:, 1:31]).long()
    model = CoLaKG(world.config, dataset, neighbors, item_embeddings, user_embeddings)
    optimizer = utils.BPRLoss(model, world.config)
    users, positives, negatives = [], [], []
    for raw_id in selected:
        item = mapping[raw_id]
        user = int(dataset.trainUser[np.flatnonzero(dataset.trainItem == item)[0]])
        known = set(dataset.allPos[user])
        negative = next(
            candidate for candidate in range(dataset.m_items) if candidate not in known
        )
        users.append(user)
        positives.append(item)
        negatives.append(negative)
    before = model.embedding_item.weight.detach().clone()
    model.train()
    loss = optimizer.stageOne(
        torch.tensor(users), torch.tensor(positives), torch.tensor(negatives)
    )
    assert np.isfinite(loss)
    assert not torch.equal(before, model.embedding_item.weight.detach())
    model.eval()
    evaluation_users = sorted(set(users))
    with torch.no_grad():
        scores = model.getUsersRating(torch.tensor(evaluation_users))
        assert (
            scores.shape == (len(evaluation_users), 3260)
            and torch.isfinite(scores).all()
        )
        for index, user in enumerate(evaluation_users):
            scores[index, dataset.allPos[user]] = -(1 << 10)
        ranking = torch.topk(scores, 20).indices
    for user, items in zip(evaluation_users, ranking.tolist()):
        assert not set(items).intersection(dataset.allPos[user])
    metrics = test_one_batch(
        (ranking, [dataset.testDict[user] for user in evaluation_users])
    )
    assert all(np.isfinite(values).all() for values in metrics.values())
    # Diagnostic values are deliberately not promoted to research metrics.
    checkpoint = output / "model.pt"
    torch.save(model.state_dict(), checkpoint)
    restored = CoLaKG(
        world.config, dataset, neighbors, item_embeddings, user_embeddings
    )
    restored.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=True)
    )
    restored.eval()
    with torch.no_grad():
        assert torch.equal(
            model.getUsersRating(torch.tensor(evaluation_users)),
            restored.getUsersRating(torch.tensor(evaluation_users)),
        )
    record = {
        "status": "PASS",
        "purpose": "engineering_only",
        "items": args.items,
        "catalog_items": 3260,
        "graph_users": 6040,
        "optimizer_steps": 1,
        "semantic_source": str(args.semantic_subset)
        if args.semantic_subset
        else "published_upstream",
        "subset_sha256": subset_hashes,
        "catalog_note": "Fresh subset rows plus published rows for other items; engineering check only"
        if args.semantic_subset
        else "Published upstream embeddings",
        "checks": [
            "cached_prompt_mapping",
            "cached_response_mapping",
            "finite_embedding_shapes",
            "optimizer_updates_weights",
            "masked_ranking",
            "upstream_metrics",
            "checkpoint_roundtrip",
        ],
        "not_validated": [
            "new_KG_extraction",
            "live_LLM_generation",
            "fresh_text_encoding",
            "semantic_provenance_of_released_embedding_rows",
            "H1_H2_H3_P1_P2_P3",
            "full_training",
        ],
        "input_sha256": {
            file.name: hashlib.sha256(file.read_bytes()).hexdigest()
            for file in sorted(data.iterdir())
            if file.suffix in (".txt", ".json", ".pt", ".csv")
        },
        "environment": {
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
        },
    }
    if args.semantic_subset:
        record["not_validated"].remove("live_LLM_generation")
        record["not_validated"].remove("fresh_text_encoding")
    (output / "result.json").write_text(json.dumps(record, indent=2) + "\n")
    manifest_path = ROOT / "artifacts/EXPERIMENT_MANIFEST.json"
    manifest = load_manifest(manifest_path)
    section = "live_semantic_smoke" if args.semantic_subset else "upstream_smoke"
    manifest.setdefault(section, {})[str(args.items)] = {
        "status": "PASS",
        "result": str((output / "result.json").relative_to(ROOT)),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"PASS: {args.items} target items; cached artifacts, one optimizer step, ranking and checkpoint roundtrip"
    )


if __name__ == "__main__":
    main()
