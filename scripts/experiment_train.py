"""Train the original CoLaKG model with one configuration's semantic vectors."""

import argparse
import json
import math
import random
import re
import sys
import time
from pathlib import Path

from experiment import DATA, digest, file_hash, read_json
from llm_subset import atomic_json
from run_baseline import UPSTREAM


def rank_metrics(top, truth, ks=(10, 20)):
    """Recall@k and NDCG@k for one user, exactly as upstream's batch metrics."""
    hits = [item in truth for item in top]
    result = {}
    for k in ks:
        dcg = sum(1 / math.log2(p + 2) for p, hit in enumerate(hits[:k]) if hit)
        idcg = sum(1 / math.log2(p + 2) for p in range(min(k, len(truth))))
        result[f"recall@{k}"] = sum(hits[:k]) / len(truth)
        result[f"ndcg@{k}"] = dcg / idcg
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument(
        "--stop-after-epoch",
        type=int,
        help="Interrupt after a saved epoch to test resume",
    )
    parser.add_argument(
        "--data-dir", type=Path, default=DATA, help="Upstream-format dataset folder"
    )
    parser.add_argument(
        "--reduced",
        action="store_true",
        help="Overnight protocol: free epoch count, MPS if available, final per-user test evaluation",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=0,
        help="Reduced: log test-file metrics every K epochs",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=0,
        help="Reduced: stop after N curve points without improvement",
    )
    args = parser.parse_args()
    if args.epochs < 1 or (not args.pilot and not args.reduced and args.epochs != 2000):
        parser.error("Research mode uses the original 2,000 epochs")
    if (args.eval_every or args.patience) and not args.reduced:
        parser.error("--eval-every and --patience need --reduced")
    if args.patience and not args.eval_every:
        parser.error("--patience needs --eval-every")
    folder = args.folder.resolve()
    data = args.data_dir.resolve()
    item_base = data / "item_emb_published.pt"
    user_base = data / "user_emb_published.pt"
    if not item_base.exists():
        item_base = DATA / "movie_embeddings_simcse_kg.pt"
        user_base = DATA / "movie_embeddings_simcse_kg_user.pt"
    sys.path.insert(0, str(UPSTREAM / "rec_code"))
    sys.argv = [
        "experiment_train",
        "--bpr_batch",
        "4096",
        "--decay",
        "0.0001",
        "--lr",
        "0.001",
        "--layer",
        "3",
        "--seed",
        str(args.seed),
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
    from Procedure import BPR_train_original, Test, test_one_batch
    from sklearn.metrics.pairwise import cosine_similarity

    if args.pilot:
        world.device = torch.device("cpu")
    elif args.reduced:
        world.device = torch.device(
            "mps" if torch.backends.mps.is_available() else "cpu"
        )
    # Native Python sampler is upstream's fallback and has serializable RNG state.
    utils.sample_ext = False
    utils.set_seed(args.seed)
    random.seed(args.seed)
    metadata = {
        "seed": args.seed,
        "epochs": args.epochs,
        "pilot": args.pilot,
        "source_sha256": file_hash(__file__),
        "config": world.config,
        "python": sys.version,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "device": str(world.device),
        "inputs": {
            str(p.name): file_hash(p)
            for p in [
                folder / "embeddings.pt",
                folder / "embedding_rows.json",
                data / "train.txt",
                data / "test.txt",
                item_base,
                user_base,
            ]
        },
    }
    if args.reduced:
        metadata["reduced"] = {
            "data_dir": str(data),
            "eval_every": args.eval_every,
            "patience": args.patience,
        }
    fingerprint = digest(metadata)
    result_file = folder / f"seed_{args.seed}.json"
    checkpoint_file = folder / f"seed_{args.seed}.pt"
    if result_file.exists():
        result = read_json(result_file)
        if result.get("fingerprint") != fingerprint:
            raise ValueError("Training inputs changed; use a new run directory")
        if (
            result.get("status") == "COMPLETE"
            and checkpoint_file.exists()
            and result.get("checkpoint_sha256") == file_hash(checkpoint_file)
        ):
            print("Valid completed seed cached:", args.seed)
            return
    rows = read_json(folder / "embedding_rows.json")
    if (data / "item_map.json").exists():
        raw_to_item = {
            entry["raw_movie_id"]: int(item)
            for item, entry in read_json(data / "item_map.json").items()
        }
    else:
        raw_to_item = dict(
            tuple(map(int, line.split()))
            for line in (DATA / "item_map.txt").read_text().splitlines()
        )
    ids = [r["item_id"] for r in rows]
    if len(set(ids)) != len(ids) or any(
        raw_to_item[r["raw_movie_id"]] != r["item_id"] for r in rows
    ):
        raise ValueError("Invalid embedding row mapping")
    item_embeddings = torch.load(item_base, map_location="cpu", weights_only=True)
    if not args.pilot and sorted(ids) != list(range(len(item_embeddings))):
        raise ValueError(
            f"Research mode requires all {len(item_embeddings):,} item embeddings"
        )
    fresh = torch.load(folder / "embeddings.pt", map_location="cpu", weights_only=True)
    if fresh.shape != (len(rows), 1024) or not torch.isfinite(fresh).all():
        raise ValueError("Invalid semantic embeddings")
    item_embeddings[ids] = fresh
    user_embeddings = torch.load(user_base, map_location="cpu", weights_only=True)
    neighbors = torch.tensor(
        np.argsort(-cosine_similarity(item_embeddings.numpy()), axis=1)[:, 1:31]
    ).long()
    dataset = Loader(path=str(data))
    model = CoLaKG(
        world.config, dataset, neighbors, item_embeddings, user_embeddings
    ).to(world.device)
    bpr = utils.BPRLoss(model, world.config)
    start_epoch, last_metrics, curve = 0, None, []
    if checkpoint_file.exists():
        # Only load locally generated checkpoints; RNG/optimizer states require pickle.
        checkpoint = torch.load(
            checkpoint_file, map_location=world.device, weights_only=False
        )
        if checkpoint["fingerprint"] != fingerprint:
            raise ValueError("Checkpoint fingerprint does not match this run")
        model.load_state_dict(checkpoint["model"])
        bpr.opt.load_state_dict(checkpoint["optimizer"])
        np.random.set_state(checkpoint["numpy_rng"])
        random.setstate(checkpoint["python_rng"])
        torch.set_rng_state(checkpoint["torch_rng"].cpu())
        if torch.cuda.is_available() and checkpoint["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(
                [state.cpu() for state in checkpoint["cuda_rng"]]
            )
        start_epoch, last_metrics = checkpoint["epoch"], checkpoint["metrics"]
        curve = checkpoint.get("curve", [])
        print("Resuming after epoch", start_epoch)
    eligible = np.flatnonzero(np.isin(dataset.trainItem, ids)) if args.pilot else None
    evaluation_users = (
        np.unique(dataset.trainUser[eligible])[:100].tolist()
        if args.pilot
        else list(dataset.testDict)
    )

    def evaluate_pilot():
        model.eval()
        with torch.no_grad():
            scores = model.getUsersRating(
                torch.tensor(evaluation_users, device=world.device)
            )
            if not torch.isfinite(scores).all():
                raise ValueError("Non-finite recommendation scores")
            for index, user in enumerate(evaluation_users):
                scores[index, dataset.allPos[user]] = -(1 << 10)
            ranking = torch.topk(scores, 20).indices.cpu()
        for user, items in zip(evaluation_users, ranking.tolist()):
            assert not set(items).intersection(dataset.allPos[user])
        values = test_one_batch(
            (ranking, [dataset.testDict[u] for u in evaluation_users])
        )
        return {
            name: values[name] / len(evaluation_users) for name in ("recall", "ndcg")
        }

    def evaluate_users(users):
        """Per-user metrics over the full catalogue, training positives masked."""
        model.eval()
        per_user = {m: [] for m in ("recall@10", "recall@20", "ndcg@10", "ndcg@20")}
        rankings = []
        with torch.no_grad():
            for start in range(0, len(users), 1024):
                batch = users[start : start + 1024]
                scores = model.getUsersRating(torch.tensor(batch, device=world.device))
                if not torch.isfinite(scores).all():
                    raise ValueError("Non-finite recommendation scores")
                for index, user in enumerate(batch):
                    scores[index, dataset.allPos[user]] = -(1 << 10)
                top = torch.topk(scores, 20).indices.cpu()
                rankings.append(top)
                for user, items in zip(batch, top.tolist()):
                    truth = set(dataset.testDict[user])
                    for name, value in rank_metrics(items, truth).items():
                        per_user[name].append(value)
        return {m: np.array(v) for m, v in per_user.items()}, torch.cat(rankings)

    def stop_early():
        if not args.patience or len(curve) <= args.patience:
            return False
        best = max(range(len(curve)), key=lambda i: curve[i]["ndcg@20"])
        return len(curve) - 1 - best >= args.patience

    stopped_epoch = None
    for epoch in range(start_epoch, args.epochs):
        if args.reduced:
            if stop_early():
                stopped_epoch = epoch
                break
            began = time.time()
            text = BPR_train_original(dataset, model, bpr, epoch)
            loss = float(re.match(r"loss([-0-9.naif]+)", text)[1])
            if not np.isfinite(loss):
                raise ValueError("Non-finite loss")
            if args.eval_every and (epoch + 1) % args.eval_every == 0:
                values, _ = evaluate_users(list(dataset.testDict))
                point = {"epoch": epoch + 1}
                point.update({m: float(v.mean()) for m, v in values.items()})
                curve.append(point)
                print("CURVE", json.dumps(point), flush=True)
            seconds = time.time() - began
            print(
                f"EPOCH {epoch + 1}/{args.epochs} loss={loss:.5f} seconds={seconds:.2f}",
                flush=True,
            )
        elif not args.pilot and epoch % 5 == 0:
            measured = Test(dataset, model, epoch)
            last_metrics = {
                f"{name}@{k}": float(measured[name][i])
                for name in ("recall", "ndcg")
                for i, k in enumerate((10, 20))
            }
        if args.pilot:
            model.train()
            selected = np.random.choice(
                eligible, size=len(ids), replace=len(eligible) < len(ids)
            )
            users, positives = dataset.trainUser[selected], dataset.trainItem[selected]
            negatives = []
            for user in users:
                negative = np.random.randint(dataset.m_items)
                while negative in dataset.allPos[user]:
                    negative = np.random.randint(dataset.m_items)
                negatives.append(negative)
            loss = bpr.stageOne(
                torch.tensor(users), torch.tensor(positives), torch.tensor(negatives)
            )
            if not np.isfinite(loss):
                raise ValueError("Non-finite loss")
            measured = evaluate_pilot()
            last_metrics = {
                f"{name}@{k}": float(measured[name][i])
                for name in ("recall", "ndcg")
                for i, k in enumerate((10, 20))
            }
        else:
            BPR_train_original(dataset, model, bpr, epoch)
        checkpoint = {
            "fingerprint": fingerprint,
            "model": model.state_dict(),
            "optimizer": bpr.opt.state_dict(),
            "epoch": epoch + 1,
            "metrics": last_metrics,
            "curve": curve,
            "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None,
        }
        temporary = checkpoint_file.with_suffix(".tmp")
        torch.save(checkpoint, temporary)
        temporary.replace(checkpoint_file)
        if not args.reduced:
            print("Epoch", epoch + 1, "saved", flush=True)
        if args.stop_after_epoch == epoch + 1 and epoch + 1 < args.epochs:
            raise SystemExit(75)
    if args.reduced:
        users = list(dataset.testDict)
        values, ranking = evaluate_users(users)
        upstream = test_one_batch((ranking, [dataset.testDict[u] for u in users]))
        for name in ("recall", "ndcg"):
            for i, k in enumerate((10, 20)):
                if not math.isclose(
                    values[f"{name}@{k}"].sum(),
                    upstream[name][i],
                    rel_tol=1e-9,
                    abs_tol=1e-9,
                ):
                    raise ValueError(
                        f"Per-user {name}@{k} disagrees with upstream metrics"
                    )
        np.savez(
            folder / f"seed_{args.seed}_users.npz", users=np.array(users), **values
        )
        last_metrics = {m: float(v.mean()) for m, v in values.items()}
    if last_metrics is None or not all(
        np.isfinite(v) and 0 <= v <= 1 for v in last_metrics.values()
    ):
        raise ValueError("Missing or invalid evaluation metrics")
    atomic_json(
        result_file,
        {
            "status": "COMPLETE",
            "fingerprint": fingerprint,
            "metadata": metadata,
            "checkpoint_sha256": file_hash(checkpoint_file),
            "metrics": last_metrics,
            "evaluation_users": len(evaluation_users),
            "evaluation_epoch": (stopped_epoch or args.epochs)
            if args.pilot or args.reduced
            else 1996,
            "curve": curve,
            "note": "Engineering only: partial semantic replacements, one subset batch per epoch, full catalog ranking"
            if args.pilot
            else "Reduced protocol: validation-chosen epoch budget, final test evaluation"
            if args.reduced
            else "Original fixed-epoch schedule; last scheduled test evaluation, no test-based selection",
        },
    )
    print(json.dumps(last_metrics))


if __name__ == "__main__":
    main()
