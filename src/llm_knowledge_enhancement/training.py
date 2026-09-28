"""Train the original CoLaKG model with one configuration's semantic vectors.

Three modes:
  research  the original fixed 2,000-epoch schedule over the full catalogue
  --pilot   engineering check: one small batch per epoch on CPU
  --reduced overnight protocol: free epoch budget, MPS when available,
            final per-user evaluation on the test split

Training is resumable: every epoch writes a checkpoint with all RNG states.
"""

import argparse
import json
import math
import random
import re
import sys
import time
from pathlib import Path

from llm_knowledge_enhancement import colakg
from llm_knowledge_enhancement.design import METRICS
from llm_knowledge_enhancement.files import (
    atomic_json,
    digest,
    file_hash,
    read_item_map,
    read_json,
)
from llm_knowledge_enhancement.paths import DATA

RESEARCH_EPOCHS = 2000
FINAL_SCHEDULED_EVALUATION = 1996
PILOT_USERS = 100
EVAL_BATCH = 1024
MASK = -(1 << 10)
RESUME_EXIT_CODE = 75


def parse_loss(text: str) -> float:
    """Upstream BPR_train_original returns 'loss<value>-<timings>'."""
    return float(re.match(r"loss(-?(?:\d+\.?\d*|nan|inf))", text)[1])


def rank_metrics(top: list[int], truth: set[int], ks=(10, 20)) -> dict[str, float]:
    """Recall@k and NDCG@k for one user, matching upstream's batch metrics."""
    hits = [item in truth for item in top]
    result = {}
    for k in ks:
        dcg = sum(1 / math.log2(p + 2) for p, hit in enumerate(hits[:k]) if hit)
        idcg = sum(1 / math.log2(p + 2) for p in range(min(k, len(truth))))
        result[f"recall@{k}"] = sum(hits[:k]) / len(truth)
        result[f"ndcg@{k}"] = dcg / idcg
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("folder", type=Path)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument(
        "--stop-after-epoch",
        type=int,
        help="Exit after this epoch (resume testing)",
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
    research = not args.pilot and not args.reduced
    if args.epochs < 1 or (research and args.epochs != RESEARCH_EPOCHS):
        parser.error("Research mode uses the original 2,000 epochs")
    if (args.eval_every or args.patience) and not args.reduced:
        parser.error("--eval-every and --patience need --reduced")
    if args.patience and not args.eval_every:
        parser.error("--patience needs --eval-every")
    return args


def metrics_at(measured: dict) -> dict[str, float]:
    return {
        f"{name}@{k}": float(measured[name][i])
        for name in ("recall", "ndcg")
        for i, k in enumerate((10, 20))
    }


def raw_to_item(data: Path) -> dict[int, int]:
    if (data / "item_map.json").exists():
        return {
            entry["raw_movie_id"]: int(item)
            for item, entry in read_json(data / "item_map.json").items()
        }
    return dict(read_item_map())


class Trainer:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.folder = args.folder.resolve()
        self.data = args.data_dir.resolve()
        self.item_base = self.data / "item_emb_published.pt"
        self.user_base = self.data / "user_emb_published.pt"
        if not self.item_base.exists():
            self.item_base = DATA / "movie_embeddings_simcse_kg.pt"
            self.user_base = DATA / "movie_embeddings_simcse_kg_user.pt"
        self.result_file = self.folder / f"seed_{args.seed}.json"
        self.checkpoint_file = self.folder / f"seed_{args.seed}.pt"
        self.curve: list[dict] = []

    def run(self) -> None:
        colakg.configure("experiment_train", colakg.training_arguments(self.args.seed))
        import numpy as np
        import torch
        import utils
        import world

        if self.args.pilot:
            world.device = torch.device("cpu")
        elif self.args.reduced:
            world.device = torch.device(
                "mps" if torch.backends.mps.is_available() else "cpu"
            )
        # Upstream's pure-Python sampler, because its RNG state can be checkpointed.
        utils.sample_ext = False
        utils.set_seed(self.args.seed)
        random.seed(self.args.seed)
        self.metadata = self.describe(world, np, torch)
        self.fingerprint = digest(self.metadata)
        if self.cached():
            print("Valid completed seed cached:", self.args.seed)
            return
        self.build(world, np, torch, utils)
        self.train(world, np, torch)

    def describe(self, world, np, torch) -> dict:
        args = self.args
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
                    self.folder / "embeddings.pt",
                    self.folder / "embedding_rows.json",
                    self.data / "train.txt",
                    self.data / "test.txt",
                    self.item_base,
                    self.user_base,
                ]
            },
        }
        if args.reduced:
            metadata["reduced"] = {
                "data_dir": str(self.data),
                "eval_every": args.eval_every,
                "patience": args.patience,
            }
        return metadata

    def cached(self) -> bool:
        if not self.result_file.exists():
            return False
        result = read_json(self.result_file)
        if result.get("fingerprint") != self.fingerprint:
            raise ValueError("Training inputs changed")
        return (
            result.get("status") == "COMPLETE"
            and self.checkpoint_file.exists()
            and result.get("checkpoint_sha256") == file_hash(self.checkpoint_file)
        )

    def build(self, world, np, torch, utils) -> None:
        from dataloader import Loader
        from model import CoLaKG
        from sklearn.metrics.pairwise import cosine_similarity

        rows = read_json(self.folder / "embedding_rows.json")
        mapping = raw_to_item(self.data)
        self.ids = [r["item_id"] for r in rows]
        if len(set(self.ids)) != len(self.ids) or any(
            mapping[r["raw_movie_id"]] != r["item_id"] for r in rows
        ):
            raise ValueError("Invalid embedding row mapping")
        item_embeddings = torch.load(
            self.item_base, map_location="cpu", weights_only=True
        )
        if not self.args.pilot and sorted(self.ids) != list(
            range(len(item_embeddings))
        ):
            raise ValueError(
                f"Research mode requires all {len(item_embeddings):,} item embeddings"
            )
        fresh = torch.load(
            self.folder / "embeddings.pt", map_location="cpu", weights_only=True
        )
        if fresh.shape != (len(rows), 1024) or not torch.isfinite(fresh).all():
            raise ValueError("Invalid semantic embeddings")
        item_embeddings[self.ids] = fresh
        user_embeddings = torch.load(
            self.user_base, map_location="cpu", weights_only=True
        )
        neighbors = torch.tensor(
            np.argsort(-cosine_similarity(item_embeddings.numpy()), axis=1)[:, 1:31]
        ).long()
        self.dataset = Loader(path=str(self.data))
        self.model = CoLaKG(
            world.config, self.dataset, neighbors, item_embeddings, user_embeddings
        ).to(world.device)
        self.bpr = utils.BPRLoss(self.model, world.config)

    def resume(self, world, np, torch) -> tuple[int, dict | None]:
        if not self.checkpoint_file.exists():
            return 0, None
        # Checkpoints are only ever produced locally; RNG and optimizer states need pickle.
        checkpoint = torch.load(
            self.checkpoint_file, map_location=world.device, weights_only=False
        )
        if checkpoint["fingerprint"] != self.fingerprint:
            raise ValueError("Checkpoint fingerprint does not match this run")
        self.model.load_state_dict(checkpoint["model"])
        self.bpr.opt.load_state_dict(checkpoint["optimizer"])
        np.random.set_state(checkpoint["numpy_rng"])
        random.setstate(checkpoint["python_rng"])
        torch.set_rng_state(checkpoint["torch_rng"].cpu())
        if torch.cuda.is_available() and checkpoint["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(
                [state.cpu() for state in checkpoint["cuda_rng"]]
            )
        self.curve = checkpoint.get("curve", [])
        print("Resuming after epoch", checkpoint["epoch"])
        return checkpoint["epoch"], checkpoint["metrics"]

    def masked_top(self, users: list[int], world, torch):
        scores = self.model.getUsersRating(torch.tensor(users, device=world.device))
        if not torch.isfinite(scores).all():
            raise ValueError("Non-finite recommendation scores")
        for index, user in enumerate(users):
            scores[index, self.dataset.allPos[user]] = MASK
        return torch.topk(scores, 20).indices.cpu()

    def evaluate_pilot(self, world, torch) -> dict:
        from Procedure import test_one_batch

        users = self.evaluation_users
        self.model.eval()
        with torch.no_grad():
            ranking = self.masked_top(users, world, torch)
        for user, items in zip(users, ranking.tolist(), strict=True):
            assert not set(items).intersection(self.dataset.allPos[user])
        values = test_one_batch((ranking, [self.dataset.testDict[u] for u in users]))
        return {name: values[name] / len(users) for name in ("recall", "ndcg")}

    def evaluate_users(self, users: list[int], world, np, torch):
        """Per-user metrics over the full catalogue, training positives masked."""
        self.model.eval()
        per_user = {m: [] for m in METRICS}
        rankings = []
        with torch.no_grad():
            for start in range(0, len(users), EVAL_BATCH):
                batch = users[start : start + EVAL_BATCH]
                top = self.masked_top(batch, world, torch)
                rankings.append(top)
                for user, items in zip(batch, top.tolist(), strict=True):
                    truth = set(self.dataset.testDict[user])
                    for name, value in rank_metrics(items, truth).items():
                        per_user[name].append(value)
        return {m: np.array(v) for m, v in per_user.items()}, torch.cat(rankings)

    def stop_early(self) -> bool:
        patience, curve = self.args.patience, self.curve
        if not patience or len(curve) <= patience:
            return False
        best = max(range(len(curve)), key=lambda i: curve[i]["ndcg@20"])
        return len(curve) - 1 - best >= patience

    def reduced_epoch(self, epoch: int, world, np, torch) -> None:
        from Procedure import BPR_train_original

        began = time.time()
        loss = parse_loss(BPR_train_original(self.dataset, self.model, self.bpr, epoch))
        if not np.isfinite(loss):
            raise ValueError("Non-finite loss")
        if self.args.eval_every and (epoch + 1) % self.args.eval_every == 0:
            values, _ = self.evaluate_users(
                list(self.dataset.testDict), world, np, torch
            )
            point = {"epoch": epoch + 1}
            point.update({m: float(v.mean()) for m, v in values.items()})
            self.curve.append(point)
            print("CURVE", json.dumps(point), flush=True)
        seconds = time.time() - began
        print(
            f"EPOCH {epoch + 1}/{self.args.epochs} loss={loss:.5f} seconds={seconds:.2f}",
            flush=True,
        )

    def pilot_epoch(self, world, np, torch) -> dict:
        dataset = self.dataset
        self.model.train()
        selected = np.random.choice(
            self.eligible,
            size=len(self.ids),
            replace=len(self.eligible) < len(self.ids),
        )
        users, positives = dataset.trainUser[selected], dataset.trainItem[selected]
        negatives = []
        for user in users:
            negative = np.random.randint(dataset.m_items)
            while negative in dataset.allPos[user]:
                negative = np.random.randint(dataset.m_items)
            negatives.append(negative)
        loss = self.bpr.stageOne(
            torch.tensor(users), torch.tensor(positives), torch.tensor(negatives)
        )
        if not np.isfinite(loss):
            raise ValueError("Non-finite loss")
        return metrics_at(self.evaluate_pilot(world, torch))

    def save_checkpoint(self, epoch: int, metrics: dict | None, np, torch) -> None:
        checkpoint = {
            "fingerprint": self.fingerprint,
            "model": self.model.state_dict(),
            "optimizer": self.bpr.opt.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "curve": self.curve,
            "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None,
        }
        temporary = self.checkpoint_file.with_suffix(".tmp")
        torch.save(checkpoint, temporary)
        temporary.replace(self.checkpoint_file)

    def final_reduced_metrics(self, world, np, torch) -> dict:
        from Procedure import test_one_batch

        users = list(self.dataset.testDict)
        values, ranking = self.evaluate_users(users, world, np, torch)
        upstream = test_one_batch((ranking, [self.dataset.testDict[u] for u in users]))
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
            self.folder / f"seed_{self.args.seed}_users.npz",
            users=np.array(users),
            **values,
        )
        return {m: float(v.mean()) for m, v in values.items()}

    def train(self, world, np, torch) -> None:
        from Procedure import BPR_train_original, Test

        args, dataset = self.args, self.dataset
        start_epoch, last_metrics = self.resume(world, np, torch)
        if args.pilot:
            self.eligible = np.flatnonzero(np.isin(dataset.trainItem, self.ids))
            self.evaluation_users = np.unique(dataset.trainUser[self.eligible])[
                :PILOT_USERS
            ].tolist()
        else:
            self.evaluation_users = list(dataset.testDict)
        stopped_epoch = None
        for epoch in range(start_epoch, args.epochs):
            if args.reduced:
                if self.stop_early():
                    stopped_epoch = epoch
                    break
                self.reduced_epoch(epoch, world, np, torch)
            elif args.pilot:
                last_metrics = self.pilot_epoch(world, np, torch)
            else:
                if epoch % 5 == 0:
                    last_metrics = metrics_at(Test(dataset, self.model, epoch))
                BPR_train_original(dataset, self.model, self.bpr, epoch)
            self.save_checkpoint(epoch + 1, last_metrics, np, torch)
            if not args.reduced:
                print("Epoch", epoch + 1, "saved", flush=True)
            if args.stop_after_epoch == epoch + 1 and epoch + 1 < args.epochs:
                raise SystemExit(RESUME_EXIT_CODE)
        if args.reduced:
            last_metrics = self.final_reduced_metrics(world, np, torch)
        if last_metrics is None or not all(
            np.isfinite(v) and 0 <= v <= 1 for v in last_metrics.values()
        ):
            raise ValueError("Missing or invalid evaluation metrics")
        self.write_result(last_metrics, stopped_epoch)
        print(json.dumps(last_metrics))

    def write_result(self, metrics: dict, stopped_epoch: int | None) -> None:
        args = self.args
        if args.pilot:
            note = "Engineering only: partial semantic replacements, one subset batch per epoch, full catalog ranking"
        elif args.reduced:
            note = "Reduced protocol: validation-chosen epoch budget, final test evaluation"
        else:
            note = "Original fixed-epoch schedule; last scheduled test evaluation, no test-based selection"
        atomic_json(
            self.result_file,
            {
                "status": "COMPLETE",
                "fingerprint": self.fingerprint,
                "metadata": self.metadata,
                "checkpoint_sha256": file_hash(self.checkpoint_file),
                "metrics": metrics,
                "evaluation_users": len(self.evaluation_users),
                "evaluation_epoch": (stopped_epoch or args.epochs)
                if args.pilot or args.reduced
                else FINAL_SCHEDULED_EVALUATION,
                "curve": self.curve,
                "note": note,
            },
        )


def main() -> None:
    Trainer(parse_args()).run()


if __name__ == "__main__":
    main()
