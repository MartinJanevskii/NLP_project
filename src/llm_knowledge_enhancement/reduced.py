"""Deriving the reduced dataset and choosing its epoch budget."""

import random
from pathlib import Path

EPOCHS_MIN, EPOCHS_MAX = 50, 400
NEAR_BEST = 0.99


def read_upstream(path: Path) -> dict[int, list[int]]:
    """Read an upstream interaction file: one 'user item item ...' line per user."""
    rows = {}
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if parts:
            rows[int(parts[0])] = [int(i) for i in parts[1:]]
    return rows


def write_upstream(path: Path, rows: dict[int, list[int]]) -> None:
    Path(path).write_text(
        "".join(f"{u} {' '.join(map(str, rows[u]))}\n" for u in sorted(rows))
    )


def sample_items(item_ids, n: int, seed: int) -> list[int]:
    return sorted(random.Random(seed).sample(list(item_ids), n))


def derive(train: dict, test: dict, keep):
    """Restrict the split to the kept items and re-index users and items.

    Repeats until every user has at least one train and one test item and every
    item occurs in train, because the upstream loader sizes the catalogue from the files.
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


def holdout(train: dict, fraction: float, seed: int):
    """Validation split taken from training data only; single-item users keep theirs."""
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


def choose_epochs(curve: list[dict], low: int = EPOCHS_MIN, high: int = EPOCHS_MAX):
    """First evaluated epoch within 1% of the best validation NDCG@20, clamped."""
    best = max(point["ndcg@20"] for point in curve)
    first = next(p["epoch"] for p in curve if p["ndcg@20"] >= NEAR_BEST * best)
    return min(max(first, low), high)
