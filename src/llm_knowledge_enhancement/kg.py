"""Knowledge graph built from the released movie metadata, and the three context strategies."""

import random
from collections import Counter, defaultdict
from itertools import combinations

from llm_knowledge_enhancement.files import digest

Edge = tuple[str, str, str]

CONTEXT_SEED = 2026
SECOND_HOP_PER_ENTITY = 10
H3_PER_ATTRIBUTE = 3
ROLES = {"was directed by": "directed", "stars": "starred in"}


def build_kg(rows: list[dict]) -> tuple[dict[str, str], dict[str, list[Edge]]]:
    """Director, first-three-actor and genre-pair facts with inverse edges.

    Movie nodes are keyed by raw MovieID so movies with the same title stay distinct.
    """
    labels, edges = {}, set()
    for row in rows:
        movie = f"movie:{row['MovieID']}"
        labels[movie] = row["Title"]
        genres = sorted(set((row["genres"] or row["Genres"] or "unknown").split("|")))
        attributes = [("was directed by", "directed", row["director"] or "unknown")]
        attributes += [
            ("stars", "starred in", actor)
            for actor in (row["actors"] or "unknown").split("|")[:3]
        ]
        attributes += (
            [("has genre", "is the genre of", genres[0])]
            if len(genres) == 1
            else [
                ("has genres", "are the genres of", " and ".join(pair))
                for pair in combinations(genres, 2)
            ]
        )
        for relation, inverse, name in attributes:
            entity = f"entity:{name}"
            labels[entity] = name
            edges.add((movie, relation, entity))
            edges.add((entity, inverse, movie))
    adjacency = defaultdict(list)
    for head, relation, tail in sorted(edges):
        adjacency[head].append((head, relation, tail))
    return labels, adjacency


def context(raw_id: int, strategy: str, adjacency: dict[str, list[Edge]]) -> dict:
    """First-hop facts plus, for H2/H3, sampled second-hop facts.

    H3 keeps only role-matched director/actor paths (at most three per attribute,
    one per neighbouring movie), dropping the broad genre and "unknown" hubs.
    It is always a subset of H2.
    """
    movie = f"movie:{raw_id}"
    first = sorted(adjacency[movie])
    second = set()
    if strategy in ("H2", "H3"):
        for _, _, entity in first:
            candidates = [edge for edge in adjacency[entity] if edge[2] != movie]
            rng = random.Random(digest([CONTEXT_SEED, raw_id, entity]))
            second.update(
                rng.sample(candidates, min(SECOND_HOP_PER_ENTITY, len(candidates)))
            )
        if strategy == "H3":
            second = _filter_second_hop(first, second)
    elif strategy != "H1":
        raise ValueError(f"Unsupported context: {strategy}")
    return {"first_hop": first, "second_hop": sorted(second)}


def _filter_second_hop(first: list[Edge], second: set[Edge]) -> set[Edge]:
    allowed = {
        (entity, ROLES[relation])
        for _, relation, entity in first
        if relation in ROLES and entity.casefold() != "entity:unknown"
    }
    kept, seen, counts = [], set(), Counter()
    for edge in sorted(second, key=lambda e: (e[1] != "directed", e)):
        if (
            (edge[0], edge[1]) in allowed
            and counts[edge[0]] < H3_PER_ATTRIBUTE
            and edge[2] not in seen
        ):
            kept.append(edge)
            seen.add(edge[2])
            counts[edge[0]] += 1
    return set(kept)
