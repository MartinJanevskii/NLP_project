"""Access to the unmodified upstream CoLaKG recommender code.

The upstream modules read their configuration from sys.argv at import time, so
configure() must run before `import world` (or anything that imports it).
"""

import sys
from collections.abc import Sequence

from llm_knowledge_enhancement.paths import UPSTREAM


def training_arguments(seed: int) -> list[str]:
    """The MovieLens settings of upstream train_movielens.sh."""
    return [
        "--bpr_batch",
        "4096",
        "--decay",
        "0.0001",
        "--lr",
        "0.001",
        "--layer",
        "3",
        "--seed",
        str(seed),
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


def configure(program: str, arguments: Sequence[str] = ()) -> None:
    sys.path.insert(0, str(UPSTREAM / "rec_code"))
    sys.argv = [program, *arguments]
