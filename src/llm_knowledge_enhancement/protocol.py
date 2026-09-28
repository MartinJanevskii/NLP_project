"""The experiment protocol: everything that determines the generated descriptions.

Its hash names the experiment folder, so runs with different prompts, model,
encoder or input data never share a folder.
"""

from pathlib import Path

from llm_knowledge_enhancement.encoder import ENCODER, ENCODER_REVISION
from llm_knowledge_enhancement.files import digest, file_hash
from llm_knowledge_enhancement.kg import CONTEXT_SEED, SECOND_HOP_PER_ENTITY
from llm_knowledge_enhancement.paths import ARTIFACTS, DATA, REVISION
from llm_knowledge_enhancement.prompts import COMMON, PROMPTS


def build_protocol(model: str, endpoint: str) -> dict:
    return {
        "version": 1,
        "upstream": REVISION,
        "prompts": PROMPTS,
        "common": COMMON,
        "encoder": [ENCODER, ENCODER_REVISION],
        "model": model,
        "endpoint": endpoint,
        "context_seed": CONTEXT_SEED,
        "second_hop_per_entity": SECOND_HOP_PER_ENTITY,
        "input_sha256": {
            p.name: file_hash(p)
            for p in sorted(DATA.iterdir())
            if p.suffix in (".txt", ".csv", ".pt")
        },
    }


def experiment_folder(protocol: dict) -> Path:
    return ARTIFACTS / "experiments" / digest(protocol)[:16]
