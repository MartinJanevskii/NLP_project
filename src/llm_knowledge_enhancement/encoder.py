"""SimCSE sentence encoder with a per-text vector cache."""

from functools import lru_cache
from pathlib import Path

from llm_knowledge_enhancement.files import atomic_json, digest, file_hash, read_json
from llm_knowledge_enhancement.paths import MODEL_CACHE

ENCODER = "princeton-nlp/sup-simcse-roberta-large"
ENCODER_REVISION = "96d164d9950b72f4ce179cb1eb3414de0910953f"
DIMENSION = 1024
BATCH_SIZE = 4


@lru_cache(maxsize=1)
def load_encoder():
    """Tokenizer and model at the pinned revision, loaded once per process."""
    from transformers import AutoModel, AutoTokenizer

    options = {"revision": ENCODER_REVISION, "cache_dir": MODEL_CACHE}
    return (
        AutoTokenizer.from_pretrained(ENCODER, **options),
        AutoModel.from_pretrained(ENCODER, **options).eval(),
    )


def embed(texts: list[str]):
    """Pooler outputs for a batch of texts, as in the original CoLaKG pipeline."""
    import torch

    tokenizer, model = load_encoder()
    inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        return model(**inputs, return_dict=True).pooler_output.cpu()


def encode(records: list[dict], rows: list[dict], folder: Path, cache: Path) -> str:
    """Write the configuration's embedding matrix and return its checksum.

    Each text vector is cached on its own, so larger runs reuse smaller ones.
    """
    import torch

    metadata_path = folder / "embedding_meta.json"
    fingerprint = digest(
        [ENCODER, ENCODER_REVISION, rows, [r["text"] for r in records]]
    )
    if metadata_path.exists() and (folder / "embeddings.pt").exists():
        metadata = read_json(metadata_path)
        if (
            metadata.get("fingerprint") == fingerprint
            and metadata.get("sha256") == file_hash(folder / "embeddings.pt")
            and read_json(folder / "embedding_rows.json") == rows
        ):
            return metadata["sha256"]
    paths = [
        cache / (digest([ENCODER, ENCODER_REVISION, record["text"]]) + ".pt")
        for record in records
    ]
    missing = []
    for index, path in enumerate(paths):
        if path.exists():
            vector = torch.load(path, map_location="cpu", weights_only=True)
            if vector.shape != (DIMENSION,) or not torch.isfinite(vector).all():
                raise ValueError(f"Corrupt vector cache: {path}")
        else:
            missing.append(index)
    for start in range(0, len(missing), BATCH_SIZE):
        batch = missing[start : start + BATCH_SIZE]
        vectors = embed([records[i]["text"] for i in batch])
        if (
            vectors.shape != (len(batch), DIMENSION)
            or not torch.isfinite(vectors).all()
        ):
            raise ValueError("Invalid encoder output")
        for index, vector in zip(batch, vectors, strict=True):
            temporary = paths[index].with_suffix(".tmp")
            torch.save(vector.clone(), temporary)
            temporary.replace(paths[index])
    matrix = torch.stack(
        [torch.load(path, map_location="cpu", weights_only=True) for path in paths]
    )
    temporary = folder / "embeddings.tmp"
    torch.save(matrix, temporary)
    temporary.replace(folder / "embeddings.pt")
    atomic_json(folder / "embedding_rows.json", rows)
    checksum = file_hash(folder / "embeddings.pt")
    atomic_json(
        metadata_path,
        {
            "fingerprint": fingerprint,
            "sha256": checksum,
            "encoder": ENCODER,
            "revision": ENCODER_REVISION,
        },
    )
    return checksum
