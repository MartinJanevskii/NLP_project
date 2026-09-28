"""Validate the original CoLaKG item preprocessing on 20, then 100 MovieLens items.

By default only the requests are written, without an API call.
  --generate            call the configured provider
  --encode              encode the cached responses
  --upstream-responses  use the published responses to check the encoder without a key
                        (never counts as validating live generation)
"""

import argparse
import hashlib
import json

from llm_knowledge_enhancement.encoder import (
    BATCH_SIZE,
    DIMENSION,
    ENCODER,
    ENCODER_REVISION,
    embed,
    load_encoder,
)
from llm_knowledge_enhancement.files import (
    atomic_json,
    file_hash,
    load_manifest,
    read_item_map,
)
from llm_knowledge_enhancement.llm import (
    DEFAULT_ENDPOINT,
    DEFAULT_MODEL,
    api_key,
    generate,
    request_id,
)
from llm_knowledge_enhancement.paths import ARTIFACTS, DATA, MANIFEST, ROOT
from llm_knowledge_enhancement.prompts import upstream_payload

USAGE_FIELDS = (
    "prompt_tokens",
    "completion_tokens",
    "prompt_cache_hit_tokens",
    "prompt_cache_miss_tokens",
)


def parse_args() -> tuple[argparse.ArgumentParser, argparse.Namespace]:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--items", type=int, choices=(20, 100), default=20)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--encode", action="store_true")
    parser.add_argument("--upstream-responses", action="store_true")
    args = parser.parse_args()
    if args.generate and args.upstream_responses:
        parser.error("--generate and --upstream-responses are mutually exclusive")
    return parser, args


def build_requests(args: argparse.Namespace) -> list[dict]:
    prompts = json.loads((DATA / "llm_input_item.json").read_text())
    return [
        {
            "raw_movie_id": raw,
            "item_id": item,
            "payload": upstream_payload(args.model, prompts[str(item)]),
        }
        for raw, item in read_item_map()[: args.items]
    ]


def collect_responses(parser, args, requests, cache) -> tuple[list[str], list[dict]]:
    if args.upstream_responses:
        published = json.loads((DATA / "llm_response_item.json").read_text())
        return [published[str(r["raw_movie_id"])] for r in requests], []
    if not (args.generate or args.encode):
        return [], []
    key = api_key()
    if args.generate and not key:
        parser.error("LLM_API_KEY or DEEPSEEK_API_KEY is not set")
    responses, records = [], []
    for request in requests:
        payload = request["payload"]
        if args.generate:
            saved = generate(args.endpoint, payload, key, cache)
        else:
            identity = request_id(args.endpoint, payload)
            saved = json.loads((cache / f"{identity}.json").read_text())
            if saved.get("request_id") != identity:
                raise ValueError("Response cache fingerprint mismatch")
        responses.append(saved["text"])
        records.append(saved)
        print(f"Response ready: {len(responses)}/{args.items}", flush=True)
    return responses, records


def usage_summary(model: str, records: list[dict]) -> dict:
    return {
        "requested_model": model,
        "served_models": sorted({record["model"] for record in records}),
        "cached_response_count": len(records),
        "tokens_for_saved_responses": {
            name: sum((record.get("usage") or {}).get(name, 0) for record in records)
            for name in USAGE_FIELDS
        },
        "note": "Usage for retained responses, not new charges on each cached rerun; failed requests are not included.",
    }


def encode_responses(args, responses, requests, validation, output, response_hash):
    import torch

    batches = [
        embed(responses[start : start + BATCH_SIZE])
        for start in range(0, len(responses), BATCH_SIZE)
    ]
    embeddings = torch.cat(batches)
    assert (
        embeddings.shape == (args.items, DIMENSION) and torch.isfinite(embeddings).all()
    )
    torch.save(embeddings, output / "embeddings.pt")
    atomic_json(output / "embedding_rows.json", expected_rows(requests))
    _, model = load_encoder()
    validation.update(
        encoding="PASS",
        encoder=ENCODER,
        encoder_revision=getattr(model.config, "_commit_hash", None),
        response_sha256=response_hash,
        embedding_sha256=file_hash(output / "embeddings.pt"),
    )
    if args.upstream_responses:
        original = torch.load(
            DATA / "movie_embeddings_simcse_kg.pt",
            map_location="cpu",
            weights_only=True,
        )
        similarity = torch.nn.functional.cosine_similarity(
            embeddings, original[[r["item_id"] for r in requests]]
        )
        validation["released_embedding_cosine_min"] = similarity.min().item()
        validation["released_embedding_cosine_mean"] = similarity.mean().item()


def expected_rows(requests: list[dict]) -> list[dict]:
    return [
        {"item_id": r["item_id"], "raw_movie_id": r["raw_movie_id"]} for r in requests
    ]


def cached_validation(previous, response_hash, output, requests) -> bool:
    embedding_file = output / "embeddings.pt"
    row_file = output / "embedding_rows.json"
    return (
        previous.get("response_sha256") == response_hash
        and previous.get("encoder") == ENCODER
        and previous.get("encoder_revision") == ENCODER_REVISION
        and previous.get("encoding") == "PASS"
        and embedding_file.exists()
        and previous.get("embedding_sha256") == file_hash(embedding_file)
        and row_file.exists()
        and json.loads(row_file.read_text()) == expected_rows(requests)
    )


def main() -> None:
    parser, args = parse_args()
    source = (
        "upstream"
        if args.upstream_responses
        else request_id(args.endpoint, {"model": args.model})[:16]
    )
    output = ARTIFACTS / "llm_subset" / source / str(args.items)
    output.mkdir(parents=True, exist_ok=True)
    cache = output.parent / "responses"
    cache.mkdir(exist_ok=True)
    if args.items == 100 and (args.generate or args.encode):
        previous = output.parent / "20/validation.json"
        if (
            not previous.exists()
            or json.loads(previous.read_text()).get("encoding") != "PASS"
        ):
            parser.error("The 20-item subset has not passed encoding")
    requests = build_requests(args)
    atomic_json(output / "requests.json", requests)
    scale = {
        "requests": len(requests),
        "model": args.model,
        "endpoint": args.endpoint,
        "prompt_characters": sum(
            len(m["content"]) for r in requests for m in r["payload"]["messages"]
        ),
        "response_word_instruction": 200,
        "purpose": "engineering_only",
    }
    atomic_json(output / "scale.json", scale)
    print(json.dumps(scale, indent=2))
    responses, records = collect_responses(parser, args, requests, cache)
    if records:
        atomic_json(output / "usage.json", usage_summary(args.model, records))
    validation_path = output / "validation.json"
    previous = (
        json.loads(validation_path.read_text()) if validation_path.exists() else {}
    )
    validation = {
        "items": args.items,
        "source": source,
        "live_generation": "COMPLETE"
        if responses and not args.upstream_responses
        else "NOT_RUN",
        "encoding": "NOT_RUN",
    }
    if responses:
        assert len(responses) == args.items and all(
            isinstance(text, str) and text.strip() for text in responses
        )
        atomic_json(
            output / "responses.json",
            [
                {
                    "item_id": r["item_id"],
                    "raw_movie_id": r["raw_movie_id"],
                    "text": text,
                }
                for r, text in zip(requests, responses, strict=True)
            ],
        )
    response_hash = hashlib.sha256(json.dumps(responses).encode()).hexdigest()
    if responses and cached_validation(previous, response_hash, output, requests):
        validation = dict(previous, live_generation=validation["live_generation"])
        print("Valid embeddings already cached:", output / "embeddings.pt")
    if args.encode and validation["encoding"] != "PASS":
        atomic_json(validation_path, validation)
        encode_responses(args, responses, requests, validation, output, response_hash)
    if args.encode or args.generate or not previous:
        atomic_json(validation_path, validation)
    manifest = load_manifest(MANIFEST)
    manifest.setdefault("original_llm_subsets", {}).setdefault(source, {})[
        str(args.items)
    ] = {
        "validation": str(validation_path.relative_to(ROOT)),
        **json.loads(validation_path.read_text()),
    }
    atomic_json(MANIFEST, manifest)
    print(
        "Saved", output.relative_to(ROOT), "—", json.loads(validation_path.read_text())
    )


if __name__ == "__main__":
    main()
