"""Validate original CoLaKG item preprocessing on 20, then 100 MovieLens items.

Default: prepare requests, without an API call. --generate calls the configured
provider. --encode encodes cached responses. --upstream-responses uses published
responses to check the encoder without a key; it never validates live generation.
"""

import argparse
import hashlib
import json
import os
from urllib.request import Request, urlopen

from run_baseline import ROOT, UPSTREAM, load_manifest

SYSTEM = (
    "Assume you are an expert in movie recommendation. You will be given a certain movie "
    "with its first-order information (in the form of triples) and some second-order "
    "relationships (movies related to this movie). Please complete the missing knowledge, "
    "summarize the movie and analyze what kind of users would like it. Your response should "
    "be a coherent paragraph and no more than 200 words."
)
ENCODER = "princeton-nlp/sup-simcse-roberta-large"
ENCODER_REVISION = "96d164d9950b72f4ce179cb1eb3414de0910953f"


def atomic_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def request_id(endpoint, payload):
    return hashlib.sha256(
        json.dumps([endpoint, payload], sort_keys=True).encode()
    ).hexdigest()


def generate(endpoint, payload, key, cache):
    identity = request_id(endpoint, payload)
    path = cache / f"{identity}.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if (
            saved.get("request_id") == identity
            and isinstance(saved.get("text"), str)
            and saved["text"].strip()
        ):
            return saved
    request = Request(
        endpoint,
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    with urlopen(request, timeout=120) as response:
        result = json.load(response)
    choice = result["choices"][0]
    text = choice["message"]["content"]
    if (
        choice.get("finish_reason") != "stop"
        or not isinstance(text, str)
        or not text.strip()
    ):
        raise ValueError(
            "Provider returned an empty, truncated or unfinished response; not cached"
        )
    saved = {
        "request_id": identity,
        "text": text,
        "usage": result.get("usage"),
        "model": result.get("model", payload["model"]),
        "endpoint": endpoint,
    }
    atomic_json(path, saved)
    return saved


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", type=int, choices=(20, 100), default=20)
    parser.add_argument("--model", default=os.environ.get("LLM_MODEL", "deepseek-chat"))
    parser.add_argument(
        "--endpoint",
        default=os.environ.get(
            "LLM_ENDPOINT", "https://api.deepseek.com/v1/chat/completions"
        ),
    )
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--encode", action="store_true")
    parser.add_argument("--upstream-responses", action="store_true")
    args = parser.parse_args()
    if args.generate and args.upstream_responses:
        parser.error("Choose published responses or live generation")
    data = UPSTREAM / "data/ml-1m"
    mapping = sorted(
        (
            tuple(map(int, line.split()))
            for line in (data / "item_map.txt").read_text().splitlines()
        ),
        key=lambda pair: pair[1],
    )
    prompts = json.loads((data / "llm_input_item.json").read_text())
    source = (
        "upstream"
        if args.upstream_responses
        else request_id(args.endpoint, {"model": args.model})[:16]
    )
    output = ROOT / "artifacts/llm_subset" / source / str(args.items)
    output.mkdir(parents=True, exist_ok=True)
    cache = output.parent / "responses"
    cache.mkdir(exist_ok=True)
    if args.items == 100 and (args.generate or args.encode):
        previous = output.parent / "20/validation.json"
        if (
            not previous.exists()
            or json.loads(previous.read_text()).get("encoding") != "PASS"
        ):
            parser.error("Encode and validate the 20-item subset first")
    requests = [
        {
            "raw_movie_id": raw,
            "item_id": item,
            "payload": {
                "model": args.model,
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": prompts[str(item)]},
                ],
                "temperature": 0.0,
                "top_p": 0.001,
                "stream": False,
            },
        }
        for raw, item in mapping[: args.items]
    ]
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
    responses = []
    generation_records = []
    if args.upstream_responses:
        published = json.loads((data / "llm_response_item.json").read_text())
        responses = [published[str(r["raw_movie_id"])] for r in requests]
    elif args.generate or args.encode:
        key = os.environ.get("LLM_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")
        if args.generate and not key:
            parser.error("Set LLM_API_KEY or DEEPSEEK_API_KEY locally; never commit it")
        for request in requests:
            payload = request["payload"]
            if args.generate:
                saved = generate(args.endpoint, payload, key, cache)
            else:
                saved = json.loads(
                    (cache / f"{request_id(args.endpoint, payload)}.json").read_text()
                )
                if saved.get("request_id") != request_id(args.endpoint, payload):
                    raise ValueError("Response cache fingerprint mismatch")
            responses.append(saved["text"])
            generation_records.append(saved)
            print(f"Response ready: {len(responses)}/{args.items}", flush=True)
    if generation_records:
        atomic_json(
            output / "usage.json",
            {
                "requested_model": args.model,
                "served_models": sorted(
                    {record["model"] for record in generation_records}
                ),
                "cached_response_count": len(generation_records),
                "tokens_for_saved_responses": {
                    name: sum(
                        (record.get("usage") or {}).get(name, 0)
                        for record in generation_records
                    )
                    for name in (
                        "prompt_tokens",
                        "completion_tokens",
                        "prompt_cache_hit_tokens",
                        "prompt_cache_miss_tokens",
                    )
                },
                "note": "Usage for retained responses, not new charges on each cached rerun; failed requests are not included.",
            },
        )
    validation_path = output / "validation.json"
    previous_validation = (
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
                for r, text in zip(requests, responses)
            ],
        )
    response_hash = hashlib.sha256(json.dumps(responses).encode()).hexdigest()
    embedding_file = output / "embeddings.pt"
    row_file = output / "embedding_rows.json"
    expected_rows = [
        {"item_id": r["item_id"], "raw_movie_id": r["raw_movie_id"]} for r in requests
    ]
    if responses and (
        previous_validation.get("response_sha256") == response_hash
        and previous_validation.get("encoder") == ENCODER
        and previous_validation.get("encoder_revision") == ENCODER_REVISION
        and previous_validation.get("encoding") == "PASS"
        and embedding_file.exists()
        and previous_validation.get("embedding_sha256")
        == hashlib.sha256(embedding_file.read_bytes()).hexdigest()
        and row_file.exists()
        and json.loads(row_file.read_text()) == expected_rows
    ):
        validation = dict(
            previous_validation, live_generation=validation["live_generation"]
        )
        print("Valid embeddings already cached:", embedding_file)
    if args.encode and validation["encoding"] != "PASS":
        import torch
        from transformers import AutoModel, AutoTokenizer

        atomic_json(validation_path, validation)
        cache_dir = ROOT / "artifacts/model_cache"
        tokenizer = AutoTokenizer.from_pretrained(
            ENCODER, revision=ENCODER_REVISION, cache_dir=cache_dir
        )
        model = AutoModel.from_pretrained(
            ENCODER, revision=ENCODER_REVISION, cache_dir=cache_dir
        ).eval()
        rows = []
        # Smaller batches only affect CPU memory usage, not the encoder or pooling.
        with torch.no_grad():
            for start in range(0, len(responses), 4):
                inputs = tokenizer(
                    responses[start : start + 4],
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                )
                rows.append(
                    model(
                        **inputs, output_hidden_states=True, return_dict=True
                    ).pooler_output.cpu()
                )
        embeddings = torch.cat(rows)
        assert (
            embeddings.shape == (args.items, 1024) and torch.isfinite(embeddings).all()
        )
        torch.save(embeddings, output / "embeddings.pt")
        atomic_json(
            output / "embedding_rows.json",
            expected_rows,
        )
        validation.update(
            encoding="PASS",
            encoder=ENCODER,
            encoder_revision=getattr(model.config, "_commit_hash", None),
            response_sha256=response_hash,
            embedding_sha256=hashlib.sha256(embedding_file.read_bytes()).hexdigest(),
        )
        if args.upstream_responses:
            original = torch.load(
                data / "movie_embeddings_simcse_kg.pt",
                map_location="cpu",
                weights_only=True,
            )
            similarity = torch.nn.functional.cosine_similarity(
                embeddings, original[[r["item_id"] for r in requests]]
            )
            validation["released_embedding_cosine_min"] = similarity.min().item()
            validation["released_embedding_cosine_mean"] = similarity.mean().item()
    if args.encode or args.generate or not previous_validation:
        atomic_json(validation_path, validation)
    manifest_path = ROOT / "artifacts/EXPERIMENT_MANIFEST.json"
    manifest = load_manifest(manifest_path)
    manifest.setdefault("original_llm_subsets", {}).setdefault(source, {})[
        str(args.items)
    ] = {
        "validation": str(validation_path.relative_to(ROOT)),
        **json.loads(validation_path.read_text()),
    }
    atomic_json(manifest_path, manifest)
    print(
        "Saved", output.relative_to(ROOT), "—", json.loads(validation_path.read_text())
    )


if __name__ == "__main__":
    main()
