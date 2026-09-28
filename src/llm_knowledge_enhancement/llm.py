"""Chat-completion client with a content-addressed response cache."""

import hashlib
import json
import os
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from llm_knowledge_enhancement.files import atomic_json

DEFAULT_MODEL = os.environ.get("LLM_MODEL", "deepseek-chat")
DEFAULT_ENDPOINT = os.environ.get(
    "LLM_ENDPOINT", "https://api.deepseek.com/v1/chat/completions"
)


def api_key() -> str | None:
    return os.environ.get("LLM_API_KEY") or os.environ.get("DEEPSEEK_API_KEY")


def request_id(endpoint: str, payload: dict) -> str:
    return hashlib.sha256(
        json.dumps([endpoint, payload], sort_keys=True).encode()
    ).hexdigest()


def post(endpoint: str, payload: dict, key: str, timeout: int = 120) -> dict:
    request = Request(
        endpoint,
        data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    with urlopen(request, timeout=timeout) as response:
        return json.load(response)


def _record(identity: str, text: str, result: dict, payload: dict, endpoint: str):
    return {
        "request_id": identity,
        "text": text,
        "usage": result.get("usage"),
        "model": result.get("model", payload["model"]),
        "endpoint": endpoint,
    }


def generate(endpoint: str, payload: dict, key: str, cache: Path) -> dict:
    """Return the cached response, or request one and cache it if it finished normally."""
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
    result = post(endpoint, payload, key)
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
    saved = _record(identity, text, result, payload, endpoint)
    atomic_json(path, saved)
    return saved


def generate_robust(
    endpoint: str,
    payload: dict,
    key: str,
    cache: Path,
    attempts: int = 4,
    pause: float = 5.0,
) -> dict:
    """generate() with retries; a response cut off at max_tokens is kept and flagged.

    At temperature 0 a length-capped answer repeats on every retry, so it is saved
    with truncated=True instead of failing the run.
    """
    last = None
    for attempt in range(attempts):
        try:
            return generate(endpoint, payload, key, cache)
        except (ValueError, URLError, TimeoutError, OSError) as error:
            last = error
            time.sleep(pause * (attempt + 1))
    try:
        result = post(endpoint, payload, key)
        choice = result["choices"][0]
        text = choice["message"]["content"]
    except (URLError, TimeoutError, OSError, KeyError, ValueError):
        text, choice, result = None, {}, {}
    if (
        choice.get("finish_reason") == "length"
        and isinstance(text, str)
        and text.strip()
    ):
        identity = request_id(endpoint, payload)
        saved = _record(identity, text, result, payload, endpoint)
        saved.update(finish_reason="length", truncated=True)
        atomic_json(cache / f"{identity}.json", saved)
        return saved
    raise RuntimeError(f"Generation failed after {attempts} attempts: {last}")
