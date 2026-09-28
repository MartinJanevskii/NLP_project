"""Request building, response caching and truncated-response handling, offline."""

import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from llm_knowledge_enhancement.llm import generate, request_id

payload = {
    "model": "deepseek-chat",
    "messages": [{"role": "user", "content": "A movie"}],
}
endpoint = "https://api.deepseek.com/v1/chat/completions"
with TemporaryDirectory() as directory:
    cache = Path(directory)
    response = {
        "choices": [{"finish_reason": "stop", "message": {"content": "A description"}}]
    }
    with patch(
        "llm_knowledge_enhancement.llm.urlopen",
        return_value=io.BytesIO(json.dumps(response).encode()),
    ) as api:
        first = generate(endpoint, payload, "test-key", cache)
        assert api.call_count == 1
        request = api.call_args.args[0]
        assert json.loads(request.data) == payload
        assert request.get_header("Authorization") == "Bearer test-key"
    with patch(
        "llm_knowledge_enhancement.llm.urlopen",
        side_effect=AssertionError("Cache must prevent API call"),
    ):
        assert generate(endpoint, payload, "test-key", cache) == first
    assert request_id(endpoint, payload) != request_id(
        endpoint, dict(payload, model="different")
    )
    assert request_id(endpoint, payload) != request_id(endpoint + "/other", payload)
    response["choices"][0]["finish_reason"] = "length"
    changed = dict(payload, model="different")
    with patch(
        "llm_knowledge_enhancement.llm.urlopen",
        return_value=io.BytesIO(json.dumps(response).encode()),
    ):
        try:
            generate(endpoint, changed, "test-key", cache)
        except ValueError:
            pass
        else:
            raise AssertionError("Truncated response accepted")
    assert not (cache / f"{request_id(endpoint, changed)}.json").exists()
print(
    "Offline request, caching and truncated-response checks passed; no live API calls"
)
