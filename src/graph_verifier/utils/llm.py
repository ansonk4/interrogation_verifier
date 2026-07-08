from __future__ import annotations

import json
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


class LLMError(RuntimeError):
    pass


ROOT = Path(__file__).resolve().parents[3]
PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompts"
DEFAULT_MODEL_CONFIG = ROOT / "model" / "openrouter" / "hy3.json"


def complete_json(prompt_name: str, data: dict[str, Any]) -> dict[str, Any]:
    prompt_path = PROMPT_DIR / prompt_name
    try:
        prompt = prompt_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise LLMError(f"missing prompt {prompt_name}: {exc}") from exc
    content = prompt + "\n\nINPUT JSON:\n" + json.dumps(data, ensure_ascii=False)
    text = complete(content)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise LLMError("LLM did not return JSON") from None
        return json.loads(text[start : end + 1])


def complete(content: str) -> str:
    try:
        config = json.loads(DEFAULT_MODEL_CONFIG.read_text(encoding="utf-8"))
    except OSError as exc:
        raise LLMError(f"model config not found: {DEFAULT_MODEL_CONFIG}") from exc
    api_key = config.get("api_key")
    base_url = str(config.get("llm_url", "")).rstrip("/")
    model = config.get("model")
    if not api_key or not base_url or not model:
        raise LLMError("model config missing api_key, llm_url, or model")
    body: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "temperature": 0,
    }
    for key, value in config.get("llm_arg", {}).get("extra_body", {}).items():
        if value not in ({}, [], None):
            body[key] = value
    request = urllib.request.Request(
        base_url + "/chat/completions",
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "https://localhost/graph-verifier",
            "X-Title": "graph-verifier",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:300]
        raise LLMError(f"HTTP {exc.code}: {detail}") from exc
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise LLMError(str(exc)) from exc
    try:
        return payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMError("malformed LLM response") from exc
