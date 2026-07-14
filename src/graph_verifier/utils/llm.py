from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


class LLMError(RuntimeError):
    pass


ROOT = Path(__file__).resolve().parents[3]
PROMPT_DIR = Path(__file__).resolve().parents[1] / "prompts"
DEFAULT_MODEL_CONFIG = ROOT / "model" / "openrouter" / "deepseek" / "deepseek-v4-flash-high.json"
DEFAULT_ATTEMPTS = 3
RETRY_BACKOFF_SECONDS = 1.0


def complete_json(
    prompt_name: str,
    data: dict[str, Any],
    model_config: str | Path = DEFAULT_MODEL_CONFIG,
    *,
    attempts: int = DEFAULT_ATTEMPTS,
) -> dict[str, Any]:
    if attempts < 1:
        raise ValueError("attempts must be at least 1")
    prompt_path = PROMPT_DIR / prompt_name
    try:
        prompt = prompt_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise LLMError(f"missing prompt {prompt_name}: {exc}") from exc
    content = prompt + "\n\nINPUT JSON:\n" + json.dumps(data, ensure_ascii=False)
    last_error: LLMError | None = None
    for attempt in range(1, attempts + 1):
        start = time.perf_counter()
        logging.info("llm start prompt=%s attempt=%s", prompt_name, attempt)
        try:
            text = complete(content, model_config)
            result = parse_json_object(text)
        except LLMError as exc:
            last_error = exc
            if attempt == attempts:
                logging.exception(
                    "llm error prompt=%s attempt=%s elapsed=%.2fs",
                    prompt_name,
                    attempt,
                    time.perf_counter() - start,
                )
                raise
            logging.warning(
                "llm retry prompt=%s attempt=%s elapsed=%.2fs backoff=%.1fs error=%s",
                prompt_name,
                attempt,
                time.perf_counter() - start,
                RETRY_BACKOFF_SECONDS * 2 ** (attempt - 1),
                exc,
            )
            time.sleep(RETRY_BACKOFF_SECONDS * 2 ** (attempt - 1))
            continue
        logging.info(
            "llm end prompt=%s attempt=%s elapsed=%.2fs",
            prompt_name,
            attempt,
            time.perf_counter() - start,
        )
        return result
    raise last_error or LLMError("LLM completion failed")


def parse_json_object(text: str) -> dict[str, Any]:
    if not isinstance(text, str):
        raise LLMError("LLM response content must be a string")
    try:
        result = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end <= start:
            raise LLMError("LLM did not return JSON") from None
        try:
            result = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise LLMError("LLM did not return valid JSON") from exc
    if not isinstance(result, dict):
        raise LLMError("LLM JSON must be an object")
    return result


def complete_agent_json(
    prompt_name: str,
    data: dict[str, Any],
    model_config: str | None,
) -> dict[str, Any]:
    if not model_config:
        raise LLMError("case has no agent_model_config")
    return complete_json(prompt_name, data, model_config)


def complete(content: str, model_config: str | Path = DEFAULT_MODEL_CONFIG) -> str:
    config_path = Path(model_config)
    if not config_path.is_absolute():
        config_path = ROOT / config_path
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise LLMError(f"model config not found: {config_path}") from exc
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
        content = payload["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMError("malformed LLM response") from exc
    if not isinstance(content, str):
        raise LLMError("LLM response content must be a string")
    return content
