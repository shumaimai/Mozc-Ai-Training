from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


@dataclass(frozen=True)
class ReviewBudget:
    max_cost_usd: float
    input_price_per_million: float
    output_price_per_million: float
    spent_usd: float = 0.0

    def can_afford(self, input_tokens: int, output_tokens: int) -> bool:
        estimated = input_tokens * self.input_price_per_million / 1_000_000
        estimated += output_tokens * self.output_price_per_million / 1_000_000
        return self.spent_usd + estimated <= self.max_cost_usd

    def charge(self, input_tokens: int, output_tokens: int) -> ReviewBudget:
        cost = input_tokens * self.input_price_per_million / 1_000_000
        cost += output_tokens * self.output_price_per_million / 1_000_000
        return ReviewBudget(
            max_cost_usd=self.max_cost_usd,
            input_price_per_million=self.input_price_per_million,
            output_price_per_million=self.output_price_per_million,
            spent_usd=self.spent_usd + cost,
        )


def review_payload(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "surface": row["record"]["surface"],
        "reading": row["record"]["reading"],
        "category": row["record"]["category"],
        "reading_confidence": row["record"]["reading_confidence"],
        "candidates": row.get("candidates", [])[:20],
        "context": row.get("context", [])[-5:],
    }


def build_request(model: str, row: dict[str, Any]) -> dict[str, Any]:
    prompt = {
        "task": "Classify this public Japanese IME dataset record. Do not invent readings or facts.",
        "record": review_payload(row),
        "allowed_decisions": [
            "accept",
            "reject_noisy_text",
            "reject_bad_reading",
            "reject_not_conversion_unit",
            "review_ambiguous",
            "dictionary_preferred",
            "rerank_preferred",
        ],
        "response_schema": {
            "decision": "one allowed decision",
            "confidence": "number from 0 to 1",
            "reason_code": "short snake_case string",
        },
    }
    return {
        "model": model,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "messages": [
            {"role": "system", "content": "Return only a valid JSON object."},
            {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
        ],
    }


def comparison_key(row: dict[str, Any]) -> str:
    record = row.get("record") or {}
    provenance = record.get("provenance") or {}
    return "|".join(
        [
            str(provenance.get("source_id", "")),
            str(record.get("surface", "")),
            str(record.get("reading", "")),
            str(row.get("action", "")),
        ]
    )


def _api_key() -> str:
    for name in ("DEEPSEEK_API_KEY", "OPENAI_API_KEY", "DASHSCOPE_API_KEY"):
        value = os.environ.get(name)
        if value:
            return value
    raise RuntimeError("DEEPSEEK_API_KEY (or OPENAI_API_KEY / DASHSCOPE_API_KEY) is not set")


def _chat_completions_url() -> str:
    base = (
        os.environ.get("DEEPSEEK_BASE_URL")
        or os.environ.get("OPENAI_BASE_URL")
        or "https://api.deepseek.com"
    ).rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    return f"{base}/chat/completions"


def _message_content(message: dict[str, Any]) -> str:
    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content
    # Some OpenAI-compatible gateways return multipart content blocks.
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
            elif isinstance(block, str):
                parts.append(block)
        joined = "".join(parts).strip()
        if joined:
            return joined
    raise RuntimeError("chat completion message content was empty")


def execute(model: str, row: dict[str, Any], timeout_seconds: int = 180) -> tuple[dict[str, Any], dict[str, int]]:
    api_key = _api_key()
    body = json.dumps(build_request(model, row), ensure_ascii=False).encode("utf-8")
    request = Request(
        _chat_completions_url(),
        data=body,
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"chat completion failed: {error.code}: {detail}") from error
    try:
        message = payload["choices"][0]["message"]
        content = _message_content(message)
        # Strip optional markdown fences some gateways add around JSON.
        stripped = content.strip()
        if stripped.startswith("```"):
            stripped = stripped.strip("`")
            if stripped.startswith("json"):
                stripped = stripped[4:]
            stripped = stripped.strip()
        verdict = json.loads(stripped)
        usage = payload.get("usage", {})
        tokens = {
            "input": int(usage.get("prompt_tokens", 0)),
            "output": int(usage.get("completion_tokens", 0)),
        }
    except (IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeError("chat completion response did not contain a valid JSON verdict") from error
    return verdict, tokens


def execute_with_retries(
    model: str,
    row: dict[str, Any],
    *,
    timeout_seconds: int = 180,
    max_attempts: int = 4,
    base_delay_seconds: float = 2.0,
) -> tuple[dict[str, Any], dict[str, int]]:
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            return execute(model, row, timeout_seconds=timeout_seconds)
        except (TimeoutError, URLError, RuntimeError) as error:
            last_error = error
            if attempt >= max_attempts:
                break
            delay = base_delay_seconds * (2 ** (attempt - 1))
            print(f"retry {attempt}/{max_attempts} after {error}; sleeping {delay:.1f}s")
            time.sleep(delay)
    assert last_error is not None
    raise last_error
