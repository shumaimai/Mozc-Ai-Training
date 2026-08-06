from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError
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


def execute(model: str, row: dict[str, Any], timeout_seconds: int = 60) -> tuple[dict[str, Any], dict[str, int]]:
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set")
    body = json.dumps(build_request(model, row), ensure_ascii=False).encode("utf-8")
    request = Request(
        "https://api.deepseek.com/chat/completions",
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
        raise RuntimeError(f"DeepSeek request failed: {error.code}: {detail}") from error
    try:
        content = payload["choices"][0]["message"]["content"]
        verdict = json.loads(content)
        usage = payload.get("usage", {})
        tokens = {
            "input": int(usage.get("prompt_tokens", 0)),
            "output": int(usage.get("completion_tokens", 0)),
        }
    except (IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeError("DeepSeek response did not contain a valid JSON verdict") from error
    return verdict, tokens
