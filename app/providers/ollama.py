"""Ollama provider wrapper."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import httpx

from app.providers.base import Provider, StandardResponse


class OllamaProvider(Provider):
    def __init__(self, base_url: Optional[str] = None):
        self.base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.client = httpx.AsyncClient(base_url=self.base_url)

    async def complete(
        self,
        messages: List[Dict[str, str]],
        model_config: Any,
        max_tokens: Optional[int] = None,
    ) -> StandardResponse:
        payload = {
            "model": model_config.model_id,
            "messages": messages,
            "stream": False,
        }
        if max_tokens is not None:
            payload["options"] = {"num_predict": max_tokens}

        response = await self.client.post("/api/chat", json=payload)
        response.raise_for_status()
        data = response.json()

        message = data.get("message", {})
        text = message.get("content", "")

        prompt_tokens = int(data.get("prompt_eval_count", 0) or 0)
        completion_tokens = int(data.get("eval_count", 0) or 0)

        return StandardResponse(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=int(data.get("total_duration", 0) / 1_000_000),
            model_id=model_config.model_id,
            provider="ollama",
            raw=data,
        )
