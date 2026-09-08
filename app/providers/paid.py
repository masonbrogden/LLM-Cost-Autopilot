from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import httpx

from app.providers.base import Provider, StandardResponse


class AnthropicProvider(Provider):
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None):
        resolved_key = (api_key or os.getenv("ANTHROPIC_API_KEY") or "").strip()
        placeholder_values = {
            "",
            "your_api_key_here",
            "sk-ant-paste-your-key-here",
            "changeme",
        }
        if resolved_key.lower() in {value.lower() for value in placeholder_values}:
            raise ValueError(
                "ANTHROPIC_API_KEY is unset or still set to a placeholder value. "
                "Set a real key before starting the app."
            )
        self.api_key = resolved_key
        self.base_url = base_url or os.getenv("PAID_BASE_URL", "https://api.anthropic.com")
        self.client = httpx.AsyncClient(
            base_url=self.base_url,
            headers={
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "Content-Type": "application/json",
            },
        )

    async def complete(
        self,
        messages: List[Dict[str, str]],
        model_config: Any,
        max_tokens: Optional[int] = None,
    ) -> StandardResponse:
        payload = {
            "model": model_config.model_id,
            "messages": messages,
            "max_tokens": max_tokens or 256,
        }

        response = await self.client.post("/v1/messages", json=payload)
        response.raise_for_status()
        data = response.json()

        text_parts = []
        for block in data.get("content", []):
            if isinstance(block, dict) and block.get("type") == "text":
                text_parts.append(block.get("text", ""))
        text = "".join(text_parts)

        usage = data.get("usage", {})
        prompt_tokens = int(usage.get("input_tokens", 0) or 0)
        completion_tokens = int(usage.get("output_tokens", 0) or 0)

        return StandardResponse(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=0,
            model_id=model_config.model_id,
            provider="anthropic",
            raw=data,
        )
