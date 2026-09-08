from unittest.mock import AsyncMock, patch

import pytest

from app.providers.base import StandardResponse
from app.providers.ollama import OllamaProvider
from app.providers.paid import AnthropicProvider


class _StubResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError("bad response")

    def json(self):
        return self._payload


@pytest.mark.asyncio
async def test_ollama_provider_normalizes_tokens():
    provider = OllamaProvider(base_url="http://localhost:11434")
    payload = {
        "message": {"content": "hello"},
        "prompt_eval_count": 11,
        "eval_count": 7,
        "total_duration": 1_250_000,
    }

    with patch.object(provider.client, "post", AsyncMock(return_value=_StubResponse(payload))):
        result = await provider.complete([
            {"role": "user", "content": "hi"}
        ], model_config=type("M", (), {"model_id": "llama3.1:8b"})())

    assert isinstance(result, StandardResponse)
    assert result.prompt_tokens == 11
    assert result.completion_tokens == 7
    assert result.provider == "ollama"


@pytest.mark.asyncio
async def test_paid_provider_normalizes_anthropic_tokens():
    provider = AnthropicProvider(api_key="test-key", base_url="https://api.anthropic.com")
    payload = {
        "content": [{"type": "text", "text": "hi there"}],
        "usage": {"input_tokens": 16, "output_tokens": 8},
        "model": "claude-haiku-4-5-20251001",
        "id": "msg_123",
    }

    with patch.object(provider.client, "post", AsyncMock(return_value=_StubResponse(payload))):
        result = await provider.complete([
            {"role": "user", "content": "hi"}
        ], model_config=type("M", (), {"model_id": "claude-haiku-4-5-20251001"})())

    assert isinstance(result, StandardResponse)
    assert result.prompt_tokens == 16
    assert result.completion_tokens == 8
    assert result.provider == "anthropic"
    assert result.text == "hi there"
