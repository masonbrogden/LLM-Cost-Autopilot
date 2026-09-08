import asyncio
import os
import sqlite3

import pytest

from app import logging_db
from app.logging_db import get_request_count, initialize_db, log_request
from app.providers.base import StandardResponse
from app.routes import proxy
from app.schemas import ChatCompletionRequest, ChatMessage, Choice, ChatCompletionResponse, Usage


def test_request_and_response_schemas_round_trip():
    req = ChatCompletionRequest(
        model="claude-haiku-4-5-20251001",
        messages=[ChatMessage(role="user", content="hello")],
        max_tokens=32,
        temperature=0.2,
    )
    assert req.model == "claude-haiku-4-5-20251001"

    resp = ChatCompletionResponse(
        model="claude-haiku-4-5-20251001",
        choices=[Choice(index=0, message=ChatMessage(role="assistant", content="hi"), finish_reason="stop")],
        usage=Usage(prompt_tokens=12, completion_tokens=3, total_tokens=15),
    )
    assert resp.choices[0].message.content == "hi"


def test_logging_db_creates_table_and_logs_row(tmp_path):
    db_path = tmp_path / "autopilot.db"
    initialize_db(str(db_path))
    request_id = "req-1"
    row_id = log_request(
        db_path=str(db_path),
        request_id=request_id,
        provider="anthropic",
        routed_model="claude-haiku-4-5-20251001",
        routed_tier="low",
        prompt_text="hello",
        response_text="hi",
        prompt_tokens=10,
        completion_tokens=2,
        actual_cost_usd=0.00002,
        baseline_cost_usd=0.00005,
        savings_usd=0.00003,
        error=None,
    )
    assert row_id == request_id
    assert get_request_count(str(db_path)) == 1


def test_proxy_route_returns_openai_compatible_shape(tmp_path, monkeypatch):
    db_path = tmp_path / "autopilot.db"
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("LLM_COST_AUTOPILOT_DB", str(db_path))

    class FakeProvider:
        async def complete(self, messages, model_config, max_tokens):
            return StandardResponse(
                text="assistant reply",
                prompt_tokens=12,
                completion_tokens=3,
                latency_ms=120,
                model_id=model_config.model_id,
                provider=model_config.provider,
                raw={"ok": True},
            )

    monkeypatch.setattr(proxy, "_provider_for_model", lambda model_id: FakeProvider())

    response = asyncio.run(
        proxy.chat_completions(
            ChatCompletionRequest(
                model="claude-haiku-4-5-20251001",
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=16,
                temperature=0.0,
            )
        )
    )

    assert response.model == "claude-haiku-4-5-20251001"
    assert response.choices[0].message.role == "assistant"
    assert response.usage.total_tokens == 15
    assert get_request_count(str(db_path)) >= 1


def test_proxy_route_logs_error_and_null_costs_when_provider_fails(tmp_path, monkeypatch):
    db_path = tmp_path / "autopilot.db"
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("LLM_COST_AUTOPILOT_DB", str(db_path))

    class FailingProvider:
        async def complete(self, messages, model_config, max_tokens):
            raise RuntimeError("provider exploded")

    monkeypatch.setattr(proxy, "_provider_for_model", lambda model_id: FailingProvider())

    with pytest.raises(Exception):
        asyncio.run(
            proxy.chat_completions(
                ChatCompletionRequest(
                    model="claude-haiku-4-5-20251001",
                    messages=[{"role": "user", "content": "Hello"}],
                    max_tokens=16,
                    temperature=0.0,
                )
            )
        )

    conn = sqlite3.connect(str(db_path))
    row = conn.execute(
        "SELECT id, request_id, error, actual_cost_usd, baseline_cost_usd, savings_usd FROM requests ORDER BY created_at DESC LIMIT 1"
    ).fetchone()
    conn.close()

    assert row is not None
    assert row[0].startswith("chatcmpl-")
    assert row[0] == row[1]
    assert row[2] == "provider exploded"
    assert row[3] is None
    assert row[4] is None
    assert row[5] is None
