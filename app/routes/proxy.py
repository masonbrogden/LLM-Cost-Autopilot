from __future__ import annotations

import os
import uuid
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException

from app.logging_db import log_request
from app.providers.base import StandardResponse
from app.providers.ollama import OllamaProvider
from app.providers.paid import AnthropicProvider
from app.registry import MODEL_REGISTRY, get_baseline_model, get_cheapest_in_tier
from app.schemas import ChatCompletionRequest, ChatCompletionResponse, ChatMessage, Choice, Usage

router = APIRouter()


def _provider_for_model(model_id: str):
    for model in MODEL_REGISTRY:
        if model.model_id == model_id:
            if model.provider == "ollama":
                return OllamaProvider()
            return AnthropicProvider(api_key=os.getenv("ANTHROPIC_API_KEY", ""))
    return AnthropicProvider(api_key=os.getenv("ANTHROPIC_API_KEY", ""))


@router.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    model_id = request.model or "claude-haiku-4-5-20251001"
    model = next((m for m in MODEL_REGISTRY if m.model_id == model_id), get_cheapest_in_tier("low"))
    if model is None:
        raise HTTPException(status_code=400, detail="Unsupported model")

    request_id = f"chatcmpl-{uuid.uuid4().hex}"
    provider = _provider_for_model(model.model_id)
    prompt_text = "\n".join(msg.content for msg in request.messages)
    db_path = os.getenv("LLM_COST_AUTOPILOT_DB", "data/autopilot.db")

    try:
        response: StandardResponse = await provider.complete(
            [{"role": msg.role, "content": msg.content} for msg in request.messages],
            model,
            max_tokens=request.max_tokens or 256,
        )
        actual_cost = (
            model.cost_per_input_token * max(response.prompt_tokens, 0)
            + model.cost_per_output_token * max(response.completion_tokens, 0)
        )
        baseline = get_baseline_model()
        baseline_cost = (
            baseline.cost_per_input_token * max(response.prompt_tokens, 0)
            + baseline.cost_per_output_token * max(response.completion_tokens, 0)
        )
        savings = baseline_cost - actual_cost
        log_request(
            db_path=db_path,
            request_id=request_id,
            provider=model.provider,
            routed_model=model.model_id,
            routed_tier=model.tier,
            prompt_text=prompt_text,
            response_text=response.text,
            prompt_tokens=response.prompt_tokens,
            completion_tokens=response.completion_tokens,
            actual_cost_usd=actual_cost,
            baseline_cost_usd=baseline_cost,
            savings_usd=savings,
            error=None,
        )
        return ChatCompletionResponse(
            id=request_id,
            model=model.model_id,
            choices=[
                Choice(
                    index=0,
                    message=ChatMessage(role="assistant", content=response.text),
                    finish_reason="stop",
                )
            ],
            usage=Usage(
                prompt_tokens=response.prompt_tokens,
                completion_tokens=response.completion_tokens,
                total_tokens=response.prompt_tokens + response.completion_tokens,
            ),
        )
    except Exception as exc:
        log_request(
            db_path=db_path,
            request_id=request_id,
            provider=model.provider,
            routed_model=model.model_id,
            routed_tier=model.tier,
            prompt_text=prompt_text,
            response_text=None,
            prompt_tokens=0,
            completion_tokens=0,
            actual_cost_usd=None,
            baseline_cost_usd=None,
            savings_usd=None,
            error=str(exc),
        )
        raise HTTPException(status_code=500, detail=str(exc)) from exc
