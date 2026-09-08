#!/usr/bin/env python3
"""Smoke-test every registry model once."""

from __future__ import annotations

import asyncio
from typing import Any

from app.registry import MODEL_REGISTRY, get_baseline_model
from app.providers.ollama import OllamaProvider
from app.providers.paid import AnthropicProvider


async def main() -> None:
    providers = {
        "local": OllamaProvider(),
        "low": AnthropicProvider(),
        "medium": AnthropicProvider(),
        "high": AnthropicProvider(),
    }

    print(f"{'Model':<30} {'Tier':<8} {'Prompt Tokens':>14} {'Completion Tokens':>17} {'Latency ms':>11} {'Cost USD':>10}")
    for model in MODEL_REGISTRY:
        provider = providers[model.tier]
        response = await provider.complete(
            [{"role": "user", "content": "Say hello in one sentence."}],
            model,
            max_tokens=32,
        )
        cost = model.cost_per_input_token * max(response.prompt_tokens, 0) + model.cost_per_output_token * max(response.completion_tokens, 0)
        print(f"{model.model_id:<30} {model.tier:<8} {response.prompt_tokens:>14} {response.completion_tokens:>17} {response.latency_ms:>11} {cost:>10.8f}")

    baseline = get_baseline_model()
    print(f"\nBaseline model: {baseline.model_id}")


if __name__ == "__main__":
    asyncio.run(main())
