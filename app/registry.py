from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class ModelConfig:
    provider: str
    model_id: str
    tier: str
    display_name: str
    cost_per_input_token: float
    cost_per_output_token: float
    max_context: int


# Source: https://platform.claude.com/docs/en/about-claude/pricing
# Verified: 2026-09-07
# Prices are given in USD per million tokens and converted to per-token values.
MODEL_REGISTRY: List[ModelConfig] = [
    ModelConfig(
        provider="ollama",
        model_id="llama3.1:8b",
        tier="local",
        display_name="Ollama llama3.1:8b",
        cost_per_input_token=0.0 / 1_000_000,
        cost_per_output_token=0.0 / 1_000_000,
        max_context=8_192,
    ),
    ModelConfig(
        provider="anthropic",
        model_id="claude-haiku-4-5-20251001",
        tier="low",
        display_name="Claude Haiku",
        cost_per_input_token=1.00 / 1_000_000,
        cost_per_output_token=5.00 / 1_000_000,
        max_context=200_000,
    ),
    ModelConfig(
        provider="anthropic",
        model_id="claude-sonnet-5",
        tier="medium",
        display_name="Claude Sonnet 5",
        cost_per_input_token=2.00 / 1_000_000,
        cost_per_output_token=10.00 / 1_000_000,
        max_context=200_000,
    ),
    ModelConfig(
        provider="anthropic",
        model_id="claude-opus-5",
        tier="high",
        display_name="Claude Opus 5",
        cost_per_input_token=5.00 / 1_000_000,
        cost_per_output_token=25.00 / 1_000_000,
        max_context=200_000,
    ),
]


def get_models_by_tier(tier: str) -> List[ModelConfig]:
    normalized = tier.lower()
    aliases = {
        "cheap": "low",
        "low": "low",
        "mid": "medium",
        "medium": "medium",
        "premium": "high",
        "high": "high",
        "local": "local",
    }
    target = aliases.get(normalized, normalized)
    return [model for model in MODEL_REGISTRY if model.tier == target]


def get_cheapest_in_tier(tier: str) -> Optional[ModelConfig]:
    models = get_models_by_tier(tier)
    if not models:
        return None
    return min(models, key=lambda model: (model.cost_per_input_token + model.cost_per_output_token))


def get_baseline_model() -> ModelConfig:
    return get_models_by_tier("high")[0]
