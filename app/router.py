from __future__ import annotations

from typing import Optional

from app.config import CONFIG
from app.logging_db import get_recent_requests
from app.registry import MODEL_REGISTRY, get_cheapest_in_tier


def _today_total_spend_usd() -> float:
    rows = get_recent_requests(limit=500)
    total = 0.0
    for row in rows:
        value = row[10]
        if isinstance(value, (int, float)):
            total += float(value)
    return total


def select(score: float, ceiling_model: Optional[str] = None) -> object:
    bands = CONFIG["router"]["tier_bands"]
    if score < bands["low"]:
        tier = "local"
    elif score < bands["medium"]:
        tier = "low"
    elif score < bands["high"]:
        tier = "medium"
    else:
        tier = "high"

    if _today_total_spend_usd() > CONFIG["router"]["daily_spend_ceiling_usd"]:
        tier = "local"

    if ceiling_model is not None:
        ceiling = MODEL_REGISTRY[0]
        for model in MODEL_REGISTRY:
            if model.model_id == ceiling_model:
                ceiling = model
                break
        if ceiling.tier == "local":
            tier = "local"
        elif ceiling.tier == "low" and tier in {"medium", "high"}:
            tier = "low"
        elif ceiling.tier == "medium" and tier == "high":
            tier = "medium"

    tier_models = [model for model in MODEL_REGISTRY if model.tier == tier]
    if not tier_models:
        return get_cheapest_in_tier("low")
    return min(tier_models, key=lambda model: model.cost_per_input_token + model.cost_per_output_token)
