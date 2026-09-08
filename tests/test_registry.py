from app.registry import (
    MODEL_REGISTRY,
    get_baseline_model,
    get_cheapest_in_tier,
    get_models_by_tier,
)


def test_registry_contains_expected_tiers():
    tiers = {entry.tier for entry in MODEL_REGISTRY}
    assert {"local", "low", "medium", "high"}.issubset(tiers)


def test_get_models_by_tier_supports_legacy_aliases():
    assert get_models_by_tier("low")
    assert get_models_by_tier("cheap")
    assert get_models_by_tier("medium")
    assert get_models_by_tier("mid")


def test_get_cheapest_in_tier_returns_lowest_cost_model():
    cheapest = get_cheapest_in_tier("low")
    assert cheapest is not None
    assert cheapest.model_id == "claude-haiku-4-5-20251001"


def test_get_baseline_model_is_opus():
    baseline = get_baseline_model()
    assert baseline is not None
    assert baseline.model_id == "claude-opus-5"
