from app.classifier import score
from app.router import select
from scripts.seed_prompts import SEED_PROMPTS


def test_classifier_returns_score_in_range():
    result = score([{"role": "user", "content": "Explain why this function is slow."}])
    assert 0.0 <= result <= 1.0


def test_router_reports_seed_agreement_percentage():
    matches = 0
    mismatches = []

    for seed in SEED_PROMPTS:
        prompt = seed["prompt"]
        expected_tier = seed["expected_tier"]
        s = score([{"role": "user", "content": prompt}])
        chosen = select(s)
        if chosen.tier == expected_tier:
            matches += 1
        else:
            mismatches.append({
                "prompt": prompt,
                "expected": expected_tier,
                "actual": chosen.tier,
                "score": s,
            })

    agreement = matches / len(SEED_PROMPTS)
    print(f"Seed agreement: {agreement:.0%}")
    if mismatches:
        print("Mismatches:")
        for mismatch in mismatches:
            print(mismatch)

    assert agreement >= 0.70
