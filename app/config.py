CONFIG = {
    "classifier": {
        "prompt_length": {"weight": 0.12, "threshold": 180},
        "code_block": {"weight": 0.18, "threshold": 0.5},
        "question_marks": {"weight": 0.06, "threshold": 1},
        "reasoning_verbs": {"weight": 0.10, "threshold": 1},
        "task_categories": {
            "local": {"weight": 0.12, "threshold": 1},
            "low": {"weight": 0.36, "threshold": 1},
            "medium": {"weight": 0.62, "threshold": 1},
            "high": {"weight": 0.90, "threshold": 1},
        },
        "structured_output": {"weight": 0.10, "threshold": 0.5},
        "turn_depth": {"weight": 0.08, "threshold": 2},
    },
    "router": {
        "tier_bands": {
            "local": 0.0,
            "low": 0.40,
            "medium": 0.70,
            "high": 1.0,
        },
        "daily_spend_ceiling_usd": 2.0,
    },
    "evaluator": {
        "sample_rate": 0.10,
        "quality_threshold": 3,
        "judge_model": "claude-opus-5",
    },
}
