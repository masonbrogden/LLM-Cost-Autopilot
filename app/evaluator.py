from __future__ import annotations

import asyncio
import inspect
import json
import os
import random
from typing import Any, Dict, Optional

from fastapi import BackgroundTasks

from app.config import CONFIG
from app.logging_db import get_recent_requests
from app.providers.paid import AnthropicProvider
from app.registry import get_baseline_model


def _today_total_spend_usd(db_path: Optional[str] = None) -> float:
    rows = get_recent_requests(db_path=db_path, limit=500)
    total = 0.0
    for row in rows:
        value = row[10]
        if isinstance(value, (int, float)):
            total += float(value)
    return total


def should_run_evaluation(db_path: Optional[str] = None) -> bool:
    target_db = db_path or os.getenv("LLM_COST_AUTOPILOT_DB")
    if _today_total_spend_usd(target_db) > CONFIG["router"]["daily_spend_ceiling_usd"]:
        return False
    sample_rate = float(CONFIG["evaluator"].get("sample_rate", 0.10))
    return random.random() < sample_rate


async def _judge_answer(prompt_text: str, cheap_answer: str, judge_model: str = "claude-opus-5") -> Dict[str, Any]:
    baseline = get_baseline_model()
    provider = AnthropicProvider(api_key=os.getenv("ANTHROPIC_API_KEY"))
    judge_prompt = (
        "You are scoring whether the cheap answer was materially worse than a strong answer. "
        "Return only valid JSON with keys 'score' and 'reason'. "
        "The score must be an integer from 1 to 5, where 1 means materially worse and 5 means equivalent or better. "
        "Do not include markdown fences or any extra text.\n\n"
        f"Prompt:\n{prompt_text}\n\nCheap answer:\n{cheap_answer}"
    )
    response = await provider.complete(
        [{"role": "user", "content": judge_prompt}],
        baseline,
        max_tokens=128,
    )
    cleaned = response.text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`\n ")
    payload = json.loads(cleaned)
    score = int(payload["score"])
    reason = str(payload.get("reason", ""))
    return {"score": score, "reason": reason}


def evaluate_request(
    request_id: str,
    prompt_text: str,
    cheap_answer: str,
    judge_model: str = "claude-opus-5",
    db_path: Optional[str] = None,
) -> Dict[str, Any]:
    target_db = db_path or os.getenv("LLM_COST_AUTOPILOT_DB", "data/autopilot.db")

    judge_fn = _judge_answer
    result = judge_fn(prompt_text, cheap_answer, judge_model=judge_model)
    if inspect.isawaitable(result):
        result = asyncio.run(result)
    if not isinstance(result, dict):
        result = {"score": 3, "reason": "judge returned an unexpected payload"}
    score_value = int(result.get("score", 3))

    escalated = 1 if score_value < CONFIG["evaluator"].get("quality_threshold", 3) else 0

    conn = __import__("sqlite3").connect(target_db)
    try:
        conn.execute(
            "UPDATE requests SET quality_score = ?, escalated = ?, judge_model = ? WHERE id = ? OR request_id = ?",
            (float(score_value), escalated, judge_model, request_id, request_id),
        )
        conn.commit()
    finally:
        conn.close()

    return {"score": score_value, "reason": result.get("reason", ""), "escalated": escalated}


def schedule_evaluation(
    request_id: str,
    prompt_text: str,
    cheap_answer: str,
    judge_model: str = "claude-opus-5",
    db_path: Optional[str] = None,
    background_tasks: Optional[BackgroundTasks] = None,
) -> None:
    target_db = db_path or os.getenv("LLM_COST_AUTOPILOT_DB", "data/autopilot.db")
    if not should_run_evaluation(target_db):
        return

    if background_tasks is not None:
        background_tasks.add_task(evaluate_request, request_id, prompt_text, cheap_answer, judge_model, db_path)
        return

    asyncio.create_task(
        _run_background_evaluation(
            request_id=request_id,
            prompt_text=prompt_text,
            cheap_answer=cheap_answer,
            judge_model=judge_model,
            db_path=db_path,
        )
    )


async def _run_background_evaluation(
    request_id: str,
    prompt_text: str,
    cheap_answer: str,
    judge_model: str = "claude-opus-5",
    db_path: Optional[str] = None,
) -> None:
    evaluate_request(
        request_id=request_id,
        prompt_text=prompt_text,
        cheap_answer=cheap_answer,
        judge_model=judge_model,
        db_path=db_path,
    )
