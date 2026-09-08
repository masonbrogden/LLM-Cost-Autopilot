import sqlite3

from app.evaluator import evaluate_request, should_run_evaluation
from app.logging_db import initialize_db, log_request


def test_evaluate_request_updates_quality_and_escalation(tmp_path, monkeypatch):
    db_path = tmp_path / "autopilot.db"
    initialize_db(str(db_path))
    request_id = "chatcmpl-test-eval"
    log_request(
        db_path=str(db_path),
        request_id=request_id,
        provider="anthropic",
        routed_model="claude-haiku-4-5-20251001",
        routed_tier="low",
        prompt_text="Explain the bug in this code",
        response_text="Short answer",
        actual_cost_usd=0.0004,
        baseline_cost_usd=0.0012,
        savings_usd=0.0008,
    )

    monkeypatch.setenv("LLM_COST_AUTOPILOT_DB", str(db_path))
    monkeypatch.setattr(
        "app.evaluator._judge_answer",
        lambda *args, **kwargs: {"score": 2, "reason": "materially worse"},
    )

    evaluate_request(request_id=request_id, prompt_text="Explain the bug in this code", cheap_answer="Short answer")

    conn = sqlite3.connect(str(db_path))
    row = conn.execute(
        "SELECT quality_score, escalated, judge_model FROM requests WHERE id = ?",
        (request_id,),
    ).fetchone()
    conn.close()

    assert row is not None
    assert row[0] == 2.0
    assert row[1] == 1
    assert row[2] == "claude-opus-5"


def test_should_run_evaluation_respects_daily_spend_ceiling(tmp_path, monkeypatch):
    db_path = tmp_path / "autopilot.db"
    initialize_db(str(db_path))
    monkeypatch.setenv("LLM_COST_AUTOPILOT_DB", str(db_path))
    log_request(
        db_path=str(db_path),
        request_id="chatcmpl-ceiling",
        provider="anthropic",
        routed_model="claude-haiku-4-5-20251001",
        routed_tier="low",
        prompt_text="text",
        response_text="ok",
        actual_cost_usd=9.0,
        baseline_cost_usd=10.0,
        savings_usd=1.0,
    )

    assert should_run_evaluation() is False
