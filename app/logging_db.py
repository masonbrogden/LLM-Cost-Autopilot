from __future__ import annotations

import os
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

DB_PATH = os.getenv("LLM_COST_AUTOPILOT_DB", str(Path("data") / "autopilot.db"))


def initialize_db(db_path: Optional[str] = None) -> str:
    path = db_path or DB_PATH
    base_dir = Path(path).parent
    base_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS requests (
            id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            request_id TEXT,
            provider TEXT,
            routed_model TEXT,
            routed_tier TEXT,
            prompt_text TEXT,
            response_text TEXT,
            prompt_tokens INTEGER,
            completion_tokens INTEGER,
            actual_cost_usd REAL,
            baseline_cost_usd REAL,
            savings_usd REAL,
            error TEXT,
            escalated INTEGER DEFAULT 0,
            quality_score REAL,
            judge_model TEXT,
            model_version TEXT
        )
        """
    )
    conn.commit()
    conn.close()
    return path


def log_request(
    db_path: Optional[str] = None,
    request_id: Optional[str] = None,
    provider: Optional[str] = None,
    routed_model: Optional[str] = None,
    routed_tier: Optional[str] = None,
    prompt_text: Optional[str] = None,
    response_text: Optional[str] = None,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    actual_cost_usd: Optional[float] = None,
    baseline_cost_usd: Optional[float] = None,
    savings_usd: Optional[float] = None,
    error: Optional[str] = None,
    escalated: int = 0,
    quality_score: Optional[float] = None,
    judge_model: Optional[str] = None,
    model_version: Optional[str] = None,
) -> str:
    request_key = request_id or f"chatcmpl-{uuid.uuid4().hex}"
    path = initialize_db(db_path or DB_PATH)
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            """
            INSERT INTO requests (
                id, created_at, request_id, provider, routed_model, routed_tier,
                prompt_text, response_text, prompt_tokens, completion_tokens,
                actual_cost_usd, baseline_cost_usd, savings_usd, error,
                escalated, quality_score, judge_model, model_version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                request_key,
                datetime.utcnow().isoformat(timespec="seconds"),
                request_key,
                provider,
                routed_model,
                routed_tier,
                prompt_text,
                response_text,
                prompt_tokens,
                completion_tokens,
                actual_cost_usd,
                baseline_cost_usd,
                savings_usd,
                error,
                escalated,
                quality_score,
                judge_model,
                model_version,
            ),
        )
        conn.commit()
        return request_key
    finally:
        conn.close()


def get_request_count(db_path: Optional[str] = None) -> int:
    path = db_path or initialize_db(DB_PATH)
    conn = sqlite3.connect(path)
    try:
        row = conn.execute("SELECT COUNT(*) FROM requests").fetchone()
        return int(row[0] if row else 0)
    finally:
        conn.close()


def get_recent_requests(db_path: Optional[str] = None, limit: int = 10):
    path = db_path or initialize_db(DB_PATH)
    conn = sqlite3.connect(path)
    try:
        return conn.execute(
            "SELECT * FROM requests ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    finally:
        conn.close()
