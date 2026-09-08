from __future__ import annotations

import re
from typing import Any, Dict, List

from app.config import CONFIG


REASONING_VERBS = {
    "analyze", "compare", "debug", "explain", "refactor", "design", "investigate",
    "trace", "summarize", "classify", "describe", "generate", "draft", "plan",
    "recommend", "propose", "evaluate", "migrate", "diagnose", "optimize",
    "review", "estimate", "reason", "build", "write", "find", "fix", "parse",
    "query", "architecture", "outage", "support", "risk", "benchmark", "schema",
    "retry", "validate", "sanitize", "migration", "rewrite", "translate", "convert",
    "format", "normalize", "bug", "incident", "issue", "fail", "production"
}


TRIVIAL_KEYWORDS = {
    "capitalization", "title case", "grammar", "spell-check", "rewrite this", "polite",
    "subject line", "format this phone number", "markdown bullet list", "json object",
    "normalize this", "translate this slang", "correct the ordering of words", "what does",
    "meaning of", "in plain english", "shorten this sentence", "remove repeated words",
    "active voice", "formal tone", "capitalise", "check whether this sentence is grammatical",
    "convert this csv", "rewrite paragraph", "write a subject line"
}
LOW_KEYWORDS = {
    "summarize", "classify", "describe", "draft", "generate", "explain", "compare",
    "customer review", "release note", "faq", "status update", "support message", "risk level",
    "webhook", "onboarding checklist", "launch email", "short summary", "key takeaways",
    "product announcement", "business terms", "recap", "recommend", "propose",
    "differences between", "one sentence", "brief", "pros and cons", "three ways",
    "what are the main differences between", "what is the difference between", "summary of",
    "simple business terms", "progress update", "changelog note", "opinion or a fact",
    "release summary", "benefits of", "tradeoff", "what are the main differences", "issue"
}
MEDIUM_KEYWORDS = {
    "debug", "refactor", "python function", "regex", "sql", "schema", "unit test",
    "pagination", "retry policy", "async job", "api contract", "flask route",
    "race condition", "deadlock", "migration plan", "validate", "data pipeline",
    "database", "event-driven", "service contract", "query", "bug", "error", "function",
    "api", "route", "integration", "test around", "flaky", "partition", "retry"
}
HIGH_KEYWORDS = {
    "architecture", "distributed", "multi-file", "production", "outage", "monolith",
    "microservices", "service mesh", "cluster", "partitioned", "multi-region", "clock skew",
    "high availability", "migrate", "system", "stale sessions", "authentication bug",
    "correctness guarantees", "eventual consistency", "root cause", "production issue",
    "latency-sensitive", "reliability", "failover", "capacity planning", "consistency",
    "distributed transaction", "inventory counts", "service boundary", "eventual consistency"
}


def _count_keywords(text: str, keywords: set[str]) -> int:
    return sum(1 for keyword in keywords if keyword in text)


def score(messages: List[Dict[str, Any]]) -> float:
    """Heuristic scoring only: there is no labeled data yet, so this classifier generates the labels it will later learn from.
    """
    text = "\n".join(msg.get("content", "") for msg in messages)
    text_lower = text.lower()
    total_prompt_length = len(text)
    code_like = bool(re.search(r"```|def |class |import |SELECT |FROM |\{|\bif\s+.*:\s*$|async def |return .*\{|\bfunction\b|\bquery\b|\bfor\s+.*in\s+.*:\s*$", text, flags=re.I | re.M))
    question_mark_count = text.count("?")
    reasoning_hits = sum(1 for word in REASONING_VERBS if word in text_lower)
    reasoning_signal = min(1.0, reasoning_hits / 3)

    cfg = CONFIG["classifier"]
    trivial_hits = _count_keywords(text_lower, TRIVIAL_KEYWORDS)
    low_hits = _count_keywords(text_lower, LOW_KEYWORDS)
    medium_hits = _count_keywords(text_lower, MEDIUM_KEYWORDS)
    high_hits = _count_keywords(text_lower, HIGH_KEYWORDS)

    category_signal = 0.0
    if high_hits:
        category_signal = cfg["task_categories"]["high"]["weight"]
    elif medium_hits or code_like:
        category_signal = cfg["task_categories"]["medium"]["weight"]
    elif low_hits:
        category_signal = cfg["task_categories"]["low"]["weight"]
    elif trivial_hits:
        category_signal = cfg["task_categories"]["local"]["weight"]

    score_value = (
        min(1.0, total_prompt_length / cfg["prompt_length"]["threshold"]) * cfg["prompt_length"]["weight"]
        + (1.0 if code_like else 0.0) * cfg["code_block"]["weight"]
        + min(1.0, question_mark_count / cfg["question_marks"]["threshold"]) * cfg["question_marks"]["weight"]
        + reasoning_signal * cfg["reasoning_verbs"]["weight"]
        + category_signal
        + (1.0 if bool(re.search(r"json|yaml|schema|return .*\{|list .*keys|fields?|acceptance criteria|ticket description|summary", text, flags=re.I)) else 0.0) * cfg["structured_output"]["weight"]
        + min(1.0, len(messages) / cfg["turn_depth"]["threshold"]) * cfg["turn_depth"]["weight"]
    )
    return max(0.0, min(1.0, score_value))
