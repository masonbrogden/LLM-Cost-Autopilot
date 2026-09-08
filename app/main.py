from __future__ import annotations

import os
from typing import Dict

from fastapi import FastAPI

from app.logging_db import initialize_db
from app.registry import get_baseline_model
from app.routes.proxy import router


def validate_runtime_config() -> None:
    api_key = (os.getenv("ANTHROPIC_API_KEY") or "").strip()
    placeholder_values = {
        "",
        "your_api_key_here",
        "sk-ant-paste-your-key-here",
        "changeme",
    }
    if api_key.lower() in {value.lower() for value in placeholder_values}:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is unset or still set to a placeholder value. "
            "Set a real key before starting the app."
        )


app = FastAPI(title="LLM Cost Autopilot")
app.include_router(router)


@app.on_event("startup")
def startup() -> None:
    validate_runtime_config()
    initialize_db(os.getenv("LLM_COST_AUTOPILOT_DB", "data/autopilot.db"))


@app.get("/health")
def health() -> Dict[str, object]:
    providers = {
        "ollama": False,
        "anthropic": bool(os.getenv("ANTHROPIC_API_KEY")),
    }
    try:
        providers["ollama"] = bool(os.getenv("OLLAMA_BASE_URL"))
    except Exception:
        pass
    return {"status": "ok", "providers": providers, "baseline_model": get_baseline_model().model_id}
