"""Base provider interfaces and response DTOs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class StandardResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int
    model_id: str
    provider: str
    raw: Dict[str, Any] = field(default_factory=dict)


class Provider(ABC):
    @abstractmethod
    async def complete(
        self,
        messages: List[Dict[str, str]],
        model_config: Any,
        max_tokens: Optional[int] = None,
    ) -> StandardResponse:
        raise NotImplementedError
