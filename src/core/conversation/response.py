from __future__ import annotations

from dataclasses import dataclass, field

from src.core.conversation.state import HintLevel


@dataclass
class SocraticResponse:

    content: str
    hint_level: HintLevel
    is_bypass_redirect: bool
    is_reveal: bool
    retrieved_context: str
    citations: list[str]
    topic_detected: str | None
    is_image_response: bool = False
    identified_structures: list[str] = field(default_factory=list)
