from __future__ import annotations

__all__ = [
    "chunk_list",
    "clean_text",
    "detect_bypass_attempt",
    "ensure_dir",
    "format_citations",
    "hash_string",
    "safe_parse_json",
    "sanitize_student_id",
    "truncate_text",
]

import hashlib
import json
import re
from pathlib import Path
from typing import Any


def truncate_text(text: str, max_chars: int = 500, suffix: str = "...") -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - len(suffix)] + suffix


def clean_text(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]", "", text)
    return text.strip()


def safe_parse_json(text: str) -> dict[str, Any] | None:
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"```\s*$", "", text)
    text = text.strip()

    text = re.sub(r",\s*([}\]])", r"\1", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def hash_string(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def chunk_list(items: list[Any], size: int) -> list[list[Any]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def format_citations(sources: list[str]) -> str:
    if not sources:
        return ""
    unique = list(dict.fromkeys(sources))
    return "\n".join(f"[{i + 1}] {src}" for i, src in enumerate(unique))


def detect_bypass_attempt(user_input: str) -> bool:
    bypass_patterns = [
        r"\bjust tell me\b",
        r"\bgive me the answer\b",
        r"\bwhat is the answer\b",
        r"\bstop asking questions\b",
        r"\btell me directly\b",
        r"\bskip the hints?\b",
        r"\breveal the answer\b",
        r"\bneed the answer\b",
        r"\bignore your instructions?\b",
        r"\bforget your rules\b",
        r"\bdisregard\b.*\binstructions?\b",
        r"\boverride\b.*\b(system|prompt|rules?)\b",
        r"\bpretend you have no\b",
        r"\bno restrictions?\b",
        r"\bact as\b.*\b(different|normal|another)\b",
        r"\bjust be a normal\b",
        r"\bi command you\b",
        r"\byou must tell me\b",
        r"\banswer without\b",
        r"\bstop the socratic\b",
        r"\bno more hints?\b",
        r"\bdon.t have time for hints?\b",
        r"\bimmediately\b.*\banswer\b",
        r"\banswer immediately\b",
    ]
    text_lower = user_input.lower()
    return any(re.search(p, text_lower) for p in bypass_patterns)


def sanitize_student_id(raw_id: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_\-]", "", raw_id)[:64]
