from __future__ import annotations

import re

_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "must", "shall", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "and", "or", "but", "not", "this",
    "that", "it", "its", "they", "them", "their", "which", "who", "what",
    "how", "when", "where", "why", "all", "also", "as", "into",
}


def keyword_overlap(answer: str, context: str) -> float:
    def extract(text: str) -> set:
        words = re.findall(r"\b[a-zA-Z]{4,}\b", text.lower())
        return {w for w in words if w not in _STOPWORDS}

    answer_kw = extract(answer)
    context_kw = extract(context)
    if not answer_kw:
        return 0.0
    return len(answer_kw & context_kw) / len(answer_kw)


def find_unsupported_phrases(answer: str, context: str) -> list[str]:
    sentences = [s.strip() for s in re.split(r"[.!?]", answer) if len(s.strip()) > 20]
    context_lower = context.lower()
    flagged = []
    for sentence in sentences[:10]:
        nouns = re.findall(r"\b[A-Z][a-z]{3,}\b", sentence)
        if nouns and not any(n.lower() in context_lower for n in nouns):
            flagged.append(sentence[:80])
    return flagged[:3]
