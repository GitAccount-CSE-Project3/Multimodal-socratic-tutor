from __future__ import annotations

TOPIC_KEYWORDS: dict[str, str] = {
    "cerebellum": "cerebellum",
    "cranial": "cranial_nerves",
    "facial nerve": "cranial_nerves",
    "median nerve": "peripheral_nervous_system",
    "ulnar": "peripheral_nervous_system",
    "radial nerve": "peripheral_nervous_system",
    "brachial": "peripheral_nervous_system",
    "hand": "hand_anatomy",
    "carpal": "hand_anatomy",
    "spinal cord": "spinal_cord",
    "dermatome": "spinal_cord",
    "basal ganglia": "basal_ganglia",
}


def detect_topic(text: str) -> str | None:
    lower = text.lower()
    for kw, topic in TOPIC_KEYWORDS.items():
        if kw in lower:
            return topic
    return None
