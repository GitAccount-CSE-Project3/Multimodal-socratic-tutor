from __future__ import annotations

import streamlit as st

from src.config.settings import get_settings
from src.core.conversation.state import PHASE_CONFIG

settings = get_settings()

STARTER_PROMPTS = [
    "What does the cerebellum do?",
    "Walk me through the median nerve pathway",
    "Quiz me on the cranial nerves",
    "Why do dermatomes matter in OT?",
]


def phase_topbar() -> None:
    phase = st.session_state.get("phase", "rapport")
    name = st.session_state.get("student_name", "Guest")
    topic = st.session_state.get("current_topic", "")
    hint = st.session_state.get("hint_level", 0)
    max_h = settings.max_hint_turns
    label, _, _ = PHASE_CONFIG.get(phase, ("Active", "", ""))

    col1, col2, col3 = st.columns([4, 2, 2])
    with col1:
        topic_str = f" · {topic.replace('_', ' ').title()}" if topic else ""
        st.write(f"**Anatomy & Neuroscience Tutor** | {name}{topic_str}")
    with col2:
        st.caption(f"Phase: {label}")
    with col3:
        if phase == "tutoring":
            if hint >= max_h:
                st.caption("Reveal unlocked")
            else:
                st.caption(f"Hint {hint}/{max_h}")
    st.divider()


def hint_indicator() -> None:
    phase = st.session_state.get("phase", "rapport")
    hint = st.session_state.get("hint_level", 0)
    max_hints = settings.max_hint_turns

    if phase != "tutoring" or not st.session_state.get("show_hints", True):
        return

    if hint >= max_hints:
        st.info("Answer reveal unlocked", icon=None)
    else:
        remaining = max_hints - hint
        st.caption(f"Hints used: {hint}/{max_hints} — {remaining} remaining")


def starter_suggestions(send_fn) -> None:
    st.write("Not sure where to start? Try one of these:")
    cols = st.columns(2)
    for i, prompt in enumerate(STARTER_PROMPTS):
        with cols[i % 2]:
            if st.button(prompt, key=f"starter_{i}", width="stretch"):
                send_fn(prompt)
