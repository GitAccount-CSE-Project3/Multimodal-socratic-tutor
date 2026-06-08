
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import streamlit as st

from src.app.views._charts import mastery_bar, radar_chart


def _section_title(text: str) -> None:
    st.subheader(text, divider=False)


def _load_memory_scores(student_id: str) -> dict[str, float]:
    try:
        import asyncio
        from src.core.memory.student_memory import StudentMemory
        return asyncio.run(StudentMemory().load(student_id)).mastery_scores
    except Exception:
        return {}


def render() -> None:
    name = st.session_state.get("student_name", "")
    student_id = st.session_state.get("student_id", "student_demo")
    topics = st.session_state.get("topics_covered", [])
    session_scores = st.session_state.get("mastery_scores", {})
    turns = st.session_state.get("turn_count", 0)
    phase = st.session_state.get("phase", "rapport")

    memory_scores = _load_memory_scores(student_id)
    scores = {**memory_scores, **session_scores}
    has_data = bool(scores)
    avg_score = sum(scores.values()) / len(scores) if scores else 0
    weak_topics = [t for t, s in scores.items() if s < 60]
    strong_count = sum(1 for s in scores.values() if s >= 80)

    st.header("Student Dashboard")
    st.caption(f"Performance overview · {name or 'Guest'}")
    st.divider()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Overall mastery", f"{int(avg_score)}%" if has_data else "—")
    c2.metric("Topics covered", len(scores) if scores else len(topics))
    c3.metric("Weak areas", len(weak_topics) if has_data else "—", delta=f"-{len(weak_topics)}" if weak_topics else None, delta_color="inverse")
    c4.metric("Turns this session", turns)

    st.write("")
    col_left, col_right = st.columns([3, 2], gap="large")

    with col_left:
        with st.container(border=True):
            _section_title("Topic mastery")
            if not has_data:
                st.info("No mastery data yet. Complete tutoring sessions — scores will appear here.", icon=None)
            else:
                if len(scores) >= 4:
                    radar_chart(scores)
                st.caption(f"{strong_count} strong (>=80%) · {len(weak_topics)} need work")
                for topic, score in sorted(scores.items(), key=lambda x: -x[1]):
                    mastery_bar(topic, score)

    with col_right:
        with st.container(border=True):
            _section_title("Needs revision")
            if not has_data:
                st.caption("No data yet.")
            elif not weak_topics:
                st.success("No weak areas — great work!")
            else:
                for topic in weak_topics:
                    st.warning(topic.replace("_", " ").title())

        with st.container(border=True):
            _section_title("Memory summary")
            if memory_scores:
                st.caption(f"Topics tracked: **{len(memory_scores)}**")
                best = max(memory_scores, key=memory_scores.get)
                st.caption(f"Strongest: **{best.replace('_', ' ').title()}** ({int(memory_scores[best])}%)")
                if weak_topics:
                    st.caption(f"Priority: **{weak_topics[0].replace('_', ' ').title()}**")
            else:
                st.caption("No cross-session memory yet.")
            phase_labels = {"rapport": "Building rapport", "tutoring": "Socratic tutoring", "assessment": "Clinical assessment", "mastery": "Mastery summary"}
            st.caption(f"Phase · **{phase_labels.get(phase, phase.title())}**")
            st.caption(f"Turns · **{turns}**")

    st.write("")
    with st.container(border=True):
        _section_title("All topics")
        if not scores:
            st.caption("No topics yet — begin a tutoring session.")
        else:
            import pandas as pd
            rows = []
            for t, s in sorted(scores.items(), key=lambda x: -x[1]):
                status = "Strong" if s >= 80 else ("Developing" if s >= 60 else "Needs work")
                rows.append({"Topic": t.replace("_", " ").title(), "Score": f"{int(s)}%", "Status": status, "Source": "This session" if t in session_scores else "Memory"})
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    st.write("")
    _section_title("Quick actions")
    ca, cb, cc = st.columns(3)
    with ca:
        if st.button("Continue tutoring", width="stretch"):
            st.session_state.current_page = "chat"
            st.rerun()
    with cb:
        if st.button("Analyse an image", width="stretch"):
            st.session_state.current_page = "images"
            st.rerun()
    with cc:
        if st.button("Reset session", width="stretch"):
            for key in ["messages", "phase", "turn_count", "hint_level", "current_topic", "topics_covered", "mastery_scores", "vision_result"]:
                st.session_state.pop(key, None)
            st.rerun()
