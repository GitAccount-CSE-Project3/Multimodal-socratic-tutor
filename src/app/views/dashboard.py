
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import plotly.graph_objects as go
import streamlit as st

INDIGO = "#6366F1"
NAVY = "#E2E8F0"
MUTED = "#94A3B8"
GRID = "#2A3650"


def _score_palette(score: float) -> tuple[str, str, str]:
    if score >= 80:
        return "#34D399", "rgba(52,211,153,.16)", "#6EE7B7"
    if score >= 60:
        return "#FBBF24", "rgba(251,191,36,.16)", "#FBBF24"
    return "#F87171", "rgba(248,113,113,.16)", "#F87171"


def _section_title(text: str) -> None:
    st.subheader(text, divider=False)


def _mastery_bar(topic: str, score: float) -> None:
    c1, c2, c3 = st.columns([3, 5, 1])
    with c1:
        st.write(topic.replace("_", " ").title())
    with c2:
        st.progress(int(score) / 100)
    with c3:
        st.write(f"{int(score)}%")


def _radar_chart(scores: dict[str, float]) -> None:
    topics = [t.replace("_", " ").title() for t in scores]
    values = [float(v) for v in scores.values()]
    theta = topics + topics[:1]
    r = values + values[:1]

    fig = go.Figure(
        go.Scatterpolar(
            r=r,
            theta=theta,
            fill="toself",
            line={"color": INDIGO, "width": 2},
            fillcolor="rgba(99,102,241,0.22)",
            hovertemplate="%{theta}: %{r:.0f}%<extra></extra>",
        )
    )
    fig.update_layout(
        polar={
            "radialaxis": {
                "range": [0, 100],
                "tickvals": [25, 50, 75, 100],
                "gridcolor": GRID,
                "tickfont": {"size": 10, "color": MUTED},
            },
            "angularaxis": {"gridcolor": GRID, "tickfont": {"size": 11, "color": NAVY}},
            "bgcolor": "rgba(0,0,0,0)",
        },
        showlegend=False,
        height=300,
        margin={"l": 50, "r": 50, "t": 20, "b": 20},
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})


def _load_memory_scores(student_id: str) -> dict[str, float]:
    """Load long-term mastery scores from StudentMemory."""
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
    with c1:
        st.metric("Overall mastery", f"{int(avg_score)}%" if has_data else "—")
    with c2:
        st.metric("Topics covered", len(scores) if scores else len(topics))
    with c3:
        st.metric(
            "Weak areas",
            len(weak_topics) if has_data else "—",
            delta=f"-{len(weak_topics)}" if weak_topics else None,
            delta_color="inverse",
        )
    with c4:
        st.metric("Turns this session", turns)

    st.write("")

    col_left, col_right = st.columns([3, 2], gap="large")

    with col_left:
        with st.container(border=True):
            _section_title("Topic mastery")
            if not has_data:
                st.info(
                    "No mastery data yet. Complete tutoring sessions and "
                    "assessments — scores will appear here.",
                    icon="📊",
                )
            else:
                if len(scores) >= 4:
                    _radar_chart(scores)
                st.caption(f"{strong_count} strong (≥80%) · {len(weak_topics)} need work")
                for topic, score in sorted(scores.items(), key=lambda x: -x[1]):
                    _mastery_bar(topic, score)

    with col_right:
        with st.container(border=True):
            _section_title("Needs revision")
            if not has_data:
                st.caption("No data yet.")
            elif not weak_topics:
                st.success("No weak areas — great work!")
            else:
                for topic in weak_topics:
                    st.warning(topic.replace("_", " ").title(), icon="⚠️")

        with st.container(border=True):
            _section_title("Memory summary")
            if memory_scores:
                st.caption(f"Topics tracked: **{len(memory_scores)}**")
                best = max(memory_scores, key=memory_scores.get)
                st.caption(
                    f"Strongest: **{best.replace('_', ' ').title()}** ({int(memory_scores[best])}%)"
                )
                if weak_topics:
                    st.caption(f"Priority: **{weak_topics[0].replace('_', ' ').title()}**")
            else:
                st.caption("No cross-session memory yet.")
            phase_labels = {
                "rapport": "Building rapport",
                "tutoring": "Socratic tutoring",
                "assessment": "Clinical assessment",
                "mastery": "Mastery summary",
            }
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
                if s >= 80:
                    status = "Strong"
                elif s >= 60:
                    status = "Developing"
                else:
                    status = "Needs work"
                source = "This session" if t in session_scores else "Memory"
                rows.append(
                    {
                        "Topic": t.replace("_", " ").title(),
                        "Score": f"{int(s)}%",
                        "Status": status,
                        "Source": source,
                    }
                )
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)

    st.write("")

    _section_title("Quick actions")
    ca, cb, cc = st.columns(3)
    with ca:
        if st.button("💬  Continue tutoring", width="stretch"):
            st.session_state.current_page = "chat"
            st.rerun()
    with cb:
        if st.button("🖼️  Analyse an image", width="stretch"):
            st.session_state.current_page = "images"
            st.rerun()
    with cc:
        if st.button("🔄  Reset session", width="stretch"):
            for key in [
                "messages",
                "phase",
                "turn_count",
                "hint_level",
                "current_topic",
                "topics_covered",
                "mastery_scores",
                "vision_result",
            ]:
                st.session_state.pop(key, None)
            st.rerun()
