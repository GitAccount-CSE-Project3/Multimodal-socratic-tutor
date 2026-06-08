from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

INDIGO = "#6366F1"
NAVY = "#E2E8F0"
MUTED = "#94A3B8"
GRID = "#2A3650"


def score_palette(score: float) -> tuple[str, str, str]:
    if score >= 80:
        return "#34D399", "rgba(52,211,153,.16)", "#6EE7B7"
    if score >= 60:
        return "#FBBF24", "rgba(251,191,36,.16)", "#FBBF24"
    return "#F87171", "rgba(248,113,113,.16)", "#F87171"


def mastery_bar(topic: str, score: float) -> None:
    c1, c2, c3 = st.columns([3, 5, 1])
    with c1:
        st.write(topic.replace("_", " ").title())
    with c2:
        st.progress(int(score) / 100)
    with c3:
        st.write(f"{int(score)}%")


def radar_chart(scores: dict[str, float]) -> None:
    topics = [t.replace("_", " ").title() for t in scores]
    values = [float(v) for v in scores.values()]
    theta = topics + topics[:1]
    r = values + values[:1]

    fig = go.Figure(
        go.Scatterpolar(
            r=r, theta=theta, fill="toself",
            line={"color": INDIGO, "width": 2},
            fillcolor="rgba(99,102,241,0.22)",
            hovertemplate="%{theta}: %{r:.0f}%<extra></extra>",
        )
    )
    fig.update_layout(
        polar={
            "radialaxis": {"range": [0, 100], "tickvals": [25, 50, 75, 100], "gridcolor": GRID, "tickfont": {"size": 10, "color": MUTED}},
            "angularaxis": {"gridcolor": GRID, "tickfont": {"size": 11, "color": NAVY}},
            "bgcolor": "rgba(0,0,0,0)",
        },
        showlegend=False, height=300,
        margin={"l": 50, "r": 50, "t": 20, "b": 20},
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
