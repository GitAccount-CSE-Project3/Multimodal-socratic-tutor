from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import streamlit as st

TOPICS = {
    "cerebellum": "Cerebellum",
    "cranial_nerves": "Cranial Nerves",
    "peripheral_nervous_system": "Peripheral Nervous System",
    "hand_anatomy": "Hand Anatomy",
    "spinal_cord": "Spinal Cord",
}

DIFFICULTIES = ["beginner", "intermediate", "advanced"]


def _generate_scenario(topic: str, difficulty: str) -> dict:
    async def _run() -> object:
        from src.core.assessment.scenario_generator import ClinicalScenarioGenerator
        gen = ClinicalScenarioGenerator()
        return await gen.generate(topic, difficulty)

    try:
        return asyncio.run(_run())
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(_run())
        loop.close()
        return result


def _evaluate_response(student_response: str, scenario: object) -> object:
    async def _run() -> object:
        from src.core.assessment.reasoning_evaluator import ReasoningEvaluator
        ev = ReasoningEvaluator()
        return await ev.evaluate(student_response, scenario)

    try:
        return asyncio.run(_run())
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(_run())
        loop.close()
        return result


def _score_bar(label: str, value: int, max_val: int, color: str) -> None:
    pct = value / max_val if max_val > 0 else 0
    st.markdown(
        f"""
<div style="margin-bottom:10px">
  <div style="display:flex;justify-content:space-between;margin-bottom:3px">
    <span style="font-size:13px;color:#334155">{label}</span>
    <span style="font-size:13px;font-weight:600;color:#1e293b">{value}/{max_val}</span>
  </div>
  <div style="background:#e2e8f0;border-radius:6px;height:10px">
    <div style="width:{pct*100:.0f}%;background:{color};height:10px;border-radius:6px"></div>
  </div>
</div>""",
        unsafe_allow_html=True,
    )


def render() -> None:
    st.header("Clinical Assessment")
    st.caption("Generate a clinical scenario, write your reasoning, and get scored feedback")
    st.divider()

    col_setup, col_work = st.columns([1, 2], gap="large")

    with col_setup:
        with st.container(border=True):
            st.subheader("Scenario settings")
            topic = st.selectbox(
                "Topic",
                options=list(TOPICS.keys()),
                format_func=lambda k: TOPICS[k],
                key="assess_topic",
            )
            difficulty = st.selectbox(
                "Difficulty",
                options=DIFFICULTIES,
                index=1,
                key="assess_difficulty",
            )
            if st.button("Generate scenario", width="stretch", type="primary"):
                with st.spinner("Building clinical case..."):
                    scenario = _generate_scenario(topic, difficulty)
                st.session_state["active_scenario"] = scenario
                st.session_state.pop("eval_result", None)
                st.rerun()

        scenario = st.session_state.get("active_scenario")
        if scenario:
            with st.container(border=True):
                st.subheader("About this assessment")
                st.caption(f"**Topic:** {TOPICS.get(scenario.topic, scenario.topic)}")
                st.caption(f"**Difficulty:** {scenario.difficulty.title()}")
                if scenario.ot_context:
                    st.info(scenario.ot_context, icon=None)

    with col_work:
        scenario = st.session_state.get("active_scenario")

        if not scenario:
            with st.container(border=True):
                st.info(
                    "Choose a topic and click **Generate scenario** to begin a clinical assessment.",
                    icon=None,
                )
            return

        with st.container(border=True):
            st.subheader("Clinical case")
            st.write(scenario.scenario_text)

        with st.container(border=True):
            st.subheader("Question")
            st.write(scenario.question)

            student_response = st.text_area(
                "Your clinical reasoning",
                height=150,
                placeholder="Describe the anatomical structures involved, the likely diagnosis, and your OT intervention approach...",
                key="assess_response",
            )

            col_btn, col_hint = st.columns([1, 1])
            with col_btn:
                submit = st.button(
                    "Submit for evaluation",
                    width="stretch",
                    type="primary",
                    disabled=not student_response.strip(),
                )
            with col_hint:
                if st.button("Take to chat", width="stretch"):
                    if "messages" not in st.session_state:
                        st.session_state.messages = []
                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": f"Let's work through this clinical case: {scenario.question}",
                            "citations": [],
                            "hint_level": 0,
                            "is_bypass_redirect": False,
                            "is_reveal": False,
                        }
                    )
                    st.session_state.current_page = "chat"
                    st.rerun()

            if submit and student_response.strip():
                with st.spinner("Evaluating your reasoning..."):
                    result = _evaluate_response(student_response, scenario)
                st.session_state["eval_result"] = result
                topics = st.session_state.setdefault("topics_covered", [])
                if scenario.topic not in topics:
                    topics.append(scenario.topic)
                scores = st.session_state.setdefault("mastery_scores", {})
                scores[scenario.topic] = result.total
                st.rerun()

        result = st.session_state.get("eval_result")
        if result:
            with st.container(border=True):
                st.subheader("Evaluation result")

                col_score, col_grade = st.columns(2)
                grade_color = {"Excellent": "#22c55e", "Good": "#3b82f6", "Pass": "#f59e0b", "Needs work": "#ef4444"}.get(
                    result.grade_label, "#64748b"
                )
                col_score.metric("Total score", f"{result.total}/100")
                col_grade.markdown(
                    f"<div style='text-align:center;padding:8px;background:{grade_color}20;border-radius:8px;"
                    f"border:1px solid {grade_color};color:{grade_color};font-weight:700;font-size:16px'>"
                    f"{result.grade_label}</div>",
                    unsafe_allow_html=True,
                )

                st.write("")
                _score_bar("Clinical accuracy", result.clinical_accuracy, 40, "#6366F1")
                _score_bar("Reasoning quality", result.reasoning_quality, 40, "#22D3EE")
                _score_bar("Terminology", result.terminology, 20, "#f59e0b")

                st.write("")
                st.caption(f"**Mastery level:** {result.mastery_level.value.title()}")
                st.write(result.feedback)

            with st.expander("Reference answer"):
                st.write(scenario.reference_answer)
