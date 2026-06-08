from __future__ import annotations

import streamlit as st

from src.config.settings import get_settings
from src.core.conversation.state import PHASE_CONFIG

settings = get_settings()


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            """
        <div style="display:flex;align-items:center;gap:11px;
                    padding-bottom:16px;margin-bottom:16px;
                    border-bottom:1px solid #e2e8f0">
            <div style="width:38px;height:38px;border-radius:10px;
                        background:linear-gradient(135deg,#6366F1 0%,#22D3EE 130%);
                        display:flex;align-items:center;justify-content:center;
                        color:white;font-weight:800;font-size:18px;flex-shrink:0;">S</div>
            <div>
                <div style="font-weight:700;font-size:15px;color:#1e293b;">socratOT</div>
                <div style="font-size:11px;color:#64748b">OT Anatomy Tutor</div>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.write("**Navigation**")
        pages = {"chat": "Tutor Chat", "dashboard": "Dashboard", "images": "Image Analysis", "settings": "Settings"}
        for key, label in pages.items():
            active = st.session_state.current_page == key
            if st.button(label, key=f"nav_{key}", width="stretch", type="primary" if active else "secondary"):
                st.session_state.current_page = key
                st.rerun()

        st.divider()

        phase = st.session_state.phase
        label, _, _ = PHASE_CONFIG.get(phase, (phase.title(), "", ""))
        st.caption(f"Phase: **{label}**")

        turns = st.session_state.turn_count
        max_turns = settings.max_session_turns
        hint_level = st.session_state.hint_level

        st.caption(f"Turn {turns} / {max_turns}")
        st.progress(min(turns / max_turns, 1.0))

        if phase == "tutoring":
            max_hints = settings.max_hint_turns
            if hint_level >= max_hints:
                st.success("Answer reveal unlocked", icon=None)
            else:
                remaining = max_hints - hint_level
                st.info(f"Hints used: {hint_level}/{max_hints} · {remaining} remaining", icon=None)

        st.divider()

        with st.expander("System info", expanded=False):
            st.caption(f"**LLM** `{settings.openai_llm_model}`")
            st.caption(f"**Vision** `{settings.openai_vision_model}`")
            embed = settings.openai_embedding_model if settings.embedding_provider.value == "openai" else settings.embedding_model.split("/")[-1]
            st.caption(f"**Provider** `{settings.llm_provider.value}`")
            st.caption(f"**Embed** `{embed}`")
            st.caption(f"**Store** `{settings.vector_store_type.value}`")
            st.caption(f"**Env** `{settings.app_env.value}`")
            if settings.using_openai and not settings.openai_api_key:
                st.warning("OPENAI_API_KEY not set in .env")

        if st.session_state.student_name:
            st.divider()
            st.caption(f"**{st.session_state.student_name}**  \nSession active")
