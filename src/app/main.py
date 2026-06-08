
from __future__ import annotations

import sys
from pathlib import Path
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import streamlit as st

st.set_page_config(
    page_title="socratOT — OT Anatomy Tutor",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="auto",
    menu_items={
        "Get Help": "https://github.com/bahodir4/multimodal-socratic-tutor",
        "About": "socratOT — Socratic AI Tutor for Occupational Therapy Education",
    },
)

from src.config.settings import get_settings  # noqa: E402  # must follow set_page_config
from src.core.conversation.state import PHASE_CONFIG  # noqa: E402
from src.utils.logger import logger  # noqa: E402

settings = get_settings()


st.markdown(
    """
<style>
.stApp { background-color: #ffffff; }

[data-testid="stSidebar"] {
    background-color: #f8fafc;
    border-right: 1px solid #e2e8f0;
}

/* hide default streamlit UI chrome */
#MainMenu, footer { visibility: hidden; }
[data-testid="stToolbarActions"], [data-testid="stDecoration"] { display: none; }

/* keep header transparent so the sidebar re-open button still works */
header[data-testid="stHeader"] {
    background: transparent;
    visibility: visible;
    pointer-events: none;
}
header[data-testid="stHeader"] * { pointer-events: auto; }

/* fix: Material icon font gets overridden by streamlit's global font rule */
[data-testid="stIconMaterial"] { font-family: 'Material Symbols Rounded' !important; }

/* make sure the sidebar expand button is actually visible and clickable */
[data-testid="stExpandSidebarButton"] {
    visibility: visible !important;
    opacity: 1 !important;
    z-index: 1000;
    display: inline-flex !important;
}

/* chat messages */
.stChatMessage { border-radius: 10px; margin-bottom: 6px; }

/* read-aloud button: small and unobtrusive */
[class*="st-key-tts_btn_"] button {
    width: auto;
    padding: 2px 10px;
    font-size: 12px;
}
</style>
""",
    unsafe_allow_html=True,
)


def _init_session() -> None:
    defaults: dict = {
        "student_id": f"session_{uuid4().hex[:12]}",
        "student_name": "",
        "student_level": "",
        "session_id": None,
        "current_page": "chat",
        "messages": [],
        "phase": "rapport",
        "turn_count": 0,
        "hint_level": 0,
        "current_topic": None,
        "is_loading": False,
        "uploaded_image": None,
        "topics_covered": [],
        "mastery_scores": {},
        "show_citations": True,
        "show_hints": True,
        "enable_tts": True,
        "enable_stt": True,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown(
            """
        <div style="display:flex;align-items:center;gap:11px;
                    padding-bottom:16px;margin-bottom:16px;
                    border-bottom:1px solid #e2e8f0">
            <div style="width:38px;height:38px;border-radius:10px;
                        background:linear-gradient(135deg,#6366F1 0%,#22D3EE 130%);
                        display:flex;align-items:center;justify-content:center;
                        color:white;font-weight:800;font-size:18px;flex-shrink:0;
                        box-shadow:0 8px 20px -8px rgba(99,102,241,.8)">S</div>
            <div>
                <div style="font-weight:700;font-size:15px;color:#1e293b;
                            letter-spacing:-.01em">socratOT</div>
                <div style="font-size:11px;color:#64748b">OT Anatomy Tutor</div>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.write("**Navigation**")
        pages = {
            "chat": "Tutor Chat",
            "dashboard": "Dashboard",
            "images": "Image Analysis",
            "settings": "Settings",
        }
        for key, label in pages.items():
            active = st.session_state.current_page == key
            if st.button(
                label,
                key=f"nav_{key}",
                width="stretch",
                type="primary" if active else "secondary",
            ):
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
            embed = (
                settings.openai_embedding_model
                if settings.embedding_provider.value == "openai"
                else settings.embedding_model.split("/")[-1]
            )
            st.caption(f"**Provider** `{settings.llm_provider.value}`")
            st.caption(f"**Embed** `{embed}`")
            st.caption(f"**Store** `{settings.vector_store_type.value}`")
            st.caption(f"**Env** `{settings.app_env.value}`")
            if settings.using_openai and not settings.openai_api_key:
                st.warning("OPENAI_API_KEY not set in .env")

        if st.session_state.student_name:
            st.divider()
            st.caption(f"**{st.session_state.student_name}**  \nSession active")


def _route() -> None:
    page = st.session_state.current_page
    if page == "chat":
        from src.app.views import chat

        chat.render()
    elif page == "dashboard":
        from src.app.views import dashboard

        dashboard.render()
    elif page == "images":
        from src.app.views import image_analysis

        image_analysis.render()
    elif page == "settings":
        from src.app.views import settings_page

        settings_page.render()
    else:
        st.error(f"Unknown page: {page}")


def main() -> None:
    logger.debug("App rendered — page={p}", p=st.session_state.get("current_page"))
    _init_session()
    _render_sidebar()
    _route()


if __name__ == "__main__":
    main()
