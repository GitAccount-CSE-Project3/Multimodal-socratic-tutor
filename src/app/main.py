
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

from src.app.sidebar import render_sidebar  # noqa: E402
from src.utils.logger import logger  # noqa: E402

st.markdown(
    """
<style>
.stApp { background-color: #ffffff; }
[data-testid="stSidebar"] { background-color: #f8fafc; border-right: 1px solid #e2e8f0; }
#MainMenu, footer { visibility: hidden; }
[data-testid="stToolbarActions"], [data-testid="stDecoration"] { display: none; }
header[data-testid="stHeader"] { background: transparent; visibility: visible; pointer-events: none; }
header[data-testid="stHeader"] * { pointer-events: auto; }
[data-testid="stIconMaterial"] { font-family: 'Material Symbols Rounded' !important; }
[data-testid="stExpandSidebarButton"] { visibility: visible !important; opacity: 1 !important; z-index: 1000; display: inline-flex !important; }
.stChatMessage { border-radius: 10px; margin-bottom: 6px; }
[class*="st-key-tts_btn_"] button { width: auto; padding: 2px 10px; font-size: 12px; }
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
    elif page == "assessment":
        from src.app.views import assessment
        assessment.render()
    elif page == "evaluation":
        from src.app.views import evaluation_dashboard
        evaluation_dashboard.render()
    elif page == "settings":
        from src.app.views import settings_page
        settings_page.render()
    else:
        st.error(f"Unknown page: {page}")


def main() -> None:
    logger.debug("App rendered — page={p}", p=st.session_state.get("current_page"))
    _init_session()
    render_sidebar()
    _route()


if __name__ == "__main__":
    main()
