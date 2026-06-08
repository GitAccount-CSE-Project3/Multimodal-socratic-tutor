from __future__ import annotations

import asyncio

import streamlit as st

from .composer import composer
from .engine import WELCOME, get_tutor_response
from .message import render_message
from .topbar import hint_indicator, phase_topbar, starter_suggestions


def render() -> None:
    phase_topbar()

    if not st.session_state.messages:
        st.session_state.messages.append({
            "role": "assistant", "content": WELCOME, "citations": [],
            "hint_level": None, "is_bypass_redirect": False, "is_reveal": False,
        })

    for idx, msg in enumerate(st.session_state.messages):
        render_message(msg, idx)

    if st.session_state.is_loading:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                st.empty()

    has_user_msg = any(m.get("role") == "user" for m in st.session_state.messages)
    if not has_user_msg and not st.session_state.is_loading:
        from .composer import send
        starter_suggestions(send)

    hint_indicator()
    composer()

    if (
        st.session_state.is_loading
        and st.session_state.messages
        and st.session_state.messages[-1]["role"] == "user"
    ):
        last_input = st.session_state.messages[-1]["content"]
        with st.spinner("socratOT is thinking..."):
            response = asyncio.run(
                get_tutor_response(last_input, st.session_state.uploaded_image)
            )

        st.session_state.messages.append(response)
        st.session_state.is_loading = False
        if st.session_state.get("uploaded_image") is not None:
            st.session_state.uploaded_image = None
            st.session_state.uploader_seq = st.session_state.get("uploader_seq", 0) + 1
        st.rerun()
