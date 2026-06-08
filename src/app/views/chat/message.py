from __future__ import annotations

import re

import streamlit as st


def clean_for_speech(text: str) -> str:
    speech = re.split(r"\n\s*Sources?:", text)[0]
    speech = re.sub(r"[*_#`>\[\]]", "", speech)
    speech = re.sub(r"\s+", " ", speech).strip()
    return speech or text


def render_audio_player(data: bytes, autoplay: bool) -> None:
    import base64

    import streamlit.components.v1 as components

    b64 = base64.b64encode(data).decode()
    auto = "autoplay" if autoplay else ""
    components.html(
        f'<audio {auto} controls style="width:100%;height:40px">'
        f'<source src="data:audio/mpeg;base64,{b64}" type="audio/mpeg"></audio>',
        height=50,
    )


def render_message(msg: dict, idx: int = 0) -> None:
    role = msg.get("role", "assistant")
    content = msg.get("content", "")
    citations = msg.get("citations", [])
    hint_lvl = msg.get("hint_level")
    is_bypass = msg.get("is_bypass_redirect", False)
    is_reveal = msg.get("is_reveal", False)

    with st.chat_message(role):
        st.markdown(content)

        if hint_lvl is not None and role == "assistant":
            hint_labels = {0: "Guiding question", 1: "Hint 1", 2: "Hint 2", 3: "Answer revealed"}
            hint_str = hint_labels.get(hint_lvl, f"Hint {hint_lvl}")
            if is_bypass:
                st.caption("Bypass attempt redirected")
            elif is_reveal:
                st.caption("Answer revealed")
            else:
                st.caption(hint_str)

        if citations and st.session_state.get("show_citations", True):
            st.caption("Sources: " + " · ".join(citations[:3]))

        if role == "assistant" and content and st.session_state.get("enable_tts"):
            audio_key = f"tts_audio_{idx}"
            if st.button("Read aloud", key=f"tts_btn_{idx}"):
                from src.core.audio.audio_service import AudioService
                try:
                    with st.spinner("Generating audio…"):
                        st.session_state[audio_key] = AudioService().synthesize(
                            clean_for_speech(content)
                        )
                    st.session_state["_tts_autoplay"] = audio_key
                except Exception as e:
                    st.session_state[audio_key] = None
                    st.error(f"Could not generate audio: {e}")
            if st.session_state.get(audio_key):
                autoplay = st.session_state.get("_tts_autoplay") == audio_key
                if autoplay:
                    del st.session_state["_tts_autoplay"]
                render_audio_player(st.session_state[audio_key], autoplay)
