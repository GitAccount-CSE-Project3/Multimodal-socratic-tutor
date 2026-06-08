from __future__ import annotations

import streamlit as st


def send(text: str) -> None:
    st.session_state.messages.append({"role": "user", "content": text})
    st.session_state.is_loading = True
    st.rerun()


def composer() -> None:
    if st.session_state.pop("_clear_composer", False):
        st.session_state.composer_text = ""
        st.session_state.pop("_last_audio_hash", None)

    st.session_state.setdefault("composer_text", "")
    st.session_state.setdefault("uploader_seq", 0)

    disabled = st.session_state.is_loading
    attached = st.session_state.get("uploaded_image")

    with st.container(
        key="composer_bar", horizontal=True, vertical_alignment="center", gap="small"
    ):
        with st.popover("+", width="content"):
            uploaded = st.file_uploader(
                "Attach an anatomy image",
                type=["png", "jpg", "jpeg", "webp"],
                label_visibility="collapsed",
                key=f"composer_img_{st.session_state.uploader_seq}",
            )
            if uploaded is not None:
                st.session_state.uploaded_image = uploaded
                st.image(uploaded, width=180, caption="Attached — add a question & send")
            if attached and st.button("Remove", key="composer_img_rm"):
                st.session_state.uploaded_image = None
                st.session_state.uploader_seq += 1
                st.rerun()

        audio = None
        with st.popover("Voice", width="content"):
            if hasattr(st, "audio_input"):
                st.caption("Record, then it transcribes into the message box.")
                audio = st.audio_input(
                    "Record", label_visibility="collapsed", key="composer_mic", disabled=disabled
                )
            else:
                st.caption("Voice input needs Streamlit >= 1.43.")

        if audio is not None:
            try:
                data = audio.getvalue()
            except Exception:
                data = b""
            if data and hash(data) != st.session_state.get("_last_audio_hash"):
                st.session_state._last_audio_hash = hash(data)
                from src.core.audio.audio_service import AudioError, AudioService
                try:
                    with st.spinner("Transcribing…"):
                        text = AudioService().transcribe(data, filename="mic.wav")
                    if text:
                        prev = st.session_state.get("composer_text", "").strip()
                        st.session_state.composer_text = f"{prev} {text}".strip() if prev else text
                        st.rerun()
                    else:
                        st.toast("No speech detected")
                except AudioError as e:
                    st.error(str(e))

        st.text_input(
            "Message",
            key="composer_text",
            placeholder="Message socratOT…",
            label_visibility="collapsed",
            disabled=disabled,
            width="stretch",
        )

        send_btn = st.button("Send", width="content", type="primary", disabled=disabled)

    if attached:
        st.caption(f"{getattr(attached, 'name', 'image')} attached — analysed on send")

    if send_btn:
        text = st.session_state.get("composer_text", "").strip()
        img = st.session_state.get("uploaded_image")
        if text or img is not None:
            st.session_state.image_question = text
            display = text if text else "Image uploaded for analysis"
            st.session_state.messages.append({"role": "user", "content": display})
            st.session_state.is_loading = True
            st.session_state._clear_composer = True
            st.rerun()
