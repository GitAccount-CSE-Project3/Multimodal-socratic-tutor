
from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import TYPE_CHECKING

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import streamlit as st

from src.config.settings import get_settings
from src.core.conversation.state import PHASE_CONFIG
from src.utils.logger import logger

if TYPE_CHECKING:
    from streamlit.runtime.uploaded_file_manager import UploadedFile

settings = get_settings()


def _phase_topbar() -> None:
    phase = st.session_state.get("phase", "rapport")
    name = st.session_state.get("student_name", "Guest")
    topic = st.session_state.get("current_topic", "")
    hint = st.session_state.get("hint_level", 0)
    max_h = settings.max_hint_turns
    label, _, _ = PHASE_CONFIG.get(phase, ("Active", "", ""))

    col1, col2, col3 = st.columns([4, 2, 2])
    with col1:
        topic_str = f" · {topic.replace('_', ' ').title()}" if topic else ""
        st.write(f"**Anatomy & Neuroscience Tutor** | {name}{topic_str}")
    with col2:
        st.caption(f"Phase: {label}")
    with col3:
        if phase == "tutoring":
            if hint >= max_h:
                st.caption("Reveal unlocked")
            else:
                st.caption(f"Hint {hint}/{max_h}")
    st.divider()


def _clean_for_speech(text: str) -> str:
    import re

    speech = re.split(r"\n\s*Sources?:", text)[0]
    speech = re.sub(r"[*_#`>\[\]]", "", speech)
    speech = re.sub(r"\s+", " ", speech).strip()
    return speech or text


def _render_message(msg: dict, idx: int = 0) -> None:
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
                            _clean_for_speech(content)
                        )
                    st.session_state["_tts_autoplay"] = audio_key
                except Exception as e:
                    st.session_state[audio_key] = None
                    st.error(f"Could not generate audio: {e}")
            if st.session_state.get(audio_key):
                autoplay = st.session_state.get("_tts_autoplay") == audio_key
                if autoplay:
                    del st.session_state["_tts_autoplay"]
                _render_audio_player(st.session_state[audio_key], autoplay)


def _render_audio_player(data: bytes, autoplay: bool) -> None:
    import base64

    import streamlit.components.v1 as components

    b64 = base64.b64encode(data).decode()
    auto = "autoplay" if autoplay else ""
    components.html(
        f'<audio {auto} controls style="width:100%;height:40px">'
        f'<source src="data:audio/mpeg;base64,{b64}" type="audio/mpeg"></audio>',
        height=50,
    )


def _hint_indicator() -> None:
    phase = st.session_state.get("phase", "rapport")
    hint = st.session_state.get("hint_level", 0)
    max_hints = settings.max_hint_turns

    if phase != "tutoring" or not st.session_state.get("show_hints", True):
        return

    if hint >= max_hints:
        st.info("Answer reveal unlocked", icon=None)
    else:
        remaining = max_hints - hint
        st.caption(f"Hints used: {hint}/{max_hints} — {remaining} remaining")


def _is_substantive(text: str, engine: object) -> bool:
    t = text.lower().strip()
    if engine.detect_topic(text):
        return True
    triggers = (
        "quiz",
        "explain",
        "what",
        "how",
        "why",
        "describe",
        "tell me about",
        "teach me",
        "list the",
        "name the",
        "where",
        "which",
        "function of",
        "?",
    )
    if "my name is" in t or t.startswith(("i am", "i'm", "hi", "hello", "hey")):
        return False
    return len(t.split()) >= 2 and any(k in t for k in triggers)


async def _get_tutor_response(user_input: str, uploaded_image: UploadedFile | None) -> dict:
    try:
        from src.core.conversation.socratic_engine import SocraticEngine
        from src.core.conversation.state import ConversationPhase

        if "_engine" not in st.session_state:
            st.session_state["_engine"] = SocraticEngine()
        engine = st.session_state["_engine"]

        if uploaded_image is not None:
            try:
                image_bytes = uploaded_image.getvalue()
            except AttributeError:
                image_bytes = uploaded_image.read()
            media_type = (getattr(uploaded_image, "type", "") or "image/jpeg").split("/")[-1]
            if media_type == "jpg":
                media_type = "jpeg"

            question = (st.session_state.pop("image_question", "") or "").strip() or None
            img_response = await engine.generate_from_image(
                image_bytes=image_bytes,
                media_type=media_type,
                user_question=question,
            )
            st.session_state.phase = "tutoring"
            if img_response.topic_detected:
                st.session_state.current_topic = img_response.topic_detected
                if img_response.topic_detected not in st.session_state.topics_covered:
                    st.session_state.topics_covered.append(img_response.topic_detected)
            st.session_state.turn_count += 1
            return {
                "role": "assistant",
                "content": img_response.content,
                "citations": img_response.citations,
                "hint_level": None,
                "is_bypass_redirect": False,
                "is_reveal": False,
                "is_image_response": True,
            }

        phase = ConversationPhase(st.session_state.get("phase", "rapport"))

        if phase == ConversationPhase.RAPPORT and _is_substantive(user_input, engine):
            phase = ConversationPhase.TUTORING
            st.session_state.phase = "tutoring"

        state = {
            "turn_count": st.session_state.get("turn_count", 0),
            "hint_level": st.session_state.get("hint_level", 0),
            "student_name": st.session_state.get("student_name", ""),
            "history": [
                {"role": m.get("role"), "content": m.get("content", "")}
                for m in st.session_state.messages[:-1][-6:]
            ],
        }

        response = await engine.generate(
            student_input=user_input,
            session_state=state,
            phase=phase,
        )

        st.session_state.turn_count += 1
        st.session_state.hint_level = int(response.hint_level)
        if response.topic_detected:
            st.session_state.current_topic = response.topic_detected
            if response.topic_detected not in st.session_state.topics_covered:
                st.session_state.topics_covered.append(response.topic_detected)

        if phase == ConversationPhase.RAPPORT and st.session_state.turn_count >= 1:
            st.session_state.phase = "tutoring"

        topic = response.topic_detected or st.session_state.get("current_topic")
        if (
            phase == ConversationPhase.TUTORING
            and topic
            and response.retrieved_context
            and user_input.strip()
        ):
            try:
                from src.core.conversation.evaluator import StudentResponseEvaluator
                from src.core.memory.student_memory import StudentMemory

                overlap = StudentResponseEvaluator().keyword_overlap(
                    user_input, response.retrieved_context
                )
                new_score = round(overlap * 100, 1)
                prev = st.session_state.mastery_scores.get(topic)
                blended = new_score if prev is None else round(0.6 * prev + 0.4 * new_score, 1)
                st.session_state.mastery_scores[topic] = blended
                await StudentMemory().update_topic_score(
                    st.session_state.get("student_id", "student_demo"), topic, blended
                )
            except Exception as e:
                logger.warning("Mastery update skipped: {e}", e=str(e))

        return {
            "role": "assistant",
            "content": response.content,
            "citations": response.citations,
            "hint_level": int(response.hint_level),
            "is_bypass_redirect": response.is_bypass_redirect,
            "is_reveal": response.is_reveal,
        }

    except Exception as e:
        logger.error("Tutor response error: {e}", e=str(e))
        return {
            "role": "assistant",
            "content": (
                "I encountered an error generating a response. "
                "Please check your OpenAI API key in settings and try again."
            ),
            "citations": [],
            "hint_level": st.session_state.get("hint_level", 0),
            "is_bypass_redirect": False,
            "is_reveal": False,
        }


STARTER_PROMPTS = [
    "What does the cerebellum do?",
    "Walk me through the median nerve pathway",
    "Quiz me on the cranial nerves",
    "Why do dermatomes matter in OT?",
]

WELCOME = (
    "Welcome! I am **socratOT**, your Socratic anatomy and neuroscience tutor. "
    "I teach through guided questioning — I will never just hand you the answer, "
    "but I will guide you to discover it yourself.\n\n"
    "Before we begin, may I ask your name and which semester of your OT programme "
    "you are currently in?"
)


def _send(text: str) -> None:
    st.session_state.messages.append({"role": "user", "content": text})
    st.session_state.is_loading = True
    st.rerun()


def _composer() -> None:
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
        with st.popover("+" , width="content"):
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
                st.caption("Voice input needs Streamlit ≥ 1.43.")
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

        send = st.button("Send", width="content", type="primary", disabled=disabled)

    if attached:
        st.caption(f"{getattr(attached, 'name', 'image')} attached — analysed on send")

    if send:
        text = st.session_state.get("composer_text", "").strip()
        img = st.session_state.get("uploaded_image")
        if text or img is not None:
            st.session_state.image_question = text
            display = text if text else "Image uploaded for analysis"
            st.session_state.messages.append({"role": "user", "content": display})
            st.session_state.is_loading = True
            st.session_state._clear_composer = True
            st.rerun()


def _starter_suggestions() -> None:
    st.write("Not sure where to start? Try one of these:")
    cols = st.columns(2)
    for i, prompt in enumerate(STARTER_PROMPTS):
        with cols[i % 2]:
            if st.button(prompt, key=f"starter_{i}", width="stretch"):
                _send(prompt)


def render() -> None:
    _phase_topbar()

    if not st.session_state.messages:
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": WELCOME,
                "citations": [],
                "hint_level": None,
                "is_bypass_redirect": False,
                "is_reveal": False,
            }
        )

    for idx, msg in enumerate(st.session_state.messages):
        _render_message(msg, idx)

    if st.session_state.is_loading:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                st.empty()

    has_user_msg = any(m.get("role") == "user" for m in st.session_state.messages)
    if not has_user_msg and not st.session_state.is_loading:
        _starter_suggestions()

    _hint_indicator()
    _composer()

    if (
        st.session_state.is_loading
        and st.session_state.messages
        and st.session_state.messages[-1]["role"] == "user"
    ):
        last_input = st.session_state.messages[-1]["content"]
        with st.spinner("socratOT is thinking..."):
            response = asyncio.run(
                _get_tutor_response(
                    last_input,
                    st.session_state.uploaded_image,
                )
            )

        st.session_state.messages.append(response)
        st.session_state.is_loading = False
        if st.session_state.get("uploaded_image") is not None:
            st.session_state.uploaded_image = None
            st.session_state.uploader_seq = st.session_state.get("uploader_seq", 0) + 1
        st.rerun()
