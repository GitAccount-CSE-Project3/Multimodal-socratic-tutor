from __future__ import annotations

from typing import TYPE_CHECKING

import streamlit as st

from src.utils.logger import logger

if TYPE_CHECKING:
    from streamlit.runtime.uploaded_file_manager import UploadedFile

WELCOME = (
    "Welcome! I am **socratOT**, your Socratic anatomy and neuroscience tutor. "
    "I teach through guided questioning — I will never just hand you the answer, "
    "but I will guide you to discover it yourself.\n\n"
    "Before we begin, may I ask your name and which semester of your OT programme "
    "you are currently in?"
)


def is_substantive(text: str, engine: object) -> bool:
    t = text.lower().strip()
    if engine.detect_topic(text):
        return True
    triggers = (
        "quiz", "explain", "what", "how", "why", "describe",
        "tell me about", "teach me", "list the", "name the",
        "where", "which", "function of", "?",
    )
    if "my name is" in t or t.startswith(("i am", "i'm", "hi", "hello", "hey")):
        return False
    return len(t.split()) >= 2 and any(k in t for k in triggers)


async def get_tutor_response(user_input: str, uploaded_image: UploadedFile | None) -> dict:
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
                image_bytes=image_bytes, media_type=media_type, user_question=question,
            )
            st.session_state.phase = "tutoring"
            if img_response.topic_detected:
                st.session_state.current_topic = img_response.topic_detected
                if img_response.topic_detected not in st.session_state.topics_covered:
                    st.session_state.topics_covered.append(img_response.topic_detected)
            st.session_state.turn_count += 1
            return {
                "role": "assistant", "content": img_response.content,
                "citations": img_response.citations, "hint_level": None,
                "is_bypass_redirect": False, "is_reveal": False, "is_image_response": True,
            }

        phase = ConversationPhase(st.session_state.get("phase", "rapport"))
        if phase == ConversationPhase.RAPPORT and is_substantive(user_input, engine):
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

        response = await engine.generate(student_input=user_input, session_state=state, phase=phase)
        st.session_state.turn_count += 1
        st.session_state.hint_level = int(response.hint_level)
        if response.topic_detected:
            st.session_state.current_topic = response.topic_detected
            if response.topic_detected not in st.session_state.topics_covered:
                st.session_state.topics_covered.append(response.topic_detected)
        if phase == ConversationPhase.RAPPORT and st.session_state.turn_count >= 1:
            st.session_state.phase = "tutoring"

        topic = response.topic_detected or st.session_state.get("current_topic")
        if phase == ConversationPhase.TUTORING and topic and response.retrieved_context and user_input.strip():
            try:
                from src.core.conversation.evaluator import StudentResponseEvaluator
                from src.core.memory.student_memory import StudentMemory
                overlap = StudentResponseEvaluator().keyword_overlap(user_input, response.retrieved_context)
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
            "role": "assistant", "content": response.content, "citations": response.citations,
            "hint_level": int(response.hint_level),
            "is_bypass_redirect": response.is_bypass_redirect, "is_reveal": response.is_reveal,
        }

    except Exception as e:
        logger.error("Tutor response error: {e}", e=str(e))
        return {
            "role": "assistant",
            "content": "I encountered an error. Please check your OpenAI API key and try again.",
            "citations": [], "hint_level": st.session_state.get("hint_level", 0),
            "is_bypass_redirect": False, "is_reveal": False,
        }
