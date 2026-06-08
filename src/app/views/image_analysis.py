
from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import streamlit as st

NAVY = "#E2E8F0"
MUTED = "#94A3B8"


def _section_title(text: str) -> None:
    st.subheader(text)


async def _run_vision_pipeline(
    image_bytes: bytes,
    media_type: str,
    user_question: str | None = None,
) -> dict:
    try:
        from src.core.multimodal.pipeline import MultimodalPipeline

        pipeline = MultimodalPipeline()
        result = await pipeline.process_image(image_bytes, media_type, user_question=user_question)
        return {
            "ok": True,
            "reply": result.socratic_reply,
            "structures": result.vision_result.structures,
            "region": result.vision_result.region,
            "confidence": result.vision_result.confidence,
            "ot_relevance": result.vision_result.ot_relevance,
            "description": result.vision_result.description,
            "questions": result.questions,
            "citations": result.citations,
            "error": result.vision_result.error,
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


def render() -> None:
    st.header("Anatomical Image Analysis")
    st.caption("Upload any anatomy diagram — GPT-4o identifies structures and generates Socratic questions")
    st.divider()

    col1, col2 = st.columns([2, 3], gap="large")

    with col1:
        with st.container(border=True):
            _section_title("Upload image")
            uploaded = st.file_uploader(
                "Upload anatomy image",
                type=["png", "jpg", "jpeg", "webp"],
                label_visibility="collapsed",
                help="Any anatomy diagram — does not need to be from the training set.",
            )
            if uploaded:
                st.image(uploaded, caption=uploaded.name, width="stretch")
                st.caption(f"{uploaded.type} · {uploaded.size // 1024} KB")

                question = st.text_input(
                    "Ask a question about this image (optional)",
                    placeholder="e.g. Which nerve is highlighted and what does it innervate?",
                    key="vision_question",
                )

                if st.button("🔬 Analyse image", width="stretch", type="primary"):
                    with st.spinner("GPT-4o analysing structures..."):
                        image_bytes = uploaded.read()
                        media_type = uploaded.type.split("/")[-1]
                        if media_type == "jpg":
                            media_type = "jpeg"

                        result = asyncio.run(
                            _run_vision_pipeline(
                                image_bytes, media_type, (question or "").strip() or None
                            )
                        )

                    st.session_state["vision_result"] = result
                    st.session_state["vision_image_name"] = uploaded.name
                    st.rerun()

    with col2:
        result = st.session_state.get("vision_result")

        if result and result.get("ok"):
            region = result.get("region", "")
            if region:
                topics = st.session_state.setdefault("topics_covered", [])
                if region not in topics:
                    topics.append(region)

            with st.container(border=True):
                _section_title("Identified structures")
                structures = result.get("structures", [])
                confidence = result.get("confidence", 0.0)

                if structures:
                    st.write(", ".join(structures))
                    st.write("")
                    col_r, col_c = st.columns(2)
                    col_r.metric("Region", region.replace("_", " ").title())
                    col_c.metric("Confidence", f"{confidence * 100:.0f}%")

                    if result.get("ot_relevance"):
                        st.caption(f"**OT relevance:** {result['ot_relevance']}")
                else:
                    st.warning("No structures identified — try a clearer image.")

            questions = result.get("questions", [])
            if questions:
                with st.container(border=True):
                    _section_title("Socratic questions")
                    for i, q in enumerate(questions, 1):
                        diff_label = getattr(q, "difficulty", "intermediate").title()
                        question_text = getattr(q, "question", str(q))
                        st.write(f"**Q{i} · {diff_label}**")
                        st.write(question_text)
                        st.write("")

                    if st.button("💬 Continue in tutor chat", width="stretch"):
                        first_q = getattr(questions[0], "question", "") if questions else ""
                        if first_q:
                            if "messages" not in st.session_state:
                                st.session_state.messages = []
                            st.session_state.messages.append(
                                {
                                    "role": "assistant",
                                    "content": result.get("reply", first_q),
                                    "citations": result.get("citations", []),
                                    "hint_level": 0,
                                    "is_bypass_redirect": False,
                                    "is_reveal": False,
                                    "is_image_response": True,
                                }
                            )
                        st.session_state.current_page = "chat"
                        st.rerun()

            if result.get("citations"):
                with st.container(border=True):
                    _section_title("Sources")
                    for c in result["citations"]:
                        st.caption(f"📄 {c}")

        elif result and not result.get("ok"):
            st.error(f"Analysis failed: {result.get('error', 'Unknown error')}")
            st.info("Ensure your OpenAI API key is set and try again.")

        else:
            with st.container(border=True):
                _section_title("How it works")
                st.info(
                    "Upload any anatomical diagram and click **Analyse image**. "
                    "GPT-4o vision will identify structures and generate "
                    "3 Socratic questions ordered by difficulty.",
                    icon="🔬",
                )
                st.caption("Supported: brain · upper extremity · hand · spinal cord · nervous system")

    st.write("")
    meta_path = Path("data/image_metadata.json")
    with st.container(border=True):
        _section_title("Training image corpus")
        if not meta_path.exists():
            st.caption(
                "Run `python scripts/download_images.py` to extract images from the OpenStax PDF."
            )
        else:
            metadata = json.loads(meta_path.read_text())
            images_dir = Path("data/images")
            available = [m for m in metadata if (images_dir / m["filename"]).exists()]
            total_kb = sum(m.get("size_kb", 0) for m in available)

            c1, c2, c3 = st.columns(3)
            c1.metric("Images", len(available))
            c2.metric("Total size", f"{total_kb / 1024:.0f} MB")
            c3.metric("Source", "OpenStax A&P 2e")

            with st.expander(f"View all {len(available)} images"):
                for m in available[:200]:
                    icon = "✅"
                    detail = (
                        m.get("region", "").replace("_", " ").title()
                        or f"page {m.get('page', '?')}"
                    )
                    st.caption(f"{icon} **{m['filename']}** · {detail} · {m.get('size_kb', 0)} KB")
