from __future__ import annotations

import asyncio

from src.config.settings import get_settings
from src.core.conversation.response import SocraticResponse
from src.core.conversation.state import ConversationPhase, HintLevel
from src.core.conversation.topic_detector import detect_topic
from src.core.rag.pipeline import RAGPipeline
from src.prompts.loader import get_prompt
from src.utils.helpers import detect_bypass_attempt, truncate_text
from src.utils.logger import logger


class SocraticEngine:

    BYPASS_RESPONSE = (
        "I understand you want the answer directly — that's natural! "
        "But research shows you retain knowledge much better when you "
        "work through it yourself. Let me guide you with another hint: {hint}"
    )

    def __init__(self, rag_pipeline: RAGPipeline | None = None, llm: object | None = None) -> None:
        self._rag = rag_pipeline or RAGPipeline()
        self._llm = llm
        self._settings = get_settings()
        self._multimodal = None

    def _get_llm(self) -> object:
        if self._llm is None:
            from src.models.llm_factory import get_llm
            self._llm = get_llm()
        return self._llm

    def _get_multimodal(self) -> object:
        if self._multimodal is None:
            from src.core.multimodal.pipeline import MultimodalPipeline
            self._multimodal = MultimodalPipeline(rag_pipeline=self._rag)
        return self._multimodal

    def detect_topic(self, text: str) -> str | None:
        return detect_topic(text)

    async def generate_from_image(self, image_bytes: bytes, media_type: str = "jpeg", session_state: dict | None = None, user_question: str | None = None) -> SocraticResponse:
        logger.info("SocraticEngine: image input ({n} bytes)", n=len(image_bytes))
        mm_result = await self._get_multimodal().process_image(image_bytes, media_type, user_question=user_question)
        return SocraticResponse(
            content=mm_result.socratic_reply, hint_level=HintLevel.NONE,
            is_bypass_redirect=False, is_reveal=False, retrieved_context=mm_result.rag_context,
            citations=mm_result.citations, topic_detected=mm_result.vision_result.region,
            is_image_response=True, identified_structures=mm_result.vision_result.structures,
        )

    async def generate(self, student_input: str, session_state: dict, phase: ConversationPhase = ConversationPhase.TUTORING) -> SocraticResponse:
        current_hint = HintLevel(session_state.get("hint_level", 0))
        is_bypass = detect_bypass_attempt(student_input)

        if phase == ConversationPhase.RAPPORT:
            return await self._handle_rapport(student_input, session_state)
        if phase == ConversationPhase.MASTERY:
            return await self._handle_mastery(session_state)

        retrieval_query = self._build_retrieval_query(student_input, session_state.get("history", []))
        rag_result = await self._rag.query(retrieval_query)
        context = rag_result.retrieval.assembled_context
        citations = rag_result.retrieval.citations
        topic = detect_topic(student_input)

        if is_bypass and not self._answer_reveal_eligible(current_hint):
            content = await self._redirect_bypass(student_input, context, current_hint)
            return SocraticResponse(content=content, hint_level=current_hint, is_bypass_redirect=True, is_reveal=False, retrieved_context=context, citations=citations, topic_detected=topic)

        if self._answer_reveal_eligible(current_hint):
            content = await self._reveal_answer(student_input, context)
            return SocraticResponse(content=content, hint_level=HintLevel.REVEALED, is_bypass_redirect=False, is_reveal=True, retrieved_context=context, citations=citations, topic_detected=topic)

        content = await self._generate_hint(student_input=student_input, context=context, hint_level=current_hint, session_state=session_state)
        next_hint = HintLevel(min(int(current_hint) + 1, int(HintLevel.LEVEL_2)))
        return SocraticResponse(content=content, hint_level=next_hint, is_bypass_redirect=False, is_reveal=False, retrieved_context=context, citations=citations, topic_detected=topic)

    async def _handle_rapport(self, student_input: str, session_state: dict) -> SocraticResponse:
        content = await self._llm_call(
            f'You are socratOT, a warm Socratic anatomy & neuroscience tutor for OT students.\n'
            f'The student just said: "{student_input}"\n\n'
            f'In 2 short sentences: acknowledge them, then invite them to name a topic to explore.\n'
            f'Ask at most ONE question. Do NOT lecture or give facts yet.'
        )
        return SocraticResponse(content=content, hint_level=HintLevel.NONE, is_bypass_redirect=False, is_reveal=False, retrieved_context="", citations=[], topic_detected=detect_topic(student_input))

    async def _handle_mastery(self, session_state: dict) -> SocraticResponse:
        history = session_state.get("history", [])
        topics = list({t["content"] for t in history if t.get("role") == "user" and detect_topic(t.get("content", ""))})
        content = get_prompt("socratic.mastery_summary", mastered_topics=", ".join(topics) if topics else "anatomy topics", weak_topics="Review sessions will be suggested next time.")
        return SocraticResponse(content=content, hint_level=HintLevel.REVEALED, is_bypass_redirect=False, is_reveal=True, retrieved_context="", citations=[], topic_detected=None)

    async def _generate_hint(self, student_input: str, context: str, hint_level: HintLevel, session_state: dict) -> str:
        turn = session_state.get("turn_count", 0)
        masked = get_prompt("socratic.knowledge_mask", topic=truncate_text(student_input, 80), context=truncate_text(context, 1500), current_turn=turn + 1, max_hint_turns=self._settings.max_hint_turns)
        if hint_level == HintLevel.NONE:
            level_instr = "This is your FIRST hint. Ask ONE focused open-ended question anchored in the context. Do NOT state any answer directly. Keep it to 1-2 sentences."
        else:
            level_instr = "This is your SECOND, STRONGER hint. Point to a specific concept in the context and ask about it — but still do NOT give the full answer. Keep it to 1-2 sentences."
        prompt = f"{masked}\n\nStudent question: \"{student_input}\"\n\n{level_instr}\nRespond with ONLY the Socratic hint — no preamble."
        return await self._llm_call(prompt)

    async def _redirect_bypass(self, student_input: str, context: str, hint_level: HintLevel) -> str:
        hint = get_prompt("socratic.hint_level_1", topic=truncate_text(student_input, 60), guiding_question="What clues from the context help you think through this?")
        return self.BYPASS_RESPONSE.format(hint=hint[:200])

    async def _reveal_answer(self, student_input: str, context: str) -> str:
        prompt = get_prompt("socratic.reveal_after_turns", direct_answer=f"Based on the context: {truncate_text(context, 800)}")
        return await self._llm_call(f"{prompt}\n\nStudent question: {student_input}\nContext: {truncate_text(context, 1000)}\n\nProvide a clear complete answer now.")

    def _answer_reveal_eligible(self, hint_level: HintLevel) -> bool:
        return hint_level >= HintLevel.LEVEL_2

    def _build_retrieval_query(self, student_input: str, history: list) -> str:
        if detect_topic(student_input):
            return student_input
        t = student_input.lower().strip()
        cues = ("how about", "what about", "and ", "what else", "what of", "and the")
        if (len(t.split()) <= 5 or t.startswith(cues)) and history:
            for h in reversed(history):
                if h.get("role") == "user" and len(h.get("content", "").split()) >= 3:
                    return f"{h['content']} {student_input}".strip()
        return student_input

    async def _llm_call(self, prompt: str) -> str:
        try:
            llm = self._get_llm()
            loop = asyncio.get_running_loop()
            resp = await loop.run_in_executor(None, lambda: llm.invoke(prompt))
            return (resp.content if hasattr(resp, "content") else str(resp)).strip()
        except Exception as e:
            logger.error("LLM call failed: {e}", e=str(e))
            return "I'm having trouble connecting right now. Please check your API key and try again."
