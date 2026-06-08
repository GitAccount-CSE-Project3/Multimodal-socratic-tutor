from __future__ import annotations

from dataclasses import dataclass

from src.core.rag._guard_utils import find_unsupported_phrases, keyword_overlap
from src.utils.helpers import safe_parse_json
from src.utils.logger import logger


@dataclass
class GuardResult:

    is_grounded: bool
    confidence: float
    flagged_phrases: list[str]
    reason: str


class HallucinationGuard:

    def __init__(self, llm: object | None = None, overlap_threshold: float = 0.15, use_llm_verification: bool = True) -> None:
        self._llm = llm
        self._overlap_threshold = overlap_threshold
        self._use_llm = use_llm_verification

    async def check(self, answer: str, context: str, query: str = "") -> GuardResult:
        if not context.strip():
            return GuardResult(is_grounded=False, confidence=0.0, flagged_phrases=[], reason="No context provided — cannot verify grounding")

        overlap_score = keyword_overlap(answer, context)
        logger.debug("Hallucination guard overlap: {s:.3f}", s=overlap_score)

        if overlap_score >= self._overlap_threshold:
            return GuardResult(is_grounded=True, confidence=min(1.0, overlap_score * 2), flagged_phrases=[], reason=f"Keyword overlap {overlap_score:.2f} above threshold")

        if self._use_llm and self._llm is not None:
            return await self._llm_verify(answer, context, query)

        return GuardResult(
            is_grounded=overlap_score > 0.05,
            confidence=overlap_score,
            flagged_phrases=find_unsupported_phrases(answer, context),
            reason=f"Low keyword overlap ({overlap_score:.2f}) — may contain hallucinations",
        )

    async def _llm_verify(self, answer: str, context: str, query: str) -> GuardResult:
        prompt = f"""You are a fact-checker for an educational AI system.

Given the following retrieved context and generated answer,
determine if the answer is grounded in the context or introduces
facts not present in the context.

Context:
{context[:2000]}

Question: {query}
Answer: {answer}

Respond ONLY with valid JSON:
{{
  "is_grounded": true or false,
  "confidence": 0.0 to 1.0,
  "reason": "brief explanation"
}}"""

        try:
            import asyncio
            loop = asyncio.get_running_loop()
            response = await loop.run_in_executor(None, lambda: self._llm.invoke(prompt))
            text = response.content if hasattr(response, "content") else str(response)
            parsed = safe_parse_json(text)
            if parsed:
                return GuardResult(
                    is_grounded=bool(parsed.get("is_grounded", False)),
                    confidence=float(parsed.get("confidence", 0.5)),
                    flagged_phrases=[],
                    reason=str(parsed.get("reason", "LLM verification")),
                )
        except Exception as e:
            logger.warning("LLM verification failed: {e}", e=str(e))

        return GuardResult(is_grounded=False, confidence=0.3, flagged_phrases=[], reason="LLM verification failed — treating as unverified")
