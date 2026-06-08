from __future__ import annotations

from src.config.settings import get_settings
from src.core.rag.vector_store import VectorStore, get_vector_store
from src.schemas.rag import RetrievalResult, RetrievedChunk
from src.utils.exceptions import RetrievalError
from src.utils.helpers import truncate_text
from src.utils.logger import logger


class Retriever:

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        top_k: int | None = None,
        min_score: float | None = None,
    ) -> None:
        settings = get_settings()
        self._store = vector_store or get_vector_store()
        self._top_k = top_k or settings.top_k_retrieval
        self._min_score = min_score or settings.min_relevance_score

    async def retrieve(self, query: str) -> RetrievalResult:
        if not query or not query.strip():
            raise RetrievalError("Query cannot be empty")

        try:
            chunks = await self._store.search(
                query=query.strip(),
                top_k=self._top_k,
                min_score=self._min_score,
            )
        except Exception as e:
            raise RetrievalError(
                "Vector store search failed",
                detail=str(e),
            ) from e

        context = self._assemble_context(chunks)
        citations = self._extract_citations(chunks)

        result = RetrievalResult(
            query=query,
            chunks=chunks,
            assembled_context=context,
            citations=citations,
        )

        logger.info(
            "Retrieved {n} chunks for query={q!r} (top score={s:.3f})",
            n=len(chunks),
            q=truncate_text(query, 60),
            s=result.top_score,
        )
        return result

    def _assemble_context(self, chunks: list[RetrievedChunk]) -> str:
        if not chunks:
            return ""

        sorted_chunks = sorted(chunks, key=lambda c: c.score, reverse=True)

        parts = []
        for i, retrieved in enumerate(sorted_chunks, 1):
            chunk = retrieved.chunk
            source_label = f"[Source {i}: {chunk.source}]"
            parts.append(f"{source_label}\n{chunk.content}")

        return "\n\n---\n\n".join(parts)

    def _extract_citations(self, chunks: list[RetrievedChunk]) -> list[str]:
        seen = set()
        citations = []
        for retrieved in chunks:
            source = retrieved.chunk.source
            if source not in seen:
                seen.add(source)
                citations.append(source)
        return citations

    @property
    def top_k(self) -> int:
        return self._top_k

    @property
    def min_score(self) -> float:
        return self._min_score
