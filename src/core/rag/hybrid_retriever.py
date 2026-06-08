from __future__ import annotations

from src.config.settings import get_settings
from src.core.rag.vector_store import VectorStore, get_vector_store
from src.schemas.rag import RetrievalResult, RetrievedChunk
from src.utils.exceptions import RetrievalError
from src.utils.helpers import truncate_text
from src.utils.logger import logger


class HybridRetriever:

    def __init__(
        self,
        vector_store: VectorStore | None = None,
        top_k: int | None = None,
        min_score: float | None = None,
        bm25_weight: float = 0.3,
        cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        use_cross_encoder: bool = True,
    ) -> None:
        settings = get_settings()
        self._store = vector_store or get_vector_store()
        self._top_k = top_k or settings.top_k_retrieval
        self._min_score = min_score or settings.min_relevance_score
        self._bm25_weight = bm25_weight
        self._cross_encoder_model = cross_encoder_model
        self._use_cross_encoder = use_cross_encoder
        self._cross_encoder = None

    def _get_cross_encoder(self) -> object:
        if self._cross_encoder is None:
            from sentence_transformers import CrossEncoder
            self._cross_encoder = CrossEncoder(self._cross_encoder_model)
        return self._cross_encoder

    def _bm25_scores(self, query: str, chunks: list[RetrievedChunk]) -> list[float]:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            return [0.0] * len(chunks)

        tokenized_corpus = [c.chunk.content.lower().split() for c in chunks]
        tokenized_query = query.lower().split()

        if not tokenized_corpus:
            return []

        bm25 = BM25Okapi(tokenized_corpus)
        scores = bm25.get_scores(tokenized_query)

        max_score = max(scores) if scores.max() > 0 else 1.0
        return [float(s / max_score) for s in scores]

    def _cross_encoder_scores(self, query: str, chunks: list[RetrievedChunk]) -> list[float]:
        ce = self._get_cross_encoder()
        pairs = [[query, c.chunk.content[:512]] for c in chunks]
        raw = ce.predict(pairs)

        import math
        sigmoid = [1 / (1 + math.exp(-float(s))) for s in raw]
        return sigmoid

    async def retrieve(self, query: str) -> RetrievalResult:
        if not query or not query.strip():
            raise RetrievalError("Query cannot be empty")

        candidate_k = min(self._top_k * 4, 20)

        try:
            chunks = await self._store.search(
                query=query.strip(),
                top_k=candidate_k,
                min_score=0.0,
            )
        except Exception as e:
            raise RetrievalError("Vector store search failed", detail=str(e)) from e

        if not chunks:
            return RetrievalResult(
                query=query,
                chunks=[],
                assembled_context="",
                citations=[],
            )

        vector_scores = [c.score for c in chunks]
        max_v = max(vector_scores) if max(vector_scores) > 0 else 1.0
        norm_vector = [s / max_v for s in vector_scores]

        bm25 = self._bm25_scores(query, chunks)
        if not bm25:
            bm25 = [0.0] * len(chunks)

        vector_weight = 1.0 - self._bm25_weight
        combined = [
            vector_weight * v + self._bm25_weight * b
            for v, b in zip(norm_vector, bm25)
        ]

        ranked = sorted(zip(combined, chunks), key=lambda x: x[0], reverse=True)

        if self._use_cross_encoder and len(ranked) > 1:
            try:
                top_candidates = [c for _, c in ranked[:10]]
                ce_scores = self._cross_encoder_scores(query, top_candidates)
                reranked = sorted(zip(ce_scores, top_candidates), key=lambda x: x[0], reverse=True)
                final = [c for _, c in reranked[: self._top_k]]
            except Exception as e:
                logger.warning("CrossEncoder reranking failed: {e}", e=str(e))
                final = [c for _, c in ranked[: self._top_k]]
        else:
            final = [c for _, c in ranked[: self._top_k]]

        final = [c for c in final if c.score >= self._min_score]

        context = self._assemble_context(final)
        citations = self._extract_citations(final)

        logger.info(
            "Hybrid retrieval: {n} chunks (bm25_weight={w}, cross_encoder={ce})",
            n=len(final),
            w=self._bm25_weight,
            ce=self._use_cross_encoder,
        )
        return RetrievalResult(
            query=query,
            chunks=final,
            assembled_context=context,
            citations=citations,
        )

    def _assemble_context(self, chunks: list[RetrievedChunk]) -> str:
        if not chunks:
            return ""
        parts = []
        for i, c in enumerate(chunks, 1):
            parts.append(f"[Source {i}: {c.chunk.source}]\n{c.chunk.content}")
        return "\n\n---\n\n".join(parts)

    def _extract_citations(self, chunks: list[RetrievedChunk]) -> list[str]:
        seen: set = set()
        citations = []
        for c in chunks:
            src = c.chunk.source
            if src not in seen:
                seen.add(src)
                citations.append(src)
        return citations

    @property
    def top_k(self) -> int:
        return self._top_k

    @property
    def min_score(self) -> float:
        return self._min_score
