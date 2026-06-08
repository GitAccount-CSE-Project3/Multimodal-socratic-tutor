from __future__ import annotations

import asyncio
import pickle
from pathlib import Path

from src.config.settings import get_settings
from src.core.rag.embedder import Embedder
from src.core.rag.stores.base import VectorStore
from src.schemas.rag import DocumentChunk, RetrievedChunk
from src.utils.exceptions import CorpusEmptyError, VectorStoreError
from src.utils.logger import logger


class FAISSVectorStore(VectorStore):

    def __init__(self) -> None:
        settings = get_settings()
        self._index_path = Path(settings.faiss_index_path)
        self._embedder = Embedder()
        self._index: object = None
        self._chunks: list[DocumentChunk] = []

    def _get_index(self, dim: int = 384) -> object:
        if self._index is not None:
            return self._index
        if self._index_path.exists():
            try:
                import faiss
                self._index = faiss.read_index(str(self._index_path))
                chunks_path = self._index_path.with_suffix(".pkl")
                if chunks_path.exists():
                    with open(chunks_path, "rb") as f:
                        self._chunks = pickle.load(f)  # noqa: S301
                logger.info("Loaded FAISS index from disk")
                return self._index
            except Exception as e:
                logger.warning("Could not load FAISS index: {e}", e=str(e))
        try:
            import faiss
            self._index = faiss.IndexFlatIP(dim)
            logger.info("Created new FAISS index (dim={d})", d=dim)
            return self._index
        except ImportError as e:
            raise VectorStoreError("faiss-cpu is not installed", detail=str(e)) from e

    async def add_chunks(self, chunks: list[DocumentChunk]) -> None:
        import faiss
        import numpy as np
        embeddings = await self._embedder.embed_chunks(chunks)
        vectors = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(vectors)
        dim = vectors.shape[1]
        index = self._get_index(dim)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, lambda: index.add(vectors))
        self._chunks.extend(chunks)
        self._index_path.parent.mkdir(parents=True, exist_ok=True)
        await loop.run_in_executor(None, lambda: faiss.write_index(index, str(self._index_path)))
        chunks_path = self._index_path.with_suffix(".pkl")
        with open(chunks_path, "wb") as f:
            pickle.dump(self._chunks, f)
        logger.info("Added {n} chunks to FAISS index", n=len(chunks))

    async def search(self, query: str, top_k: int = 5, min_score: float = 0.0) -> list[RetrievedChunk]:
        import faiss
        import numpy as np
        if not self._chunks:
            raise CorpusEmptyError("FAISS index is empty")
        query_vec = await self._embedder.embed_query(query)
        q = np.array([query_vec], dtype=np.float32)
        faiss.normalize_L2(q)
        index = self._get_index()
        k = min(top_k, len(self._chunks))
        loop = asyncio.get_running_loop()
        scores, indices = await loop.run_in_executor(None, lambda: index.search(q, k))
        retrieved = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or score < min_score:
                continue
            retrieved.append(RetrievedChunk(chunk=self._chunks[idx], score=round(float(score), 4)))
        return retrieved

    async def count(self) -> int:
        return len(self._chunks)

    async def reset(self) -> None:
        self._index = None
        self._chunks = []
        if self._index_path.exists():
            self._index_path.unlink()
        logger.warning("FAISS index reset")
