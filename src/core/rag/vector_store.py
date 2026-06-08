from __future__ import annotations

import asyncio
import pickle
from abc import ABC, abstractmethod
from functools import lru_cache
from pathlib import Path

from src.config.settings import VectorStoreType, get_settings
from src.core.rag.embedder import Embedder
from src.schemas.rag import DocumentChunk, RetrievedChunk
from src.utils.exceptions import CorpusEmptyError, VectorStoreError
from src.utils.logger import logger


class VectorStore(ABC):

    @abstractmethod
    async def add_chunks(self, chunks: list[DocumentChunk]) -> None:
        ...

    @abstractmethod
    async def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> list[RetrievedChunk]:
        ...

    @abstractmethod
    async def count(self) -> int:
        ...

    @abstractmethod
    async def reset(self) -> None:
        ...


def _make_noop_embedding_function() -> object:
    from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

    class _NoOpEmbeddingFunction(EmbeddingFunction):
        def __call__(self, input: Documents) -> Embeddings:
            raise RuntimeError(
                "Embeddings are computed externally (OpenAI) — pass "
                "embeddings/query_embeddings to Chroma explicitly."
            )

    return _NoOpEmbeddingFunction()


class ChromaVectorStore(VectorStore):

    COLLECTION_NAME = "socratot_anatomy"

    def __init__(self) -> None:
        settings = get_settings()
        self._persist_dir = str(settings.chroma_persist_path)
        self._embedder = Embedder()
        self._client: object = None
        self._collection: object = None

    def _get_collection(self) -> object:
        if self._collection is not None:
            return self._collection

        try:
            import logging
            import os

            os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
            logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)
            logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)

            import chromadb
            from chromadb.config import Settings as ChromaSettings

            self._client = chromadb.PersistentClient(
                path=self._persist_dir,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
            self._collection = self._client.get_or_create_collection(
                name=self.COLLECTION_NAME,
                metadata={"hnsw:space": "cosine"},
                embedding_function=_make_noop_embedding_function(),
            )
            logger.info(
                "ChromaDB collection ready: {name} at {path}",
                name=self.COLLECTION_NAME,
                path=self._persist_dir,
            )
            return self._collection

        except ImportError as e:
            raise VectorStoreError(
                "chromadb is not installed",
                detail=str(e),
            ) from e
        except Exception as e:
            raise VectorStoreError(
                "Failed to initialise ChromaDB",
                detail=str(e),
            ) from e

    async def add_chunks(self, chunks: list[DocumentChunk]) -> None:
        if not chunks:
            return

        collection = self._get_collection()

        embeddings = await self._embedder.embed_chunks(chunks)

        ids = [str(chunk.id) for chunk in chunks]
        documents = [chunk.content for chunk in chunks]
        metadatas = [
            {
                "source": chunk.source,
                "chapter": chunk.chapter or "",
                "topic_tags": ",".join(chunk.topic_tags),
                "token_count": chunk.token_count,
            }
            for chunk in chunks
        ]

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            lambda: collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            ),
        )
        logger.info("Added {n} chunks to ChromaDB", n=len(chunks))

    async def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> list[RetrievedChunk]:
        collection = self._get_collection()

        count = await self.count()
        if count == 0:
            raise CorpusEmptyError(
                "Vector store is empty",
                detail="Run scripts/ingest_corpus.py first",
            )

        query_embedding = await self._embedder.embed_query(query)

        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(
            None,
            lambda: collection.query(
                query_embeddings=[query_embedding],
                n_results=min(top_k, count),
                include=["documents", "metadatas", "distances"],
            ),
        )

        retrieved: list[RetrievedChunk] = []
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]
        ids = results.get("ids", [[]])[0]

        for _doc_id, doc, meta, dist in zip(ids, documents, metadatas, distances):
            score = max(0.0, 1.0 - (dist / 2.0))

            if score < min_score:
                continue

            chunk = DocumentChunk(
                content=doc,
                source=meta.get("source", ""),
                chapter=meta.get("chapter") or None,
                topic_tags=meta.get("topic_tags", "").split(","),
                token_count=int(meta.get("token_count", 0)),
            )

            retrieved.append(RetrievedChunk(chunk=chunk, score=round(score, 4)))

        logger.debug(
            "ChromaDB search: query={q!r}, results={n}",
            q=query[:50],
            n=len(retrieved),
        )
        return retrieved

    async def count(self) -> int:
        try:
            collection = self._get_collection()
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, collection.count)
        except Exception:
            return 0

    async def reset(self) -> None:
        if self._client is not None:
            self._client.delete_collection(self.COLLECTION_NAME)
            self._collection = None
        logger.warning("ChromaDB collection reset — all chunks deleted")


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
                import pickle

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

    async def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> list[RetrievedChunk]:
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

        retrieved: list[RetrievedChunk] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or score < min_score:
                continue
            retrieved.append(
                RetrievedChunk(
                    chunk=self._chunks[idx],
                    score=round(float(score), 4),
                )
            )

        return retrieved

    async def count(self) -> int:
        return len(self._chunks)

    async def reset(self) -> None:
        self._index = None
        self._chunks = []
        if self._index_path.exists():
            self._index_path.unlink()
        logger.warning("FAISS index reset")


@lru_cache(maxsize=1)
def get_vector_store() -> VectorStore:
    settings = get_settings()
    if settings.vector_store_type == VectorStoreType.FAISS:
        logger.info("Using FAISS vector store")
        return FAISSVectorStore()
    logger.info("Using ChromaDB vector store")
    return ChromaVectorStore()
