from __future__ import annotations

import asyncio

from src.config.settings import get_settings
from src.core.rag.embedder import Embedder
from src.core.rag.stores.base import VectorStore
from src.schemas.rag import DocumentChunk, RetrievedChunk
from src.utils.exceptions import CorpusEmptyError, VectorStoreError
from src.utils.logger import logger


def _make_noop_embedding_function() -> object:
    from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

    class _NoOpEmbeddingFunction(EmbeddingFunction):
        def __call__(self, input: Documents) -> Embeddings:
            raise RuntimeError("Embeddings are computed externally — pass them explicitly.")

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
            logger.info("ChromaDB ready: {name} at {path}", name=self.COLLECTION_NAME, path=self._persist_dir)
            return self._collection
        except ImportError as e:
            raise VectorStoreError("chromadb is not installed", detail=str(e)) from e
        except Exception as e:
            raise VectorStoreError("Failed to initialise ChromaDB", detail=str(e)) from e

    async def add_chunks(self, chunks: list[DocumentChunk]) -> None:
        if not chunks:
            return
        collection = self._get_collection()
        embeddings = await self._embedder.embed_chunks(chunks)
        ids = [str(c.id) for c in chunks]
        documents = [c.content for c in chunks]
        metadatas = [
            {"source": c.source, "chapter": c.chapter or "", "topic_tags": ",".join(c.topic_tags), "token_count": c.token_count}
            for c in chunks
        ]
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, lambda: collection.add(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas))
        logger.info("Added {n} chunks to ChromaDB", n=len(chunks))

    async def search(self, query: str, top_k: int = 5, min_score: float = 0.0) -> list[RetrievedChunk]:
        collection = self._get_collection()
        count = await self.count()
        if count == 0:
            raise CorpusEmptyError("Vector store is empty", detail="Run scripts/ingest_corpus.py first")
        query_embedding = await self._embedder.embed_query(query)
        loop = asyncio.get_running_loop()
        results = await loop.run_in_executor(
            None,
            lambda: collection.query(query_embeddings=[query_embedding], n_results=min(top_k, count), include=["documents", "metadatas", "distances"]),
        )
        retrieved: list[RetrievedChunk] = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]
        ids = results.get("ids", [[]])[0]
        for _id, doc, meta, dist in zip(ids, docs, metas, dists):
            score = max(0.0, 1.0 - (dist / 2.0))
            if score < min_score:
                continue
            chunk = DocumentChunk(
                content=doc, source=meta.get("source", ""), chapter=meta.get("chapter") or None,
                topic_tags=meta.get("topic_tags", "").split(","), token_count=int(meta.get("token_count", 0)),
            )
            retrieved.append(RetrievedChunk(chunk=chunk, score=round(score, 4)))
        logger.debug("ChromaDB search: {q!r} -> {n} results", q=query[:50], n=len(retrieved))
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
        logger.warning("ChromaDB collection reset")
