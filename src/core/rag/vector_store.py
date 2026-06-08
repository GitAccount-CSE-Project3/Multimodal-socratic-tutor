from __future__ import annotations

from functools import lru_cache

from src.config.settings import VectorStoreType, get_settings
from src.core.rag.stores.base import VectorStore
from src.core.rag.stores.chroma import ChromaVectorStore
from src.core.rag.stores.faiss import FAISSVectorStore
from src.utils.logger import logger


@lru_cache(maxsize=1)
def get_vector_store() -> VectorStore:
    settings = get_settings()
    if settings.vector_store_type == VectorStoreType.FAISS:
        logger.info("Using FAISS vector store")
        return FAISSVectorStore()
    logger.info("Using ChromaDB vector store")
    return ChromaVectorStore()
