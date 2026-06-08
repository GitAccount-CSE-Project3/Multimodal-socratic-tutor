from __future__ import annotations

from abc import ABC, abstractmethod

from src.schemas.rag import DocumentChunk, RetrievedChunk


class VectorStore(ABC):

    @abstractmethod
    async def add_chunks(self, chunks: list[DocumentChunk]) -> None:
        ...

    @abstractmethod
    async def search(self, query: str, top_k: int = 5, min_score: float = 0.0) -> list[RetrievedChunk]:
        ...

    @abstractmethod
    async def count(self) -> int:
        ...

    @abstractmethod
    async def reset(self) -> None:
        ...
