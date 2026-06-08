from __future__ import annotations

from enum import Enum
from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class AppEnv(str, Enum):
    DEVELOPMENT = "development"
    PRODUCTION = "production"

class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"

class LLMProvider(str, Enum):
    OPENAI = "openai"

class EmbeddingProvider(str, Enum):
    OPENAI = "openai"
    SENTENCE_TRANSFORMERS = "sentence-transformers"

class EmbeddingDevice(str, Enum):
    CPU = "cpu"
    MPS = "mps"
    CUDA = "cuda"

class VectorStoreType(str, Enum):
    CHROMA = "chroma"
    FAISS = "faiss"

_INT_BOUNDS = {
    "chunk_size": (128, 2048),
    "chunk_overlap": (0, 256),
    "top_k_retrieval": (1, 20),
    "max_hint_turns": (1, 5),
    "max_session_turns": (5, 100),
}

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_env: AppEnv = Field(default=AppEnv.DEVELOPMENT)
    app_debug: bool = Field(default=True)
    app_log_level: LogLevel = Field(default=LogLevel.INFO)

    llm_provider: LLMProvider = Field(default=LLMProvider.OPENAI)

    openai_api_key: str | None = Field(default=None)
    openai_llm_model: str = Field(default="gpt-4o-mini")
    openai_vision_model: str = Field(default="gpt-4o")
    openai_embedding_model: str = Field(default="text-embedding-3-small")

    openai_tts_model: str = Field(default="gpt-4o-mini-tts")
    openai_stt_model: str = Field(default="gpt-4o-mini-transcribe")
    tts_voice: str = Field(default="alloy")

    embedding_provider: EmbeddingProvider = Field(default=EmbeddingProvider.OPENAI)
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    embedding_device: EmbeddingDevice = Field(default=EmbeddingDevice.CPU)

    vector_store_type: VectorStoreType = Field(default=VectorStoreType.CHROMA)
    chroma_persist_dir: str = Field(default="./data/processed/chroma_db")
    faiss_index_path: str = Field(default="./data/processed/faiss_index/index.bin")

    chunk_size: int = Field(default=512, ge=128, le=2048)
    chunk_overlap: int = Field(default=64, ge=0, le=256)
    top_k_retrieval: int = Field(default=5, ge=1, le=20)
    min_relevance_score: float = Field(default=0.35, ge=0.0, le=1.0)

    max_hint_turns: int = Field(default=2, ge=1, le=5)
    max_session_turns: int = Field(default=30, ge=5, le=100)
    socratic_strict_mode: bool = Field(default=True)

    database_url: str = Field(default="sqlite+aiosqlite:///./data/socratot.db")

    streamlit_port: int = Field(default=8501)
    streamlit_host: str = Field(default="0.0.0.0")  # noqa: S104  # intentional: container/LAN access

    log_dir: str = Field(default="./logs")
    log_rotation: str = Field(default="10 MB")

    eval_dataset_path: str = Field(default="./evaluation/ground_truth.jsonl")

    @field_validator(
        "chunk_size",
        "chunk_overlap",
        "top_k_retrieval",
        "max_hint_turns",
        "max_session_turns",
        mode="before",
    )
    @classmethod
    def _clamp_int_range(cls, v: object, info: object) -> int:
        lo, hi = _INT_BOUNDS[info.field_name]
        try:
            n = int(v)
        except (TypeError, ValueError):
            return lo
        return max(lo, min(hi, n))

    @field_validator("min_relevance_score", mode="before")
    @classmethod
    def _clamp_relevance(cls, v: object) -> float:
        try:
            return max(0.0, min(1.0, float(v)))
        except (TypeError, ValueError):
            return 0.35

    @field_validator("chunk_overlap")
    @classmethod
    def overlap_less_than_chunk(cls, v: int, info: object) -> int:
        data = getattr(info, "data", {})
        chunk_size = data.get("chunk_size", 512)
        if v >= chunk_size:
            return max(0, chunk_size - 1)
        return v

    @property
    def is_production(self) -> bool:
        return self.app_env == AppEnv.PRODUCTION

    @property
    def is_development(self) -> bool:
        return self.app_env == AppEnv.DEVELOPMENT

    @property
    def using_openai(self) -> bool:
        return self.llm_provider == LLMProvider.OPENAI

    @property
    def chroma_persist_path(self) -> Path:
        return Path(self.chroma_persist_dir)

    @property
    def log_path(self) -> Path:
        return Path(self.log_dir)

    @property
    def database_path(self) -> Path:
        url = self.database_url
        if "///" in url:
            return Path(url.split("///")[-1])
        return Path("./data/socratot.db")

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

def _load_streamlit_secrets() -> None:
    try:
        import streamlit as st

        for key, val in st.secrets.items():
            import os

            if key.upper() not in os.environ:
                os.environ[key.upper()] = str(val)
    except Exception:  # noqa: S110  # no st.secrets when not on Streamlit Cloud
        pass

try:
    _load_streamlit_secrets()
    get_settings.cache_clear()
except Exception:  # noqa: S110  # secrets loading is best-effort
    pass
