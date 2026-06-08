from __future__ import annotations

__all__ = ["LLMFactory", "get_llm", "get_vision_llm"]

from functools import lru_cache
from typing import Any

from langchain_core.language_models import BaseLLM

from src.config.settings import get_settings
from src.utils.exceptions import LLMUnavailableError
from src.utils.logger import logger

_DEFAULT_TEMPERATURE: float = 0.7


class LLMFactory:


    @staticmethod
    def _build_openai(model_name: str, **kwargs: Any) -> BaseLLM:
        try:
            from langchain_openai import ChatOpenAI

            settings = get_settings()
            if not settings.openai_api_key:
                raise LLMUnavailableError(
                    "OPENAI_API_KEY not set",
                    detail="Add OPENAI_API_KEY=sk-... to your .env file",
                )
            return ChatOpenAI(
                model=model_name,
                api_key=settings.openai_api_key,
                temperature=_DEFAULT_TEMPERATURE,
                **kwargs,
            )
        except ImportError as e:
            raise LLMUnavailableError(
                "langchain-openai not installed",
                detail="Run: pip install langchain-openai",
            ) from e


    @classmethod
    def get_llm(cls, **kwargs: Any) -> BaseLLM:
        settings = get_settings()
        logger.info("LLM: OpenAI {m}", m=settings.openai_llm_model)
        return cls._build_openai(settings.openai_llm_model, **kwargs)

    @classmethod
    def get_vision_llm(cls, **kwargs: Any) -> BaseLLM:
        settings = get_settings()
        logger.info("Vision LLM: OpenAI {m}", m=settings.openai_vision_model)
        return cls._build_openai(settings.openai_vision_model, **kwargs)


@lru_cache(maxsize=1)
def get_llm() -> BaseLLM:
    return LLMFactory.get_llm()


@lru_cache(maxsize=1)
def get_vision_llm() -> BaseLLM:
    return LLMFactory.get_vision_llm()
