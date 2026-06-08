from __future__ import annotations

__all__ = [
    "AudioSynthesisError",
    "AudioTranscriptionError",
    "ConfigurationError",
    "CorpusEmptyError",
    "DatabaseError",
    "EmbeddingError",
    "ImageAnalysisError",
    "ImageProcessingError",
    "InvalidPhaseTransitionError",
    "LLMResponseParseError",
    "LLMTimeoutError",
    "LLMUnavailableError",
    "ModelNotFoundError",
    "RetrievalError",
    "SessionExpiredError",
    "SessionNotFoundError",
    "SocratOTError",
    "UnsupportedImageFormatError",
    "VectorStoreError",
]


class SocratOTError(Exception):
    pass

    def __init__(self, message: str, detail: str = "") -> None:
        super().__init__(message)
        self.message = message
        self.detail = detail

    def __str__(self) -> str:
        if self.detail:
            return f"{self.message} — {self.detail}"
        return self.message


class LLMUnavailableError(SocratOTError):
    pass


class LLMTimeoutError(SocratOTError):
    pass


class LLMResponseParseError(SocratOTError):
    pass


class ModelNotFoundError(SocratOTError):
    pass


class VectorStoreError(SocratOTError):
    pass


class EmbeddingError(SocratOTError):
    pass


class CorpusEmptyError(SocratOTError):
    pass


class RetrievalError(SocratOTError):
    pass


class ImageProcessingError(SocratOTError):
    pass


class ImageAnalysisError(SocratOTError):
    pass


class UnsupportedImageFormatError(SocratOTError):
    pass


class SessionNotFoundError(SocratOTError):
    pass


class SessionExpiredError(SocratOTError):
    pass


class InvalidPhaseTransitionError(SocratOTError):
    pass


class AudioTranscriptionError(SocratOTError):
    pass


class AudioSynthesisError(SocratOTError):
    pass


class DatabaseError(SocratOTError):
    pass


class ConfigurationError(SocratOTError):
    pass
