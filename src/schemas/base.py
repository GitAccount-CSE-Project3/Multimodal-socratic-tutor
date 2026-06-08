from __future__ import annotations

__all__ = ["BaseSchema", "TimestampedSchema", "IdentifiedSchema"]

from datetime import datetime, timezone
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field


class BaseSchema(BaseModel):

    model_config = ConfigDict(
        frozen=False,
        validate_assignment=True,
        populate_by_name=True,
        str_strip_whitespace=True,
    )


class TimestampedSchema(BaseSchema):

    created_at: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))


class IdentifiedSchema(TimestampedSchema):

    id: UUID = Field(default_factory=uuid4)
