"""Base configuration class."""

from pydantic import BaseModel, ConfigDict


class BaseConfig(BaseModel):
    """Frozen, validated base config."""

    model_config = ConfigDict(frozen=True, extra="forbid")
