"""Data configuration."""

from pathlib import Path

from pydantic import Field

from novozymes.config.base import BaseConfig


class DataConfig(BaseConfig):
    """Data loading and preprocessing configuration."""

    train_path: Path = Path("data/raw/train.csv")
    val_fraction: float = Field(default=0.1, ge=0.0, le=1.0)
    batch_size: int = Field(default=32, gt=0)
    max_sequence_length: int = Field(default=1000, gt=0)
