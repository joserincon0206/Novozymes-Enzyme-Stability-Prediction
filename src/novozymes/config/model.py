"""Model configuration."""

from pydantic import Field

from novozymes.config.base import BaseConfig
from novozymes.constants import StrategyEnum


class ModelConfig(BaseConfig):
    """Model strategy and hyperparameter configuration."""

    strategy: StrategyEnum
    esm_model_name: str = "facebook/esm2_t6_8M_UR50D"
    hidden_dim: int = Field(default=256, gt=0)
    dropout: float = Field(default=0.1, ge=0.0, le=1.0)
    learning_rate: float = Field(default=1e-4, gt=0)


class FineTuneModelConfig(ModelConfig):
    """Extended config for fine-tuning strategies."""

    unfreeze_last_n_layers: int = Field(default=6, gt=0)
    esm_lr: float = Field(default=1e-5, gt=0)
