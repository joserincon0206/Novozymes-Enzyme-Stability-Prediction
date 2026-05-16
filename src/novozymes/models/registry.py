"""Strategy registry — maps StrategyEnum to implementations."""

from novozymes.config.model import ModelConfig
from novozymes.constants import LossEnum, StrategyEnum
from novozymes.models.base import ModelStrategy
from novozymes.models.esm_finetune import ESMFinetuneStrategy
from novozymes.models.esm_frozen import ESMFrozenStrategy

STRATEGY_REGISTRY: dict[StrategyEnum, type[ModelStrategy]] = {
    StrategyEnum.ESM_FROZEN: ESMFrozenStrategy,
    StrategyEnum.ESM_FINETUNE: ESMFinetuneStrategy,
}


def build_strategy(
    strategy_enum: StrategyEnum,
    loss_enum: LossEnum = LossEnum.MSE,
    num_classes: int = 50,
    config: ModelConfig | None = None,
) -> ModelStrategy:
    """Build a model strategy from enum."""
    if config is None:
        config = ModelConfig(strategy=strategy_enum)
    return STRATEGY_REGISTRY[strategy_enum](
        config=config, loss_enum=loss_enum, num_classes=num_classes
    )
