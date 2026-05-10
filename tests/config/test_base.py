"""Tests for configuration system (enum-driven, type-safe)."""

import pytest
from pydantic import ValidationError

from novozymes.config.data import DataConfig
from novozymes.config.model import FineTuneModelConfig, ModelConfig
from novozymes.config.training import LossConfig, TrainingConfig
from novozymes.constants import LossEnum, StrategyEnum

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def data_config() -> DataConfig:
    """Valid DataConfig."""
    return DataConfig()


@pytest.fixture
def model_config() -> ModelConfig:
    """Valid ModelConfig."""
    return ModelConfig(strategy=StrategyEnum.ESM_FROZEN)


@pytest.fixture
def finetune_model_config() -> FineTuneModelConfig:
    """Valid FineTuneModelConfig."""
    return FineTuneModelConfig(strategy=StrategyEnum.ESM_FINETUNE)


@pytest.fixture
def loss_config() -> LossConfig:
    """Valid LossConfig."""
    return LossConfig(loss=LossEnum.MSE)


@pytest.fixture
def training_config() -> TrainingConfig:
    """Valid TrainingConfig."""
    return TrainingConfig()


# ============================================================================
# Invariant Tests (apply to all configs)
# ============================================================================


@pytest.mark.parametrize(
    "config",
    [
        DataConfig(),
        ModelConfig(strategy=StrategyEnum.ESM_FROZEN),
        LossConfig(loss=LossEnum.MSE),
        TrainingConfig(),
    ],
    ids=["DataConfig", "ModelConfig", "LossConfig", "TrainingConfig"],
)
def test_all_configs_are_frozen(config) -> None:
    """All configs are immutable."""
    with pytest.raises((ValidationError, AttributeError, TypeError)):
        config.foo = "bar"  # type: ignore


def test_model_config_forbids_extra_fields() -> None:
    """Extra fields rejected."""
    with pytest.raises(ValidationError):
        ModelConfig(strategy=StrategyEnum.ESM_FROZEN, extra_field="value")  # type: ignore


# ============================================================================
# Enum Acceptance Tests (parameterized)
# ============================================================================


@pytest.mark.parametrize("strategy", StrategyEnum)
def test_model_config_accepts_all_strategies(strategy: StrategyEnum) -> None:
    """ModelConfig accepts all StrategyEnum members."""
    cfg = ModelConfig(strategy=strategy)
    assert cfg.strategy == strategy


@pytest.mark.parametrize("loss", LossEnum)
def test_loss_config_accepts_all_losses(loss: LossEnum) -> None:
    """LossConfig accepts all LossEnum members."""
    cfg = LossConfig(loss=loss)
    assert cfg.loss == loss


# ============================================================================
# Validation Tests (edge cases, parameterized)
# ============================================================================


@pytest.mark.parametrize("batch_size", [0, -1, -100])
def test_data_config_batch_size_positive(batch_size: int) -> None:
    """batch_size must be positive."""
    with pytest.raises(ValidationError):
        DataConfig(batch_size=batch_size)


@pytest.mark.parametrize("val_fraction", [-0.1, 1.1, 2.0])
def test_data_config_val_fraction_in_range(val_fraction: float) -> None:
    """val_fraction must be in [0, 1]."""
    with pytest.raises(ValidationError):
        DataConfig(val_fraction=val_fraction)


# ============================================================================
# Required Field Tests
# ============================================================================


def test_model_config_requires_strategy() -> None:
    """strategy is required."""
    with pytest.raises(ValidationError):
        ModelConfig()  # type: ignore


def test_loss_config_requires_loss() -> None:
    """loss is required."""
    with pytest.raises(ValidationError):
        LossConfig()  # type: ignore


# ============================================================================
# Inheritance Tests
# ============================================================================


def test_finetune_model_config_inherits_from_model_config(
    finetune_model_config: FineTuneModelConfig,
) -> None:
    """FineTuneModelConfig has ModelConfig fields."""
    assert hasattr(finetune_model_config, "strategy")
    assert hasattr(finetune_model_config, "hidden_dim")
    assert hasattr(finetune_model_config, "unfreeze_last_n_layers")
    assert hasattr(finetune_model_config, "esm_lr")
