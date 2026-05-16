"""Tests for ModelStrategy ABC and all registered strategies."""

import random

import pytest
import torch
from torch import nn

from novozymes.constants import LossEnum, StrategyEnum
from novozymes.models.base import ModelStrategy
from novozymes.models.registry import build_strategy

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def make_sequences(n: int, length: int = 20) -> list[str]:
    """Generate random protein sequences."""
    return ["".join(random.choices(AMINO_ACIDS, k=length)) for _ in range(n)]


class TestModelStrategyABC:
    """ModelStrategy ABC defines the interface."""

    def test_cannot_instantiate_abc(self) -> None:
        """ABC cannot be instantiated directly."""
        with pytest.raises(TypeError):
            ModelStrategy()  # type: ignore[abstract]


class TestStrategyRegistry:
    """All registered strategies satisfy the ModelStrategy contract."""

    @pytest.mark.parametrize("strategy_enum", StrategyEnum)
    def test_registry_returns_model_strategy(self, strategy_enum: StrategyEnum) -> None:
        """Every registered strategy is a ModelStrategy subclass."""
        strategy = build_strategy(strategy_enum, loss_enum=LossEnum.MSE)
        assert isinstance(strategy, ModelStrategy)
        assert isinstance(strategy, nn.Module)

    @pytest.mark.parametrize("strategy_enum", StrategyEnum)
    @pytest.mark.parametrize("loss_enum", LossEnum)
    def test_forward_returns_tensor(self, strategy_enum: StrategyEnum, loss_enum: LossEnum) -> None:
        """forward() returns a tensor for every strategy × loss combination."""
        strategy = build_strategy(strategy_enum, loss_enum=loss_enum, num_classes=20)
        sequences = make_sequences(4)
        ph = torch.rand(4)
        out = strategy(sequences, ph)
        assert isinstance(out, torch.Tensor)

    @pytest.mark.parametrize("strategy_enum", StrategyEnum)
    @pytest.mark.parametrize("batch_size", [1, 4])
    def test_forward_batch_dim_preserved(
        self, strategy_enum: StrategyEnum, batch_size: int
    ) -> None:
        """Output batch dimension matches input."""
        strategy = build_strategy(strategy_enum, loss_enum=LossEnum.MSE)
        sequences = make_sequences(batch_size)
        ph = torch.rand(batch_size)
        out = strategy(sequences, ph)
        assert out.shape[0] == batch_size

    @pytest.mark.parametrize("strategy_enum", StrategyEnum)
    def test_configure_optimizers_returns_list(self, strategy_enum: StrategyEnum) -> None:
        """configure_optimizers() returns a list of optimizer configs."""
        strategy = build_strategy(strategy_enum, loss_enum=LossEnum.MSE)
        optimizers = strategy.configure_optimizers()
        assert isinstance(optimizers, list)
        assert len(optimizers) > 0

    @pytest.mark.parametrize("strategy_enum", StrategyEnum)
    def test_regression_head_outputs_scalar(self, strategy_enum: StrategyEnum) -> None:
        """With MSE loss, output is [B, 1]."""
        strategy = build_strategy(strategy_enum, loss_enum=LossEnum.MSE)
        sequences = make_sequences(4)
        ph = torch.rand(4)
        out = strategy(sequences, ph)
        assert out.shape == (4, 1)

    @pytest.mark.parametrize("strategy_enum", StrategyEnum)
    @pytest.mark.parametrize("loss_enum", [LossEnum.CORAL, LossEnum.COGOL])
    def test_ordinal_head_outputs_num_classes(
        self, strategy_enum: StrategyEnum, loss_enum: LossEnum
    ) -> None:
        """With ordinal loss, output is [B, num_classes]."""
        strategy = build_strategy(strategy_enum, loss_enum=loss_enum, num_classes=20)
        sequences = make_sequences(4)
        ph = torch.rand(4)
        out = strategy(sequences, ph)
        assert out.shape == (4, 20)
