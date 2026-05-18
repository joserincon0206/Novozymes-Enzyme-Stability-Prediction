"""Tests for EnzymeLightningModule."""

import random

import pytest
import torch

from novozymes.constants import LossEnum, StrategyEnum
from novozymes.models.registry import build_strategy
from novozymes.training.lightning_module import EnzymeLightningModule
from novozymes.training.losses.registry import build_loss

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def make_sequences(n: int, length: int = 20) -> list[str]:
    """Generate random protein sequences."""
    return ["".join(random.choices(AMINO_ACIDS, k=length)) for _ in range(n)]


def make_batch(
    batch_size: int = 4,
) -> tuple[list[str], torch.Tensor, torch.Tensor]:
    """Generate a synthetic batch."""
    sequences = make_sequences(batch_size)
    ph = torch.rand(batch_size)
    tm = torch.rand(batch_size) * 100
    return sequences, ph, tm


class TestEnzymeLightningModule:
    """EnzymeLightningModule wraps strategy + loss."""

    @pytest.mark.parametrize("strategy_enum", StrategyEnum)
    @pytest.mark.parametrize("loss_enum", LossEnum)
    def test_training_step_returns_loss(
        self, strategy_enum: StrategyEnum, loss_enum: LossEnum
    ) -> None:
        """training_step returns a scalar loss tensor."""
        strategy = build_strategy(strategy_enum, loss_enum=loss_enum, num_classes=20)
        loss_fn = build_loss(loss_enum, num_classes=20)
        module = EnzymeLightningModule(strategy=strategy, loss_fn=loss_fn)
        batch = make_batch()
        loss = module.training_step(batch)
        assert isinstance(loss, torch.Tensor)
        assert loss.shape == ()
        assert torch.isfinite(loss)

    @pytest.mark.parametrize("strategy_enum", StrategyEnum)
    @pytest.mark.parametrize("loss_enum", LossEnum)
    def test_validation_step_returns_loss(
        self, strategy_enum: StrategyEnum, loss_enum: LossEnum
    ) -> None:
        """validation_step returns a scalar loss tensor."""
        strategy = build_strategy(strategy_enum, loss_enum=loss_enum, num_classes=20)
        loss_fn = build_loss(loss_enum, num_classes=20)
        module = EnzymeLightningModule(strategy=strategy, loss_fn=loss_fn)
        batch = make_batch()
        loss = module.validation_step(batch)
        assert isinstance(loss, torch.Tensor)
        assert loss.shape == ()

    @pytest.mark.parametrize("strategy_enum", StrategyEnum)
    def test_configure_optimizers_delegates_to_strategy(
        self, strategy_enum: StrategyEnum
    ) -> None:
        """configure_optimizers delegates to strategy."""
        strategy = build_strategy(strategy_enum, loss_enum=LossEnum.MSE)
        loss_fn = build_loss(LossEnum.MSE)
        module = EnzymeLightningModule(strategy=strategy, loss_fn=loss_fn)
        result = module.configure_optimizers()
        assert isinstance(result, list)
        assert len(result) > 0

    @pytest.mark.parametrize("strategy_enum", StrategyEnum)
    def test_to_scalar_identity_for_regression(self, strategy_enum: StrategyEnum) -> None:
        """_to_scalar is identity for [B, 1] inputs."""
        strategy = build_strategy(strategy_enum, loss_enum=LossEnum.MSE)
        loss_fn = build_loss(LossEnum.MSE)
        module = EnzymeLightningModule(strategy=strategy, loss_fn=loss_fn)
        preds = torch.randn(4, 1)
        result = module._to_scalar(preds)
        assert torch.allclose(result, preds)

    @pytest.mark.parametrize("strategy_enum", StrategyEnum)
    def test_to_scalar_collapses_ordinal(self, strategy_enum: StrategyEnum) -> None:
        """_to_scalar collapses [B, K] ordinal to [B, 1]."""
        strategy = build_strategy(strategy_enum, loss_enum=LossEnum.MSE)
        loss_fn = build_loss(LossEnum.MSE)
        module = EnzymeLightningModule(strategy=strategy, loss_fn=loss_fn)
        preds = torch.randn(4, 20)
        result = module._to_scalar(preds)
        assert result.shape == (4, 1)
