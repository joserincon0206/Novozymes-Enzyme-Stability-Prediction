"""Tests for LossStrategy ABC and all registered losses."""

import pytest
import torch

from novozymes.constants import HeadEnum, LossEnum
from novozymes.training.losses.base import LossStrategy
from novozymes.training.losses.registry import build_loss


class TestLossStrategyABC:
    """LossStrategy ABC defines the interface."""

    def test_cannot_instantiate_abc(self) -> None:
        """ABC cannot be instantiated directly."""
        with pytest.raises(TypeError):
            LossStrategy()  # type: ignore[abstract]


class TestLossRegistry:
    """All registered losses satisfy the LossStrategy contract."""

    @pytest.mark.parametrize("loss_enum", LossEnum)
    def test_registry_returns_loss_strategy(self, loss_enum: LossEnum) -> None:
        """Every registered loss is a LossStrategy subclass."""
        loss = build_loss(loss_enum, num_classes=20)
        assert isinstance(loss, LossStrategy)

    @pytest.mark.parametrize("loss_enum", LossEnum)
    def test_required_head_returns_head_enum(self, loss_enum: LossEnum) -> None:
        """required_head returns a valid HeadEnum."""
        loss = build_loss(loss_enum, num_classes=20)
        assert isinstance(loss.required_head, HeadEnum)

    @pytest.mark.parametrize("loss_enum", LossEnum)
    def test_call_returns_scalar_tensor(self, loss_enum: LossEnum) -> None:
        """__call__ returns a scalar loss tensor."""
        loss = build_loss(loss_enum, num_classes=20)
        if loss.required_head == HeadEnum.REGRESSION:
            preds = torch.randn(8, 1)
        else:
            preds = torch.randn(8, 20)
        targets = torch.rand(8) * 100
        result = loss(preds, targets)
        assert result.shape == ()
        assert result.dtype == torch.float32

    @pytest.mark.parametrize("loss_enum", LossEnum)
    def test_loss_is_finite(self, loss_enum: LossEnum) -> None:
        """Loss value is finite for reasonable inputs."""
        loss = build_loss(loss_enum, num_classes=20)
        if loss.required_head == HeadEnum.REGRESSION:
            preds = torch.randn(8, 1)
        else:
            preds = torch.randn(8, 20)
        targets = torch.rand(8) * 100
        result = loss(preds, targets)
        assert torch.isfinite(result)

    @pytest.mark.parametrize("loss_enum", LossEnum)
    def test_loss_requires_grad(self, loss_enum: LossEnum) -> None:
        """Loss supports backpropagation."""
        loss = build_loss(loss_enum, num_classes=20)
        if loss.required_head == HeadEnum.REGRESSION:
            preds = torch.randn(8, 1, requires_grad=True)
        else:
            preds = torch.randn(8, 20, requires_grad=True)
        targets = torch.rand(8) * 100
        result = loss(preds, targets)
        result.backward()
        assert preds.grad is not None

    @pytest.mark.parametrize("loss_enum", LossEnum)
    @pytest.mark.parametrize("batch_size", [1, 4, 16])
    def test_loss_works_with_different_batch_sizes(
        self, loss_enum: LossEnum, batch_size: int
    ) -> None:
        """Loss works across batch sizes."""
        loss = build_loss(loss_enum, num_classes=20)
        if loss.required_head == HeadEnum.REGRESSION:
            preds = torch.randn(batch_size, 1)
        else:
            preds = torch.randn(batch_size, 20)
        targets = torch.rand(batch_size) * 100
        result = loss(preds, targets)
        assert result.shape == ()
