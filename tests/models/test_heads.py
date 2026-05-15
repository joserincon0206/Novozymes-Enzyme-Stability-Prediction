"""Tests for Head ABC and all registered heads."""

import pytest
import torch
from torch import nn

from novozymes.constants import HeadEnum
from novozymes.models.heads.base import Head
from novozymes.models.heads.registry import build_head


class TestHeadABC:
    """Head ABC defines the interface."""

    def test_cannot_instantiate_abc(self) -> None:
        """ABC cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Head()  # type: ignore[abstract]


class TestHeadRegistry:
    """All registered heads satisfy the Head contract."""

    @pytest.mark.parametrize("head_enum", HeadEnum)
    def test_registry_returns_head(self, head_enum: HeadEnum) -> None:
        """Every registered head is a Head subclass."""
        head = build_head(head_enum, input_dim=320, num_classes=50)
        assert isinstance(head, Head)
        assert isinstance(head, nn.Module)

    @pytest.mark.parametrize("head_enum", HeadEnum)
    def test_forward_returns_tensor(self, head_enum: HeadEnum) -> None:
        """forward() returns a tensor."""
        head = build_head(head_enum, input_dim=320, num_classes=50)
        x = torch.randn(8, 320)
        out = head(x)
        assert isinstance(out, torch.Tensor)

    @pytest.mark.parametrize("head_enum", HeadEnum)
    @pytest.mark.parametrize("batch_size", [1, 4, 8])
    def test_forward_batch_dim_preserved(self, head_enum: HeadEnum, batch_size: int) -> None:
        """Output batch dimension matches input."""
        head = build_head(head_enum, input_dim=128, num_classes=20)
        x = torch.randn(batch_size, 128)
        out = head(x)
        assert out.shape[0] == batch_size

    def test_regression_head_outputs_scalar(self) -> None:
        """RegressionHead outputs [B, 1]."""
        head = build_head(HeadEnum.REGRESSION, input_dim=128, num_classes=1)
        x = torch.randn(4, 128)
        out = head(x)
        assert out.shape == (4, 1)

    @pytest.mark.parametrize("head_enum", [HeadEnum.CORAL, HeadEnum.COGOL])
    @pytest.mark.parametrize("num_classes", [10, 20, 50])
    def test_ordinal_heads_output_num_classes(self, head_enum: HeadEnum, num_classes: int) -> None:
        """Ordinal heads output [B, num_classes]."""
        head = build_head(head_enum, input_dim=128, num_classes=num_classes)
        x = torch.randn(4, 128)
        out = head(x)
        assert out.shape == (4, num_classes)

    @pytest.mark.parametrize("head_enum", HeadEnum)
    def test_forward_returns_float_tensor(self, head_enum: HeadEnum) -> None:
        """Output is float tensor."""
        head = build_head(head_enum, input_dim=64, num_classes=10)
        x = torch.randn(4, 64)
        out = head(x)
        assert out.dtype == torch.float32
