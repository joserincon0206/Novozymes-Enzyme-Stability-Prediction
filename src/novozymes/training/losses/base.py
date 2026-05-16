"""LossStrategy ABC — interface for all loss functions."""

from abc import ABC, abstractmethod

import torch

from novozymes.constants import HeadEnum


class LossStrategy(ABC):
    """Abstract loss strategy.

    All subclasses must accept num_classes in __init__ for uniform dispatch.
    """

    def __init__(self, num_classes: int) -> None:
        self.num_classes = num_classes

    @abstractmethod
    def __call__(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute loss.

        Args:
            preds: [B, 1] for regression, [B, num_classes] for ordinal
            targets: [B] float tm values

        Returns:
            Scalar loss tensor
        """

    @property
    @abstractmethod
    def required_head(self) -> HeadEnum:
        """Which head this loss requires."""
