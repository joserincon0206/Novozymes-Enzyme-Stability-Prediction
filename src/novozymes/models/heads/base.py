"""Head ABC — interface for all prediction heads."""

from abc import ABCMeta, abstractmethod

import torch
from torch import nn


class Head(nn.Module, metaclass=ABCMeta):
    """Abstract prediction head."""

    @abstractmethod
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """Map features to predictions.

        Args:
            features: [B, D] feature tensor

        Returns:
            [B, 1] for scalar heads, [B, num_classes] for ordinal heads
        """
