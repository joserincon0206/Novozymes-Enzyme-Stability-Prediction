"""ModelStrategy ABC — interface for all model strategies."""

from abc import ABCMeta, abstractmethod

import torch
from torch import nn


class ModelStrategy(nn.Module, metaclass=ABCMeta):
    """Abstract model strategy.

    Each strategy owns its own feature extraction, head, and optimizer config.
    forward() takes raw sequences — tokenization is strategy-specific.
    """

    @abstractmethod
    def forward(self, sequences: list[str], ph: torch.Tensor) -> torch.Tensor:
        """Run forward pass.

        Args:
            sequences: raw protein sequences [B]
            ph: pH values [B]

        Returns:
            [B, 1] for scalar heads, [B, num_classes] for ordinal heads
        """

    @abstractmethod
    def configure_optimizers(self) -> list[dict[str, object]]:
        """Return optimizer config(s).

        Returns:
            List of dicts with 'optimizer' key (and optional 'scheduler').
        """
