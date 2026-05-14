"""FeatureExtractor ABC — interface for all feature extractors."""

from abc import ABC, abstractmethod

import torch


class FeatureExtractor(ABC):
    """Extract features from protein sequences."""

    @abstractmethod
    def extract(self, sequences: list[str], ph: torch.Tensor) -> torch.Tensor:
        """Extract features from sequences and pH values.

        Args:
            sequences: list of protein sequences [B]
            ph: pH values tensor [B]

        Returns:
            Feature tensor [B, D]
        """

    @property
    @abstractmethod
    def output_dim(self) -> int:
        """Dimensionality of the output feature vector."""
