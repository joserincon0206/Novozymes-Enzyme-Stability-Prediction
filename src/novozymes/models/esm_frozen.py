"""ESMFrozenStrategy — frozen ESM-2 backbone + pluggable head."""

import torch
from torch import optim

from novozymes.config.model import ModelConfig
from novozymes.constants import LossEnum
from novozymes.features.esm_embedding import ESMEmbeddingExtractor
from novozymes.models.base import ModelStrategy
from novozymes.models.heads.registry import build_head
from novozymes.training.losses.registry import build_loss


class ESMFrozenStrategy(ModelStrategy):
    """Frozen ESM-2 embeddings + trainable head.

    Feature extraction runs under torch.no_grad().
    Only the head parameters are optimized.
    """

    def __init__(
        self,
        config: ModelConfig,
        loss_enum: LossEnum,
        num_classes: int = 50,
    ) -> None:
        super().__init__()
        self.config = config
        self.extractor = ESMEmbeddingExtractor(config.esm_model_name)
        loss = build_loss(loss_enum, num_classes=num_classes)
        self.head = build_head(loss.required_head, self.extractor.output_dim, num_classes)

    def forward(self, sequences: list[str], ph: torch.Tensor) -> torch.Tensor:
        """Extract frozen embeddings, pass through head."""
        features = self.extractor.extract(sequences, ph)
        result: torch.Tensor = self.head(features)
        return result

    def configure_optimizers(self) -> list[dict[str, object]]:
        """Optimize head parameters only."""
        optimizer = optim.AdamW(self.head.parameters(), lr=self.config.learning_rate)
        return [{"optimizer": optimizer}]
