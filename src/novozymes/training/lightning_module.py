"""EnzymeLightningModule — thin orchestrator wrapping strategy + loss."""

from typing import Any

import lightning as L
import torch

from novozymes.models.base import ModelStrategy
from novozymes.training.losses.base import LossStrategy


class EnzymeLightningModule(L.LightningModule):
    """Lightning module that delegates to ModelStrategy and LossStrategy.

    Never changes when new strategies or losses are added.
    """

    def __init__(self, strategy: ModelStrategy, loss_fn: LossStrategy) -> None:
        super().__init__()
        self.strategy = strategy
        self.loss_fn = loss_fn

    def _shared_step(
        self, batch: tuple[list[str], torch.Tensor, torch.Tensor], stage: str
    ) -> torch.Tensor:
        """Shared logic for training and validation steps."""
        sequences, ph, tm = batch
        preds = self.strategy(sequences, ph)
        loss = self.loss_fn(preds, tm)
        self.log(f"{stage}/loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def training_step(
        self, batch: tuple[list[str], torch.Tensor, torch.Tensor], batch_idx: int = 0
    ) -> torch.Tensor:
        """Training step."""
        return self._shared_step(batch, "train")

    def validation_step(
        self, batch: tuple[list[str], torch.Tensor, torch.Tensor], batch_idx: int = 0
    ) -> torch.Tensor:
        """Validation step."""
        return self._shared_step(batch, "val")

    def configure_optimizers(self) -> Any:
        """Delegate to strategy."""
        return self.strategy.configure_optimizers()

    def _to_scalar(self, preds: torch.Tensor) -> torch.Tensor:
        """Collapse ordinal outputs to scalar predictions.

        For [B, 1]: identity.
        For [B, K]: expected ordinal rank via sigmoid + sum.
        """
        if preds.shape[-1] > 1:
            probs = torch.sigmoid(preds)
            return probs.sum(dim=-1, keepdim=True)
        return preds
