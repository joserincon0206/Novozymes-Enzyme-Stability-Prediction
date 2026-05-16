"""MSELoss — baseline regression loss."""

import torch
from torch.nn import functional as F

from novozymes.constants import HeadEnum
from novozymes.training.losses.base import LossStrategy


class MSELoss(LossStrategy):
    """Mean squared error loss for scalar regression."""

    def __init__(self, num_classes: int = 1) -> None:
        super().__init__(num_classes)

    @property
    def required_head(self) -> HeadEnum:
        return HeadEnum.REGRESSION

    def __call__(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """MSE between [B, 1] predictions and [B] targets."""
        return F.mse_loss(preds.squeeze(-1), targets)
