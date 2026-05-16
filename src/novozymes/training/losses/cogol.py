"""CoGOLLoss — Combination of Group Ordinal Losses."""

import torch
from torch.nn import functional as F

from novozymes.constants import HeadEnum
from novozymes.training.losses.base import LossStrategy


class CoGOLLoss(LossStrategy):
    """CoGOL ordinal loss — cross-entropy over ordinal class assignments.

    Discretizes continuous tm targets into class labels,
    then applies standard cross-entropy.
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__(num_classes)

    @property
    def required_head(self) -> HeadEnum:
        return HeadEnum.COGOL

    def __call__(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """CoGOL loss between [B, K] logits and [B] continuous targets.

        Targets are discretized into class labels [B] (integers 0..K-1).
        """
        # Discretize targets into class indices [B]
        t_min = targets.min()
        t_max = targets.max()
        t_range = t_max - t_min
        if t_range == 0:
            class_labels = torch.zeros(targets.shape[0], dtype=torch.long, device=targets.device)
        else:
            normalized = (targets - t_min) / t_range
            class_labels = (normalized * (self.num_classes - 1)).long()
            class_labels = class_labels.clamp(0, self.num_classes - 1)

        return F.cross_entropy(preds, class_labels)
