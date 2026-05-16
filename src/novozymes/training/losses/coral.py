"""CORALLoss — Consistent Rank Logits ordinal regression loss."""

import torch
from torch.nn import functional as F

from novozymes.constants import HeadEnum
from novozymes.training.losses.base import LossStrategy


class CORALLoss(LossStrategy):
    """CORAL ordinal loss — binary cross-entropy over K threshold classifiers.

    Discretizes continuous tm targets into ordinal labels,
    then applies BCE to each threshold independently.
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__(num_classes)

    @property
    def required_head(self) -> HeadEnum:
        return HeadEnum.CORAL

    def __call__(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """CORAL loss between [B, K] logits and [B] continuous targets.

        Targets are discretized into ordinal labels: for each threshold k,
        label is 1 if target > bin_edge[k], else 0.
        """
        # Discretize targets into ordinal labels [B, K]
        t_min = targets.min()
        t_max = targets.max()
        # Avoid division by zero for constant targets
        t_range = t_max - t_min
        if t_range == 0:
            t_range = torch.tensor(1.0)
        edges = torch.linspace(
            float(t_min), float(t_max), self.num_classes + 1, device=targets.device
        )
        ordinal_labels = torch.zeros(targets.shape[0], self.num_classes, device=targets.device)
        for k in range(self.num_classes):
            ordinal_labels[:, k] = (targets > edges[k + 1]).float()

        return F.binary_cross_entropy_with_logits(preds, ordinal_labels)
