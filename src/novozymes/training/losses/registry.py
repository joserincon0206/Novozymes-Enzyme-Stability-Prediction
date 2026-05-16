"""Loss registry — maps LossEnum to implementations."""

from novozymes.constants import LossEnum
from novozymes.training.losses.base import LossStrategy
from novozymes.training.losses.cogol import CoGOLLoss
from novozymes.training.losses.coral import CORALLoss
from novozymes.training.losses.mse import MSELoss

LOSS_REGISTRY: dict[LossEnum, type[LossStrategy]] = {
    LossEnum.MSE: MSELoss,
    LossEnum.CORAL: CORALLoss,
    LossEnum.COGOL: CoGOLLoss,
}


def build_loss(loss_enum: LossEnum, num_classes: int = 50) -> LossStrategy:
    """Build a loss from enum."""
    return LOSS_REGISTRY[loss_enum](num_classes=num_classes)
