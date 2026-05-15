"""Head registry — maps HeadEnum to implementations."""

from novozymes.constants import HeadEnum
from novozymes.models.heads.base import Head
from novozymes.models.heads.cogol import CoGOLHead
from novozymes.models.heads.coral import CORALHead
from novozymes.models.heads.regression import RegressionHead

HEAD_REGISTRY: dict[HeadEnum, type[Head]] = {
    HeadEnum.REGRESSION: RegressionHead,
    HeadEnum.CORAL: CORALHead,
    HeadEnum.COGOL: CoGOLHead,
}


def build_head(head_enum: HeadEnum, input_dim: int, num_classes: int) -> Head:
    """Build a head from enum."""
    head_cls = HEAD_REGISTRY[head_enum]
    if head_enum == HeadEnum.REGRESSION:
        return head_cls(input_dim=input_dim)
    return head_cls(input_dim=input_dim, num_classes=num_classes)
