"""Type-safe enums for strategy, loss, and head selection."""

from enum import Enum


class StrategyEnum(str, Enum):
    """Valid model strategies."""

    ESM_FROZEN = "esm_frozen"
    ESM_FINETUNE = "esm_finetune"


class LossEnum(str, Enum):
    """Valid loss functions."""

    MSE = "mse"
    CORAL = "coral"
    COGOL = "cogol"


class HeadEnum(str, Enum):
    """Valid output heads."""

    REGRESSION = "regression"
    CORAL = "coral"
    COGOL = "cogol"


class FeatureExtractorEnum(str, Enum):
    """Valid feature extractors."""

    ESM_EMBEDDING = "esm_embedding"


class StageEnum(str, Enum):
    """Training stages."""

    FIT = "fit"
    TEST = "test"
    PREDICT = "predict"
