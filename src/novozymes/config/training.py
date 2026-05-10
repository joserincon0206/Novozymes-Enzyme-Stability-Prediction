"""Training configuration."""

from pydantic import Field

from novozymes.config.base import BaseConfig
from novozymes.constants import LossEnum


class LossConfig(BaseConfig):
    """Loss function configuration."""

    loss: LossEnum
    num_classes: int = Field(default=50, gt=0)


class TrainingConfig(BaseConfig):
    """Training hyperparameter configuration."""

    max_epochs: int = Field(default=50, gt=0)
    early_stopping_patience: int = Field(default=10, gt=0)
    mlflow_experiment_name: str = "enzyme-stability"
    mlflow_tracking_uri: str = "http://localhost:5000"


class OptunaConfig(BaseConfig):
    """Optuna hyperparameter tuning configuration."""

    n_trials: int = Field(default=20, gt=0)
    n_jobs: int = Field(default=1, gt=0)
    timeout: int = Field(default=3600, gt=0)
