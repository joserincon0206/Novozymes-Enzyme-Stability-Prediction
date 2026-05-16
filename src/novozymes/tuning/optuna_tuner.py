"""OptunaTuner — hyperparameter optimization with Optuna + MLflow."""

from typing import Any

import lightning as L
import optuna
from lightning.pytorch.callbacks import EarlyStopping
from optuna.samplers import TPESampler

from novozymes.config.data import DataConfig
from novozymes.config.model import ModelConfig
from novozymes.config.training import OptunaConfig, TrainingConfig
from novozymes.constants import LossEnum, StrategyEnum
from novozymes.data.datamodule import EnzymeDataModule
from novozymes.models.registry import build_strategy
from novozymes.training.lightning_module import EnzymeLightningModule
from novozymes.training.losses.registry import build_loss
from novozymes.utils.logging import get_mlflow_logger
from novozymes.utils.seed import seed_everything


class OptunaTuner:
    """Optuna HPO tuner — one MLflow run per trial."""

    def __init__(
        self,
        data_config: DataConfig,
        training_config: TrainingConfig,
        optuna_config: OptunaConfig,
    ) -> None:
        self.data_config = data_config
        self.training_config = training_config
        self.optuna_config = optuna_config

    def _objective(self, trial: optuna.Trial) -> float:
        """Single Optuna trial."""
        seed_everything(42)

        # Sample hyperparameters
        strategy = trial.suggest_categorical("strategy", [s.value for s in StrategyEnum])
        loss = trial.suggest_categorical("loss", [le.value for le in LossEnum])
        lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        hidden_dim = trial.suggest_categorical("hidden_dim", [128, 256, 512])
        dropout = trial.suggest_float("dropout", 0.0, 0.5)
        num_classes = trial.suggest_categorical("num_classes", [10, 20, 50, 100])

        strategy_enum = StrategyEnum(strategy)
        loss_enum = LossEnum(loss)

        # Build
        model_config = ModelConfig(
            strategy=strategy_enum,
            learning_rate=lr,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )

        datamodule = EnzymeDataModule(self.data_config)
        datamodule.setup()

        model_strategy = build_strategy(
            strategy_enum, loss_enum=loss_enum, num_classes=num_classes, config=model_config
        )
        loss_fn = build_loss(loss_enum, num_classes=num_classes)
        module = EnzymeLightningModule(strategy=model_strategy, loss_fn=loss_fn)

        logger = get_mlflow_logger(
            experiment_name=f"{self.training_config.mlflow_experiment_name}_hpo",
            run_name=f"trial_{trial.number}",
            tracking_uri=self.training_config.mlflow_tracking_uri,
        )

        callbacks: list[Any] = [
            EarlyStopping(monitor="val/loss", patience=5, mode="min"),
        ]

        trainer = L.Trainer(
            max_epochs=self.training_config.max_epochs,
            logger=logger,
            callbacks=callbacks,
            accelerator="auto",
            enable_checkpointing=False,
        )
        trainer.fit(module, datamodule.train_dataloader(), datamodule.val_dataloader())

        val_loss = trainer.callback_metrics.get("val/loss")
        if val_loss is None:
            return float("inf")
        return float(val_loss)

    def run(self) -> optuna.Study:
        """Run the HPO study."""
        study = optuna.create_study(
            direction="minimize",
            sampler=TPESampler(),
        )
        study.optimize(
            self._objective,
            n_trials=self.optuna_config.n_trials,
            n_jobs=self.optuna_config.n_jobs,
            timeout=self.optuna_config.timeout,
        )
        return study
