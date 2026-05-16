"""Entry point: single training run."""

from pathlib import Path

import lightning as L
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from novozymes.config.data import DataConfig
from novozymes.config.model import ModelConfig
from novozymes.config.training import LossConfig, TrainingConfig
from novozymes.constants import LossEnum, StrategyEnum
from novozymes.data.datamodule import EnzymeDataModule
from novozymes.models.registry import build_strategy
from novozymes.training.lightning_module import EnzymeLightningModule
from novozymes.training.losses.registry import build_loss
from novozymes.utils.logging import get_mlflow_logger
from novozymes.utils.seed import seed_everything


def train(
    strategy: StrategyEnum = StrategyEnum.ESM_FROZEN,
    loss: LossEnum = LossEnum.MSE,
    train_path: Path = Path("data/raw/train.csv"),
    num_classes: int = 50,
    max_epochs: int = 50,
    batch_size: int = 32,
    seed: int = 42,
) -> None:
    """Run a single training experiment."""
    seed_everything(seed)

    data_config = DataConfig(train_path=train_path, batch_size=batch_size)
    model_config = ModelConfig(strategy=strategy)
    loss_config = LossConfig(loss=loss, num_classes=num_classes)
    training_config = TrainingConfig(max_epochs=max_epochs)

    # Data
    datamodule = EnzymeDataModule(data_config)
    datamodule.setup()

    # Model
    model_strategy = build_strategy(
        strategy, loss_enum=loss, num_classes=loss_config.num_classes, config=model_config
    )
    loss_fn = build_loss(loss, num_classes=loss_config.num_classes)
    module = EnzymeLightningModule(strategy=model_strategy, loss_fn=loss_fn)

    # Logger
    logger = get_mlflow_logger(
        experiment_name=training_config.mlflow_experiment_name,
        run_name=f"{strategy.value}_{loss.value}",
        tracking_uri=training_config.mlflow_tracking_uri,
    )

    # Callbacks
    callbacks = [
        EarlyStopping(
            monitor="val/loss", patience=training_config.early_stopping_patience, mode="min"
        ),
        ModelCheckpoint(monitor="val/loss", mode="min", save_top_k=1),
    ]

    # Train
    trainer = L.Trainer(
        max_epochs=training_config.max_epochs,
        logger=logger,
        callbacks=callbacks,
        accelerator="auto",
    )
    trainer.fit(module, datamodule.train_dataloader(), datamodule.val_dataloader())


if __name__ == "__main__":
    train()
