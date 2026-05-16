"""Entry point: Optuna HPO study."""

from pathlib import Path

from novozymes.config.data import DataConfig
from novozymes.config.training import OptunaConfig, TrainingConfig
from novozymes.tuning.optuna_tuner import OptunaTuner


def tune(
    train_path: Path = Path("data/raw/train.csv"),
    n_trials: int = 20,
    max_epochs: int = 10,
) -> None:
    """Run HPO study."""
    data_config = DataConfig(train_path=train_path)
    training_config = TrainingConfig(max_epochs=max_epochs)
    optuna_config = OptunaConfig(n_trials=n_trials)

    tuner = OptunaTuner(
        data_config=data_config,
        training_config=training_config,
        optuna_config=optuna_config,
    )
    study = tuner.run()

    print("\n=== Best Trial ===")
    print(f"  Value: {study.best_trial.value}")
    print(f"  Params: {study.best_trial.params}")


if __name__ == "__main__":
    tune()
