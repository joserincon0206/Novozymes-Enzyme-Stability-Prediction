"""Entry point: generate submission.csv from a trained checkpoint."""

from pathlib import Path

import pandas as pd
import torch

from novozymes.config.model import ModelConfig
from novozymes.constants import LossEnum, StrategyEnum
from novozymes.models.registry import build_strategy
from novozymes.training.lightning_module import EnzymeLightningModule
from novozymes.training.losses.registry import build_loss


def predict(
    checkpoint_path: Path,
    test_path: Path = Path("data/raw/test.csv"),
    output_path: Path = Path("submission.csv"),
    strategy: StrategyEnum = StrategyEnum.ESM_FROZEN,
    loss: LossEnum = LossEnum.MSE,
    num_classes: int = 50,
) -> None:
    """Generate predictions on test set."""
    # Load test data
    test_df = pd.read_csv(test_path)
    sequences = test_df["protein_sequence"].tolist()
    ph = torch.tensor(test_df["pH"].values, dtype=torch.float32)

    # Build model and load weights
    model_config = ModelConfig(strategy=strategy)
    model_strategy = build_strategy(
        strategy, loss_enum=loss, num_classes=num_classes, config=model_config
    )
    loss_fn = build_loss(loss, num_classes=num_classes)
    module = EnzymeLightningModule.load_from_checkpoint(
        str(checkpoint_path), strategy=model_strategy, loss_fn=loss_fn
    )
    module.eval()

    # Predict
    with torch.no_grad():
        preds = module.strategy(sequences, ph)
        scalar_preds = module._to_scalar(preds).squeeze(-1)

    # Write submission
    submission = pd.DataFrame(
        {
            "seq_id": test_df["seq_id"],
            "tm": scalar_preds.cpu().numpy(),
        }
    )
    submission.to_csv(output_path, index=False)
    print(f"Submission written to {output_path} ({len(submission)} rows)")


if __name__ == "__main__":
    predict(checkpoint_path=Path("best.ckpt"))
