"""Enzyme datamodule for data loading."""

from typing import Any

import torch
from torch.utils.data import DataLoader, random_split

from novozymes.config.data import DataConfig
from novozymes.data.base import DataModule
from novozymes.data.dataset import EnzymeDataset


class EnzymeDataModule(DataModule):
    """Data module for enzyme stability dataset."""

    def __init__(self, config: DataConfig) -> None:
        """Initialize with DataConfig."""
        self.config = config
        self.train_dataset: Any = None
        self.val_dataset: Any = None

    def setup(self) -> None:
        """Split data into train/val."""
        full_dataset = EnzymeDataset(self.config.train_path)
        total = len(full_dataset)
        val_size = int(total * self.config.val_fraction)
        train_size = total - val_size

        self.train_dataset, self.val_dataset = random_split(
            full_dataset,
            [train_size, val_size],
            generator=None,
        )

    def train_dataloader(self) -> DataLoader:
        """Train dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=self._collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        """Validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=self._collate_fn,
        )

    @staticmethod
    def _collate_fn(batch: list[Any]) -> tuple[list[str], torch.Tensor, torch.Tensor]:
        """Collate batch: keep sequences as strings, stack tensors."""
        sequences = [item[0] for item in batch]
        ph = [item[1] for item in batch]
        tm = [item[2] for item in batch]

        ph_tensor = torch.tensor(ph, dtype=torch.float32)
        tm_tensor = torch.tensor(tm, dtype=torch.float32)

        return sequences, ph_tensor, tm_tensor
