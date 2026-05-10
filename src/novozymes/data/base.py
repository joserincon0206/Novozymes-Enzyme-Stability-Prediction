"""Base datamodule interface."""

from abc import ABC, abstractmethod

from torch.utils.data import DataLoader


class DataModule(ABC):
    """Abstract datamodule interface."""

    @abstractmethod
    def setup(self) -> None:
        """Prepare train/val datasets."""

    @abstractmethod
    def train_dataloader(self) -> DataLoader:
        """Return train dataloader."""

    @abstractmethod
    def val_dataloader(self) -> DataLoader:
        """Return validation dataloader."""
