"""Tests for data module (dataset and datamodule)."""

import random
import string
import tempfile
from pathlib import Path

import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from novozymes.config.data import DataConfig
from novozymes.data.datamodule import EnzymeDataModule
from novozymes.data.dataset import EnzymeDataset

# ============================================================================
# Synthetic Data Generation
# ============================================================================


def make_sequences(n: int, length: int = 10) -> list[str]:
    """Generate n random protein sequences."""
    return ["".join(random.choices(string.ascii_uppercase, k=length)) for _ in range(n)]


def make_csv(n: int = 10, path: Path | None = None) -> Path:
    """Generate temp CSV with n random rows. No hardcoded data."""
    sequences = make_sequences(n)
    df = pd.DataFrame(
        {
            "seq_id": [f"seq_{i:03d}" for i in range(n)],
            "protein_sequence": sequences,
            "pH": [6.0 + random.random() * 2.0 for _ in range(n)],
            "data_source": [random.choice(["A", "B", "C"]) for _ in range(n)],
            "tm": [30.0 + random.random() * 40.0 for _ in range(n)],
        }
    )
    if path is None:
        f = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
        path = Path(f.name)
        f.close()
    df.to_csv(path, index=False)
    return path


# ============================================================================
# EnzymeDataset Tests
# ============================================================================


class TestEnzymeDataset:
    """EnzymeDataset loads CSV and returns (sequence, pH, tm) tuples."""

    def test_dataset_length(self) -> None:
        """Length matches CSV rows."""
        csv_path = make_csv(n=10)
        dataset = EnzymeDataset(csv_path)
        assert len(dataset) == 10

    def test_dataset_item_shape(self) -> None:
        """Each item is (str, float, float)."""
        csv_path = make_csv(n=5)
        dataset = EnzymeDataset(csv_path)
        seq, ph, tm = dataset[0]
        assert isinstance(seq, str)
        assert isinstance(ph, float)
        assert isinstance(tm, float)

    @pytest.mark.parametrize("index", [0, 2, 4])
    def test_dataset_item_valid(self, index: int) -> None:
        """Items are valid across indices."""
        csv_path = make_csv(n=10)
        dataset = EnzymeDataset(csv_path)
        seq, ph, tm = dataset[index]
        assert len(seq) > 0
        assert 4.0 <= ph <= 9.0
        assert 20.0 <= tm <= 80.0


# ============================================================================
# EnzymeDataModule Tests
# ============================================================================


class TestEnzymeDataModule:
    """EnzymeDataModule creates train/val splits and dataloaders."""

    @pytest.mark.parametrize("val_fraction", [0.1, 0.2, 0.3])
    def test_train_val_split(self, val_fraction: float) -> None:
        """Train + val splits cover all data."""
        csv_path = make_csv(n=100)
        cfg = DataConfig(train_path=csv_path, val_fraction=val_fraction)
        dm = EnzymeDataModule(cfg)
        dm.setup()

        total = len(dm.train_dataset) + len(dm.val_dataset)
        assert total == 100

        expected_val = int(100 * val_fraction)
        expected_train = 100 - expected_val
        assert len(dm.val_dataset) == expected_val
        assert len(dm.train_dataset) == expected_train

    @pytest.mark.parametrize("batch_size", [1, 4, 8])
    def test_train_dataloader_batch_size(self, batch_size: int) -> None:
        """Train dataloader respects batch_size."""
        csv_path = make_csv(n=20)
        cfg = DataConfig(train_path=csv_path, batch_size=batch_size)
        dm = EnzymeDataModule(cfg)
        dm.setup()

        loader = dm.train_dataloader()
        assert isinstance(loader, DataLoader)
        assert loader.batch_size == batch_size

    def test_batch_structure(self) -> None:
        """Batch is (list[str], Tensor[pH], Tensor[tm])."""
        csv_path = make_csv(n=10)
        cfg = DataConfig(train_path=csv_path, batch_size=4)
        dm = EnzymeDataModule(cfg)
        dm.setup()

        loader = dm.train_dataloader()
        sequences, ph, tm = next(iter(loader))

        assert isinstance(sequences, list)
        assert all(isinstance(s, str) for s in sequences)
        assert isinstance(ph, torch.Tensor)
        assert isinstance(tm, torch.Tensor)
        assert ph.shape == (len(sequences),)
        assert tm.shape == (len(sequences),)

    def test_val_dataloader_exists(self) -> None:
        """val_dataloader() returns valid DataLoader."""
        csv_path = make_csv(n=20)
        cfg = DataConfig(train_path=csv_path)
        dm = EnzymeDataModule(cfg)
        dm.setup()

        loader = dm.val_dataloader()
        assert isinstance(loader, DataLoader)
        assert len(loader) > 0
