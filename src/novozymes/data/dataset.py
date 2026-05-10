"""Enzyme dataset."""

from pathlib import Path

import pandas as pd
from torch.utils.data import Dataset


class EnzymeDataset(Dataset):
    """Protein thermostability dataset."""

    def __init__(self, csv_path: Path) -> None:
        """Load CSV with columns: protein_sequence, pH, tm."""
        self.df = pd.read_csv(csv_path)

    def __len__(self) -> int:
        """Number of samples."""
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple[str, float, float]:
        """Return (sequence, pH, tm)."""
        row = self.df.iloc[idx]
        sequence = row["protein_sequence"]
        ph = float(row["pH"])
        tm = float(row["tm"])
        return sequence, ph, tm
