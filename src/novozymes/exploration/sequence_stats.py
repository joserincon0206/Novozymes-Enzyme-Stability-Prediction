"""SequenceAnalyzer — sequence length distribution and AA frequency."""

from collections import Counter
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


class SequenceAnalyzer:
    """Analyze protein sequence statistics."""

    def analyze(self, df: pd.DataFrame) -> dict[str, Any]:
        """Compute sequence length distribution and AA frequency."""
        lengths = df["protein_sequence"].str.len()

        all_chars: Counter[str] = Counter()
        total_chars = 0
        for seq in df["protein_sequence"]:
            all_chars.update(seq)
            total_chars += len(seq)

        aa_frequency: dict[str, float] = {}
        if total_chars > 0:
            aa_frequency = {aa: count / total_chars for aa, count in sorted(all_chars.items())}

        return {
            "length_mean": float(lengths.mean()),
            "length_std": float(lengths.std()),
            "length_min": int(lengths.min()),
            "length_max": int(lengths.max()),
            "length_median": float(lengths.median()),
            "aa_frequency": aa_frequency,
        }

    def plot_length_distribution(self, df: pd.DataFrame) -> Any:
        """Histogram of sequence lengths."""
        lengths = df["protein_sequence"].str.len()
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(lengths, bins=30, edgecolor="black", alpha=0.7)
        ax.set_xlabel("Sequence Length")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Sequence Length Distribution (n={len(df)})")
        plt.close(fig)
        return fig
