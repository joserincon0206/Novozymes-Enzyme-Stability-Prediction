"""BinAdvisor — recommends num_classes for CORAL/CoGOL ordinal losses."""

import math
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


class BinAdvisor:
    """Recommend bin counts for ordinal regression losses."""

    def recommend(self, tm_series: pd.Series) -> dict[str, int]:
        """Compute bin recommendations using multiple strategies.

        Returns dict with:
            sturges: Sturges rule (log2(n) + 1)
            freedman_diaconis: Freedman-Diaconis rule (2*IQR/n^(1/3))
            recommended: median of the two strategies
        """
        n = len(tm_series)
        sturges = math.ceil(math.log2(n) + 1)

        q1 = float(tm_series.quantile(0.25))
        q3 = float(tm_series.quantile(0.75))
        iqr = q3 - q1
        tm_range = float(tm_series.max() - tm_series.min())

        if iqr > 0:
            bin_width = 2.0 * iqr / (n ** (1.0 / 3.0))
            fd = max(1, math.ceil(tm_range / bin_width))
        else:
            fd = sturges

        recommended = (sturges + fd) // 2

        return {
            "sturges": sturges,
            "freedman_diaconis": fd,
            "recommended": max(1, recommended),
        }

    def plot_bins(self, tm_series: pd.Series, num_classes: int) -> Any:
        """Show histogram with bin edges overlaid."""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(tm_series, bins=num_classes, edgecolor="black", alpha=0.7)
        ax.set_xlabel("Thermostability (tm)")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Bin Distribution (num_classes={num_classes})")
        plt.close(fig)
        return fig
