"""Label distribution analysis."""

from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


class LabelAnalyzer:
    """Analyze label (tm) distribution."""

    def analyze(self, df: pd.DataFrame) -> dict[str, Any]:
        """Compute label statistics."""
        tm = df["tm"]
        return {
            "mean": float(tm.mean()),
            "std": float(tm.std()),
            "min": float(tm.min()),
            "max": float(tm.max()),
            "median": float(tm.median()),
            "quantile_05": float(tm.quantile(0.05)),
            "quantile_25": float(tm.quantile(0.25)),
            "quantile_75": float(tm.quantile(0.75)),
            "quantile_95": float(tm.quantile(0.95)),
            "count": int(len(tm)),
        }

    def detect_outliers(
        self,
        df: pd.DataFrame,
        iqr_multiplier: float = 1.5,
    ) -> pd.DataFrame:
        """Detect outliers using IQR method (robust to outliers).

        Values beyond Q1 - iqr_multiplier*IQR or Q3 + iqr_multiplier*IQR are outliers.
        Default iqr_multiplier=1.5 is standard (1.0 for extreme outliers).
        """
        tm = df["tm"]
        q1 = tm.quantile(0.25)
        q3 = tm.quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - iqr_multiplier * iqr
        upper_bound = q3 + iqr_multiplier * iqr
        mask = (tm < lower_bound) | (tm > upper_bound)
        result: pd.DataFrame = df[mask]
        return result

    def plot_histogram(self, df: pd.DataFrame) -> Any:
        """Plot histogram of tm values."""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.hist(df["tm"], bins=30, edgecolor="black", alpha=0.7)
        ax.set_xlabel("Thermostability (tm)")
        ax.set_ylabel("Frequency")
        ax.set_title(f"Label Distribution (n={len(df)})")
        plt.close(fig)
        return fig
