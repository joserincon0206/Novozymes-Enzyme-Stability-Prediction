"""GroupBalanceAnalyzer — data_source balance and pH vs tm analysis."""

from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


class GroupBalanceAnalyzer:
    """Analyze data_source balance and pH-tm relationship."""

    def analyze(self, df: pd.DataFrame) -> dict[str, dict[str, float]]:
        """Per data_source: count, tm mean/std, pH mean/std."""
        result: dict[str, dict[str, float]] = {}
        for source, group in df.groupby("data_source"):
            result[str(source)] = {
                "count": len(group),
                "tm_mean": float(group["tm"].mean()),
                "tm_std": float(group["tm"].std()),
                "ph_mean": float(group["pH"].mean()),
                "ph_std": float(group["pH"].std()),
            }
        return result

    def plot_tm_by_source(self, df: pd.DataFrame) -> Any:
        """Boxplot of tm per data_source."""
        fig, ax = plt.subplots(figsize=(10, 6))
        sources = df["data_source"].unique()
        data = [df[df["data_source"] == s]["tm"].values for s in sources]
        ax.boxplot(data, labels=sources)
        ax.set_xlabel("Data Source")
        ax.set_ylabel("Thermostability (tm)")
        ax.set_title("tm Distribution by Data Source")
        plt.close(fig)
        return fig

    def plot_ph_vs_tm(self, df: pd.DataFrame) -> Any:
        """Scatter plot of pH vs tm."""
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.scatter(df["pH"], df["tm"], alpha=0.5, s=10)
        ax.set_xlabel("pH")
        ax.set_ylabel("Thermostability (tm)")
        ax.set_title("pH vs tm")
        plt.close(fig)
        return fig
