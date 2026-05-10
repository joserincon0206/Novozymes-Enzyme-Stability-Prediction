"""Tests for data exploration module."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from novozymes.exploration.label_stats import LabelAnalyzer

# ============================================================================
# Synthetic Data Generation
# ============================================================================


def make_label_csv(n: int, mean: float = 50.0, std: float = 5.0) -> Path:
    """Generate CSV with tm labels."""
    import random

    df = pd.DataFrame(
        {
            "seq_id": [f"seq_{i:03d}" for i in range(n)],
            "protein_sequence": ["MKLAVL"] * n,
            "pH": [7.0] * n,
            "data_source": ["A"] * n,
            "tm": [mean + random.gauss(0, std) for _ in range(n)],
        }
    )
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False)
    path = Path(f.name)
    f.close()
    df.to_csv(path, index=False)
    return path


# ============================================================================
# LabelAnalyzer Tests
# ============================================================================


class TestLabelAnalyzer:
    """LabelAnalyzer computes label statistics."""

    @pytest.mark.parametrize("n", [10, 50, 100])
    def test_analyze_returns_dict(self, n: int) -> None:
        """analyze() returns dict with stats for any dataset size."""
        csv_path = make_label_csv(n=n)
        df = pd.read_csv(csv_path)
        analyzer = LabelAnalyzer()
        stats = analyzer.analyze(df)
        assert isinstance(stats, dict)

    @pytest.mark.parametrize("n", [10, 50, 100])
    def test_analyze_has_required_keys(self, n: int) -> None:
        """Stats dict has mean, std, min, max, quantiles."""
        csv_path = make_label_csv(n=n)
        df = pd.read_csv(csv_path)
        analyzer = LabelAnalyzer()
        stats = analyzer.analyze(df)
        assert "mean" in stats
        assert "std" in stats
        assert "min" in stats
        assert "max" in stats

    def test_analyze_values_reasonable(self) -> None:
        """Stats are reasonable for normal distribution."""
        csv_path = make_label_csv(n=200, mean=50.0, std=5.0)
        df = pd.read_csv(csv_path)
        analyzer = LabelAnalyzer()
        stats = analyzer.analyze(df)
        assert 45.0 < stats["mean"] < 55.0  # Close to 50
        assert 3.0 < stats["std"] < 7.0  # Close to 5

    @pytest.mark.parametrize("n", [10, 50, 100])
    def test_detect_outliers_returns_dataframe(self, n: int) -> None:
        """detect_outliers() returns DataFrame for any size."""
        csv_path = make_label_csv(n=n)
        df = pd.read_csv(csv_path)
        analyzer = LabelAnalyzer()
        outliers = analyzer.detect_outliers(df, iqr_multiplier=1.5)
        assert isinstance(outliers, pd.DataFrame)

    def test_detect_outliers_finds_extreme_values(self) -> None:
        """Finds values outside IQR bounds (robust to outliers)."""
        df = pd.DataFrame(
            {
                "tm": [48.0, 48.0, 48.0, 49.0, 50.0, 51.0, 52.0, 49.0, 48.0, 20000.0],
            }
        )
        analyzer = LabelAnalyzer()
        outliers = analyzer.detect_outliers(df, iqr_multiplier=1.5)
        assert len(outliers) == 1

    @pytest.mark.parametrize("n", [10, 50, 100, 200])
    def test_plot_histogram_works_on_any_size(self, n: int) -> None:
        """plot_histogram() works on any dataset size."""
        csv_path = make_label_csv(n=n)
        df = pd.read_csv(csv_path)
        analyzer = LabelAnalyzer()
        fig = analyzer.plot_histogram(df)
        assert fig is not None
