"""Tests for BinAdvisor — recommends num_classes for CORAL/CoGOL."""

import random

import pandas as pd
import pytest

from novozymes.exploration.bin_advisor import BinAdvisor


def make_tm_series(n: int, mean: float = 50.0, std: float = 10.0) -> pd.Series:
    """Generate synthetic tm series."""
    return pd.Series([mean + random.gauss(0, std) for _ in range(n)])


class TestBinAdvisor:
    """BinAdvisor recommends bin counts for ordinal losses."""

    @pytest.mark.parametrize("n", [50, 200, 1000])
    def test_recommend_returns_dict(self, n: int) -> None:
        """recommend() returns dict with binning strategies."""
        tm = make_tm_series(n)
        advisor = BinAdvisor()
        result = advisor.recommend(tm)
        assert isinstance(result, dict)

    @pytest.mark.parametrize("n", [50, 200, 1000])
    def test_recommend_has_required_keys(self, n: int) -> None:
        """Result contains sturges, freedman_diaconis, and recommended."""
        tm = make_tm_series(n)
        advisor = BinAdvisor()
        result = advisor.recommend(tm)
        assert "sturges" in result
        assert "freedman_diaconis" in result
        assert "recommended" in result

    @pytest.mark.parametrize("n", [50, 200, 1000])
    def test_recommend_values_are_positive_ints(self, n: int) -> None:
        """All recommended bin counts are positive integers."""
        tm = make_tm_series(n)
        advisor = BinAdvisor()
        result = advisor.recommend(tm)
        for key in ("sturges", "freedman_diaconis", "recommended"):
            assert isinstance(result[key], int)
            assert result[key] > 0

    def test_sturges_formula_correctness(self) -> None:
        """Sturges rule: ceil(log2(n) + 1)."""
        import math

        for n in [64, 128, 256, 512]:
            tm = make_tm_series(n)
            advisor = BinAdvisor()
            result = advisor.recommend(tm)
            expected = math.ceil(math.log2(n) + 1)
            assert result["sturges"] == expected

    @pytest.mark.parametrize("num_classes", [10, 20, 50])
    def test_plot_bins_returns_figure(self, num_classes: int) -> None:
        """plot_bins() returns a matplotlib figure."""
        tm = make_tm_series(200)
        advisor = BinAdvisor()
        fig = advisor.plot_bins(tm, num_classes)
        assert fig is not None
