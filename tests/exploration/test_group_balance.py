"""Tests for GroupBalanceAnalyzer — data_source balance and pH vs tm."""

import random

import pandas as pd
import pytest

from novozymes.exploration.group_balance import GroupBalanceAnalyzer


def make_grouped_df(
    sources: list[str],
    counts: list[int],
) -> pd.DataFrame:
    """Generate synthetic DataFrame with multiple data sources."""
    rows = []
    for source, count in zip(sources, counts, strict=False):
        for i in range(count):
            rows.append(
                {
                    "seq_id": f"{source}_{i:03d}",
                    "protein_sequence": "MKLAVL",
                    "pH": 7.0 + random.gauss(0, 1.0),
                    "data_source": source,
                    "tm": 50.0 + random.gauss(0, 5.0),
                }
            )
    return pd.DataFrame(rows)


class TestGroupBalanceAnalyzer:
    """GroupBalanceAnalyzer checks data_source balance."""

    @pytest.mark.parametrize(
        "sources,counts",
        [
            (["A", "B"], [50, 50]),
            (["A", "B", "C"], [100, 50, 25]),
            (["A"], [200]),
        ],
    )
    def test_analyze_returns_dict(self, sources: list[str], counts: list[int]) -> None:
        """analyze() returns dict for any source configuration."""
        df = make_grouped_df(sources, counts)
        analyzer = GroupBalanceAnalyzer()
        result = analyzer.analyze(df)
        assert isinstance(result, dict)

    @pytest.mark.parametrize(
        "sources,counts",
        [
            (["A", "B"], [50, 50]),
            (["A", "B", "C"], [100, 50, 25]),
        ],
    )
    def test_analyze_covers_all_sources(self, sources: list[str], counts: list[int]) -> None:
        """Result has an entry for every data_source."""
        df = make_grouped_df(sources, counts)
        analyzer = GroupBalanceAnalyzer()
        result = analyzer.analyze(df)
        for source in sources:
            assert source in result

    def test_analyze_per_source_has_stats(self) -> None:
        """Each source entry has count, tm_mean, tm_std, ph_mean, ph_std."""
        df = make_grouped_df(["A", "B"], [100, 100])
        analyzer = GroupBalanceAnalyzer()
        result = analyzer.analyze(df)
        for source_stats in result.values():
            assert "count" in source_stats
            assert "tm_mean" in source_stats
            assert "tm_std" in source_stats
            assert "ph_mean" in source_stats
            assert "ph_std" in source_stats

    def test_analyze_counts_match(self) -> None:
        """Reported counts match actual group sizes."""
        df = make_grouped_df(["A", "B", "C"], [100, 50, 25])
        analyzer = GroupBalanceAnalyzer()
        result = analyzer.analyze(df)
        assert result["A"]["count"] == 100
        assert result["B"]["count"] == 50
        assert result["C"]["count"] == 25

    def test_plot_tm_by_source_returns_figure(self) -> None:
        """plot_tm_by_source() returns a matplotlib figure."""
        df = make_grouped_df(["A", "B"], [50, 50])
        analyzer = GroupBalanceAnalyzer()
        fig = analyzer.plot_tm_by_source(df)
        assert fig is not None

    def test_plot_ph_vs_tm_returns_figure(self) -> None:
        """plot_ph_vs_tm() returns a matplotlib figure."""
        df = make_grouped_df(["A", "B"], [50, 50])
        analyzer = GroupBalanceAnalyzer()
        fig = analyzer.plot_ph_vs_tm(df)
        assert fig is not None
