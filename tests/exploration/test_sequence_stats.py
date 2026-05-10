"""Tests for SequenceAnalyzer — sequence length, AA frequency, mutations."""

import random

import pandas as pd
import pytest

from novozymes.exploration.sequence_stats import SequenceAnalyzer

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def make_sequence_df(n: int, min_len: int = 50, max_len: int = 500) -> pd.DataFrame:
    """Generate synthetic DataFrame with random protein sequences."""
    sequences = []
    for _ in range(n):
        length = random.randint(min_len, max_len)
        seq = "".join(random.choices(AMINO_ACIDS, k=length))
        sequences.append(seq)

    return pd.DataFrame(
        {
            "seq_id": [f"seq_{i:03d}" for i in range(n)],
            "protein_sequence": sequences,
            "pH": [7.0 + random.gauss(0, 1.0) for _ in range(n)],
            "data_source": ["A"] * n,
            "tm": [50.0 + random.gauss(0, 5.0) for _ in range(n)],
        }
    )


class TestSequenceAnalyzer:
    """SequenceAnalyzer computes sequence statistics."""

    @pytest.mark.parametrize("n", [10, 50, 200])
    def test_analyze_returns_dict(self, n: int) -> None:
        """analyze() returns dict for any dataset size."""
        df = make_sequence_df(n)
        analyzer = SequenceAnalyzer()
        result = analyzer.analyze(df)
        assert isinstance(result, dict)

    @pytest.mark.parametrize("n", [10, 50, 200])
    def test_analyze_has_length_stats(self, n: int) -> None:
        """Result contains length distribution stats."""
        df = make_sequence_df(n)
        analyzer = SequenceAnalyzer()
        result = analyzer.analyze(df)
        assert "length_mean" in result
        assert "length_std" in result
        assert "length_min" in result
        assert "length_max" in result

    def test_analyze_length_values_reasonable(self) -> None:
        """Length stats fall within expected range."""
        df = make_sequence_df(200, min_len=100, max_len=200)
        analyzer = SequenceAnalyzer()
        result = analyzer.analyze(df)
        assert result["length_min"] >= 100
        assert result["length_max"] <= 200
        assert 100 <= result["length_mean"] <= 200

    @pytest.mark.parametrize("n", [10, 50, 200])
    def test_analyze_has_aa_frequency(self, n: int) -> None:
        """Result contains amino acid frequency table."""
        df = make_sequence_df(n)
        analyzer = SequenceAnalyzer()
        result = analyzer.analyze(df)
        assert "aa_frequency" in result
        assert isinstance(result["aa_frequency"], dict)

    def test_aa_frequency_sums_to_one(self) -> None:
        """AA frequencies should sum to approximately 1.0."""
        df = make_sequence_df(100)
        analyzer = SequenceAnalyzer()
        result = analyzer.analyze(df)
        total = sum(result["aa_frequency"].values())
        assert abs(total - 1.0) < 0.01

    def test_plot_length_distribution_returns_figure(self) -> None:
        """plot_length_distribution() returns a matplotlib figure."""
        df = make_sequence_df(100)
        analyzer = SequenceAnalyzer()
        fig = analyzer.plot_length_distribution(df)
        assert fig is not None
