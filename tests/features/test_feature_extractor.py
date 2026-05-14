"""Tests for FeatureExtractor ABC and all registered extractors."""

import random

import pytest
import torch

from novozymes.constants import FeatureExtractorEnum
from novozymes.features.base import FeatureExtractor
from novozymes.features.registry import build_feature_extractor

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def make_sequences(n: int, length: int = 20) -> list[str]:
    """Generate random protein sequences."""
    return ["".join(random.choices(AMINO_ACIDS, k=length)) for _ in range(n)]


class TestFeatureExtractorABC:
    """FeatureExtractor ABC defines the interface."""

    def test_cannot_instantiate_abc(self) -> None:
        """ABC cannot be instantiated directly."""
        with pytest.raises(TypeError):
            FeatureExtractor()  # type: ignore[abstract]


class TestFeatureExtractorRegistry:
    """All registered extractors satisfy the FeatureExtractor contract."""

    @pytest.mark.parametrize("extractor_enum", FeatureExtractorEnum)
    def test_registry_returns_feature_extractor(self, extractor_enum: FeatureExtractorEnum) -> None:
        """Every registered extractor is a FeatureExtractor subclass."""
        extractor = build_feature_extractor(extractor_enum)
        assert isinstance(extractor, FeatureExtractor)

    @pytest.mark.parametrize("extractor_enum", FeatureExtractorEnum)
    def test_output_dim_is_positive(self, extractor_enum: FeatureExtractorEnum) -> None:
        """output_dim returns a positive integer."""
        extractor = build_feature_extractor(extractor_enum)
        assert extractor.output_dim > 0

    @pytest.mark.parametrize("extractor_enum", FeatureExtractorEnum)
    @pytest.mark.parametrize("batch_size", [1, 4])
    def test_extract_returns_correct_shape(
        self, extractor_enum: FeatureExtractorEnum, batch_size: int
    ) -> None:
        """extract() returns [B, D] tensor."""
        extractor = build_feature_extractor(extractor_enum)
        sequences = make_sequences(batch_size)
        ph = torch.rand(batch_size)
        embeddings = extractor.extract(sequences, ph)
        assert embeddings.shape == (batch_size, extractor.output_dim)

    @pytest.mark.parametrize("extractor_enum", FeatureExtractorEnum)
    def test_extract_returns_float_tensor(self, extractor_enum: FeatureExtractorEnum) -> None:
        """Embeddings are float tensors."""
        extractor = build_feature_extractor(extractor_enum)
        sequences = make_sequences(4)
        ph = torch.rand(4)
        embeddings = extractor.extract(sequences, ph)
        assert embeddings.dtype == torch.float32

    @pytest.mark.parametrize("extractor_enum", FeatureExtractorEnum)
    def test_extract_is_deterministic(self, extractor_enum: FeatureExtractorEnum) -> None:
        """Same input produces same output (frozen model)."""
        extractor = build_feature_extractor(extractor_enum)
        sequences = make_sequences(4)
        ph = torch.rand(4)
        out1 = extractor.extract(sequences, ph)
        out2 = extractor.extract(sequences, ph)
        assert torch.allclose(out1, out2)
