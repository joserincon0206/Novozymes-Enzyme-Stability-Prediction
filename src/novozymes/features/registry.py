"""Feature extractor registry — maps FeatureExtractorEnum to implementations."""

from novozymes.constants import FeatureExtractorEnum
from novozymes.features.base import FeatureExtractor
from novozymes.features.esm_embedding import ESMEmbeddingExtractor

FEATURE_EXTRACTOR_REGISTRY: dict[FeatureExtractorEnum, type[FeatureExtractor]] = {
    FeatureExtractorEnum.ESM_EMBEDDING: ESMEmbeddingExtractor,
}


def build_feature_extractor(extractor_enum: FeatureExtractorEnum) -> FeatureExtractor:
    """Build a feature extractor from enum."""
    return FEATURE_EXTRACTOR_REGISTRY[extractor_enum]()
