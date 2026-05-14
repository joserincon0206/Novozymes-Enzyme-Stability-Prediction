"""ESMEmbeddingExtractor — frozen ESM-2 mean-pooled embeddings + pH."""

import torch
from transformers import AutoModel, AutoTokenizer

from novozymes.features.base import FeatureExtractor


class ESMEmbeddingExtractor(FeatureExtractor):
    """Extract mean-pooled ESM-2 embeddings concatenated with pH."""

    def __init__(self, model_name: str = "facebook/esm2_t6_8M_UR50D") -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self._esm_dim: int = self.model.config.hidden_size

    @property
    def output_dim(self) -> int:
        """ESM hidden size + 1 (pH)."""
        return self._esm_dim + 1

    def extract(self, sequences: list[str], ph: torch.Tensor) -> torch.Tensor:
        """Mean-pool ESM-2 token embeddings, concatenate pH.

        Args:
            sequences: protein sequences [B]
            ph: pH values [B]

        Returns:
            [B, esm_dim + 1] feature tensor
        """
        tokens = self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        with torch.no_grad():
            outputs = self.model(**tokens)

        # Mean-pool over sequence length, masking padding tokens
        attention_mask = tokens["attention_mask"].unsqueeze(-1)
        hidden_states = outputs.last_hidden_state
        masked = hidden_states * attention_mask
        summed = masked.sum(dim=1)
        counts = attention_mask.sum(dim=1)
        mean_pooled = summed / counts

        # Concatenate pH
        ph_col = ph.unsqueeze(-1).float()
        return torch.cat([mean_pooled, ph_col], dim=-1)
