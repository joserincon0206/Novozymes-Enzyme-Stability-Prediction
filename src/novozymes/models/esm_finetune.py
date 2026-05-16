"""ESMFinetuneStrategy — ESM-2 with last N layers unfrozen + pluggable head."""

import torch
from torch import optim
from transformers import AutoModel, AutoTokenizer

from novozymes.config.model import ModelConfig
from novozymes.constants import LossEnum
from novozymes.models.base import ModelStrategy
from novozymes.models.heads.registry import build_head
from novozymes.training.losses.registry import build_loss


class ESMFinetuneStrategy(ModelStrategy):
    """ESM-2 with last N layers unfrozen + trainable head.

    Uses two param groups: backbone at esm_lr, head at learning_rate.
    """

    def __init__(
        self,
        config: ModelConfig,
        loss_enum: LossEnum,
        num_classes: int = 50,
    ) -> None:
        super().__init__()
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.esm_model_name)
        self.backbone = AutoModel.from_pretrained(config.esm_model_name)

        # Freeze all, then unfreeze last N layers
        for param in self.backbone.parameters():
            param.requires_grad = False

        layers = self.backbone.encoder.layer
        unfreeze_n = min(getattr(config, "unfreeze_last_n_layers", 2), len(layers))
        for layer in layers[-unfreeze_n:]:
            for param in layer.parameters():
                param.requires_grad = True

        esm_dim: int = self.backbone.config.hidden_size
        output_dim = esm_dim + 1  # +1 for pH

        loss = build_loss(loss_enum, num_classes=num_classes)
        self.head = build_head(loss.required_head, output_dim, num_classes)

    def forward(self, sequences: list[str], ph: torch.Tensor) -> torch.Tensor:
        """Tokenize, run backbone, mean-pool, concatenate pH, pass through head."""
        tokens = self.tokenizer(
            sequences,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        # Move tokens to same device as backbone
        device = next(self.backbone.parameters()).device
        tokens = {k: v.to(device) for k, v in tokens.items()}

        outputs = self.backbone(**tokens)

        attention_mask = tokens["attention_mask"].unsqueeze(-1)
        hidden_states = outputs.last_hidden_state
        masked = hidden_states * attention_mask
        summed = masked.sum(dim=1)
        counts = attention_mask.sum(dim=1)
        mean_pooled = summed / counts

        ph_col = ph.unsqueeze(-1).float().to(device)
        features = torch.cat([mean_pooled, ph_col], dim=-1)

        result: torch.Tensor = self.head(features)
        return result

    def configure_optimizers(self) -> list[dict[str, object]]:
        """Two param groups: backbone at esm_lr, head at learning_rate."""
        backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
        esm_lr = getattr(self.config, "esm_lr", 1e-5)
        optimizer = optim.AdamW(
            [
                {"params": backbone_params, "lr": esm_lr},
                {"params": self.head.parameters(), "lr": self.config.learning_rate},
            ]
        )
        return [{"optimizer": optimizer}]
