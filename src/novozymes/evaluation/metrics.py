"""Evaluation metrics — Evaluator ABC and SpearmanEvaluator."""

from abc import ABC, abstractmethod

import torch
from scipy.stats import spearmanr


class Evaluator(ABC):
    """Abstract evaluator."""

    @abstractmethod
    def compute(self, preds: torch.Tensor, targets: torch.Tensor) -> float:
        """Compute metric between predictions and targets."""


class SpearmanEvaluator(Evaluator):
    """Spearman rank correlation — the Kaggle evaluation metric."""

    def compute(self, preds: torch.Tensor, targets: torch.Tensor) -> float:
        """Compute Spearman's rho between predictions and targets."""
        result = spearmanr(preds.detach().cpu().numpy(), targets.detach().cpu().numpy())
        return float(result.statistic)
