"""Tests for Evaluator ABC and all registered evaluators."""

import pytest
import torch

from novozymes.evaluation.metrics import Evaluator, SpearmanEvaluator


class TestEvaluatorABC:
    """Evaluator ABC defines the interface."""

    def test_cannot_instantiate_abc(self) -> None:
        """ABC cannot be instantiated directly."""
        with pytest.raises(TypeError):
            Evaluator()  # type: ignore[abstract]


class TestSpearmanEvaluator:
    """SpearmanEvaluator computes Spearman rank correlation."""

    def test_perfect_correlation(self) -> None:
        """Perfectly ranked predictions give rho = 1.0."""
        evaluator = SpearmanEvaluator()
        preds = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        targets = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        rho = evaluator.compute(preds, targets)
        assert abs(rho - 1.0) < 1e-5

    def test_reversed_correlation(self) -> None:
        """Reversed predictions give rho = -1.0."""
        evaluator = SpearmanEvaluator()
        preds = torch.tensor([5.0, 4.0, 3.0, 2.0, 1.0])
        targets = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        rho = evaluator.compute(preds, targets)
        assert abs(rho - (-1.0)) < 1e-5

    def test_no_correlation(self) -> None:
        """Uncorrelated predictions give rho near 0."""
        evaluator = SpearmanEvaluator()
        preds = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        targets = torch.tensor([3.0, 5.0, 1.0, 4.0, 2.0])
        rho = evaluator.compute(preds, targets)
        assert abs(rho) < 0.5

    def test_returns_float(self) -> None:
        """compute() returns a float."""
        evaluator = SpearmanEvaluator()
        preds = torch.rand(10)
        targets = torch.rand(10)
        rho = evaluator.compute(preds, targets)
        assert isinstance(rho, float)

    @pytest.mark.parametrize("n", [5, 10, 50, 100])
    def test_rho_in_valid_range(self, n: int) -> None:
        """Spearman rho is always in [-1, 1]."""
        evaluator = SpearmanEvaluator()
        preds = torch.rand(n)
        targets = torch.rand(n)
        rho = evaluator.compute(preds, targets)
        assert -1.0 <= rho <= 1.0
