"""Module evaluation — Métriques et pipeline d'évaluation."""

from evaluation.metrics import (
    precision_at_k,
    recall_at_k,
    mean_reciprocal_rank,
    rouge_l_score,
    compute_faithfulness,
    abstention_accuracy,
)
from evaluation.run_eval import run_evaluation

__all__ = [
    "precision_at_k",
    "recall_at_k",
    "mean_reciprocal_rank",
    "rouge_l_score",
    "compute_faithfulness",
    "abstention_accuracy",
    "run_evaluation",
]
