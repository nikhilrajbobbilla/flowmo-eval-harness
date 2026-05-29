"""
Set-overlap metrics for evaluating extracted entity sets against ground truth.

Precision, Recall, F1, and F0.5 - the four numbers that make set-based eval
debuggable. F0.5 is included because in many real-world eval problems
(extraction, retrieval, code review) precision hurts more than recall:
a wrong suggestion poisons trust faster than a missing one rebuilds it.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class PRF:
    precision: float
    recall: float
    f1: float
    f05: float
    true_positive: int
    false_positive: int
    false_negative: int


def _norm(s: str) -> str:
    return s.strip().lower()


def score(predicted: Iterable[str], ground_truth: Iterable[str]) -> PRF:
    """Compute precision, recall, F1, F0.5 between two string sets."""
    pred = {_norm(p) for p in predicted if p and p.strip()}
    truth = {_norm(g) for g in ground_truth if g and g.strip()}

    tp = len(pred & truth)
    fp = len(pred - truth)
    fn = len(truth - pred)

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    f1 = _f_beta(precision, recall, beta=1.0)
    f05 = _f_beta(precision, recall, beta=0.5)

    return PRF(
        precision=precision,
        recall=recall,
        f1=f1,
        f05=f05,
        true_positive=tp,
        false_positive=fp,
        false_negative=fn,
    )


def _f_beta(precision: float, recall: float, beta: float) -> float:
    """F-beta score. beta < 1 weights precision higher; beta > 1 weights recall."""
    if precision == 0 and recall == 0:
        return 0.0
    beta_sq = beta * beta
    return (1 + beta_sq) * precision * recall / (beta_sq * precision + recall)
