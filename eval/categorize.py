"""
Categorize each eval failure into one of three buckets:

  - training_gap   : ground-truth item missed; concept appears common enough
                     that retraining or fine-tuning the model should fix it.
  - retrieval_issue: ground-truth item missed; relevant context exists in
                     the knowledge surface but did not reach the prompt.
  - prompt_fix    : model produced an item NOT in ground truth (hallucination
                     or misclassification); usually fixable by tightening the
                     prompt or adding a structural rule.

The point of categorization is debuggable failure signal. An aggregate
LLM-judge score tells you "C scored 4.07 and B scored 3.65" but not which
of the three problems is responsible. Categorized signal lets you pick the
right intervention without re-reading every failure case.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class CategorizedFailure:
    item: str
    category: str  # "training_gap" | "retrieval_issue" | "prompt_fix"
    reason: str


def categorize(
    predicted: Iterable[str],
    ground_truth: Iterable[str],
    retrievable: Iterable[str] | None = None,
    common_vocabulary: Iterable[str] | None = None,
) -> list[CategorizedFailure]:
    """Categorize precision/recall failures.

    Args:
        predicted: items the model returned
        ground_truth: items the model should have returned
        retrievable: items present in the retrieval surface (RAG context,
                     tool schema, etc.) - drives retrieval_issue vs training_gap
                     attribution
        common_vocabulary: items the model has been shown frequently during
                           training - if a missed item is in here, it's a
                           training gap, otherwise retrieval issue
    """
    pred = {p.strip().lower() for p in predicted if p.strip()}
    truth = {g.strip().lower() for g in ground_truth if g.strip()}
    retr = {r.strip().lower() for r in (retrievable or []) if r.strip()}
    common = {c.strip().lower() for c in (common_vocabulary or []) if c.strip()}

    failures: list[CategorizedFailure] = []

    # False negatives (recall failures): in truth, not in pred
    for item in truth - pred:
        if item in retr:
            failures.append(CategorizedFailure(
                item=item,
                category="retrieval_issue",
                reason="present in retrievable surface but not surfaced to model",
            ))
        elif item in common:
            failures.append(CategorizedFailure(
                item=item,
                category="training_gap",
                reason="common-vocabulary item the model did not produce",
            ))
        else:
            # Default: treat as retrieval issue (model may not have had the signal)
            failures.append(CategorizedFailure(
                item=item,
                category="retrieval_issue",
                reason="missing item not in common vocabulary; assume context gap",
            ))

    # False positives (precision failures): in pred, not in truth
    for item in pred - truth:
        failures.append(CategorizedFailure(
            item=item,
            category="prompt_fix",
            reason="hallucinated or out-of-scope item; tighten prompt constraints",
        ))

    return failures


def summary(failures: list[CategorizedFailure]) -> dict[str, int]:
    """Count failures by category."""
    counts: dict[str, int] = {
        "training_gap": 0,
        "retrieval_issue": 0,
        "prompt_fix": 0,
    }
    for f in failures:
        counts[f.category] = counts.get(f.category, 0) + 1
    return counts
