from eval.categorize import categorize, summary
from eval.metrics import score


def test_perfect_match():
    prf = score(["a", "b", "c"], ["a", "b", "c"])
    assert prf.precision == 1.0
    assert prf.recall == 1.0
    assert prf.f1 == 1.0
    assert prf.f05 == 1.0


def test_case_insensitive_and_whitespace_tolerant():
    prf = score(["A ", " b", "C"], ["a", "b", "c"])
    assert prf.precision == 1.0
    assert prf.recall == 1.0


def test_partial_overlap():
    prf = score(["a", "b", "x"], ["a", "b", "c"])
    assert prf.true_positive == 2
    assert prf.false_positive == 1
    assert prf.false_negative == 1
    assert abs(prf.precision - 2 / 3) < 1e-9
    assert abs(prf.recall - 2 / 3) < 1e-9


def test_f05_weights_precision_more_than_recall():
    # Same precision and recall -> F1 == F0.5
    same = score(["a", "b"], ["a", "b"])
    assert same.f1 == same.f05

    # Higher recall than precision -> F1 should exceed F0.5
    pred = ["a", "b", "c", "d"]  # 2 correct + 2 wrong -> precision 0.5
    truth = ["a", "b"]            # both found -> recall 1.0
    prf = score(pred, truth)
    assert prf.precision == 0.5
    assert prf.recall == 1.0
    assert prf.f1 > prf.f05


def test_empty_inputs():
    prf = score([], [])
    assert prf.precision == 0.0
    assert prf.recall == 0.0
    assert prf.f1 == 0.0
    assert prf.f05 == 0.0


def test_categorize_recall_failure_is_retrieval_issue_when_in_retrievable():
    failures = categorize(
        predicted=["a"],
        ground_truth=["a", "b"],
        retrievable=["a", "b"],
        common_vocabulary=["a"],
    )
    assert len(failures) == 1
    assert failures[0].item == "b"
    assert failures[0].category == "retrieval_issue"


def test_categorize_recall_failure_is_training_gap_when_common_not_retrievable():
    failures = categorize(
        predicted=["a"],
        ground_truth=["a", "b"],
        retrievable=["a"],
        common_vocabulary=["a", "b"],
    )
    assert any(f.category == "training_gap" and f.item == "b" for f in failures)


def test_categorize_precision_failure_is_prompt_fix():
    failures = categorize(
        predicted=["a", "x"],
        ground_truth=["a"],
    )
    assert any(f.category == "prompt_fix" and f.item == "x" for f in failures)


def test_summary_counts_by_category():
    failures = categorize(
        predicted=["a", "x"],
        ground_truth=["a", "b"],
        retrievable=["a", "b"],
    )
    s = summary(failures)
    assert s["retrieval_issue"] == 1
    assert s["prompt_fix"] == 1
    assert s["training_gap"] == 0
