"""
End-to-end eval runner. Two modes:

  python -m eval.run_eval --fixtures   (CI-friendly, no API key needed)
  python -m eval.run_eval --live       (calls OpenAI, requires OPENAI_API_KEY)

In both modes the runner:
  1. Loads golden_set.jsonl
  2. Produces a predicted skill set per case
  3. Scores P / R / F1 / F0.5 per case and in aggregate
  4. Categorizes failures into training_gap / retrieval_issue / prompt_fix
  5. Compares aggregates against thresholds.json
  6. Exits non-zero if any threshold is breached - this is the CI gate

The point: every prompt or model change must keep these numbers within
tolerance, or the PR cannot merge. Drift gets caught at the source.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from eval.categorize import categorize, summary
from eval.judge import extract_skills, fixture_output_for
from eval.metrics import score

ROOT = Path(__file__).parent
GOLDEN_PATH = ROOT / "golden_set.jsonl"
FIXTURES_PATH = ROOT / "fixtures.json"
THRESHOLDS_PATH = ROOT / "thresholds.json"


def _load_golden() -> list[dict]:
    cases: list[dict] = []
    with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    return cases


def _load_thresholds() -> dict:
    with open(THRESHOLDS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def run(mode: str = "fixtures") -> int:
    cases = _load_golden()
    thresholds = _load_thresholds()

    total_p = total_r = total_f1 = total_f05 = 0.0
    cat_counts = {"training_gap": 0, "retrieval_issue": 0, "prompt_fix": 0}
    total_failures = 0
    total_items = 0

    print(f"Running {len(cases)} cases in {mode} mode\n")

    for case in cases:
        case_id = case["id"]
        truth = case["ground_truth"]
        retrievable = case.get("retrievable", [])
        common = case.get("common_vocabulary", [])

        if mode == "live":
            predicted = extract_skills(case["jd"])
        else:
            predicted = fixture_output_for(case_id, str(FIXTURES_PATH))

        prf = score(predicted, truth)
        failures = categorize(predicted, truth, retrievable, common)
        cat = summary(failures)

        total_p += prf.precision
        total_r += prf.recall
        total_f1 += prf.f1
        total_f05 += prf.f05
        for k, v in cat.items():
            cat_counts[k] = cat_counts.get(k, 0) + v
        total_failures += len(failures)
        total_items += len(truth)

        print(f"  {case_id:25s}  P={prf.precision:.2f}  R={prf.recall:.2f}  F1={prf.f1:.2f}  F0.5={prf.f05:.2f}  failures={len(failures)}")
        for fail in failures:
            print(f"      - {fail.category:18s}  {fail.item}")

    n = len(cases)
    avg_p = total_p / n
    avg_r = total_r / n
    avg_f1 = total_f1 / n
    avg_f05 = total_f05 / n
    denom = max(total_items, 1)
    rates = {
        "training_gap_rate": cat_counts["training_gap"] / denom,
        "retrieval_issue_rate": cat_counts["retrieval_issue"] / denom,
        "prompt_fix_rate": cat_counts["prompt_fix"] / denom,
    }

    print(
        f"\nAGGREGATE  P={avg_p:.3f}  R={avg_r:.3f}  F1={avg_f1:.3f}  F0.5={avg_f05:.3f}"
    )
    print(
        "FAILURE RATES "
        f"training_gap={rates['training_gap_rate']:.2%}  "
        f"retrieval_issue={rates['retrieval_issue_rate']:.2%}  "
        f"prompt_fix={rates['prompt_fix_rate']:.2%}"
    )

    breaches: list[str] = []
    if avg_p < thresholds["min_precision"]:
        breaches.append(f"precision {avg_p:.3f} < {thresholds['min_precision']}")
    if avg_r < thresholds["min_recall"]:
        breaches.append(f"recall {avg_r:.3f} < {thresholds['min_recall']}")
    if avg_f1 < thresholds["min_f1"]:
        breaches.append(f"f1 {avg_f1:.3f} < {thresholds['min_f1']}")
    if avg_f05 < thresholds["min_f05"]:
        breaches.append(f"f0.5 {avg_f05:.3f} < {thresholds['min_f05']}")
    if rates["training_gap_rate"] > thresholds["max_training_gap_rate"]:
        breaches.append(
            f"training_gap_rate {rates['training_gap_rate']:.2%} > {thresholds['max_training_gap_rate']:.2%}"
        )
    if rates["retrieval_issue_rate"] > thresholds["max_retrieval_issue_rate"]:
        breaches.append(
            f"retrieval_issue_rate {rates['retrieval_issue_rate']:.2%} > {thresholds['max_retrieval_issue_rate']:.2%}"
        )
    if rates["prompt_fix_rate"] > thresholds["max_prompt_fix_rate"]:
        breaches.append(
            f"prompt_fix_rate {rates['prompt_fix_rate']:.2%} > {thresholds['max_prompt_fix_rate']:.2%}"
        )

    if breaches:
        print("\nGATE FAILED")
        for b in breaches:
            print(f"  {b}")
        return 1

    print("\nGATE PASSED")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the FlowMo eval harness")
    parser.add_argument(
        "--live",
        action="store_true",
        help="call the live model (requires OPENAI_API_KEY)",
    )
    parser.add_argument(
        "--fixtures",
        action="store_true",
        help="use pre-recorded fixture outputs (default, CI-friendly)",
    )
    args = parser.parse_args()

    mode = "live" if args.live else "fixtures"
    return run(mode=mode)


if __name__ == "__main__":
    sys.exit(main())
