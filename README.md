# flowmo-eval-harness

A small, opinionated LLM-as-judge evaluation harness with a golden-set regression suite, **F1 and F0.5 scoring**, and **categorized failure tagging**. The CI gate fails if any tracked aggregate drifts past its threshold, so prompt or model changes either improve the metrics or do not merge.

This is a public extract of the eval discipline I use inside [FlowMoAI](https://flowmoai.com) and a 6,500-user enterprise RAG system. The example task here is **skill extraction from job descriptions**, but the harness is generic — any "string-in, set-of-strings-out" model fits.

## Why this exists

LLM apps drift silently. A prompt tweak that looks like an improvement on three eyeballed examples can quietly tank precision on the ones nobody re-checks. An LLM-as-judge that runs ad-hoc in a notebook does not catch the regression at PR time, only at user-report time.

The fix is the same one every other quality regression problem has needed for thirty years: **a versioned ground-truth set, deterministic metrics, and a gate that runs on every PR**. The novel part for LLMs is that the gate also needs to tell you *why* a case failed, because a single aggregate score loses the signal you would use to actually fix the model.

## The four numbers

For every case, the harness reports:

| metric | what it answers |
|---|---|
| **Precision** | Of the items the model returned, what fraction were correct? |
| **Recall** | Of the items it should have returned, what fraction did it return? |
| **F1** | Harmonic mean of precision and recall (equal weight). |
| **F0.5** | Harmonic mean weighted toward precision (precision matters more). |

**Why F0.5 is in here:** in many real product surfaces (code review suggestions, extracted facts shown to a user, retrieval results passed to a second-stage model) a wrong item poisons trust faster than a missing one rebuilds it. F0.5 makes that asymmetry visible. If your model drops 0.05 F1 but holds F0.5, you might be okay; if it holds F1 but loses F0.5, you are quietly getting noisier.

## Categorized failure tagging

An aggregate score tells you "model B beat model A by 0.04 F1." It does not tell you whether B is winning because it covers more rare items or because A hallucinates a few common ones. Without that, you cannot pick the right intervention.

Every failure in this harness is tagged as one of:

- **`training_gap`** — the model missed a common-vocabulary item. The right fix is retraining, fine-tuning, or adding the term to your prompt's worked-example block.
- **`retrieval_issue`** — the model missed an item that was actually present in the retrieval surface (RAG context, tool schema, function manifest). The right fix is upstream — chunking, ranking, or the retrieval prompt.
- **`prompt_fix`** — the model produced an item that was *not* in ground truth. Usually a hallucination or a too-loose interpretation. Tighten the prompt's "only include" constraint.

The CI gate has per-category rate ceilings so you cannot accidentally trade one failure mode for another.

## Architecture

```
golden_set.jsonl    ──►  judge.py        ──►  predicted set
                        (extract skills)
                                                │
ground_truth ────────────────────────────────►  metrics.py     ──►  P / R / F1 / F0.5
                                                │
retrievable, common_vocab ──────────────────►  categorize.py  ──►  per-failure tag
                                                │
                                                ▼
                                          run_eval.py
                                          (compare against thresholds.json)
                                                │
                                                ▼
                                    pass / fail → exit code → CI gate
```

## Quick start

```bash
# 1. install
python -m pip install -r requirements.txt

# 2. run the gate in fixture mode (no API key needed)
python -m eval.run_eval --fixtures

# 3. run against a live model (requires OPENAI_API_KEY in your env)
export OPENAI_API_KEY=sk-...your-key...
python -m eval.run_eval --live

# 4. unit tests
pytest -q

# 5. scan for accidentally committed secrets
python scripts/scan_secrets.py
```

The CI workflow at `.github/workflows/eval-gate.yml` runs the secret scan, the unit tests, and the fixture-mode eval on every PR. **No API keys are needed in CI.**

## Repository layout

```
eval/
  golden_set.jsonl    versioned ground truth (5 example cases here)
  fixtures.json       pre-recorded model outputs for deterministic CI
  thresholds.json     per-metric and per-category failure ceilings
  metrics.py          P / R / F1 / F0.5
  categorize.py       training_gap / retrieval_issue / prompt_fix
  judge.py            example model wrapper (OpenAI gpt-4o-mini)
  run_eval.py         end-to-end runner; exit code is the gate

tests/
  test_metrics.py     metric edge cases + categorization behavior

scripts/
  scan_secrets.py     pre-push scan for committed keys

.github/workflows/
  eval-gate.yml       CI gate: secret scan + tests + eval
```

## Adapting it to your problem

Swap `eval/judge.py` for whatever model you actually want to evaluate. Anything that takes a string and returns an iterable of strings works. The metrics and categorization modules are model-agnostic.

To extend to per-case rubrics instead of set overlap (e.g. code review quality), keep the categorization vocabulary but replace `metrics.py` with a per-case judge prompt that returns structured `{score, justification}` JSON. The thresholds and gate logic stay the same.

## What is *not* here

This is the harness, not the model. The example skill-extraction task is intentionally small. Three things are deliberately omitted because they belong in the consuming app, not the harness:

- The production prompt for FlowMo's actual resume-tailoring pipeline.
- The full FlowMo golden set (much larger; some cases are user-derived).
- Any API keys, secrets, or live credentials. The `.gitignore` blocks `.env`, `*.api_keys`, and similar; `scripts/scan_secrets.py` runs in CI and locally.

## License

MIT. See `LICENSE`.
