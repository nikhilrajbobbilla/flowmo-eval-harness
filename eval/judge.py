"""
Minimal OpenAI-backed judge that extracts skills from a job-description string.

This is intentionally simple - the harness is the point, not the model. The
contract is just: input string -> set of skill strings. Swap this module for
any model you actually want to evaluate.

No key is read at import time. The OPENAI_API_KEY environment variable is
required only when run_eval() is called against a live model. The CI gate
runs against pre-recorded fixture outputs so no API key is needed there.
"""
from __future__ import annotations

import json
import os
from typing import Iterable

_SYSTEM = (
    "You are extracting the technical skills explicitly required by a job "
    "description. Return JSON {\"skills\": [\"...\", ...]}. Lowercase each "
    "skill. Include only skills literally present or directly implied (e.g. "
    "\"Spark\" implies \"PySpark\" only if pyspark is mentioned). Do not "
    "invent adjacent skills."
)


def extract_skills(jd_text: str, model: str = "gpt-4o-mini") -> list[str]:
    """Live LLM call. Requires OPENAI_API_KEY in env."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY not set. The CI gate uses fixture outputs and does "
            "not call live. Set the env var to run against a real model."
        )

    # Imported here so the module imports cleanly without the package installed.
    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    resp = client.chat.completions.create(
        model=model,
        temperature=0.0,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": _SYSTEM},
            {"role": "user", "content": jd_text},
        ],
    )
    content = resp.choices[0].message.content or "{}"
    parsed = json.loads(content)
    raw = parsed.get("skills", [])
    return [s.strip().lower() for s in raw if isinstance(s, str) and s.strip()]


def fixture_output_for(case_id: str, fixtures_path: str) -> list[str]:
    """Read a pre-recorded model output for the given case from fixtures.

    The CI gate uses this so it can run deterministically without API access.
    """
    with open(fixtures_path, "r", encoding="utf-8") as f:
        fixtures = json.load(f)
    return [s.strip().lower() for s in fixtures.get(case_id, [])]
