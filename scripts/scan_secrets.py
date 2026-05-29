"""
Pre-push safety check: scan the working tree for likely secrets.

Run via:  python scripts/scan_secrets.py

Exits 1 if any pattern matches. Patterns cover OpenAI / Anthropic / AWS
key formats and generic high-entropy 32+ char strings near "key" or "token".
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Skip generated and dependency dirs
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", "venv", ".venv", "node_modules"}

# Skip this script itself (it contains the patterns)
SKIP_FILES = {"scripts/scan_secrets.py"}

PATTERNS = [
    (r"sk-[A-Za-z0-9]{20,}", "OpenAI-style secret key"),
    (r"sk-ant-[A-Za-z0-9_\-]{20,}", "Anthropic API key"),
    (r"AKIA[0-9A-Z]{16}", "AWS access key ID"),
    (r"aws_secret_access_key\s*=\s*[A-Za-z0-9/+=]{20,}", "AWS secret access key"),
    (r"-----BEGIN [A-Z ]*PRIVATE KEY-----", "Private key block"),
    (r"(?i)(api[_-]?key|secret|token)\s*[:=]\s*[\"\'][A-Za-z0-9_\-]{32,}[\"\']",
     "Generic key/secret/token assignment"),
]


def scan_file(path: Path) -> list[tuple[int, str, str]]:
    hits: list[tuple[int, str, str]] = []
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return hits
    for lineno, line in enumerate(text.splitlines(), 1):
        for pattern, label in PATTERNS:
            if re.search(pattern, line):
                hits.append((lineno, label, line.strip()[:120]))
    return hits


def main() -> int:
    total = 0
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT).as_posix()
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if rel in SKIP_FILES:
            continue
        hits = scan_file(path)
        for lineno, label, line in hits:
            print(f"{rel}:{lineno}  {label}\n  {line}")
            total += 1
    if total:
        print(f"\n{total} potential secret(s) found. Do NOT push.")
        return 1
    print("scan_secrets: clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
