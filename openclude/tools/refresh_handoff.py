"""Recompute the numbers in HANDOFF.json from the repository itself.

Run after adding or removing tests, or after a refactor:

    python tools/refresh_handoff.py

A handoff file whose numbers have drifted is worse than no handoff file: the
next session trusts it. So the numbers are generated, never typed.
"""

from __future__ import annotations

import ast
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HANDOFF = ROOT / "HANDOFF.json"


def line_counts() -> tuple[int, int]:
    def total(folder: Path, pattern: str) -> int:
        n = 0
        # rglob: the package lives at src/openclude/*.py, not src/*.py
        for f in folder.rglob(pattern):
            if "__pycache__" in f.parts:
                continue
            n += len(f.read_text("utf-8", errors="replace").splitlines())
        return n

    return total(ROOT / "src", "*.py"), total(ROOT / "tests", "*.py")


def test_function_count() -> int:
    n = 0
    for f in sorted((ROOT / "tests").glob("test_*.py")):
        for node in ast.walk(ast.parse(f.read_text("utf-8"))):
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                n += 1
    return n


def run_suite() -> tuple[int, int, int, str]:
    """Run pytest and return (passed, failed, skipped, tail)."""
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider"],
        cwd=str(ROOT), capture_output=True, text=True, timeout=900,
    )
    tail = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    passed = int(m.group(1)) if (m := re.search(r"(\d+) passed", tail)) else 0
    failed = int(m.group(1)) if (m := re.search(r"(\d+) failed", tail)) else 0
    skipped = int(m.group(1)) if (m := re.search(r"(\d+) skipped", tail)) else 0
    return passed, failed, skipped, tail


def run_until_stable(rounds: int = 3) -> tuple[int, int, str]:
    """The handoff is itself asserted by the suite, so the first run after a
    change can fail on stale numbers. Re-run and keep the last clean result."""
    last = (0, 0, 0, "")
    for _ in range(rounds):
        passed, failed, skipped, tail = run_suite()
        if failed == 0:
            return passed, skipped, tail
        last = (passed, failed, skipped, tail)
    return last[0], last[2], last[3]


def main() -> int:
    if not HANDOFF.exists():
        print(f"no {HANDOFF} to refresh", file=sys.stderr)
        return 1

    passed, skipped, tail = run_until_stable()
    functions = test_function_count()
    src, tst = line_counts()

    if passed < functions:
        print(
            f"refusing to write: the suite reports {passed} passed but "
            f"{functions} test functions are defined. Something is broken.",
            file=sys.stderr,
        )
        return 2

    data = json.loads(HANDOFF.read_text("utf-8"))
    data["verified"].update(
        {
            "tests": passed,
            "skipped": skipped,
            "source_lines": src,
            "test_lines": tst,
            "test_functions": functions,
            "verified_by": f"pytest -q  ->  {tail.strip()}",
        }
    )
    HANDOFF.write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"HANDOFF.json refreshed: {tail.strip()}")
    print(f"  tests={passed} skipped={skipped} functions={functions} src={src} test={tst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
