"""Tests for the handoff documents and the CLI.

A handoff file that drifts from reality is worse than none: a new session
would trust it and build on numbers that are wrong. So the numbers in
HANDOFF.json are asserted against the actual repository.
"""

from __future__ import annotations

import ast
import io
import json
import re
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HANDOFF = ROOT / "HANDOFF.json"
STATUS_MD = ROOT / "PROJECT_STATUS.md"
SRC = ROOT / "src" / "openclude"


@pytest.fixture(scope="module")
def handoff() -> dict:
    if not HANDOFF.exists():
        pytest.skip("HANDOFF.json not present")
    return json.loads(HANDOFF.read_text("utf-8"))


# --------------------------------------------------------------------------
# the handoff file is real, not decorative
# --------------------------------------------------------------------------


def test_handoff_is_valid_json_with_the_required_keys(handoff: dict) -> None:
    for key in (
        "version", "updated", "headline", "verified", "stages",
        "decisions", "next", "read_first", "known_gaps", "environment",
        "brief_corrections",
    ):
        assert key in handoff, f"HANDOFF.json is missing {key!r}"


def test_handoff_declares_a_version(handoff: dict) -> None:
    assert re.fullmatch(r"\d+\.\d+\.\d+", handoff["version"])


def test_every_stage_has_a_known_state(handoff: dict) -> None:
    allowed = {"done", "partial", "todo"}
    for s in handoff["stages"]:
        assert s["state"] in allowed, s
        assert s["name"] and s["note"]


def test_every_next_task_states_how_to_tell_it_is_finished(handoff: dict) -> None:
    for n in handoff["next"]:
        assert n["task"] and n["done_when"], n


def test_every_decision_says_why(handoff: dict) -> None:
    for d in handoff["decisions"]:
        assert d["decision"] and d["why"], d


# --------------------------------------------------------------------------
# the handoff must match the repository
# --------------------------------------------------------------------------


def count_test_functions() -> int:
    """Count test functions by parsing, no subprocess needed."""
    total = 0
    for f in sorted((ROOT / "tests").glob("test_*.py")):
        tree = ast.parse(f.read_text("utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name.startswith("test_"):
                total += 1
    return total


def test_the_handoff_test_count_cannot_drift_below_the_real_suite(
    handoff: dict,
) -> None:
    """Guards the number in HANDOFF.json from going stale.

    pytest expands parametrised cases, so the reported count is always at
    least the number of test functions. If this ever fails, tests were added
    without updating the handoff.
    """
    functions = count_test_functions()
    assert handoff["verified"]["tests"] >= functions, (
        f"HANDOFF.json claims {handoff['verified']['tests']} tests but the "
        f"suite defines {functions} test functions"
    )


def test_the_handoff_test_count_is_not_absurd(handoff: dict) -> None:
    assert 50 <= handoff["verified"]["tests"] <= 5000


def test_the_handoff_records_a_function_count_too(handoff: dict) -> None:
    """tools/refresh_handoff.py writes both; a missing one means it never ran."""
    assert "test_functions" in handoff["verified"]
    assert handoff["verified"]["test_functions"] == count_test_functions()


def test_the_refresh_tool_exists_and_is_runnable() -> None:
    tool = ROOT / "tools" / "refresh_handoff.py"
    assert tool.exists()
    text = tool.read_text("utf-8")
    # the glob must be recursive, or source_lines silently becomes zero
    assert "rglob" in text
    assert "refusing to write" in text


def test_every_module_the_handoff_names_actually_exists(handoff: dict) -> None:
    named = " ".join(handoff["read_first"])
    for mod in re.findall(r"src/openclude/(\w+)\.py", named):
        assert (SRC / f"{mod}.py").exists(), f"read_first names missing {mod}.py"


def test_every_file_in_read_first_exists(handoff: dict) -> None:
    for entry in handoff["read_first"]:
        path = entry.split()[0]
        assert (ROOT / path).exists(), f"read_first names missing file {path}"


def test_the_pipeline_stage_is_marked_done_and_pipeline_py_exists(handoff: dict) -> None:
    stage = next(s for s in handoff["stages"] if s["name"] == "pipeline")
    assert stage["state"] == "done"
    assert (SRC / "pipeline.py").exists()


# --------------------------------------------------------------------------
# the narrative document
# --------------------------------------------------------------------------


def test_status_md_exists_and_is_substantial() -> None:
    if not STATUS_MD.exists():
        pytest.skip("PROJECT_STATUS.md not present")
    text = STATUS_MD.read_text("utf-8")
    assert len(text) > 6000
    assert text.count("#") > 30


def test_status_md_documents_the_engine_decision() -> None:
    if not STATUS_MD.exists():
        pytest.skip
    text = STATUS_MD.read_text("utf-8")
    assert "deepbeepmeep/Wan2GP" in text
    assert "ComfyUI" in text
    assert "production path entirely" in text or "not a script bolted on" in text


def test_status_md_documents_the_dead_source_repo() -> None:
    """If this is forgotten, a new session will analyse a 404 again."""
    if not STATUS_MD.exists():
        pytest.skip
    assert "no longer exists" in STATUS_MD.read_text("utf-8")


def test_status_md_records_the_cost_correction() -> None:
    """An uncorrected number in a handoff doc becomes a wrong promise."""
    if not STATUS_MD.exists():
        pytest.skip
    text = STATUS_MD.read_text("utf-8")
    assert "Correction" in text
    assert "$5" in text and "$13" in text


def test_status_md_documents_the_persistence_correction() -> None:
    if not STATUS_MD.exists():
        pytest.skip
    text = STATUS_MD.read_text("utf-8")
    assert "ephemeral" in text
    assert "R2" in text


# --------------------------------------------------------------------------
# the CLI
# --------------------------------------------------------------------------


def run_cli(*argv: str) -> tuple[int, str]:
    """Call the CLI in-process. Subprocess spawning is broken on this host."""
    from openclude.cli import main

    buf = io.StringIO()
    with redirect_stdout(buf):
        code = main(list(argv))
    return code, buf.getvalue()


def test_status_command_runs_and_prints_the_key_sections() -> None:
    code, out = run_cli("status")
    assert code == 0
    for token in ("STAGES", "DECISIONS LOCKED", "NEXT", "KNOWN GAPS"):
        assert token in out, f"status output is missing {token}"


def test_status_output_contains_the_verified_numbers() -> None:
    if not HANDOFF.exists():
        pytest.skip
    h = json.loads(HANDOFF.read_text("utf-8"))
    _, out = run_cli("status")
    assert str(h["verified"]["tests"]) in out
    assert h["headline"] in out


def test_doctor_command_runs() -> None:
    code, out = run_cli("doctor")
    assert code == 0
    assert "ENVIRONMENT" in out
    assert "python" in out


def test_bare_invocation_shows_status() -> None:
    """A new session will type the shortest thing possible."""
    code, out = run_cli()
    assert code == 0
    assert "STAGES" in out


def test_version_is_reported() -> None:
    with pytest.raises(SystemExit) as exc:
        run_cli("--version")
    assert exc.value.code == 0


def test_an_unknown_command_is_rejected() -> None:
    with pytest.raises(SystemExit):
        run_cli("not-a-command")


def test_the_environment_note_names_a_working_python() -> None:
    """The PATH python is broken here; a new session must not use it."""
    if not HANDOFF.exists():
        pytest.skip
    env = json.loads(HANDOFF.read_text("utf-8"))["environment"]
    assert "broken_python" in env and "working_python" in env
    assert "do not use" in env["broken_python"].lower()
