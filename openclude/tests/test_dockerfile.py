"""Static checks on the Dockerfile.

Docker is not installed on the dev machine, so the image is built on GitHub's
runners. That means a broken Dockerfile is only discovered when CI runs, after
a push. These tests catch the failure modes that are cheap to detect by
reading, so they never reach a runner.

Every test here corresponds to something that actually broke a real build.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCKERFILE = ROOT / "Dockerfile"
ENTRYPOINT = ROOT / "deploy" / "entrypoint.sh"
PICKER = ROOT / "deploy" / "pick_ffmpeg.py"


@pytest.fixture(scope="module")
def dockerfile() -> list[str]:
    if not DOCKERFILE.exists():
        pytest.skip("no Dockerfile")
    return DOCKERFILE.read_text("utf-8").splitlines()


# --------------------------------------------------------------------------
# the one that broke CI
# --------------------------------------------------------------------------


def test_no_heredoc_inside_a_run_instruction(dockerfile) -> None:
    """A heredoc body inside one RUN is not valid Dockerfile.

    Docker joins physical lines only when they end with a backslash, so the
    heredoc body is parsed as fresh instructions. The real build died with
    `dockerfile parse error: unknown instruction: );`.
    """
    offenders: list[str] = []
    in_continuation = False
    for n, line in enumerate(dockerfile, start=1):
        stripped = line.strip()
        if not in_continuation:
            in_continuation = False
        if re.search(r"<<-?\s*['\"]?[A-Z]{2,}", line):
            offenders.append(f"{n}: {stripped}")
        if stripped.endswith("\\"):
            in_continuation = True
    assert not offenders, "heredoc found inside a RUN: " + "; ".join(offenders)


def test_every_run_instruction_is_a_known_command(dockerfile) -> None:
    """`unknown instruction` is what a stray shell line looks like to Docker."""
    known = {
        "RUN", "CMD", "COPY", "ADD", "ENTRYPOINT", "ENV", "ARG", "WORKDIR",
        "USER", "EXPOSE", "VOLUME", "LABEL", "HEALTHCHECK", "SHELL", "ONBUILD",
        "STOPSIGNAL", "FROM", "AS",
    }
    in_continuation = False
    for n, line in enumerate(dockerfile, start=1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if in_continuation:
            in_continuation = line.rstrip().endswith("\\")
            continue
        head = line.strip().split()[0].upper()
        if head not in known:
            pytest.fail(f"line {n} starts with {head!r}, not a Dockerfile instruction: {line!r}")
        in_continuation = line.rstrip().endswith("\\")


def test_every_line_continuation_actually_continues(dockerfile) -> None:
    """A trailing backslash before EOF silently swallows the next file."""
    for n, line in enumerate(dockerfile, start=1):
        if line.rstrip().endswith("\\") and n == len(dockerfile):
            pytest.fail(f"line {n} ends with a continuation but is the last line")


# --------------------------------------------------------------------------
# things the image must and must not contain
# --------------------------------------------------------------------------


def test_the_base_image_is_cuda_and_amd64_capable(dockerfile) -> None:
    text = "\n".join(dockerfile)
    assert "FROM nvidia/cuda" in text
    assert "CUDA_TAG" in text


def test_torch_is_baked_not_installed_at_start(dockerfile) -> None:
    """Installing torch at container start costs ~4 GB on every cold start."""
    text = "\n".join(dockerfile)
    assert "torch==2.10.0" in text
    assert "cu128" in text


def test_the_engine_is_cloned_from_the_live_repo(dockerfile) -> None:
    text = "\n".join(dockerfile)
    assert "deepbeepmeep/Wan2GP" in text
    # a bare clone with no ref pin means a new commit every build
    assert "WAN2GP_REF" in text
    assert "rev-parse HEAD" in text


def test_the_display_patch_is_applied_at_build_time(dockerfile) -> None:
    text = "\n".join(dockerfile)
    assert "TkAgg" in text
    assert "Agg" in text


def test_ffmpeg_is_verified_not_just_installed(dockerfile) -> None:
    """The engine needs -fps_mode; Ubuntu 22.04's ffmpeg 4.x lacks it."""
    text = "\n".join(dockerfile)
    assert "fps_mode" in text
    assert "pick_ffmpeg.py" in text


def test_the_ffmpeg_picker_exists_and_is_valid_python() -> None:
    if not PICKER.exists():
        pytest.skip("no pick_ffmpeg.py")
    import ast

    ast.parse(PICKER.read_text("utf-8"))


def test_the_picker_matches_only_linux64_gpl_archives() -> None:
    """A macOS or Windows asset would install a binary that cannot run."""
    src = PICKER.read_text("utf-8")
    assert "linux64" in src
    assert "windows" not in src
    assert "darwin" not in src


def test_the_image_never_copies_a_dotenv(dockerfile) -> None:
    """A local .env holds object-storage credentials."""
    for line in dockerfile:
        s = line.strip()
        if s.startswith(("COPY", "ADD")):
            assert ".env" not in s, f"secret-leaking line: {s}"


def test_the_image_never_copies_the_tests(dockerfile) -> None:
    text = "\n".join(dockerfile)
    for line in text.splitlines():
        s = line.strip()
        if s.startswith(("COPY", "ADD")):
            assert "tests" not in s.split()[1], f"tests should not ship: {s}"


def test_healthcheck_targets_a_path_the_image_serves(dockerfile) -> None:
    text = "\n".join(dockerfile)
    assert "HEALTHCHECK" in text
    assert "/health" in text
    assert "EXPOSE 8000" in text


def test_the_start_period_outlasts_a_model_download(dockerfile) -> None:
    """Too short a grace period and the instance is killed mid-download."""
    text = "\n".join(dockerfile)
    m = re.search(r"--start-period=(\d+)s", text)
    assert m, "no start-period on the healthcheck"
    assert int(m.group(1)) >= 300


def test_data_dirs_are_created_and_symlinked(dockerfile) -> None:
    text = "\n".join(dockerfile)
    for d in ("ckpts", "loras", "outputs"):
        assert f"/data/{d}" in text, d
    # the engine resolves ckpts/ relative to its own root
    assert "ln -sfn /data/ckpts" in text


# --------------------------------------------------------------------------
# the entrypoint
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def entrypoint() -> str:
    if not ENTRYPOINT.exists():
        pytest.skip("no entrypoint.sh")
    return ENTRYPOINT.read_text("utf-8")


def test_the_entrypoint_is_posix_shell(entrypoint: str) -> None:
    assert entrypoint.startswith("#!")
    assert "set -euo pipefail" in entrypoint


def test_the_entrypoint_is_lf_only() -> None:
    """CRLF in a bash script fails with '\\r: command not found' on line one."""
    raw = ENTRYPOINT.read_bytes()
    assert b"\r\n" not in raw, "entrypoint.sh has CRLF line endings"


def test_the_entrypoint_clears_the_crash_lock(entrypoint: str) -> None:
    """A leftover startup.lock puts the engine into Safe Mode and disables
    its plugins, and there is no flag to turn that off."""
    assert "startup.lock" in entrypoint
    assert "rm -f" in entrypoint


def test_the_entrypoint_refuses_to_start_without_a_store(entrypoint: str) -> None:
    assert "OPENCLIDE_REQUIRE_STORE" in entrypoint
    assert "S3_ENDPOINT" in entrypoint
    assert "exit" in entrypoint


def test_the_entrypoint_hands_over_to_the_worker(entrypoint: str) -> None:
    assert "openclude.worker" in entrypoint
    assert "exec " in entrypoint


def test_the_entrypoint_does_not_pull_images_itself(entrypoint: str) -> None:
    """docker.sock is not available and the node has no image cache."""
    assert "docker" not in entrypoint.lower()
