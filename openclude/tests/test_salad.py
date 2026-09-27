"""Tests for the SaladCloud control script.

The dangerous property here is not "does it build a correct request" but
"can it ever spend money without being told to". These tests are mostly about
that.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from openclude import salad
from openclude.salad import Config, SaladError, group_spec, main

ROOT = Path(__file__).resolve().parents[1]


def env(**over) -> dict[str, str]:
    base = {
        "SALAD_API_KEY": "test-key-not-real",
        "SALAD_ORGANIZATION": "acme",
        "SALAD_PROJECT": "studio",
        "OPENCLIDE_IMAGE": "ghcr.io/eg250k-del/openclude:latest",
    }
    base.update(over)
    return base


@pytest.fixture
def clean_env(monkeypatch):
    for k in list(os.environ):
        if k.startswith(("SALAD_", "OPENCLIDE_", "S3_")):
            monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(os, "environ", env())
    yield


# --------------------------------------------------------------------------
# safety: nothing billable without an explicit flag
# --------------------------------------------------------------------------


def test_apply_without_the_flag_refuses(clean_env) -> None:
    assert main(["apply"]) == 2


def test_the_group_name_alone_never_applies(clean_env) -> None:
    """Even a well-formed invocation must not create anything by accident."""
    for argv in (["apply"], ["apply", "--yes"], ["APPLY"], ["apply", "-f"]):
        assert main(argv) != 0, argv


def test_preflight_and_plan_need_no_apply_flag(clean_env) -> None:
    assert main(["plan"]) == 0


# --------------------------------------------------------------------------
# credentials: never printed, never guessed
# --------------------------------------------------------------------------


def test_a_missing_key_is_a_clear_error_not_a_guess(clean_env) -> None:
    os.environ.pop("SALAD_API_KEY", None)
    with pytest.raises(SaladError, match="SALAD_API_KEY"):
        Config.from_env()


def test_a_missing_organization_is_never_inferred(clean_env) -> None:
    os.environ.pop("SALAD_ORGANIZATION", None)
    with pytest.raises(SaladError, match="SALAD_ORGANIZATION"):
        Config.from_env()


def test_every_missing_name_is_reported_at_once(clean_env) -> None:
    for k in ("SALAD_API_KEY", "SALAD_ORGANIZATION", "SALAD_PROJECT", "OPENCLIDE_IMAGE"):
        os.environ.pop(k, None)
    with pytest.raises(SaladError) as exc:
        Config.from_env()
    for name in ("SALAD_API_KEY", "SALAD_ORGANIZATION", "SALAD_PROJECT", "OPENCLIDE_IMAGE"):
        assert name in str(exc.value)


def test_the_error_tells_the_user_where_the_key_comes_from(clean_env) -> None:
    os.environ.pop("SALAD_API_KEY", None)
    with pytest.raises(SaladError, match="environment variable"):
        Config.from_env()


def test_the_key_is_never_in_the_repr(clean_env) -> None:
    """`api_key` is declared repr=False precisely so a traceback cannot leak it."""
    cfg = Config.from_env()
    assert "test-key-not-real" not in repr(cfg)
    assert "api_key" not in repr(cfg)


def test_the_key_is_never_in_the_group_spec(clean_env) -> None:
    spec = group_spec(Config.from_env())
    blob = json.dumps(spec)
    assert "test-key-not-real" not in blob


def test_no_secret_is_printed_by_plan(clean_env, capsys) -> None:
    os.environ["S3_SECRET_ACCESS_KEY"] = "super-secret-value"
    assert main(["plan"]) == 0
    out = capsys.readouterr().out
    assert "super-secret-value" not in out
    assert "***set***" in out


# --------------------------------------------------------------------------
# cost
# --------------------------------------------------------------------------


def test_minimum_replicas_is_zero_so_idle_costs_nothing(clean_env) -> None:
    """A running replica bills every second. Idle must be free."""
    cfg = Config.from_env()
    assert cfg.min_replicas == 0
    assert main(["plan"]) == 0


def test_plan_prints_the_cost_ceiling(clean_env, capsys) -> None:
    main(["plan"])
    out = capsys.readouterr().out
    assert "COST CEILING" in out
    assert "$0.00/h" in out


# --------------------------------------------------------------------------
# the spec itself
# --------------------------------------------------------------------------


def test_the_spec_targets_amd64(clean_env) -> None:
    assert group_spec(Config.from_env())["image"]["architecture"] == "amd64"


def test_the_spec_asks_for_profile_3_for_24gb(clean_env) -> None:
    """Profile 3 is the engine's own label for 24 GB VRAM; 5 is a 10 GB choice."""
    spec = group_spec(Config.from_env())
    assert spec["environment_variables"]["WAN2GP_PROFILE"] == "3"


def test_the_spec_uses_sage_attention(clean_env) -> None:
    assert group_spec(Config.from_env())["environment_variables"]["WAN2GP_ATTENTION"] == "sage2"


def test_the_spec_asks_for_a_large_disk(clean_env) -> None:
    """The node must have room for the weights plus a 720p working set."""
    assert group_spec(Config.from_env())["resources"]["storage"] >= 100_000


def test_the_startup_probe_allows_a_long_model_download(clean_env) -> None:
    """A short probe threshold kills the instance mid-download."""
    probe = group_spec(Config.from_env())["probes"]["startup"]["http"]
    assert probe["failure_threshold"] * probe["period_seconds"] >= 300


def test_the_spec_requires_object_storage(clean_env) -> None:
    """The container must refuse to start without a durable store."""
    assert group_spec(Config.from_env())["environment_variables"]["OPENCLIDE_REQUIRE_STORE"] == "1"


def test_the_spec_restarts_on_failure(clean_env) -> None:
    assert group_spec(Config.from_env())["restart"]["condition"] == "always"


def test_the_spec_exposes_one_http_port(clean_env) -> None:
    net = group_spec(Config.from_env())["network"]
    assert net["port"] == 8000
    assert net["protocol"] == "http"


def test_the_spec_is_json_serialisable(clean_env) -> None:
    json.dumps(group_spec(Config.from_env()))


# --------------------------------------------------------------------------
# dispatch
# --------------------------------------------------------------------------


def test_an_unknown_command_prints_help(clean_env) -> None:
    assert main([]) == 1
    assert main(["nonsense"]) == 1
    assert main(["preflight", "plan"]) == 1


def test_every_documented_command_is_implemented() -> None:
    for name in ("preflight", "plan", "apply", "status", "logs"):
        assert name in salad.COMMANDS
        assert callable(salad.COMMANDS[name])


def test_the_dockerfile_pins_the_engine() -> None:
    root = ROOT
    docker = root / "Dockerfile"
    if not docker.exists():
        pytest.skip("no Dockerfile")
    text = docker.read_text("utf-8")
    assert "deepbeepmeep/Wan2GP" in text
    assert "torch==2.10.0" in text
    # a build-time-only patch, with its verification
    assert "TkAgg" in text
    assert "fps_mode" in text          # the reason for the static ffmpeg


def test_the_entrypoint_refuses_to_start_without_a_store() -> None:
    sh = ROOT / "deploy" / "entrypoint.sh"
    if not sh.exists():
        pytest.skip("no entrypoint")
    text = sh.read_text("utf-8")
    assert "OPENCLIDE_REQUIRE_STORE" in text
    assert "startup.lock" in text      # the Safe Mode trap
    assert "set -euo pipefail" in text


def test_the_dockerfile_exposes_the_health_port() -> None:
    docker = ROOT / "Dockerfile"
    if not docker.exists():
        pytest.skip
    text = docker.read_text("utf-8")
    assert "EXPOSE 8000" in text
    assert "/health" in text
